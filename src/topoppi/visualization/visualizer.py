import json
import logging
import os
import re

import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import KDTree

from topoppi.config import VisualizationConfig
from topoppi.interactions.interaction_engine import load_prolif_data
from topoppi.interactions.metadata import INTERACTION_COLORS, INTERACTION_TYPES

logger = logging.getLogger("Visualizer")

class InterfaceVisualizer:
    def __init__(self, chain_A_atoms, chain_A_coords, chain_B_coords, chain_B_atoms=None, 
                 pdb_file=None, chain_a_id=None, chain_b_id=None, prolif_file=None, config: VisualizationConfig | None = None):
        """
        Args:
            chain_A_atoms: List of Bio.PDB Atom objects for Chain A
            chain_A_coords: Numpy array (N,3) for Chain A
            chain_B_coords: Numpy array (M,3) for Chain B
            chain_B_atoms: List of Bio.PDB Atom objects for Chain B
            chain_a_id: Chain ID string for A (e.g. "A")
            chain_b_id: Chain ID string for B (e.g. "B")
            prolif_file: Path to the ProLIF interaction JSON
        """
        self.config = config or VisualizationConfig()
        self.config.validate()
        self.atoms_A = chain_A_atoms
        self.coords_A = chain_A_coords
        self.coords_B = chain_B_coords
        self.atoms_B = chain_B_atoms
        self.residue_lookup_A = self._build_residue_lookup(self.atoms_A)
        self.residue_lookup_B = self._build_residue_lookup(self.atoms_B) if self.atoms_B else {}
        
        # Only build tree if we don't have external interaction data
        self.tree_B = KDTree(self.coords_B) if not prolif_file else None
        self.artist_map = {}

        # Ensure IDs are stripped of whitespace for comparison
        self.chain_a_id = str(chain_a_id).strip() if chain_a_id else None
        self.chain_b_id = str(chain_b_id).strip() if chain_b_id else None

        # --- Interaction categories (canonicalized from ProLIF output) ---
        self.interaction_types = list(INTERACTION_TYPES)
        self.interaction_colors = dict(INTERACTION_COLORS)
        self.patch_fill_palette = [
            '#5A8DEE', '#58C4A3', '#F6C667', '#E78AC3', '#7CC6FE',
            '#B8DE6F', '#FFA06B', '#A78BFA', '#7BDFF2', '#FF9AA2'
        ]

        # Data store: {res_seq: set([interaction_types]) }
        self.prolif_data = None
        self.prolif_partners = {}
        
        if prolif_file and os.path.exists(prolif_file):
            self._load_prolif_data(prolif_file)
        elif prolif_file:
            logger.warning(f"ProLIF file not found: {prolif_file}. Falling back to heuristics.")
            if self.config.use_geometric_interaction_fallback and self.tree_B is None:
                self.tree_B = KDTree(self.coords_B)

        # --- Fallback Chemical Definitions ---
        self.charged_pos = {'ARG', 'LYS', 'HIS'} 
        self.charged_neg = {'ASP', 'GLU'}
        self.aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        self.hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'CYS', 'TYR'}
        self.cation_atoms = {'NZ', 'NH1', 'NH2', 'ND1', 'NE2'} 
        self.anion_atoms = {'OD1', 'OD2', 'OE1', 'OE2', 'OXT'} 
        self.polar_atoms = {'N', 'O', 'S', 'F'}
        self.sulfur_atoms = {'SG', 'SD'}

    def _build_residue_lookup(self, atoms):
        lookup = {}
        if not atoms:
            return lookup
        for atom in atoms:
            parent = atom.get_parent()
            seq = parent.get_id()[1]
            if seq not in lookup:
                lookup[seq] = parent.get_resname()
        return lookup

    def _format_residue_label(self, res_name, res_seq):
        if res_name:
            return f"{str(res_name).title()}{res_seq}"
        return str(res_seq)

    def _build_label_text(self, mode, res_a_seq, res_a_name, partner_counts):
        a_label = self._format_residue_label(res_a_name, res_a_seq)
        if mode == 'chain_a':
            return a_label
        if not partner_counts:
            return a_label if mode == 'pair' else "N/A"
        best_b_seq = max(partner_counts.items(), key=lambda item: item[1])[0]
        b_name = self.residue_lookup_B.get(best_b_seq, "UNK")
        b_label = self._format_residue_label(b_name, best_b_seq)
        if mode == 'chain_b':
            return b_label
        return f"{a_label}-{b_label}"

    def _extract_seq_num(self, val):
        """Robustly extract integer sequence number from strings like '100', '100A'."""
        try:
            return int(val)
        except (ValueError, TypeError):
            match = re.match(r"^(-?\d+)", str(val))
            if match:
                return int(match.group(1))
            return None

    def _load_prolif_data(self, json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                expected_a = str(payload.get("chain_a", "")).strip()
                expected_b = str(payload.get("chain_b", "")).strip()
                if expected_a and self.chain_a_id and expected_a != self.chain_a_id:
                    logger.warning("ProLIF chain_a mismatch (%s != %s). Falling back to geometric heuristics.", expected_a, self.chain_a_id)
                    self.prolif_data = None
                    if self.tree_B is None:
                        self.tree_B = KDTree(self.coords_B)
                    return
                if expected_b and self.chain_b_id and expected_b != self.chain_b_id:
                    logger.warning("ProLIF chain_b mismatch (%s != %s). Falling back to geometric heuristics.", expected_b, self.chain_b_id)
                    self.prolif_data = None
                    if self.tree_B is None:
                        self.tree_B = KDTree(self.coords_B)
                    return
            data, partners = load_prolif_data(json_path)
            if not data:
                logger.warning("No ProLIF interactions found. Falling back to geometric heuristics.")
                self.prolif_data = None
                if self.tree_B is None:
                    self.tree_B = KDTree(self.coords_B)
                return
            self.prolif_data = data
            self.prolif_partners = partners
            logger.info(f"Loaded ProLIF interactions for {len(data)} Chain-{self.chain_a_id} residues.")
        except Exception as e:
            logger.error(f"Failed to load ProLIF JSON: {e}")
            self.prolif_data = None 
            if self.tree_B is None: self.tree_B = KDTree(self.coords_B)

    def _get_interaction_type_heuristic(self, atom_A, atom_B, dist):
        if dist > self.config.vdw_distance: return None
        
        res_A = atom_A.get_parent().get_resname()
        res_B = atom_B.get_parent().get_resname()
        name_A = atom_A.get_name()
        name_B = atom_B.get_name()
        elem_A = atom_A.element.upper()
        elem_B = atom_B.element.upper()
        
        is_ani_A = ((res_A in self.charged_neg and name_A in self.anion_atoms) or name_A == 'OXT')
        is_ani_B = ((res_B in self.charged_neg and name_B in self.anion_atoms) or name_B == 'OXT')
        is_cat_A = (res_A in self.charged_pos and name_A in self.cation_atoms)
        is_cat_B = (res_B in self.charged_pos and name_B in self.cation_atoms)
        is_aro_A = (res_A in self.aromatic and name_A not in ['CA', 'C', 'O', 'N'])
        is_aro_B = (res_B in self.aromatic and name_B not in ['CA', 'C', 'O', 'N'])

        if dist < self.config.ionic_distance:
            if dist < self.config.strong_ionic_distance and ((is_cat_A and is_ani_B) or (is_ani_A and is_cat_B)): return 'Cationic'
            if is_aro_A or is_aro_B:
                if (is_aro_A and is_cat_B) or (is_cat_A and is_aro_B): return 'PiCation'
            if (is_cat_A and is_ani_B) or (is_ani_A and is_cat_B): return 'Anionic'

        if dist < self.config.hydrogen_bond_distance:
            if elem_A in self.polar_atoms and elem_B in self.polar_atoms: return 'HydrogenBond'
            if (elem_A == 'C' and elem_B in self.polar_atoms) or (elem_B == 'C' and elem_A in self.polar_atoms): return 'HydrogenBond'

        if dist < self.config.pi_stack_distance and (is_aro_A or is_aro_B):
            if (is_aro_A and name_B in self.sulfur_atoms) or (name_A in self.sulfur_atoms and is_aro_B): return 'PiStacking'

        if dist < self.config.aromatic_distance:
            if is_aro_A and is_aro_B: return 'PiStacking'
            if dist < self.config.pi_stack_distance:
                is_ali_A = (elem_A == 'C' and not is_aro_A and name_A not in ['CA', 'C'])
                is_ali_B = (elem_B == 'C' and not is_aro_B and name_B not in ['CA', 'C'])
                if (is_aro_A and is_ali_B) or (is_ali_A and is_aro_B): return 'Hydrophobic'
            if dist < self.config.pi_stack_distance and elem_A == 'C' and elem_B == 'C' and name_A not in ['CA', 'C'] and name_B not in ['CA', 'C']:
                 if (not is_aro_A) and (not is_aro_B): return 'Hydrophobic'

        return None

    def plot_patches(self, patches, output_file=None, show=True, style_config=None):
        if not patches: return None
        self.artist_map = {}
        style = {
            'color': 'red', 'font_family': 'sans-serif', 'font_size': 9, 
            'color_by_type': False, 'active_types': self.interaction_types,
            'show_labels': True, 'label_mode': 'chain_a', 'avoid_label_overlap': True,
            'use_uv_atlas': True,
            'label_offsets': {},
            'mesh_fill_alpha': self.config.mesh_fill_alpha,
            'mesh_line_alpha': self.config.mesh_line_alpha,
        }
        if style_config: style.update(style_config)
        interaction_colors = dict(self.interaction_colors)
        interaction_colors.update(style.get('interaction_colors') or {})
        style['interaction_colors'] = interaction_colors

        n_patches = len(patches)
        use_uv_atlas = bool(style.get('use_uv_atlas', True))
        if use_uv_atlas:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = np.asarray([axes], dtype=object)
        else:
            n_cols = min(3, n_patches)
            n_rows = int(np.ceil(n_patches / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.2 * n_rows))
            axes = np.atleast_1d(axes).flatten()
        fig.patch.set_facecolor('white')

        logger.info(
            "Visualizing %d patches in %s mode.",
            n_patches,
            "atlas" if use_uv_atlas else "separate",
        )

        used_interactions = set()
        for i, patch in enumerate(patches):
            ax = axes[0] if use_uv_atlas else axes[i]
            ax.set_facecolor('#fcfcfd')
            found = self._draw_single_patch(ax, patch, i + 1, style, use_uv_atlas=use_uv_atlas)
            used_interactions.update(found)
            if not use_uv_atlas:
                ax.set_title(f"Patch {i + 1}", fontsize=10, weight='semibold', color='#2f3640')
                ax.set_aspect('equal')
                ax.axis('off')

        if not use_uv_atlas:
            for extra_ax in axes[n_patches:]:
                extra_ax.axis('off')

        if style['color_by_type'] and used_interactions:
            legend_handles = []
            for t in self.interaction_types:
                if t in used_interactions:
                    legend_handles.append(mpatches.Patch(color=style['interaction_colors'].get(t, 'gray'), label=t))
            fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(legend_handles), 5), frameon=False)

        if use_uv_atlas:
            axes[0].set_title("Global UV Interaction Map", fontsize=13, weight='semibold', color='#2f3640', pad=14)
            axes[0].set_aspect('equal')
            axes[0].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if output_file: plt.savefig(output_file, dpi=300)
        if show: plt.show()
        return fig

    def _draw_single_patch(self, ax, patch, patch_id, style, use_uv_atlas=True):
        found_types = set()
        if use_uv_atlas:
            uv = patch.metadata.get('uv_global')
        else:
            uv = patch.metadata.get('uv')
        if uv is None:
            uv = patch.metadata.get('uv_global')
        if uv is None: return found_types

        patch_color = self.patch_fill_palette[(patch_id - 1) % len(self.patch_fill_palette)]
        triangles_2d = uv[np.asarray(patch.faces, dtype=np.int64)]
        fill = PolyCollection(
            triangles_2d,
            facecolors=patch_color,
            edgecolors='none',
            alpha=float(style.get('mesh_fill_alpha', 0.22)),
            zorder=0
        )
        ax.add_collection(fill)
        ax.triplot(
            uv[:, 0], uv[:, 1], patch.faces,
            color='#2f3b52',
            alpha=float(style.get('mesh_line_alpha', 0.60)),
            lw=0.70,
            zorder=1
        )
        
        residue_data = self._collect_patch_residue_data(patch, uv, include_types=style['color_by_type'])

        label_records = []

        for res_id, data in residue_data.items():
            uv_array = np.array(data['uvs'])
            u_center, v_center = np.mean(uv_array, axis=0)
            types = data['types']

            best_type = None
            if style['color_by_type']:
                active_list = style.get('active_types', [])
                if types:
                    best_rank = len(self.interaction_types)
                    for t in types:
                        if t not in active_list:
                            continue
                        if t in self.interaction_types:
                            rank = self.interaction_types.index(t)
                            if rank < best_rank:
                                best_rank, best_type = rank, t

                if best_type is None:
                    continue
                final_color = style['interaction_colors'].get(best_type, '#FFC0CB')
                found_types.add(best_type)
            else:
                final_color = style['color']

            chain_obj = self.atoms_A[0].get_parent().get_parent()
            try:
                res_obj = chain_obj[res_id]
                res_name = res_obj.get_resname()
            except KeyError:
                 res_name = "UNK"
            label_text = self._build_label_text(
                style.get('label_mode', 'chain_a'),
                res_id[1],
                res_name,
                data.get('partners', {})
            )
            res_name_for_id = self._format_residue_label(res_name, res_id[1])

            uid = f"{patch_id}_{res_name_for_id}"
            
            sc = ax.scatter(u_center, v_center, c=final_color, s=80, edgecolors='white', zorder=10, picker=5)
            sc.set_gid(uid)
            txt = None
            connector = None
            if style.get('show_labels', True):
                default_pos = (u_center, v_center + self.config.label_offset)
                offset = style.get('label_offsets', {}).get(uid, (0.0, 0.0))
                text_pos = (default_pos[0] + offset[0], default_pos[1] + offset[1])
                txt = ax.text(
                    text_pos[0],
                    text_pos[1],
                    label_text,
                    fontsize=style['font_size'],
                    fontname=style['font_family'],
                    ha='center',
                    fontweight='bold',
                    color='#2f3640',
                    zorder=20,
                )
                txt.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
                txt.set_gid(uid)
                connector, = ax.plot([u_center, text_pos[0]], [v_center, text_pos[1]], linestyle=(0, (2, 2)),
                                     color='dimgray', lw=0.8, alpha=0.8, zorder=6)
                label_records.append({'text': txt, 'scatter': sc, 'connector': connector})
            self.artist_map[uid] = {'scatter': sc, 'text': txt, 'connector': connector}

        if style.get('show_labels', True) and style.get('avoid_label_overlap', True) and label_records:
            self._relax_labels(ax, label_records)

        center = uv.mean(axis=0)
        ax.text(center[0], center[1], f"P{patch_id}", fontsize=8, color='#4b5563', alpha=0.35)
        return found_types

    def _collect_patch_residue_data(self, patch, uv, include_types=True):
        patch_tree = KDTree(patch.vertices)
        dists_A_to_patch, vertex_indices = patch_tree.query(self.coords_A)
        on_patch_mask = dists_A_to_patch < self.config.on_patch_distance

        if self.prolif_data is not None:
            candidate_indices = np.where(on_patch_mask)[0]
        elif self.tree_B:
            dists_A_to_B_coarse, _ = self.tree_B.query(self.coords_A)
            interaction_mask = dists_A_to_B_coarse < self.config.coarse_interaction_distance
            candidate_indices = np.where(on_patch_mask & interaction_mask)[0]
        else:
            candidate_indices = []

        residue_data = {}
        for idx in candidate_indices:
            atom_A = self.atoms_A[idx]
            parent = atom_A.get_parent()
            res_id = parent.get_id()
            res_seq = res_id[1]

            u, v = uv[vertex_indices[idx]]
            if res_id not in residue_data:
                residue_data[res_id] = {'uvs': [], 'types': set(), 'partners': {}}
            residue_data[res_id]['uvs'].append([u, v])
            partner_counts = residue_data[res_id]['partners']
            if self.prolif_data is not None:
                for p_seq, p_count in self.prolif_partners.get(res_seq, {}).items():
                    partner_counts[p_seq] = partner_counts.get(p_seq, 0) + p_count
            elif self.atoms_B and self.tree_B:
                nearby_partner_indices = self.tree_B.query_ball_point(self.coords_A[idx], r=self.config.partner_search_distance)
                for b_idx in nearby_partner_indices:
                    b_seq = self.atoms_B[b_idx].get_parent().get_id()[1]
                    partner_counts[b_seq] = partner_counts.get(b_seq, 0) + 1

            if include_types:
                if self.prolif_data is not None and res_seq in self.prolif_data:
                    residue_data[res_id]['types'].update(self.prolif_data[res_seq])
                elif self.atoms_B and self.tree_B:
                    nearby_b_indices = self.tree_B.query_ball_point(self.coords_A[idx], r=self.config.partner_search_distance)
                    for b_idx in nearby_b_indices:
                        dist = np.linalg.norm(self.coords_A[idx] - self.coords_B[b_idx])
                        i_type = self._get_interaction_type_heuristic(atom_A, self.atoms_B[b_idx], dist)
                        if i_type:
                            residue_data[res_id]['types'].add(i_type)
                            b_seq = self.atoms_B[b_idx].get_parent().get_id()[1]
                            partner_counts[b_seq] = partner_counts.get(b_seq, 0) + 1
        return residue_data

    def count_patch_points(self, patch):
        uv = patch.metadata.get('uv_global')
        if uv is None:
            uv = patch.metadata.get('uv')
        if uv is None:
            return 0
        residue_data = self._collect_patch_residue_data(patch, uv, include_types=True)
        interacting_residue_count = 0
        for data in residue_data.values():
            if data.get('types'):
                interacting_residue_count += 1
        return interacting_residue_count

    def _relax_labels(self, ax, label_records, max_iter=80):
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for _ in range(max_iter):
            moved = False
            bboxes = [rec['text'].get_window_extent(renderer=renderer).expanded(1.05, 1.1) for rec in label_records]
            for i in range(len(label_records)):
                for j in range(i + 1, len(label_records)):
                    b1, b2 = bboxes[i], bboxes[j]
                    if not b1.overlaps(b2):
                        continue
                    moved = True
                    dx = min(b1.x1 - b2.x0, b2.x1 - b1.x0) * 0.5
                    dy = min(b1.y1 - b2.y0, b2.y1 - b1.y0) * 0.5
                    sign_x = 1 if b1.x0 <= b2.x0 else -1
                    sign_y = 1 if b1.y0 <= b2.y0 else -1
                    self._move_text_display(ax, label_records[i]['text'], -sign_x * dx, -sign_y * dy)
                    self._move_text_display(ax, label_records[j]['text'], sign_x * dx, sign_y * dy)

            if not moved:
                break
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

        for rec in label_records:
            pt = rec['scatter'].get_offsets()[0]
            tx, ty = rec['text'].get_position()
            if rec.get('connector') is not None:
                rec['connector'].set_data([pt[0], tx], [pt[1], ty])

    def _move_text_display(self, ax, txt, dx, dy):
        cur_data = txt.get_position()
        cur_disp = ax.transData.transform(cur_data)
        new_disp = (cur_disp[0] + dx, cur_disp[1] + dy)
        new_data = ax.transData.inverted().transform(new_disp)
        txt.set_position((new_data[0], new_data[1]))
