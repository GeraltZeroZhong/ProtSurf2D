import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree
import logging

logger = logging.getLogger("Visualizer")

class InterfaceVisualizer:
    def __init__(self, chain_A_atoms, chain_A_coords, chain_B_coords, chain_B_atoms=None, 
                 pdb_file=None, chain_a_id=None, chain_b_id=None):
        self.atoms_A = chain_A_atoms
        self.coords_A = chain_A_coords
        self.coords_B = chain_B_coords
        self.atoms_B = chain_B_atoms
        
        self.tree_B = KDTree(self.coords_B)
        self.artist_map = {}

        # --- Interaction Categories ---
        self.interaction_types = [
            'Electrostatic',    # Includes Salt Bridge, Pi-Cation
            'Hydrogen Bond',    # Includes Conventional & Carbon H-Bonds
            'Pi-Sulfur',        # Specific request
            'Hydrophobic',      # Includes Alkyl, Pi-Alkyl, Pi-Pi
            'Others'            # Disulfide, Metal, VdW, etc.
        ]
        
        self.interaction_colors = {
            'Electrostatic': '#FFA500',     # Orange
            'Hydrogen Bond': '#0000FF',     # Blue
            'Pi-Sulfur': '#FFD700',         # Gold
            'Hydrophobic': '#808080',       # Gray
            'Others': '#FFC0CB'             # Pink
        }

        # --- Chemical Definitions ---
        self.charged_pos = {'ARG', 'LYS', 'HIS'} 
        self.charged_neg = {'ASP', 'GLU'}
        self.aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}
        self.hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'CYS', 'TYR'}
        
        self.cation_atoms = {'NZ', 'NH1', 'NH2', 'ND1', 'NE2'} 
        self.anion_atoms = {'OD1', 'OD2', 'OE1', 'OE2', 'OXT'} 
        self.polar_atoms = {'N', 'O', 'S', 'F'}
        self.sulfur_atoms = {'SG', 'SD'}

    def _get_interaction_type(self, atom_A, atom_B, dist):
        """
        Simplified classification logic.
        """
        if dist > 6.0: return None
        
        res_A = atom_A.get_parent().get_resname()
        res_B = atom_B.get_parent().get_resname()
        name_A = atom_A.get_name()
        name_B = atom_B.get_name()
        elem_A = atom_A.element.upper()
        elem_B = atom_B.element.upper()
        
        # --- Pre-calc ---
        is_ani_A = ((res_A in self.charged_neg and name_A in self.anion_atoms) or name_A == 'OXT')
        is_ani_B = ((res_B in self.charged_neg and name_B in self.anion_atoms) or name_B == 'OXT')
        is_cat_A = (res_A in self.charged_pos and name_A in self.cation_atoms)
        is_cat_B = (res_B in self.charged_pos and name_B in self.cation_atoms)
        
        is_aro_A = (res_A in self.aromatic and name_A not in ['CA', 'C', 'O', 'N'])
        is_aro_B = (res_B in self.aromatic and name_B not in ['CA', 'C', 'O', 'N'])

        # 1. Electrostatic (Salt Bridge & Pi-Cation)
        if dist < 5.0:
            # Salt Bridge (< 4.0A)
            if dist < 4.0 and ((is_cat_A and is_ani_B) or (is_ani_A and is_cat_B)):
                return 'Electrostatic'
            # Pi-Cation (< 5.0A)
            if is_aro_A or is_aro_B:
                if (is_aro_A and is_cat_B) or (is_cat_A and is_aro_B):
                    return 'Electrostatic'
            # General Attractive Charge
            if (is_cat_A and is_ani_B) or (is_ani_A and is_cat_B):
                return 'Electrostatic'

        # 2. Hydrogen Bond (Conventional & Carbon)
        if dist < 3.8:
            # Conventional
            if elem_A in self.polar_atoms and elem_B in self.polar_atoms:
                return 'Hydrogen Bond'
            # Carbon H-Bond
            if (elem_A == 'C' and elem_B in self.polar_atoms) or \
               (elem_B == 'C' and elem_A in self.polar_atoms):
                return 'Hydrogen Bond'

        # 3. Pi-Sulfur (Explicit request)
        if dist < 4.5 and (is_aro_A or is_aro_B):
            if (is_aro_A and name_B in self.sulfur_atoms) or \
               (name_A in self.sulfur_atoms and is_aro_B):
                return 'Pi-Sulfur'

        # 4. Hydrophobic Group (Pi-Pi, Pi-Alkyl, Alkyl, General Hydrophobic)
        if dist < 5.5:
            # Pi-Pi Stacking
            if is_aro_A and is_aro_B:
                return 'Hydrophobic'
            
            # Pi-Alkyl (< 4.5A)
            if dist < 4.5:
                is_aliphatic_A = (elem_A == 'C' and not is_aro_A and name_A not in ['CA', 'C'])
                is_aliphatic_B = (elem_B == 'C' and not is_aro_B and name_B not in ['CA', 'C'])
                if (is_aro_A and is_aliphatic_B) or (is_aliphatic_A and is_aro_B):
                    return 'Hydrophobic'

            # Alkyl / General Hydrophobic (< 4.5A)
            if dist < 4.5 and elem_A == 'C' and elem_B == 'C':
                if name_A not in ['CA', 'C'] and name_B not in ['CA', 'C']:
                    # Alkyl
                    is_aliphatic_A = (elem_A == 'C' and not is_aro_A)
                    is_aliphatic_B = (elem_B == 'C' and not is_aro_B)
                    if is_aliphatic_A and is_aliphatic_B:
                        return 'Hydrophobic'
                    # General
                    if res_A in self.hydrophobic and res_B in self.hydrophobic:
                        return 'Hydrophobic'

        # 5. Others (Disulfide, Metal, VdW)
        # Disulfide (< 2.5A)
        if dist < 2.5 and res_A == 'CYS' and res_B == 'CYS' and name_A == 'SG' and name_B == 'SG':
            return 'Others'
            
        if dist < 5.0:
            return 'Others'

        return None

    def plot_patches(self, patches, output_file=None, show=True, style_config=None):
        if not patches:
            return None
        
        self.artist_map = {}
        # Changed 'Arial' to 'sans-serif' to avoid font not found warnings on Linux/WSL
        style = {
            'color': 'red', 'font_family': 'sans-serif', 'font_size': 9, 
            'color_by_type': False, 'active_types': self.interaction_types
        }
        if style_config: style.update(style_config)

        n_patches = len(patches)
        fig, axes = plt.subplots(1, n_patches, figsize=(5 * n_patches, 6))
        if n_patches == 1: axes = [axes]
        
        logger.info(f"Visualizing {n_patches} patches.")

        used_interactions = set()
        for i, patch in enumerate(patches):
            found = self._draw_single_patch(axes[i], patch, i+1, style)
            used_interactions.update(found)

        if style['color_by_type'] and used_interactions:
            legend_handles = []
            for t in self.interaction_types:
                if t in used_interactions:
                    legend_handles.append(mpatches.Patch(color=self.interaction_colors.get(t, 'gray'), label=t))
            fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(legend_handles), 5), frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if output_file: plt.savefig(output_file, dpi=300)
        if show: plt.show()
        return fig

    def _draw_single_patch(self, ax, patch, patch_id, style):
        found_types = set()
        uv = patch.metadata.get('uv')
        if uv is None: return found_types

        ax.triplot(uv[:, 0], uv[:, 1], patch.faces, color='gray', alpha=0.15, lw=0.5, zorder=1)
        
        patch_tree = KDTree(patch.vertices)
        dists_A_to_patch, vertex_indices = patch_tree.query(self.coords_A)
        on_patch_mask = dists_A_to_patch < 3.0 
        dists_A_to_B_coarse, _ = self.tree_B.query(self.coords_A)
        interaction_mask = dists_A_to_B_coarse < 8.0
        candidate_indices = np.where(on_patch_mask & interaction_mask)[0]
        
        residue_data = {} 
        
        for idx in candidate_indices:
            atom_A = self.atoms_A[idx]
            res_id = atom_A.get_parent().get_id() 
            u, v = uv[vertex_indices[idx]]
            
            if res_id not in residue_data: residue_data[res_id] = {'uvs': [], 'types': set()}
            residue_data[res_id]['uvs'].append([u, v])
            
            if style['color_by_type'] and self.atoms_B:
                nearby_b_indices = self.tree_B.query_ball_point(self.coords_A[idx], r=6.0)
                for b_idx in nearby_b_indices:
                    dist = np.linalg.norm(self.coords_A[idx] - self.coords_B[b_idx])
                    i_type = self._get_interaction_type(atom_A, self.atoms_B[b_idx], dist)
                    if i_type: residue_data[res_id]['types'].add(i_type)

        for res_id, data in residue_data.items():
            uv_array = np.array(data['uvs'])
            u_center, v_center = np.mean(uv_array, axis=0)
            types = data['types']
            
            best_type = None
            if style['color_by_type']:
                active_list = style.get('active_types', [])
                if types:
                    best_rank = 999
                    for t in types:
                        if t not in active_list: continue
                        if t in self.interaction_types:
                            rank = self.interaction_types.index(t)
                            if rank < best_rank:
                                best_rank, best_type = rank, t
                
                if best_type is None: continue
                final_color = self.interaction_colors.get(best_type, '#FFC0CB')
                found_types.add(best_type)
            else:
                final_color = style['color']

            res_obj = self.atoms_A[candidate_indices[0]].get_parent().get_parent()[res_id]
            res_name = f"{res_obj.get_resname()}{res_id[1]}"
            uid = f"{patch_id}_{res_name}"
            
            sc = ax.scatter(u_center, v_center, c=final_color, s=80, edgecolors='white', zorder=10, picker=5)
            sc.set_gid(uid)
            txt = ax.text(u_center, v_center + 0.04, res_name, fontsize=style['font_size'], 
                          fontname=style['font_family'], ha='center', fontweight='bold', color='darkred', picker=True)
            txt.set_gid(uid)
            self.artist_map[uid] = {'scatter': sc, 'text': txt}

        ax.set_title(f"Patch {patch_id}")
        ax.set_aspect('equal')
        ax.axis('off')
        return found_types
