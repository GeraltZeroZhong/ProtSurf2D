import csv
import os
from tkinter import messagebox

from src.io_loader import PDBLoader
from src.surface import SurfaceGenerator
from src.topology import TopologyManager
from src.parameterization import Parameterizer
from src.uv_optimizer import OptCutsUVOptimizer, UVOptimizerConfig
from src.visualizer import InterfaceVisualizer
from src.interaction_engine import generate_prolif_interactions


class WorkflowMixin:
    def _generate_auto_csv(self, folder):
        self.log("Scanning folder for PDB files to generate CSV...")
        pdbs = [f for f in os.listdir(folder) if f.lower().endswith('.pdb')]
        if not pdbs:
            return False, "No .pdb files found in the folder."

        targets = []
        for f in pdbs:
            path = os.path.join(folder, f)
            try:
                loader = PDBLoader(path)
                chains = [c.id for c in loader.model]
                if len(chains) >= 2:
                    targets.append([os.path.splitext(f)[0], chains[0], chains[1], "Auto_Generated"])
                else:
                    print(f"Skipping {f}: Found {len(chains)} chains, need at least 2.")
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if not targets:
            return False, "Found PDBs, but none had >= 2 chains."

        csv_path = os.path.join(folder, "benchmark_targets.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["PDB", "Chain_A", "Chain_B", "Category"])
                writer.writerows(targets)
            return True, csv_path
        except Exception as e:
            return False, f"Failed to write CSV: {e}"

    def start_benchmark(self):
        messagebox.showinfo(
            "Benchmark Removed",
            "Benchmark workflow has been removed. "
            "Please use the joint optimization pipeline directly from Run.",
        )
        self.log("Benchmark module removed; skipping benchmark workflow.")

    def generate_prolif_interactions(self, pdb_path, chain_a, chain_b):
        self.log("Checking ProLIF requirements...")
        output_json = generate_prolif_interactions(pdb_path, chain_a, chain_b)
        if output_json:
            self.root.after(0, lambda: self.entry_prolif.delete(0, "end"))
            self.root.after(0, lambda: self.entry_prolif.insert(0, output_json))
            return output_json
        self.log("ProLIF interaction generation skipped/failed. Falling back to geometric heuristics.")
        return None

    def run_pipeline(self, params):
        try:
            prolif_file = params.get('prolif')
            if not prolif_file or not os.path.exists(prolif_file):
                generated_json = self.generate_prolif_interactions(params['path'], params['chain_a'], params['chain_b'])
                prolif_file = generated_json if generated_json else None

            self.log("Loading PDB structure...")
            loader = PDBLoader(params['path'])
            coords_A, atoms_A = loader.get_chain_data(params['chain_a'])
            coords_B, atoms_B = loader.get_chain_data(params['chain_b'])

            self.log("Generating molecular surface...")
            surf_gen = SurfaceGenerator(coords_A)
            mesh_A = surf_gen.generate_mesh(grid_resolution=params['res'], sigma=params['sigma'])
            if mesh_A is None:
                raise ValueError("Surface generation failed.")

            self.log("Extracting interface patches...")
            param = Parameterizer()
            optimizer = OptCutsUVOptimizer(
                UVOptimizerConfig(
                    optcuts_bin=params.get('optcuts_bin', "OptCuts_bin"),
                    patch_gap=params.get('patch_gap', 0.08),
                )
            )
            viz = InterfaceVisualizer(
                chain_A_atoms=atoms_A,
                chain_A_coords=coords_A,
                chain_B_coords=coords_B,
                chain_B_atoms=atoms_B,
                chain_a_id=params['chain_a'],
                chain_b_id=params['chain_b'],
                prolif_file=prolif_file
            )

            cutoff_value = params['cutoff']
            if params.get('auto_cutoff', False):
                cutoff_value = self._search_best_cutoff(
                    mesh_A, coords_B, param, optimizer, viz,
                    params['cutoff_start'], params['cutoff_end'], params['cutoff_step'], params['min_points']
                )
                self.log(f"Auto-selected cutoff = {cutoff_value:.2f} Å")

            topo = TopologyManager(mesh_A, coords_B)
            patches = topo.get_interface_patches(distance_cutoff=cutoff_value)
            if not patches:
                raise ValueError(f"No interface found with cutoff {cutoff_value:.2f}.")

            self.log(f"Flattening {len(patches)} patches...")
            valid_patches = self._parameterize_and_optimize_patches(patches, param, optimizer)
            if not valid_patches:
                raise ValueError("LSCM Parameterization failed for all patches.")

            display_patches, valid_count, invalid_count = self._split_interfaces_by_point_count(valid_patches, viz, params['min_points'])
            self.log(f"Interface count summary (min points = {params['min_points']}): valid={valid_count}, invalid={invalid_count}")

            if params.get('filter_valid_only', True):
                selected_patches = display_patches
                if not selected_patches:
                    raise ValueError(f"All interfaces are invalid (point count < {params['min_points']}).")
                self.log("Display mode: valid interfaces only.")
            else:
                selected_patches = valid_patches
                self.log("Display mode: all interfaces (including invalid).")

            self.log("Rendering visualization...")
            self.cached_viz = viz
            self.cached_patches = selected_patches
            self.root.after(0, lambda: self.finish_success())
        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.show_error(msg))

    def _parameterize_and_optimize_patches(self, patches, parameterizer, optimizer):
        valid_patches = []
        for p in patches:
            uv = parameterizer.flatten_patch(p)
            if uv is not None:
                p.metadata['uv'] = uv
                valid_patches.append(p)
        if not valid_patches:
            return []
        result = optimizer.optimize_patches(valid_patches)
        self._log_joint_report(optimizer)
        return result

    def _log_joint_report(self, optimizer):
        report = getattr(optimizer, "get_last_report", lambda: {})()
        if not report:
            self.log("Joint optimization report unavailable.")
            return
        pq = report.get("parameterization_quality", {})
        tc = report.get("topology_complexity", {})
        au = report.get("atlas_usability", {})
        se = report.get("stability_efficiency", {})
        self.log(
            "[JointReport] flip={:.4f}, dist(mean/max/p95)=({:.4f}/{:.4f}/{:.4f})".format(
                float(pq.get("flip_rate_mean", 1.0)),
                float(pq.get("distortion", {}).get("mean", float("inf"))),
                float(pq.get("distortion", {}).get("max", float("inf"))),
                float(pq.get("distortion", {}).get("p95", float("inf"))),
            )
        )
        self.log(
            "[JointReport] seam_len={:.3f}, charts={}, overlap={:.4f}, padding_viol={}, util={:.4f}".format(
                float(tc.get("seam_total_length", 0.0)),
                int(tc.get("chart_count", 0)),
                float(au.get("overlap_area", 0.0)),
                int(au.get("padding_violations", 0)),
                float(au.get("utilization", 0.0)),
            )
        )
        self.log(
            "[JointReport] obj_drop={:.4f}, total_time={:.3f}s, failure_rate={:.3f}".format(
                float(se.get("objective_drop", 0.0)),
                float(se.get("total_time_sec", 0.0)),
                float(se.get("failure_rate", 0.0)),
            )
        )

    def _split_interfaces_by_point_count(self, patches, viz, min_points):
        valid = []
        invalid = 0
        for p in patches:
            point_count = viz.count_patch_points(p)
            p.metadata['point_count'] = point_count
            if point_count >= min_points:
                valid.append(p)
            else:
                invalid += 1
        return valid, len(valid), invalid

    def _generate_cutoff_values(self, start, end, step):
        if step <= 0:
            raise ValueError("Cutoff step must be > 0.")
        if end < start:
            raise ValueError("Cutoff end must be >= cutoff start.")
        values = []
        cur = start
        while cur <= end + 1e-8:
            values.append(round(cur, 6))
            cur += step
        return values

    def _search_best_cutoff(self, mesh_A, coords_B, parameterizer, optimizer, viz, start, end, step, min_points):
        candidates = self._generate_cutoff_values(start, end, step)
        best = None
        for cutoff in candidates:
            topo = TopologyManager(mesh_A, coords_B)
            patches = topo.get_interface_patches(distance_cutoff=cutoff)
            if not patches:
                self.log(f"[Cutoff Search] cutoff={cutoff:.2f}: no interfaces")
                continue
            processed = self._parameterize_and_optimize_patches(patches, parameterizer, optimizer)
            if not processed:
                self.log(f"[Cutoff Search] cutoff={cutoff:.2f}: parameterization failed")
                continue
            valid_patches, valid_count, invalid_count = self._split_interfaces_by_point_count(processed, viz, min_points)
            valid_points = sum(p.metadata.get('point_count', 0) for p in valid_patches)
            self.log(f"[Cutoff Search] cutoff={cutoff:.2f}: valid={valid_count}, invalid={invalid_count}, valid_points={valid_points}")
            if valid_count == 0:
                continue
            if best is None or valid_count < best['valid_count'] or (valid_count == best['valid_count'] and valid_points > best['valid_points']):
                best = {'cutoff': cutoff, 'valid_count': valid_count, 'valid_points': valid_points}

        if best is None:
            raise ValueError("Cutoff search failed: no candidate produced valid interfaces. Please adjust range/step/min-points.")
        self.log(f"[Cutoff Search] selected cutoff={best['cutoff']:.2f} (valid={best['valid_count']}, valid_points={best['valid_points']})")
        return best['cutoff']
