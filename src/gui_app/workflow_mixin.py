import csv
import os
import threading
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
        input_path = self.entry_file.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid PDB file or Folder.")
            return

        is_batch = os.path.isdir(input_path)
        chain_a = self.entry_chain_a.get().strip()
        chain_b = self.entry_chain_b.get().strip()
        if not is_batch and (not chain_a or not chain_b):
            messagebox.showerror("Error", "Please define Chain A and Chain B for single file mode.")
            return

        self.btn_run.config(state="disabled")
        self.btn_bench.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.progress.start(10)
        self.log(f"Running Benchmark ({'Batch Folder' if is_batch else 'Single File'})...")

        def run_thread():
            try:
                import benchmark
                if is_batch:
                    pdb_dir = input_path
                    target_csv = os.path.join(pdb_dir, "benchmark_targets.csv")
                    if not os.path.exists(target_csv):
                        self.root.after(0, lambda: self.log("CSV missing. Generating from PDBs..."))
                        success, msg = self._generate_auto_csv(pdb_dir)
                        if not success:
                            raise Exception(msg)
                        self.root.after(0, lambda: self.log("CSV generated."))

                    tasks = []
                    with open(target_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            tasks.append(row)
                    total = len(tasks)
                    if total == 0:
                        raise Exception("CSV file is empty.")

                    self.root.after(0, lambda: self.log(f"Found {total} tasks."))
                    for i, task in enumerate(tasks):
                        pdb_id = task['PDB']
                        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
                        if not os.path.exists(pdb_file):
                            pdb_file = os.path.join(pdb_dir, pdb_id)
                        if not os.path.exists(pdb_file):
                            print(f"Skipping {pdb_id}, file missing.")
                            continue
                        self.root.after(0, lambda idx=i: self.log(f"Benchmarking {idx + 1}/{total}: {pdb_id}..."))
                        try:
                            runner = benchmark.BenchmarkRunner(pdb_file, task['Chain_A'], task['Chain_B'])
                            runner.output_root = os.path.join(pdb_dir, "benchmark_results")
                            runner.category = task.get('Category', 'Uncategorized')
                            runner.category_dir = os.path.join(runner.output_root, runner.category)
                            if not os.path.exists(runner.category_dir):
                                os.makedirs(runner.category_dir)
                            runner.output_csv = os.path.join(runner.category_dir, f"{pdb_id}_benchmark.csv")
                            runner.run()
                        except Exception as e:
                            print(f"Error on {pdb_id}: {e}")
                    msg = f"Batch Benchmark Complete!\nResults saved in {os.path.join(pdb_dir, 'benchmark_results')}"
                else:
                    runner = benchmark.BenchmarkRunner(input_path, chain_a, chain_b)
                    runner.run()
                    msg = "Single Benchmark Complete!\nSaved to default folder."

                self.root.after(0, lambda: messagebox.showinfo("Success", msg))
                self.root.after(0, lambda: self.log("Benchmark Finished."))
            except Exception as e:
                err_msg = str(e)
                print(e)
                self.root.after(0, lambda: messagebox.showerror("Benchmark Error", err_msg))
            finally:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.btn_run.config(state="normal"))
                self.root.after(0, lambda: self.btn_bench.config(state="normal"))

        threading.Thread(target=run_thread, daemon=True).start()

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
                    enabled=params.get('enable_joint_opt', True),
                    use_optcuts=params.get('use_optcuts', False),
                    optcuts_bin=params.get('optcuts_bin', "OptCuts_bin"),
                    overlap_weight=params.get('overlap_weight', 1.0),
                    max_iterations=params.get('uv_max_iter', 60)
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
            self.root.after(0, lambda: self.show_error(str(e)))

    def _parameterize_and_optimize_patches(self, patches, parameterizer, optimizer):
        valid_patches = []
        for p in patches:
            uv = parameterizer.flatten_patch(p)
            if uv is not None:
                p.metadata['uv'] = uv
                valid_patches.append(p)
        if not valid_patches:
            return []
        return optimizer.optimize_patches(valid_patches)

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
