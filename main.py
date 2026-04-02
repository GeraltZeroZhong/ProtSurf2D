import argparse
import sys
import os
import time
import logging

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.io.io_loader import PDBLoader
from src.mesh.surface import SurfaceGenerator
from src.mesh.topology import TopologyManager
from src.mesh.parameterization import Parameterizer
from src.optimization.uv_optimizer import OptCutsUVOptimizer, UVOptimizerConfig
from src.visualization.visualizer import InterfaceVisualizer
from src.interactions.interaction_engine import generate_prolif_interactions

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="ProtSurf2D: Protein Interface 2D Map Generator")
    parser.add_argument("pdb_file", help="Path to input PDB/CIF file")
    parser.add_argument("-A", "--chain_a", required=True, help="Chain ID for the surface (Receptor)")
    parser.add_argument("-B", "--chain_b", required=True, help="Chain ID for the ligand")
    parser.add_argument("--prolif", "--arpeggio", dest="prolif", help="Path to ProLIF interaction JSON (Optional)", default=None)
    parser.add_argument("--cutoff", type=float, default=5.0, help="Interface distance cutoff (Angstroms)")
    parser.add_argument("--res", type=float, default=1.0, help="Grid resolution for surface generation (Angstroms)")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    parser.add_argument("--output", "-o", default="interface_map.png", help="Output image filename")
    parser.add_argument("--disable-optcuts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--optcuts-bin", default="OptCuts_bin", help="Path/name for OptCuts executable")
    parser.add_argument("--patch-gap", type=float, default=0.08, help="Padding/min-gap between charts in global UV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("Main")

    start_time = time.time()

    # --- Step 0: ProLIF Handling ---
    prolif_file = args.prolif
    
    # If path is empty or invalid, try to generate it
    if not prolif_file or not os.path.exists(prolif_file):
        generated_json = generate_prolif_interactions(args.pdb_file, args.chain_a, args.chain_b, logger)
        if generated_json:
            prolif_file = generated_json
        else:
            prolif_file = None # Explicitly None to trigger heuristic fallback in Visualizer

    # --- Step 1: Data Loading ---
    logger.info(f"Loading {args.pdb_file}...")
    try:
        loader = PDBLoader(args.pdb_file)
        coords_A, atoms_A = loader.get_chain_data(args.chain_a)
        coords_B, atoms_B = loader.get_chain_data(args.chain_b)
        logger.info(f"Loaded Chain {args.chain_a}: {len(coords_A)} atoms")
        logger.info(f"Loaded Chain {args.chain_b}: {len(coords_B)} atoms")
    except Exception as e:
        logger.error(f"Failed to load PDB data: {e}")
        sys.exit(1)

    # --- Step 2: Surface Generation ---
    logger.info("Generating SES Surface for Chain A...")
    surf_gen = SurfaceGenerator(coords_A)
    mesh_A = surf_gen.generate_mesh(grid_resolution=args.res, sigma=args.sigma)
    
    if mesh_A is None or len(mesh_A.vertices) == 0:
        logger.error("Failed to generate surface mesh.")
        sys.exit(1)

    # --- Step 3: Topology Extraction ---
    logger.info("Extracting interface patches...")
    topo_mgr = TopologyManager(mesh_A, coords_B)
    patches = topo_mgr.get_interface_patches(distance_cutoff=args.cutoff)
    
    if not patches:
        logger.error("No interface patches found. Try increasing --cutoff.")
        sys.exit(0)

    # --- Step 4: Parameterization (LSCM) ---
    logger.info(f"Parameterizing {len(patches)} patches...")
    valid_patches = []
    param = Parameterizer()
    
    for i, patch in enumerate(patches):
        logger.info(f"  Flattening Patch {i+1} ({len(patch.vertices)} verts)...")
        uv = param.flatten_patch(patch)
        
        if uv is not None:
            # Attach UV data to the mesh object for the visualizer
            patch.metadata['uv'] = uv
            valid_patches.append(patch)
        else:
            logger.warning(f"  Skipping Patch {i+1} due to parameterization failure.")

    if not valid_patches:
        logger.error("All patches failed to parameterize.")
        sys.exit(1)

    # --- Step 5: OptCuts UV Optimization ---
    if args.disable_optcuts:
        logger.error("--disable-optcuts is no longer supported. This pipeline now requires OptCuts.")
        sys.exit(2)
    logger.info("Running OptCuts UV optimization...")
    optimizer = OptCutsUVOptimizer(
        UVOptimizerConfig(
            optcuts_bin=args.optcuts_bin,
            patch_gap=args.patch_gap,
        )
    )
    valid_patches = optimizer.optimize_patches(valid_patches)
    report = optimizer.get_last_report() if hasattr(optimizer, "get_last_report") else {}
    if report:
        logger.info(f"Joint report: {report}")

    # --- Step 6: Visualization ---
    logger.info("Visualizing results...")
    
    # Updated signature to support ProLIF
    viz = InterfaceVisualizer(
        chain_A_atoms=atoms_A, 
        chain_A_coords=coords_A, 
        chain_B_coords=coords_B, 
        chain_B_atoms=atoms_B,
        chain_a_id=args.chain_a,
        chain_b_id=args.chain_b,
        prolif_file=prolif_file
    )
    
    viz.plot_patches(valid_patches, output_file=args.output)

    elapsed = time.time() - start_time
    logger.info(f"Done! Pipeline finished in {elapsed:.2f}s. Saved to {args.output}")

if __name__ == "__main__":
    main()
