import numpy as np
import warnings
from Bio.PDB import PDBParser, MMCIFParser, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress Biopython warnings about discontinuous chains etc.
warnings.simplefilter('ignore', PDBConstructionWarning)

class NotHetero(Select):
    """Filter to remove water and heteroatoms, keeping only standard residues."""
    def accept_residue(self, residue):
        # standard residues have blank id[0]
        return residue.id[0] == " "

class PDBLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        if file_path.endswith('.cif'):
            self.parser = MMCIFParser(QUIET=True)
        else:
            self.parser = PDBParser(QUIET=True)
        
        self.structure = self.parser.get_structure("P", file_path)
        self.model = self.structure[0] # Always take first model

    def get_chain_data(self, chain_id):
        """
        Extract coordinates and atom objects for a specific chain.
        Returns:
            coords (np.ndarray): (N, 3) coordinates
            atoms (list): List of Bio.PDB.Atom objects (aligned with coords)
        """
        if chain_id not in self.model:
            raise ValueError(f"Chain {chain_id} not found in PDB.")

        chain = self.model[chain_id]
        
        # Filter atoms: remove HOH, use only N, CA, C, O, CB etc.
        # We generally keep all heavy atoms for surface generation
        atoms = []
        coords = []
        
        for residue in chain:
            # Skip heteroatoms (water, ions)
            if residue.id[0] != " ":
                continue
                
            for atom in residue:
                atoms.append(atom)
                coords.append(atom.get_coord())
                
        return np.array(coords), atoms

# --- Unit Test ---
if __name__ == "__main__":
    print("This module is a helper for loading PDB files.")
    print("Please use main.py to run the tool.")
