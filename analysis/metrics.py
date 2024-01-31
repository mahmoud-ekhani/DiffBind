from rdkit import Chem
from rdkit.Chem.rdchem import Mol



def rdmol_to_smiles(rdmol: Mol) -> str:
    """
    Converts an RDKit molecule object to its SMILES representation.

    The function removes stereochemistry and hydrogen atoms from the molecule 
    before converting it to a SMILES string.

    Args:
        rdmol (Mol): An RDKit molecule object.

    Returns:
        str: The SMILES string representation of the molecule.
    """
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)
