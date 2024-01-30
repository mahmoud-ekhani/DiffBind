import tempfile

import torch
import numpy as np
from openbabel import openbabel
from rdkit import Chem

import utils
from constants import bonds1, bonds2, bonds3, margin1, margin2, margin3, \
    bond_dict


def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write the xyz file
        utils.write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        # Read the sdf file using RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    # Build a new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.get_atoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond.GetBondType())
    return mol


def get_bond_order_batch(atoms1, atoms2, distances, dataset_info, margin1=0.1, margin2=0.1, margin3=0.1):
    """
    Calculate bond orders based on atomic distances and dataset information.

    Args:
        atoms1 (ndarray or Tensor): Indices of the first set of atoms involved in potential bonds.
        atoms2 (ndarray or Tensor): Indices of the second set of atoms involved in potential bonds.
        distances (ndarray or Tensor): Atomic distances between atoms1 and atoms2.
        dataset_info (dict): Dataset information containing bond thresholds (bonds1, bonds2, bonds3).
        margin1 (float, optional): Margin for single bond classification. Default is 0.1.
        margin2 (float, optional): Margin for double bond classification. Default is 0.1.
        margin3 (float, optional): Margin for triple bond classification. Default is 0.1.

    Returns:
        Tensor: Bond types for each pair of atoms, where:
        - 0: No bond
        - 1: Single bond
        - 2: Double bond
        - 3: Triple bond

    Note:
        This function calculates bond orders based on atomic distances and predefined bond thresholds.
        It assigns bond types to each pair of atoms in 'atoms1' and 'atoms2' based on the distances and
        thresholds provided in 'dataset_info'. The bond types are represented as integers: 0 for no bond,
        1 for single bond, 2 for double bond, and 3 for triple bond.

    Example:
        >>> atoms1 = np.array([0, 1, 2])
        >>> atoms2 = np.array([1, 2, 3])
        >>> distances = np.array([1.2, 1.5, 1.0])
        >>> dataset_info = {
        ...     'bonds1': np.array([1.0, 1.1, 1.2, 1.3]),
        ...     'bonds2': np.array([1.4, 1.5, 1.6, 1.7]),
        ...     'bonds3': np.array([1.8, 1.9, 2.0, 2.1])
        ... }
        >>> bond_types = get_bond_order_batch(atoms1, atoms2, distances, dataset_info)
        >>> print(bond_types)
        tensor([1, 2, 1])

    """
    if isinstance(atoms1, np.ndarray):
        atoms1 = torch.from_numpy(atoms1)
    if isinstance(atoms2, np.ndarray):
        atoms2 = torch.from_numpy(atoms2)
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    distances = 100 * distances  # We change the metric

    bonds1 = torch.tensor(dataset_info['bonds1'], device=atoms1.device)
    bonds2 = torch.tensor(dataset_info['bonds2'], device=atoms1.device)
    bonds3 = torch.tensor(dataset_info['bonds3'], device=atoms1.device)

    bond_types = torch.zeros_like(atoms1)  # 0: No bond

    # Single
    bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1

    # Double (note that already assigned single bonds will be overwritten)
    bond_types[distances < bonds2[atoms1, atoms2] + margin2] = 2

    # Triple
    bond_types[distances < bonds3[atoms1, atoms2] + margin3] = 3

    return bond_types


def make_mol_edm(positions, atom_types, dataset_info, add_coords):
    """
    Equivalent to EDM's way of building RDKit molecules.
    """
    n = len(positions)

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    pos = positions.unsqueeze(0) # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1) # remove batch and flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
    E = torch.tril(E_full, diagonal=-1) # Warning: graph should be directed
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(),
                    bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])
        
    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()
                                     ))
        mol.AddConformer(conf)

    return mol
                                

def build_molecule(positions, atom_types, dataset_info, add_coords=False,
                   use_openbabel=True):
    """
    Builds RDKit molecule.
    Args:
        positions: N x 3
        atom_type: N
        dataset_info: dict
        add_coords: Add conformer to mol (always add if use_openbabel=True)
        use_openbabel: use OpenBabel to generate the bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions,
                                 atom_types,
                                 dataset_info['atom_decoder']
                                 )
        
    else:
        mol = make_mol_edm(positions,
                           atom_types,
                           dataset_info, 
                           add_coords)
    
    return mol