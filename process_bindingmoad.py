import warnings
import random
import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB.Structure import Structure
from rdkit import Chem
from rdkit.Chem import QED
from openbabel import openbabel
from scipy.ndimage import gaussian_filter


from geometry_utils import get_bb_transform
from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params
import utils


dataset_info = dataset_params['bindingmoad']
amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

class Model0(Select):
    def accept_model(self, model):
        return model.id == 0


def read_label_file(csv_path: Path) -> dict:
    """
    Read BindingMOAD's labels file and organize ligand data into a nested dictionary

    Args:
        csv_path (Path): Path to the 'every.csv' file.

    Returns:
        dict: A nested dictionary representing ligands.
            - First level: EC number.
            - Second level: PDB ID.
            - Thrid level: A list containing ligand information as [ligand name, validity, SMILES string]

    Example:
        {
            'EC_number1': {
                'PDB_ID1': [['ligand_name1', 'validity1', 'SMILES1'], ['ligand_name2', 'validity2', 'SMILES2']],
                'PDB_ID2': [['ligand_name3', 'validity3', 'SMILES3']]
            },
            'EC_number2': {
                'PDB_ID3': [['ligand_name4', 'validity4', 'SMILES4']]
            }
        }
    """
    ligand_dict = {}

    with open(csv_path, 'r') as f:
        for line in f.readlines():
            row = line.split(',')

            # A new protein class
            if len(row[0]) > 0:
                curr_class = row[0]
                ligand_dict[curr_class] = {}
                continue

            # A new protein ID
            if len(row[2]) > 0:
                curr_prot = row[2]
                ligand_dict[curr_class][curr_prot] = []
                continue

            # A new small molecule
            if len(row[3]) > 0:
                ligand_dict[curr_class][curr_prot].append(
                    # [ligand name, validity, SMILES string]
                    [row[3], row[4], row[9]]
                )

    return ligand_dict


def compute_druglikeness(ligand_dict: dict) -> dict:
    """
    Computes the Quantitative Esimate of Drug-likeness (QED) for molecules in the ligand_dict
    and adds it to each molecule's information.

    Args:
        ligand_dict (dict): A nested dictionary containing ligand information.
            - First level: Protein family.
            - Second level: Protein ID.
            - Third level: List containing [ligand name, validity, SMILES string].

    Returns:
        dict: The same ligand dictionary with the QED value appended to each molecule's list.

    Note:
        QED is a measure of drug-likeness, with higher values indicating molecules
        that are more likely to have drug-like properties.

    Example:
        {
            'ProteinFamily1': {
                'ProteinID1': [
                    ['ligand_name1', 'validity1', 'SMILES1', qed_value1],
                    ['ligand_name2', 'validity2', 'SMILES2', qed_value2]
                ],
                'ProteinID2': [
                    ['ligand_name3', 'validity3', 'SMILES3', qed_value3]
                ]
            },
            'ProteinFamily2': {
                'ProteinID3': [
                    ['ligand_name4', 'validity4', 'SMILES4', qed_value4]
                ]
            }
        }
    """
    for prot, mol_list in tqdm([(prot, mol_list)
                                for family in ligand_dict.values()
                                for prot, ligand_list in family.items()
                                for mol_list in ligand_list
                                ]):
        smiles = mol_list[2]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_id = f"{prot}_{mol_list}"
            warnings.warn(f"Could not construct molecule {mol_id} from SMILES string '{smiles}'")
            continue
        qed = QED.qed(mol)
        mol_list.append(qed)

    return ligand_dict


def filter_and_flatten(ligand_dict: dict, qed_thresh: float, max_occurences: int, seed: int) -> list:
    """
    Flattens a nested dictionary of ligands and filters molecules based on specified criteria.

    Args:
        ligand_dict (dict): A nested dictionary of ligands.
        qed_thresh (float): The minimum QED threshold for molecules to be included.
        max_occurences (int): The maximum allowed occurrences of a molecule.
        seed (int): Seed for random shuffling.

    Returns:
        list: A list of tuples (family, protein, molecule) containing filtered molecules.

    Notes:
        - The function flattens the nested dictionary structure and shuffles the examples randomly.
        - Molecules are filtered based on the following criteria:
            - Must have 'valid' as the second element in the molecule list.
            - Must have more than three elements in the molecule list.
            - Must have a QED value greater than or equal to qed_thresh.
            - The number of occurrences of a molecule must be less than max_occurences.

    Example:
        [
            ('Family1', 'Protein1', ['Molecule1:1', 'valid', 'SMILES1', qed_value1]),
            ('Family2', 'Protein2', ['Molecule2:1', 'valid', 'SMILES2', qed_value2]),
            ('Family2', 'Protein3', ['Molecule2:2', 'valid', 'SMILES3', qed_value3]),
            ...
        ]
    """
    flatten_examples = [(family, prot, mol_list) 
                    for family in ligand_dict.values()
                    for prot, lig_list in family.items()
                    for mol_list in lig_list]

    # Shuffle the flatten_examples randomly
    random.seed(seed)
    random.shuffle(flatten_examples)

    mol_name_counter = defaultdict(int)
    print("Filtering the compounds ...")

    return [
        (family, prot, mol_list)
        for family, prot, mol_list in flatten_examples
        if mol_list[1] == 'valid' and len(mol_list) > 3 and mol_list[-1] > qed_thresh 
        and mol_name_counter[mol_list[0].split(':')[0]] < max_occurences 
        and not mol_name_counter.update({mol_list[0].split(':')[0] : mol_name_counter[mol_list[0].split(':')[0]] + 1})
        ]


def split_by_ec_number(data_list: list, n_val: int, n_test: int, ec_level: int = 1) -> dict:
    """
    Split a dataset into training, validation, and test sets based on EC numbers.

    Args:
        data_list (list): A list of molecules in the form of (family, protein, molecule_list) tuples.
        n_val (int): Number of examples for the validation set.
        n_test (int): Number of examples for the test set.
        ec_level (int): The level in the EC hierarchy at which the split is performed.

    Returns:
        dict: A dictionary containing 'train', 'validation', and 'test' sets of data.

    Notes:
        - The dataset is divided into sets based on the hierarchical structure of EC numbers.
        - Molecules from the same EC subfamily (up to the specified level) are grouped together.
        - The split ensures that the validation and test sets contain a balanced representation
          of subfamilies.
    """
    examples_per_class = defaultdict(int)
    for family, _, _ in data_list:
        sub_family = '.'.join(family.split('.')[:ec_level])
        examples_per_class[sub_family] += 1

    assert sum(examples_per_class.values()) == len(data_list)

    sorted_classes = sorted(examples_per_class.items(), key=lambda x: x[1], reverse=True)

    val_classes, test_classes = set(), set()
    val_count, test_count = 0, 0

    for sub_family, num_examples in sorted_classes:
        if val_count + num_examples <= n_val:
            val_classes.add(sub_family)
            val_count += num_examples
        elif val_count >= n_val and test_count + num_examples <= n_test:
            test_classes.add(sub_family)
            test_count += num_examples

    train_classes = set(examples_per_class.keys()) - val_classes - test_classes

    # Split the data based on the classes
    data_split = {
        'train': [data for data in data_list if '.'.join(data[0].split('.')[:ec_level]) in train_classes],
        'val': [data for data in data_list if '.'.join(data[0].split('.')[:ec_level]) in val_classes],
        'test': [data for data in data_list if '.'.join(data[0].split('.')[:ec_level]) in test_classes]
    }

    assert sum(len(set_) for set_ in data_split.values()) == len(data_list)

    return data_split


def ligand_list_to_dict(ligand_list: list) -> dict:
    """
    Convert a list of ligands into a dictionary with proteins as keys and ligands as values.

    Args:
        ligand_list (list): A list of ligands represented as (family, protein, ligand_list) tuples.

    Returns:
        dict: A dictionary where protein identifiers are keys, and associated ligand lists are values.

    Notes:
        - The input list is typically structured as [(family, protein, ligand_list), ...].
        - This function rearranges the data into a dictionary for easy access to ligands per protein.
    """
    out_dict = defaultdict(list)
    for _, prot, lig in ligand_list:
        out_dict[prot].append(lig)
    return out_dict


def process_ligand_and_pocket(pdb_struct: Structure, pdbfile: Path, ligand_name: str, 
                              ligand_chain: str, ligand_resi: int, dist_cutoff: float, 
                              ca_only: bool, compute_quaternion: bool = False):
    """
    Processes the ligand and its surrounding protein pocket from a PDB structure.

    Args:
        pdb_struct (Structure): Bio.PDB structure object representing the protein.
        pdbfile (Path): Path to the PDB file.
        ligand_name (str): Name of the ligand.
        ligand_chain (str): Chain identifier of the ligand.
        ligand_resi (int): Residue number of the ligand.
        dist_cutoff (float): Distance cutoff for defining pocket-ligand interactions.
        ca_only (bool): If True, use only alpha carbon atoms for pocket representation.
        compute_quaternion (bool, optional): If True, compute quaternion for pocket backbone atoms.
            Default is False.

    Returns:
        tuple: A tuple containing two dictionaries:
            - `ligand_data` with ligand coordinates and features.
            - `pocket_data` with pocket coordinates, features, and residue identifiers.
            Optional `pocket_quaternion` is included in `pocket_data` if compute_quaternion is True.

    Raises:
        KeyError: If chain or atoms are not found in the PDB structure.
        ValueError: If invalid values are encountered in quaternion computation.
        AssertionError: If ligand names do not match.

    Notes:
        - This function processes ligand and pocket data from a PDB structure.
        - It computes coordinates, one-hot encodings, and optional quaternion transformations.
        - The ligand and pocket data are returned in separate dictionaries.
        - Optional quaternion is computed if `compute_quaternion` is True.
    """

    try:
        residues = {obj.id[1]: obj
                    for obj in pdb_struct[0][ligand_chain].get_residues()
                    }
    except KeyError as e:
        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
   
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"
    
    # Remove H if it is not inlcuded in the atom_dict. The other atom types
    # that are not allowed should remain because the entire ligand will be 
    # removed from the dataset.
    lig_atoms = [atom for atom in ligand.get_atoms()
                 if (atom.element.capitalize() in atom_dict or 
                     atom.element != 'H')
                ]
    lig_coords = np.array([atom.get_coord() 
                           for atom in lig_atoms]
                           )
    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[atom.element.capitalize()]).squeeze()
            for atom in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(
            f'Ligand atom {e} is not in atom dict ({pdbfile}, '
            f'{ligand_name}:{ligand_chain}:{ligand_resi})'
        )
    
    # Find the interacting pocket residues based on distance cut-off
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    # Compute transform of the canonical reference frame
    n_xyz = np.array([res['N'].get_coord() for res in pocket_residues])
    ca_xyz = np.array([res['CA'].get_coord() for res in pocket_residues])
    c_xyz = np.array([res['C'].get_coord() for res in pocket_residues])

    if compute_quaternion:
        quaternion, c_alpha = get_bb_transform(n_xyz, ca_xyz, c_xyz)
        if np.any(np.isnan(quaternion)):
            raise ValueError(
                'Invalid value in quaternion ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})'
            )
    else:
        c_alpha = ca_xyz

    if ca_only:
        pocket_coords = c_alpha
        try: 
            pocket_one_hot = np.stack([
                np.eye(1, len(amino_acid_dict),
                       amino_acid_dict[three_to_one(res.get_resname())]).squeeze()
                       for res in pocket_residues
            ])
        except KeyError as e:
            raise KeyError(
                f'{e} is no in the amino acid dict ({pdbfile} '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})'
            )
    else:
        pocket_atoms = [atom for res in pocket_residues for atom in res.get_atoms()
                        if (atom.element.capitalize() in atom_dict or atom.element != 'H')]
        
        pocket_coords = np.array([atom.get_coord() for atom in pocket_atoms])
        try:
            pocket_one_hot = np.stack([
                np.eye(1, len(atom_dict), atom_dict[atom.element.capitalize()]).squeeze()
                for atom in pocket_atoms
            ])
        except KeyError as e:
            raise KeyError(
                f'Pocket atom {e} is not in atom dict ({pdbfile}, '
                f'{ligand_name}:{ligand_chain:{ligand_resi}})'
            )
    
    pocket_ids = [f'{res.parent_id}:{res.id[1]}' for res in pocket_residues]

    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }

    pocket_data = {
        'pocket_coords': pocket_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids,
    }
    if compute_quaternion:
        pocket_data['pocket_quaternion'] = quaternion
    return ligand_data, pocket_data


def compute_smiles(positions: np.ndarray, one_hot: np.ndarray, mask: np.ndarray, dataset_info) -> list:
    """
    Compute SMILES representations for molecules based on atomic positions, one-hot encoding, and masks.

    Args:
        positions (np.ndarray): Atomic positions as a NumPy array of shape (n_atoms, 3).
        one_hot (np.ndarray): One-hot encoding of atom types as a NumPy array of shape (n_atoms, n_atom_types).
        mask (np.ndarray): Mask indicating sections of the molecule as a NumPy array of shape (n_atoms,).
        dataset_info: Additional information about the dataset (not specified in the original code).

    Returns:
        list: List of SMILES representations for molecules.

    Note:
        This function relies on external functions 'build_molecule' and 'rdmol_to_smiles' for molecule construction
        and conversion to SMILES, respectively.

    Example:
        >>> positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> one_hot = np.array([[1, 0], [0, 1], [1, 0]])
        >>> mask = np.array([0, 0, 1])
        >>> dataset_info = {}  # Additional dataset information
        >>> smiles_list = compute_smiles(positions, one_hot, mask, dataset_info)
    """
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)

        # BasicMolecularMetrics() computes SMILES after sanitization
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            continue

        mol = rdmol_to_smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    """
    Calculate the joint distribution of the number of nodes in ligands and pockets.

    Args:
        lig_mask (ndarray): Mask indicating the ligand associated with each node.
        pocket_mask (ndarray): Mask indicating the pocket associated with each node.
        smooth_sigma (float, optional): Standard deviation for Gaussian smoothing. Default is None.

    Returns:
        ndarray: Joint histogram of the number of nodes in ligands and pockets.

    Note:
        This function calculates the joint distribution of the number of nodes in ligands and pockets based on
        the provided masks. It first counts the number of nodes associated with each ligand and pocket.
        Then, it constructs a joint histogram where each cell (i, j) represents the number of occurrences
        where there are i nodes in the ligand and j nodes in the pocket.

    Example:
        >>> lig_mask = np.array([0, 0, 1, 1, 2, 2])
        >>> pocket_mask = np.array([0, 1, 0, 1, 2, 2])
        >>> joint_histogram = get_n_nodes(lig_mask, pocket_mask)
        >>> print(joint_histogram)
        array([[1, 1, 0],
               [0, 2, 2],
               [0, 0, 2]])
    """
    # The Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((
        np.max(n_nodes_lig) + 1,
        np.max(n_nodes_pocket) + 1
    ))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'The original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')
    
    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0
        )
        
        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    """
    Calculate bond length arrays based on atom mapping.

    Args:
        atom_mapping (dict): A dictionary that maps atom names to indices.

    Returns:
        list: A list of three 2D NumPy arrays representing bond lengths between atoms.
              The arrays correspond to single, double, and triple bonds, respectively.

    Note:
        This function calculates bond length arrays based on the provided atom mapping.
        The resulting list contains three 2D arrays where each element represents the bond length
        between two atoms. The arrays represent single, double, and triple bonds.

    Example:
        >>> atom_mapping = {'C': 0, 'H': 1, 'O': 2}
        >>> bond_arrays = get_bond_length_arrays(atom_mapping)
        >>> print(bond_arrays[0])  # Single bond lengths
        array([[0.0, 1.0, 0.0],
               [1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])
        >>> print(bond_arrays[1])  # Double bond lengths
        array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])
        >>> print(bond_arrays[2])  # Triple bond lengths
        array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])
    """
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f'bonds{i+1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    """
    Calculate Lennard-Jones radii matrix based on atom mapping.

    Args:
        atom_mapping (dict): A dictionary that maps atom names to indices.

    Returns:
        np.ndarray: A 2D NumPy array representing Lennard-Jones radii between atoms.

    Note:
        This function calculates the Lennard-Jones radii matrix based on the provided atom mapping.
        The resulting 2D NumPy array represents the Lennard-Jones radii between pairs of atoms.

    Example:
        >>> atom_mapping = {'C': 0, 'H': 1, 'O': 2}
        >>> LJ_rm = get_lennard_jones_rm(atom_mapping)
        >>> print(LJ_rm)
        array([[1.7, 0.9, 1.5],
               [0.9, 1.2, 1.0],
               [1.5, 1.0, 1.5]])
    """
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bonds_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bonds_lengths.append(bond_dict[a1][a2])
            
            if len(all_bonds_lengths) > 0:
                # Take the shortest possible bond length because slightly larger
                # values are not penalized as much
                bond_len = min(all_bonds_lengths)
            else:
                bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    """
    Calculate histograms of atom and amino acid types based on one-hot encodings.

    Args:
        lig_one_hot (np.ndarray): One-hot encoding of ligand atom types.
        pocket_one_hot (np.ndarray): One-hot encoding of pocket amino acid types.
        atom_encoder (dict): A dictionary that maps atom types to indices.
        aa_encoder (dict): A dictionary that maps amino acid types to indices.

    Returns:
        tuple: A tuple containing two dictionaries:
            - atom_counts (dict): Histogram of atom types.
            - aa_counts (dict): Histogram of amino acid types.

    Note:
        This function calculates histograms of atom and amino acid types based on the provided
        one-hot encodings and encoding dictionaries. The resulting dictionaries represent the
        counts of each atom and amino acid type in the respective encodings.

    Example:
        >>> atom_encoder = {'C': 0, 'H': 1, 'O': 2}
        >>> aa_encoder = {'ALA': 0, 'LEU': 1, 'GLY': 2}
        >>> lig_one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> pocket_one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> atom_counts, aa_counts = get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder)
        >>> print(atom_counts)
        {'C': 1, 'H': 1, 'O': 1}
        >>> print(aa_counts)
        {'ALA': 1, 'LEU': 1, 'GLY': 1}
    """
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=Path, default='data/moad')
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--qed_thresh', type=float, default=0.3)
    parser.add_argument('--max_occurences', type=int, default=50)
    parser.add_argument('--num_val', type=int, default=300)
    parser.add_argument('--num_test', type=int, default=300)
    parser.add_argument('--dist_cutoff', type=float, default=8.0)
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--make_split', action='store_true')
    args = parser.parse_args()

    pdb_dir = args.basedir / 'BindingMOAD_2020/'

    # Create the output directory
    if args.outdir is None:
        suffix = '' if 'H' in atom_dict else '_noH'
        suffix = '_ca_only' if args.ca_only else '_full'
        processed_dir = Path(args.basedir, f"processed{suffix}")
    else:
        processed_dir = args.outdir
    
    processed_dir.mkdir(exist_ok=True, parents=True)

    # Split the dataset into training, validation, and test sets
    if args.make_split:
        # Process the labels file
        csv_path = args.basedir  / 'every.csv'
        ligand_dict = read_label_file(csv_path)
        ligand_dict = compute_druglikeness(ligand_dict)
        filtered_examples = filter_and_flatten(
            ligand_dict, args.qed_thresh, args.max_occurences, args.random_seed)
        print(f"{len(filtered_examples)} examples after filtering.")

        # Split the dataset based on the EC number
        data_split = split_by_ec_number(filtered_examples,
                                        args.num_val,
                                        args.num_test)
    else:
        # Use the precomputed data split
        data_split = {}
        for split in ['test', 'val', 'train']:
            with open(f"data/moad_{split}.txt", 'r') as f:
                pocket_ids = f.read().split(',')
            # (family, prot, molecule tuple)
            data_split[split] = [(None, x.split('_')[0][:4], (x.split('_')[1],))
                                 for x in pocket_ids]
            
    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    n_samples_after = {}
    # Load and process the PDB files
    for split in data_split.keys():
        lig_coords, lig_one_hot, lig_mask, pocket_coords, pocket_one_hot,\
        pocket_mask, pdb_and_mol_ids, receptors, count = [[] for _ in range(9)], 0

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        n_tot = len(data_split[split])
        pair_dict = ligand_list_to_dict(data_split[split])

        tic = time()
        num_failed = 0
        with tqdm(total=n_tot) as pbar:
            for p in pair_dict:

                pdb_successful = set()
                # Try all available .bio files   
                for pdbfile in sorted(pdb_dir.glob(f"{p.lower()}.bio*")):

                    # Skip the pdb_file if all ligands have been processed already
                    if len(pdb_successful) == len(pair_dict[p]):
                        continue

                    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
                    struct_copy = pdb_struct.copy()
                    n_bio_successful = 0
                    for m in pair_dict[p]:

                        # Skip if the ligand is already processed
                        if m[0] in pdb_successful:
                            continue

                        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                        ligand_resi = int(ligand_resi)

                        try:
                            ligand_data, pocket_data = process_ligand_and_pocket(
                                pdb_struct,
                                pdbfile,
                                ligand_name,
                                ligand_chain,
                                ligand_resi,
                                dist_cutoff=args.dist_cutoff,
                                ca_only=args.ca_only
                            )
                        except (KeyError, AssertionError, FileNotFoundError,
                                IndexError, ValueError) as e:
                            continue

                        pdb_and_mol_ids.append(f"{p}_{m[0]}")
                        receptors.append(pdbfile.name)
                        lig_coords.append(ligand_data['lig_coords'])
                        lig_one_hot.append(ligand_data['lig_one_hot'])
                        lig_mask.append(
                            count * np.ones(len(ligand_data['lig_coords']))
                        )
                        pocket_coords.append(pocket_data['pocket_coords'])
                        # pocket_quaternion.append(
                        #     pocket_data['pocket_quaternion']
                        # )
                        pocket_one_hot.append(
                            pocket_data['pocket_one_hot']
                        )
                        pocket_mask.append(
                            count * np.ones(len(pocket_data['pocket_coords']))
                        )
                        count += 1

                        pdb_successful.add(m[0])
                        n_bio_successful += 1

                        # Save additional files for affinity analysis
                        if split in {'val', 'test'}:
                            
                            # Remove the ligand from receptor
                            try:
                                struct_copy[0][ligand_chain].detach_child((f"H_{ligand_name}",
                                                                           ligand_resi,
                                                                           ' '))
                            except KeyError:
                                warnings.warn(f"Could not find ligand {(f'H_{ligand_name}', ligand_resi, ' ')}" \
                                              f" in {pdbfile}")
                                continue

                            # Create SDF file
                            atom_types = [atom_decoder[np.argmax(index)]
                                          for index in ligand_data['lig_one_hot']]
                            xyz_file = Path(pdb_sdf_dir, 'tmp.xyz')
                            utils.write_xyz_file(ligand_data['lig_coords'], 
                                                 atom_types,
                                                 xyz_file)
                            obConversion = openbabel.OBConversion()
                            obConversion.SetInAndOutFormats("xyz", "sdf")
                            mol = openbabel.OBMol()
                            obConversion.ReadFile(mol, str(xyz_file))
                            xyz_file.unlink()

                            name = f"{p}-{pdbfile.suffix[1:]}_{m[0]}"
                            sdf_file = Path(pdb_sdf_dir, f'{name}.sdf')
                            obConversion.WriteFile(mol, str(sdf_file))

                            # Specify pocket residues
                            with open(Path(pdb_sdf_dir, f'{name}.txt'), 'w') as f:
                                f.write(' '.join(pocket_data['pocket_ids']))

                    if split in {'val', 'test'} and n_bio_successful > 0:
                        # Create receptor PDB file
                        pdb_file_out = Path(pdb_sdf_dir, 
                                            f'{p}-{pdbfile.suffix[1:]}.pdb'
                                            )
                        io = PDBIO()
                        io.set_structure(struct_copy)
                        io.save(str(pdb_file_out), select=Model0())
                
                pbar.update(len(pair_dict[p]))
                num_failed += (len(pair_dict[p]) - len(pdb_successful))
                pbar.set_description(f'#failed: {num_failed}')

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        np.savez(processed_dir / f'{split}.npz', 
                 names=pdb_and_mol_ids, 
                 receptors=receptors,
                 lig_coords=lig_coords,
                 lig_one_hot=lig_one_hot,
                 lig_mask=lig_mask,
                 pocket_coords=pocket_coords,
                 pocket_one_hot=pocket_one_hot,
                 pocket_mask=pocket_mask
                 )
        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic)/60.0:.2f} minutes")
    
    # Compute statistics and additionanl information
    with np.load(processed_dir / 'train.npz', allow_pickle=True) as data:
        lig_mask = data['lig_mask']
        pocket_mask = data['pocket_mask']
        lig_coords = data['lig_coords']
        lig_one_hot = data['lig_one_hot']
        pocket_one_hot = data['pocket_one_hot']

    # Compute smiles for all training examples
    train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask)
    np.save(processed_dir / 'train_smiles.npy', train_smiles)

    # Joint histogram of number of ligand and pocket bonds
    n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)

    # Convert bond length dictionaries to arrays for batch processing
    bonds1, bonds2, bonds3 = get_bond_length_arrays(atom_dict)

    rm_LJ = get_lennard_jones_rm(atom_dict)

    # Generate histograms of ligand and pocket node types
    atom_hist, aa_hist = get_type_histograms(lig_one_hot, 
                                             pocket_one_hot,
                                             atom_dict,
                                             amino_acid_dict)
    
    # Create summary strings:
    # Data for summary
    summary_data = {
        'Before processing': {
            'num_samples train': n_train_before,
            'num_samples val': n_val_before,
            'num_samples test': n_test_before
        },
        'After processing': {
            'num_samples train': n_samples_after['train'],
            'num_samples val': n_samples_after['val'],
            'num_samples test': n_samples_after['test']
        },
        'Info': {
            'atom_encoder': atom_dict,
            'atom_decoder': list(atom_dict.keys()),
            'aa_encoder': amino_acid_dict,
            'aa_decoder': list(amino_acid_dict.keys()),
            'bonds1': bonds1.tolist(),
            'bonds2': bonds2.tolist(),
            'bonds3': bonds3.tolist(),
            'lennard_jones_rm': rm_LJ.tolist(),
            'atom_hist': atom_hist,
            'aa_hist': aa_hist,
            'n_nodes': n_nodes.tolist()
        }
    }

    # Create summary string
    summary_string = '# SUMMARY\n\n'
    for section, data in summary_data.items():
        summary_string += f'# {section}\n'
        for key, value in data.items():
            summary_string += f'{key}: {value}\n'
        summary_string += '\n'

    # Write summary to text file
    with open(processed_dir / 'summary.txt', 'w') as f:
        f.write(summary_string)

    # Print summary
    print(summary_string)





                        
                                