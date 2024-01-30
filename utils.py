from typing import List

def write_xyz_file(coords: List[List[float]], atom_types: List[str], filename: str) -> None:
    """
    Write atomic coordinates and atom types to an XYZ file.

    Args:
        coords (List[List[float]]): A list of coordinate triplets, each as [x, y, z].
        atom_types (List[str]): A list of atom types corresponding to the coordinates.
        filename (str): The name of the XYZ file to be created.

    Returns:
        None

    Raises:
        AssertionError: If the length of 'coords' is not equal to the length of 'atom_types'.

    Example:
        >>> coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        >>> atom_types = ['C', 'O']
        >>> write_xyz_file(coords, atom_types, 'molecule.xyz')
    """
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i][0]:.3f} {coords[i][1]:.3f} {coords[i][2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)
