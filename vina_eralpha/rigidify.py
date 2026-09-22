"""Turn a flexible Vina ligand pdbqt into a rigid one.

Vina searches torsions by default, so docking two conformers of the same
molecule returns nearly the same score -- measured at 0.03-0.09 kcal/mol spread,
about 1% of the between-compound spread. To ask "which conformer fits best" the
ligand has to be held rigid, which is what Glide's rigid mode does.

obabel -xr writes a receptor-style pdbqt with no torsion tree and Vina rejects
it. The ligand format needs a tree, so this keeps every atom in ROOT and
declares zero torsional degrees of freedom.
"""
import sys

def rigidify(src, dst):
    atoms = [l for l in open(src) if l.startswith(("ATOM", "HETATM"))]
    with open(dst, "w") as f:
        f.write("ROOT\n")
        f.writelines(atoms)
        f.write("ENDROOT\nTORSDOF 0\n")
    return len(atoms)

if __name__ == "__main__":
    print(rigidify(sys.argv[1], sys.argv[2]), "atoms")
