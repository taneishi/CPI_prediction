#!/usr/bin/env python
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger

def create_atoms(mol):
    '''Create a list of atom (e.g., hydrogen and oxygen) IDs considering the aromaticity.'''
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    '''Create a dictionary, which each key is a node ID and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs.'''
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    '''Extract the r-radius subgraphs (i.e., fingerprints) from a molecular graph using Weisfeiler-Lehman algorithm.'''

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            '''Update each node ID considering its neighboring nodes and edges (i.e., r-radius subgraphs or fingerprints).'''
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            '''Also update each edge ID considering two nodes on its both sides.'''
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def main():
    dataset = 'human'
    # dataset = 'celegans'

    # radius = 0  # w/o fingerprints (i.e., atoms).
    # radius = 1
    radius = 2
    # radius = 3

    # ngram = 2
    ngram = 3
    
    with open('../dataset/%s/original/data.txt' % dataset, 'r') as f:
        data_list = f.read().strip().split('\n')

    '''Exclude data contains '.' in the SMILES format.'''
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    compounds, adjacencies, proteins, interactions = [], [], [], []

    for index, data in enumerate(data_list, 1):
        smiles, sequence, interaction = data.strip().split()

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        words = split_sequence(sequence, ngram)
        proteins.append(words)

        interactions.append(np.array([float(interaction)]))

        print('\r%5d/%5d' % (index, len(data_list)), end='')
    print('')

    dir_input = ('../dataset/%s/input/radius%d_ngram%d/' % (dataset, radius, ngram))

    # Create a dataset and split it into train/test.
    dataset_ = zip(compounds, adjacencies, proteins, interactions)
    dataset_train, dataset_test = train_test_split(list(dataset_), train_size=0.8, test_size=0.2, shuffle=True, stratify=interactions)

    np.savez_compressed('dataset.npz', 
            dataset_train=dataset_train, dataset_test=dataset_test,
            n_fingerprint=len(fingerprint_dict), n_word=len(word_dict))

    print('The preprocess of ' + dataset + ' dataset has finished!')

if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    main()
