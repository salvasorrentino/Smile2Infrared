import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt


def n_atoms(smile):
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()
    return num_atoms


def n_bond(smile):
    mol = Chem.MolFromSmiles(smile)
    num_bonds = mol.GetNumBonds()
    return num_bonds


dtf_results = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')

dtf_results['n_atoms'] = dtf_results.apply(lambda row: n_atoms(row['smile']), axis=1)
dtf_results['n_bond'] = dtf_results.apply(lambda row: n_bond(row['smile']), axis=1)

plt.figure()
plt.scatter(dtf_results['n_atoms'], dtf_results['sis'], color='blue', label='Predizioni')
plt.xlabel('Number of Atoms')
plt.ylabel('Sis Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(dtf_results['n_bond'], dtf_results['sis'], color='blue', label='Predizioni')
plt.xlabel('Number of Bond')
plt.ylabel('Sis Value')
plt.legend()
plt.grid(True)
plt.show()