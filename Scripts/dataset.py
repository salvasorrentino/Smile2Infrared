import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
from tqdm import tqdm
from rdkit import Chem 
from Scripts.utils import featurize_with_retry
from Scripts.script_features.custom_featurizer import custom_featuriz
import deepchem as dc
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class MoleculeDataset(Dataset, object):
    # @property
    # def processed_paths(self):
    #     return self._processed_paths

    def __init__(self, root, filename, target_col, additional_feats=True, test=None, transform=None,
                 pre_transform=None, path_proc=None, global_feature_col=None,
                 featuriz: str = 'MolGraphConv', remove_bad_mol: bool = False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        # if processed_paths is None:
        #     processed_paths = []
        self.test = test
        self.filename = filename
        self.target_col = target_col
        self.additional_feats = additional_feats
        self.path_proc = path_proc
        self.featuriz = featuriz
        self.remove_bad_mol = remove_bad_mol
        self.data = self.load_data()
        self.global_feature_col = global_feature_col
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    def load_data(self):

        data = pd.read_pickle(rf'data/raw/{self.filename}').reset_index()

        data = self.filter_data(data)
        data = data.drop('index', axis=1)
        data = data.reset_index()
        data = data.drop('index', axis=1)

        return data

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        # self.data = pd.read_pickle(self.raw_paths[0]).reset_index()

        if self.test == 'test':
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.test == 'validation':
            return [f'data_validation_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def filter_data(self, data):
        if self.remove_bad_mol:
            mol_to_remove = pd.read_pickle(rf'data/raw/remove_molecules_all.pickle')
            merged_df = pd.merge(data, mol_to_remove, on='SMILE', how='left', indicator=True)
            filtered_df = merged_df[merged_df['_merge'] == 'left_only']
            data = filtered_df.drop(columns=['_merge'])
            print('Bad Molecules removed')

        data = (data.assign(smile_len=lambda x: x.SMILE.apply(len)).query("smile_len > 1").
                drop("smile_len", axis=1))
        return data

    @staticmethod
    def molecule_from_smiles(smiles):
        # MolFromSmiles(m, sanitize=True) should be equivalent to
        # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)

        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        
        return molecule
    
    def process(self):
        
        # data_raw = pd.read_pickle(self.raw_paths[0]).reset_index()

        # self.data = MoleculeDataset.filter_data(data_raw)

        # featurizer = custom_featuriz(self.featuriz)
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        # featurizer = dc.feat.PagtnMolGraphFeaturizer()
        # featurizer = dc.feat.DMPNNFeaturizer()
        i = 0
        # lst_error = []
        for index, row in tqdm(self.data.iterrows()):
            # Featurize molecule
            mol = MoleculeDataset.molecule_from_smiles(row["SMILE"])
            try:
                # f = featurizer._featurize(mol)
                f = featurize_with_retry(featurizer, mol)
            except:
                print(f'Smile n°{index}')
                print(row[f"SMILE"])
                print(mol)
                raise ValueError("Gnoooooo")

            data = f.to_pyg_graph()             # old
            data.y = self._get_label(row[self.target_col])
            data.smiles = row["SMILE"]

            # Graph Level Feats
            selected_cols = self.global_feature_col
            selected_values = []

            for val in row[selected_cols].values:
                if isinstance(val, list):
                    selected_values.extend(val)
                else:
                    selected_values.append(val)

            # selected_values = row[selected_cols].values.astype(np.float64)
            selected_values = np.array(selected_values, dtype=np.float64)
            torch_tensor = torch.from_numpy(selected_values)
            data.graph_level_feats = torch_tensor

            if self.test == 'validation':
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_validation_{i}.pt'))
            elif self.test == 'test':
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_test_{i}.pt'))
            else:
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_{i}.pt'))
            i += 1
        print('Dataset Done!')

    def _get_label(self, label):
        return torch.tensor(label, dtype=torch.float64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test == 'test':
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        elif self.test == 'validation':
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_validation_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.path_proc)

    # @processed_paths.setter
    # def processed_paths(self, value):
    #     self._processed_paths = value
