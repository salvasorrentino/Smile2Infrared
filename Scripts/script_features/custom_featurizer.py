from deepchem import feat
import Scripts.script_features.custom_featurizer_utils


def custom_featuriz(config):
    if config == 'Pagtn':
        CustomFeaturizer = feat.PagtnMolGraphFeaturizer
    elif config == 'DMPNN':
        CustomFeaturizer = feat.DMPNNFeaturizer
    else:
        CustomFeaturizer = feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
    return CustomFeaturizer
