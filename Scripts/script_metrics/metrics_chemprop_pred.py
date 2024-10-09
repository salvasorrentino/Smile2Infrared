import pandas as pd
from Scripts.utils_model.utils_metrics import metrics_ir_peaks, metrics_spectra, make_conv_matrix


chemprop_prediction = pd.read_pickle(r'data/results/chemprop_predictions/chemprop_ir_their_model_their_computed_mol.pickle')
our_prediction = pd.read_pickle(r'data/results/pred_spectra_mol2raman_ir.pickle')

result_df = chemprop_prediction

leng = list(range(400, 4001, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

result_df = metrics_ir_peaks(result_df)

result_df = metrics_spectra(result_df, conv, leng, true_col='IR_SPECTRUM')

# result_df.to_pickle(r'data/results/chemprop_ir_their_model_their_computed_mol.pickle')

result_df.sort_values('sis', ascending=False, inplace=True)
