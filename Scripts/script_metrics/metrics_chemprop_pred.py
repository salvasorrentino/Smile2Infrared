import pandas as pd
from Scripts.utils import convolve_with_lorentzian, generate_lorentzian_kernel, calc_cos_sim, spectral_information_similarity, \
    f1_score_mod, precision_score_mod, recall_score_mod, fnr_score_mod, make_conv_matrix
from Scripts.plot_spectra_function import plot_two_spectrum
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def apply_find_peaks(signal, prom):
    mask, _ = find_peaks(signal, prominence=prom)
    peaks_signal = np.zeros_like(signal)
    peaks_signal[mask] = np.array(signal)[mask]
    return peaks_signal


def metrics_raman_peaks(result_df, prom):
    result_df['precision_10cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom), axis=1)
    result_df['precision_15cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                         tolerance=7.5), axis=1)
    result_df['precision_20cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                         tolerance=10), axis=1)
    result_df['recall_10cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom),
                                                  axis=1)
    result_df['recall_15cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                               tolerance=7.5), axis=1)
    result_df['recall_20cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                               tolerance=10), axis=1)
    result_df['fnr_10cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom), axis=1)
    result_df['fnr_15cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                         tolerance=7.5), axis=1)
    result_df['fnr_20cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                         tolerance=10), axis=1)
    result_df['F1_10cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom), axis=1)
    result_df['F1_15cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                       tolerance=7.5), axis=1)
    result_df['F1_20cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], prominence=prom,
                                                                       tolerance=10), axis=1)
    return result_df


def metrics_ir_peaks(result_df):
    result_df['precision_30cm^-1'] = result_df.apply(
        lambda row: precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
    result_df['precision_20cm^-1'] = result_df.apply(
        lambda row: precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

    result_df['recall_30cm^-1'] = result_df.apply(
        lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
    result_df['recall_20cm^-1'] = result_df.apply(
        lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

    result_df['fnr_30cm^-1'] = result_df.apply(
        lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
    result_df['fnr_20cm^-1'] = result_df.apply(
        lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

    result_df['F1_30cm^-1'] = result_df.apply(
        lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
    result_df['F1_20cm^-1'] = result_df.apply(
        lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)
    return result_df


def metrics_spectra(result_df, conv, leng, true_col='RAMAN_SPECTRUM_CONV', pred_col='raman_pred'):
    result_df['sis'] = result_df.apply(lambda row: spectral_information_similarity(pd.array(row[true_col]),
                                                                                   pd.array(row[pred_col]), conv,
                                                                                   frequencies=leng), axis=1)
    result_df['R2'] = result_df.apply(lambda row: r2_score(row[true_col], row[pred_col]), axis=1)
    result_df['MAE'] = result_df.apply(lambda row: mean_squared_error(row[true_col], row[pred_col]), axis=1)
    result_df['RMSE'] = result_df.apply(lambda row: root_mean_squared_error(row[true_col], row[pred_col]), axis=1)
    result_df['cos_sim'] = result_df.apply(lambda row: calc_cos_sim(row[pred_col], row[true_col]), axis=1)
    return result_df


def excel_data_raman(result_df):
    excel = [result_df['precision_10cm^-1'].mean(), result_df['precision_15cm^-1'].mean(), result_df['precision_20cm^-1'].mean(),
            result_df['recall_10cm^-1'].mean(), result_df['recall_15cm^-1'].mean(), result_df['recall_20cm^-1'].mean(),
            result_df['fnr_10cm^-1'].mean(), result_df['fnr_15cm^-1'].mean(), result_df['fnr_20cm^-1'].mean(),
            result_df['F1_10cm^-1'].mean(), result_df['F1_15cm^-1'].mean(), result_df['F1_20cm^-1'].mean(),
            result_df['sis'].mean(), result_df['cos_sim'].mean(),
            result_df['MAE'].mean(), result_df['RMSE'].mean(), result_df['R2'].mean()]
    return excel


def excel_data_ir(result_df):
    excel = [result_df['precision_20cm^-1'].mean(), result_df['precision_30cm^-1'].mean(),
             result_df['recall_20cm^-1'].mean(), result_df['recall_30cm^-1'].mean(),
             result_df['fnr_20cm^-1'].mean(), result_df['fnr_30cm^-1'].mean(),
             result_df['F1_20cm^-1'].mean(), result_df['F1_30cm^-1'].mean(),
             result_df['sis'].mean(), result_df['cos_sim'].mean(),
             result_df['MAE'].mean(), result_df['RMSE'].mean(), result_df['R2'].mean()]
    return excel


chemprop_prediction = pd.read_pickle(r'data/results/chemprop_predictions/chemprop_ir_their_model_their_computed_mol.pickle')
our_prediction = pd.read_pickle(r'data/results/pred_spectra_mol2raman_ir.pickle')
# chemprop_prediction = pd.read_pickle(r'data/results/chemprop_predictions/chemprop_raman_10cm_conv_pred.pickle')
# our_prediction = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')

result_df = chemprop_prediction

# gamma = 7.5
# prom = 0.05
# kernel_size = 600
# lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(400, 4001, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

# result_df = metrics_raman_peaks(result_df)
result_df = metrics_ir_peaks(result_df)

result_df = metrics_spectra(result_df, conv, leng, true_col='RAMAN_SPECTRUM')

# result_df.to_pickle(r'data/results/chemprop_ir_their_model_their_computed_mol.pickle')

result_df.sort_values('sis', ascending=False, inplace=True)

# i = 600
# str_smile = result_df.SMILES.iloc[i]
# # plot_spectrum(result_df['RAMAN_SPECTRUM_CONV'].iloc[i], result_df['raman_pred'].iloc[i], start=301, stop=3500, fill=False)
#
# plt.figure()
# plt.plot(result_df['RAMAN_SPECTRUM_CONV'].iloc[i]/result_df['RAMAN_SPECTRUM_CONV'].iloc[i].sum())
# plt.plot(np.array(result_df['raman_pred'].iloc[i])/sum(result_df['raman_pred'].iloc[i]))
# plt.plot(np.array(our_prediction[our_prediction.smile == str_smile].raman_pred.iloc[0])/
#          sum(our_prediction[our_prediction.smile == str_smile].raman_pred.iloc[0]))
# plt.show()
excel_raman = excel_data_ir(result_df)
# print('Smile2Raman F1:', our_prediction['F1_30cm^-1'].mean())
# print('Smile2Raman sis:', our_prediction['sis'].mean())
result_df.sort_values('sis', ascending=False, inplace=True)
print('Chemprop F1:', result_df['F1_30cm^-1'].mean())
print('Chemprop sis:', result_df['sis'].mean())

i = 8521
str_smile = result_df.SMILES.iloc[i]
plt.figure()
plt.plot(result_df['RAMAN_SPECTRUM'].iloc[i]/result_df['RAMAN_SPECTRUM'].iloc[i].sum())
plt.plot(np.array(result_df['raman_pred'].iloc[i])/sum(result_df['raman_pred'].iloc[i]))
plt.show()

# plt.plot(np.array(our_prediction[our_prediction.smile == str_smile].raman_pred.iloc[0])/
#          sum(our_prediction[our_prediction.smile == str_smile].raman_pred.iloc[0]))


plt.figure()
# plt.plot(our_prediction[our_prediction.smile=='COC(=O)C(C)C']['raman_pred'].iloc[0][50:]/
#          our_prediction[our_prediction.smile=='COC(=O)C(C)C']['raman_pred'].iloc[0][50:].sum())
plt.plot(result_df[result_df.SMILES=='COC(=O)C(C)C']['raman_pred'].iloc[0][:-250]/
         result_df[result_df.SMILES=='COC(=O)C(C)C']['raman_pred'].iloc[0][:-250].sum())
plt.plot(result_df[result_df.SMILES=='COC(=O)C(C)C']['RAMAN_SPECTRUM'].iloc[0][:-250]/
         result_df[result_df.SMILES=='COC(=O)C(C)C']['RAMAN_SPECTRUM'].iloc[0][:-250].sum())
# plt.plot(our_prediction[our_prediction.smile=='COC(=O)C(C)C']['RAMAN_SPECTRUM_CONV'].iloc[0][50:]/
#          our_prediction[our_prediction.smile=='COC(=O)C(C)C']['RAMAN_SPECTRUM_CONV'].iloc[0][50:].sum())

plt.figure()
# plt.plot(our_prediction[our_prediction.smile=='CCOc1ccccc1']['raman_pred'].iloc[0][50:]/
#          our_prediction[our_prediction.smile=='CCOc1ccccc1']['raman_pred'].iloc[0][50:].sum())
# plt.plot(result_df[result_df.SMILES=='CCOc1ccccc1']['raman_pred'].iloc[0][:-250]/
#          result_df[result_df.SMILES=='CCOc1ccccc1']['raman_pred'].iloc[0][:-250].sum())
plt.plot(np.linspace(400, 3500, 1551), result_df[result_df.SMILES=='CCOc1ccccc1']['RAMAN_SPECTRUM'].iloc[0][:-250]/
         result_df[result_df.SMILES=='CCOc1ccccc1']['RAMAN_SPECTRUM'].iloc[0][:-250].sum())
plt.plot(np.linspace(400, 3500, 1550), our_prediction[our_prediction.smile=='CCOc1ccccc1']['RAMAN_SPECTRUM_CONV'].iloc[0][50:]/
         our_prediction[our_prediction.smile=='CCOc1ccccc1']['RAMAN_SPECTRUM_CONV'].iloc[0][50:].sum())



