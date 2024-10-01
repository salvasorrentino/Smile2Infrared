import pandas as pd
import numpy as np
from Scripts.utils import generate_lorentzian_kernel, make_conv_matrix, sim, rescale, f1_score_mod, \
    convolve_with_lorentzian, keep_peaks_prom, lorentz_conv, spectral_information_similarity, \
     count_matched_peaks, sim_dir, precision_score_mod, recall_score_mod, calc_cos_sim, fnr_score_mod
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from numpy.linalg import norm
from numpy import dot
import pickle


raman_spectrum = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')

result_df = pd.read_parquet(r'data/predictions/pred_spectra_predictions_fingerprint_500_2100_feat_numpeak_loss_8651.parquet')
raman_spectrum['RAMAN_SPECTRUM_1'] = raman_spectrum.apply(lambda row: row.RAMAN_SPECTRUM[100:900], axis=1)
leng = list(range(501, 2100, 2))

# result_df = pd.read_parquet(r'data/predictions/pred_spectra_predictions_ch_1900_3500_feat_numpeak_loss_8651.parquet')
# raman_spectrum['RAMAN_SPECTRUM_1'] = raman_spectrum.apply(lambda row: row.RAMAN_SPECTRUM[800:1600], axis=1)
# leng = list(range(1901, 3500, 2))

result_df = pd.merge(result_df, raman_spectrum, left_on='smile', right_on='SMILE', how='inner')
result_df = result_df.drop('SMILE', axis=1)

gamma = 2.5
kernel_size = 200
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)

conv = make_conv_matrix(std_dev=10, frequencies=leng)

# Rescale
result_df['raman_pred']=result_df.apply(lambda row:  rescale(row.RAMAN_SPECTRUM_1, row.raman_pred), axis=1)

# Keep peaks based on prominence and predicted number of peaks
result_df['raman_pred']=result_df.apply(lambda row:  keep_peaks_prom(row.raman_pred, round(row.pred_num_peak)), axis=1)

# Calculation of Precision, Recall, FNR, F1
result_df['precision_10cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['precision_15cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=7.5), axis=1)
result_df['precision_20cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=10), axis=1)
result_df['recall_10cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['recall_15cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=7.5), axis=1)
result_df['recall_20cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=10), axis=1)
result_df['fnr_10cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['fnr_15cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=7.5), axis=1)
result_df['fnr_20cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=10), axis=1)
result_df['F1_10cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['F1_15cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=7.5), axis=1)
result_df['F1_20cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM_1'], row['raman_pred'], tolerance=10), axis=1)

# Convolve RAMAN_SPECTRUM and prediction with a lorentzian
# result_df['raman_pred']=result_df.apply(lambda row:  convolve_with_lorentzian(row.raman_pred, lorentzian_kernel), axis=1)
# result_df['RAMAN_SPECTRUM_1']=result_df.apply(lambda row:  convolve_with_lorentzian(row.RAMAN_SPECTRUM_1, lorentzian_kernel), axis=1)
result_df['raman_pred']=result_df.apply(lambda row:  lorentz_conv(row.raman_pred, len(row.raman_pred)), axis=1)
result_df['RAMAN_SPECTRUM_1']=result_df.apply(lambda row:  lorentz_conv(row.RAMAN_SPECTRUM_1, len(row.RAMAN_SPECTRUM_1)), axis=1)

# Calculation of metrics
result_df['sis'] = result_df.apply(lambda row: spectral_information_similarity(row['RAMAN_SPECTRUM_1'], row['raman_pred'], conv, frequencies=leng), axis=1)
result_df['R2'] = result_df.apply(lambda row: r2_score(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['MAE'] = result_df.apply(lambda row: mean_squared_error(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['RMSE'] = result_df.apply(lambda row: root_mean_squared_error(row['RAMAN_SPECTRUM_1'], row['raman_pred']), axis=1)
result_df['cos_sim'] = result_df.apply(lambda row: calc_cos_sim(row['raman_pred'], row['RAMAN_SPECTRUM_1']), axis=1)

excel = [result_df['precision_10cm^-1'].mean(), result_df['precision_15cm^-1'].mean(), result_df['precision_20cm^-1'].mean(),
        result_df['recall_10cm^-1'].mean(), result_df['recall_15cm^-1'].mean(), result_df['recall_20cm^-1'].mean(),
        result_df['fnr_10cm^-1'].mean(), result_df['fnr_15cm^-1'].mean(), result_df['fnr_20cm^-1'].mean(),
        result_df['F1_10cm^-1'].mean(), result_df['F1_15cm^-1'].mean(), result_df['F1_20cm^-1'].mean(),
        result_df['sis'].mean(), result_df['cos_sim'].mean(),
        result_df['MAE'].mean(), result_df['RMSE'].mean(), result_df['R2'].mean()]

# result_df.sort_values('sis', inplace=True, ascending=False)