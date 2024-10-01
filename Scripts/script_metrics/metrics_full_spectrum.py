import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Scripts.utils import generate_lorentzian_kernel, make_conv_matrix, rescale, f1_score_mod, \
    convolve_with_lorentzian, keep_peaks_prom, sim_dir, precision_score_mod, \
    recall_score_mod,  calc_cos_sim, fnr_score_mod
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import pickle
from Scripts.plot_spectra_function import plot_spectrum


def process_arrays(df1, df2, col1, col2, new_column):
    # Ensure the columns are numpy arrays of the correct length
    assert all(df1[col1].apply(lambda x: len(x) == 800)), "All elements in col1 should be arrays of length 900"
    assert all(df2[col2].apply(lambda x: len(x) == 800)), "All elements in col2 should be arrays of length 900"

    def create_combined_array(arr1, arr2):
        first_700 = arr1[:700]
        last_700 = arr2[-700:]

        last_200_of_450 = arr1[-100:]
        first_200_of_900 = arr2[:100]

        mean_200 = (last_200_of_450 + first_200_of_900) / 2

        combined_array = np.concatenate([first_700, mean_200, last_700])

        return combined_array

    df1[new_column] = [create_combined_array(arr1, arr2) for arr1, arr2 in zip(df1[col1], df2[col2])]

    return df1


# Load Dataframe number of peaks predicted and raman spectrum predicted, for lower and higher spectra
df_down = pd.read_parquet(r'data/predictions/pred_spectra_predictions_fingerprint_500_2100_feat_numpeak_'
                          r'daylight_morgan_loss_8651.parquet')
df_up = pd.read_parquet(r'data/predictions/pred_spectra_predictions_ch_1900_3500_feat_numpeak_'
                        r'daylight_morgan_loss_8651.parquet')


n_peak_down = pd.read_pickle(r'data/predictions/pred_pred_num_peak_down.pickle')
n_peak_down = n_peak_down['test'].rename(columns={'TRUE_NUM_PEAK': 'raman_true_num_peak_down'})
n_peak_down = n_peak_down.rename(columns={'PRED_NUM_PEAK': 'raman_pred_num_peak_down'})

n_peak_up = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')
n_peak_up = n_peak_up['test'].rename(columns={'TRUE_NUM_PEAK': 'raman_true_num_peak_up'})
n_peak_up = n_peak_up.rename(columns={'PRED_NUM_PEAK': 'raman_pred_num_peak_up'})

raman_spectrum = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')
raman_spectrum['RAMAN_SPECTRUM_1'] = raman_spectrum['RAMAN_SPECTRUM'].apply(lambda row: row[100:900])
raman_spectrum['RAMAN_SPECTRUM_2'] = raman_spectrum['RAMAN_SPECTRUM'].apply(lambda row: row[800:])

# Add number of peaks predicted to use keep_peaks_prom
df_down = pd.merge(df_down, n_peak_down[['SMILE', 'raman_pred_num_peak_down']], left_on='smile', right_on='SMILE', how='left')
df_up = pd.merge(df_up, n_peak_up[['SMILE', 'raman_pred_num_peak_up']], left_on='smile', right_on='SMILE', how='left')
df_down = df_down.drop(['SMILE', 'pred_num_peak'], axis=1)
df_up = df_up.drop(['SMILE', 'pred_num_peak'], axis=1)

# Rescale raman_pred and raman_true
df_down['raman_pred'] = df_down['raman_pred'].apply(lambda row: rescale(range(0, 800), row))
df_up['raman_pred'] = df_up['raman_pred'].apply(lambda row: rescale(range(0, 800), row))
df_down['raman_true'] = df_down['raman_true'].apply(lambda row: rescale(range(0, 800), row))
df_up['raman_true'] = df_up['raman_true'].apply(lambda row: rescale(range(0, 800), row))

# # Rescale Intensities (only if the intensities had been rescaled in the split.py)
# df_down['raman_pred'] = df_down.apply(lambda row: row['raman_pred']/5, axis=1)
# df_down['raman_true'] = df_down.apply(lambda row: row['raman_true']/5, axis=1)

# Apply keep_peaks_prom
df_down['raman_pred'] = df_down.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.raman_pred_num_peak_down)), axis=1)
df_up['raman_pred'] = df_up.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.raman_pred_num_peak_up)), axis=1)

# Merge into a unique Dataframe for all Spectrum, first 700 down + 200 mean up&down + last 700
result_df = process_arrays(df_down, df_up, 'raman_pred', 'raman_pred', 'raman_pred')
result_df = process_arrays(df_down, df_up, 'raman_true', 'raman_true', 'raman_true')
result_df = pd.merge(result_df, raman_spectrum[['SMILE', 'RAMAN_SPECTRUM_1', 'RAMAN_SPECTRUM_2']],
                     left_on='smile', right_on='SMILE', how='inner',)
result_df = process_arrays(result_df, result_df, 'RAMAN_SPECTRUM_1', 'RAMAN_SPECTRUM_2', 'RAMAN_SPECTRUM')
result_df = result_df.drop(['RAMAN_SPECTRUM_1', 'RAMAN_SPECTRUM_2'], axis=1)

gamma = 2.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(501, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

# Tanimoto analysis code
with open(r'data/tanimoto_filterd_list_daylight.pkl', 'rb') as file:
    dayligth_dict = pickle.load(file)

with open(r'data/tanimoto_filterd_list_morgan.pkl', 'rb') as file:
    morgan_dict = pickle.load(file)

# Load a dataframe with the 10 most similar molecule (Tanimoto similiraty for Daylight or Morgan Fingerprint)
dtf_tan = pd.read_parquet('data/mol_tanimoto_daylight_fingerprint_with_spectrum.parquet')
# dtf_tan = pd.read_parquet('data/mol_tanimoto_morgan_fingerprint_with_spectrum.parquet')

## Load dictionaries with the test SMILEs as keys and the top 10 most similar molecules with the relative
## tanimoto similarity value
# dct_tan = pd.read_pickle('data/tanimoto_dictionary_morgan.pkl')
dct_tan = pd.read_pickle('data/tanimoto_dictionary_daylight.pkl')

## Filtered Dataframe with molecules which have values of Tanimoto similarity under a certain treshold
# result_df = result_df[result_df['smile'].isin(morgan_dict)]
# dtf_tan = dtf_tan[dtf_tan['SMILE'].isin(morgan_dict)]

# Metrics F1
result_df['precision_10cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                                 row['raman_pred']), axis=1)
result_df['precision_15cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                                 row['raman_pred'],
                                                                                 tolerance=7.5), axis=1)
result_df['precision_20cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                                 row['raman_pred'],
                                                                                 tolerance=10), axis=1)
result_df['recall_10cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                           row['raman_pred']), axis=1)
result_df['recall_15cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                           row['raman_pred'], tolerance=7.5), axis=1)
result_df['recall_20cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                           row['raman_pred'], tolerance=10), axis=1)
result_df['fnr_10cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred']), axis=1)
result_df['fnr_15cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'],
                                                                     row['raman_pred'], tolerance=7.5), axis=1)
result_df['fnr_20cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'],
                                                                     row['raman_pred'], tolerance=10), axis=1)
result_df['F1_10cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred']), axis=1)
result_df['F1_15cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'],
                                                                   row['raman_pred'], tolerance=7.5), axis=1)
result_df['F1_20cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'],
                                                                   row['raman_pred'], tolerance=10), axis=1)

# Metrics
result_df['sis'] = result_df.apply(lambda row: sim_dir(row['RAMAN_SPECTRUM'], row['raman_pred'], conv, freq=leng,
                                                       lor_kernel=lorentzian_kernel), axis=1)
# Convolve spectra with lorentzian curves
result_df['raman_pred'] = result_df.apply(lambda row: convolve_with_lorentzian(row['raman_pred'],
                                                                               lorentzian_kernel), axis=1)
result_df['RAMAN_SPECTRUM'] = result_df.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'],
                                                                                   lorentzian_kernel), axis=1)

# Other Metrics: R^2, MAE, RMSE, Cosine Sim
result_df['R2'] = result_df.apply(lambda row: r2_score(row['RAMAN_SPECTRUM'], row['raman_pred']), axis=1)
result_df['MAE'] = result_df.apply(lambda row: mean_squared_error(row['RAMAN_SPECTRUM'], row['raman_pred']), axis=1)
result_df['RMSE'] = result_df.apply(lambda row: root_mean_squared_error(row['RAMAN_SPECTRUM'], row['raman_pred']), axis=1)
result_df['cos_sim'] = result_df.apply(lambda row: calc_cos_sim(row['raman_pred'], row['RAMAN_SPECTRUM']), axis=1)

# Optional line to save the full spectrum after the postprocessing
result_df.to_pickle(r'data/results/pred_spectra_mol2raman.pickle')

# dtf_tan Metrics
dtf_tan['tanimoto_spectra_10'] = dtf_tan.apply(lambda row: row['tanimoto_spectra_10'][100:1600], axis=1)
dtf_tan['RAMAN_SPECTRUM'] = dtf_tan.apply(lambda row: row['RAMAN_SPECTRUM'][100:1600], axis=1)
dtf_tan['sis'] = dtf_tan.apply(lambda row: sim_dir(row['RAMAN_SPECTRUM'], row['tanimoto_spectra_10'], conv,
                                                   freq=leng, lor_kernel=lorentzian_kernel), axis=1)
dtf_tan['precision_10cm^-1'] = dtf_tan.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                             row['tanimoto_spectra_10']), axis=1)
dtf_tan['precision_15cm^-1'] = dtf_tan.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                             row['tanimoto_spectra_10'], tolerance=7.5), axis=1)
dtf_tan['precision_20cm^-1'] = dtf_tan.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'],
                                                                             row['tanimoto_spectra_10'], tolerance=10), axis=1)
dtf_tan['recall_10cm^-1'] = dtf_tan.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                       row['tanimoto_spectra_10']), axis=1)
dtf_tan['recall_15cm^-1'] = dtf_tan.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                       row['tanimoto_spectra_10'], tolerance=7.5), axis=1)
dtf_tan['recall_20cm^-1'] = dtf_tan.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'],
                                                                       row['tanimoto_spectra_10'], tolerance=10), axis=1)
dtf_tan['fnr_10cm^-1'] = dtf_tan.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['tanimoto_spectra_10']), axis=1)
dtf_tan['fnr_15cm^-1'] = dtf_tan.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'],
                                                                 row['tanimoto_spectra_10'], tolerance=7.5), axis=1)
dtf_tan['fnr_20cm^-1'] = dtf_tan.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'],
                                                                 row['tanimoto_spectra_10'], tolerance=10), axis=1)
dtf_tan['F1_10cm^-1'] = dtf_tan.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'],
                                                               row['tanimoto_spectra_10']), axis=1)
dtf_tan['F1_15cm^-1'] = dtf_tan.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'],
                                                               row['tanimoto_spectra_10'], tolerance=7.5), axis=1)
dtf_tan['F1_20cm^-1'] = dtf_tan.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'],
                                                               row['tanimoto_spectra_10'], tolerance=10), axis=1)
dtf_tan['raman_pred_tan'] = dtf_tan.apply(lambda row: convolve_with_lorentzian(row['tanimoto_spectra_10'],
                                                                               lorentzian_kernel), axis=1)
dtf_tan['RAMAN_SPECTRUM'] = dtf_tan.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'],
                                                                               lorentzian_kernel), axis=1)
dtf_tan['R2'] = dtf_tan.apply(lambda row: r2_score(row['RAMAN_SPECTRUM'], row['raman_pred_tan']), axis=1)
dtf_tan['MAE'] = dtf_tan.apply(lambda row: mean_squared_error(row['RAMAN_SPECTRUM'], row['raman_pred_tan']), axis=1)
dtf_tan['RMSE'] = dtf_tan.apply(lambda row: root_mean_squared_error(row['RAMAN_SPECTRUM'], row['raman_pred_tan']), axis=1)
dtf_tan['cos_sim'] = dtf_tan.apply(lambda row: calc_cos_sim(row['raman_pred_tan'], row['RAMAN_SPECTRUM']), axis=1)

# Save Tanimoto Dataframe
dtf_tan.to_pickle(r'data/results/tanimoto_spectra.pickle')


# Plot of Tanimoto and prediciton
lst_use = list(set(result_df['smile']) & set(dtf_tan['SMILE']))
# str_use = lst_use[2000]
result_df.sort_values('F1_10cm^-1', ascending=False, inplace=True)
str_use = result_df.SMILE.iloc[33]
# str_use = r'CCC[C@@H](O)C(F)(F)F'

plot_spectrum(result_df[result_df['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]/
              sum(result_df[result_df['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]),
              result_df[result_df['smile'] == str_use]['raman_pred'].iloc[0] /
              sum(result_df[result_df['smile'] == str_use]['raman_pred'].iloc[0]), 501, 3500, fill=False,
              rescale=1)

# Plot of True Spectrum and Tanimoto Spectrum
plt.figure()
plt.plot(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]/
         sum(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]))
plt.plot(dtf_tan[dtf_tan['SMILE']==str_use]['raman_pred_tan'].iloc[0]/
         sum(dtf_tan[dtf_tan['SMILE']==str_use]['raman_pred_tan'].iloc[0]))
plt.legend(['raman_true', 'raman_tan'])

dct_tan_tmp = {k: list(v.values())[0] for k, v in dct_tan.items()}
# dtf_tan_tmp = pd.DataFrame.from_dict(dct_tan_tmp, orient='index')
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(list(range(len(dct_tan_tmp))), dtf_tan_tmp[0])
#
#
# fig, ax = plt.subplots(1, 1, figsize=(25, 15), dpi=300)
plt.figure(dpi=200)
plt.hist(result_df['F1_15cm^-1'], bins=45, alpha=0.85, color='tab:red')
plt.hist(dtf_tan['F1_15cm^-1'], bins=45, alpha=0.85, color='tab:blue')
plt.xlabel('F1 - tolerance 15 $cm^{-1}$')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions', 'Average on Tanimoto similarity'])
#
# plt.figure(dpi=200)
# plt.hist(result_df[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='tab:red')
# plt.hist(dtf_tan[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='tab:blue')
# plt.xlabel('F1 - tolerance 15 $cm^{-1}$')
# plt.ylabel('Number of molecules')
# plt.legend(['Model Predictions', 'Average on Tanimoto similarity'])
#
#
# dtf_data = pd.read_pickle('data/raw/dtf_data_smile_no_conv.pickle')
# plt.figure(dpi=200)
# # str_use = dtf_data.SMILE.iloc[9517]
# str_use = 'C[C@H]1OCCC(=C1)C'
# plt.plot(np.linspace(501, 3500, 1500),
#          dtf_data[dtf_data['SMILE'] == str_use]['RAMAN_SPECTRUM'].iloc[0][100:], color='tab:blue')
# (dtf_data[dtf_data['SMILE'] == str_use]['RAMAN_SPECTRUM'].iloc[0][100:]!=0).sum()
# plt.ylabel('Intensity (a.u.)')
# plt.xlabel('Raman shift ($cm^{-1}$)')

