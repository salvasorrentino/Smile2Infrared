import pandas as pd
import matplotlib.pyplot as plt
from Scripts.plot_spectra_function import plot_two_spectrum, plot_three_spectrum
import numpy as np

dtf_chemprop_spectra = pd.read_pickle(r'data/results/pred_our_spectra_chemprop_raman_conv.pickle')
dtf_smile2ramam_spectra = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')
dtf_tanimoto = pd.read_pickle(r'data/results/tanimoto_spectra.pickle')
dtf_smile2ramam_IR_spectra = pd.read_pickle(r'data/results/pred_spectra_mol2raman_IR.pickle')
dtf_data = pd.read_pickle('data/raw/dtf_data_smile.pickle')

# Check if the test SMILEs are the same
lst_use = list(set(dtf_smile2ramam_spectra['smile']) & set(dtf_chemprop_spectra['SMILES']) & set(dtf_tanimoto['SMILE']))

# Sort the detaframe by metric value
dtf_smile2ramam_spectra.sort_values('F1_10cm^-1', ascending=False, inplace=True)
str_use = dtf_smile2ramam_spectra.SMILE.iloc[33]

plot_two_spectrum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]/
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]),
              dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0] /
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0]), 501, 3500, fill=False,
              rescale=1)

plot_two_spectrum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]/
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]),
              dtf_tanimoto[dtf_tanimoto['SMILE'] == str_use]['raman_pred_tan'].iloc[0] /
              sum(dtf_tanimoto[dtf_tanimoto['SMILE'] == str_use]['raman_pred_tan'].iloc[0]), 501, 3500, fill=False,
              rescale=1)

plt.figure()
# for i in range(10):
#     str_use = dtf_tanimoto[dtf_tanimoto['SMILE']=='CC(C)(C)C1(O)CC1'][f'top_{i+1}'].iloc[0]
#     plt.plot(dtf_data[dtf_data['SMILE']==str_use]['RAMAN_SPECTRUM'].iloc[0][100:])
plt.plot(dtf_tanimoto[dtf_tanimoto['SMILE']=='CO/C=N/c1nonc1']['raman_pred_tan'].iloc[0])
plt.plot(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']=='CO/C=N/c1nonc1']['RAMAN_SPECTRUM'].iloc[0])
plt.show()


# Sort Dataframe by SIS value and select a particular molecule
dtf_chemprop_spectra.sort_values('sis', ascending=False, inplace=True)
str_use = dtf_chemprop_spectra.SMILES.iloc[2]

plot_two_spectrum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]),
            np.array(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0])/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0]), 501, 3500, fill=False)

#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from Scripts.utils import generate_lorentzian_kernel, make_conv_matrix, sim, rescale, f1_score_mod, \
#     convolve_with_lorentzian, keep_peaks_prom, lorentz_conv, spectral_information_similarity, \
#     sim_dir, precision_score_mod, recall_score_mod,  calc_cos_sim, fnr_score_mod
# from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
# from Scripts.plot_spectra_function import plot_spectrum
#
# gamma = 5
# kernel_size = 600
# lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
# leng = list(range(301, 3500, 2))
# conv = make_conv_matrix(std_dev=10, frequencies=leng)
#
# dtf_smile2ramam_IR_spectra['raman_pred'] = dtf_smile2ramam_IR_spectra.apply(lambda row: row['raman_pred'] + 0.001,
#                                                                             axis=1)
# dtf_smile2ramam_IR_spectra['sis'] = dtf_smile2ramam_IR_spectra.apply(lambda row:
#                                                                      sim_dir(row['RAMAN_SPECTRUM'], row['raman_pred'],
#                                                                              conv, freq=leng,
#                                                                              lor_kernel=lorentzian_kernel), axis=1)

plot_three_spectrum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]),
            np.array(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0])/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0]),
            np.array(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0])/
            sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0]), start=501,
                    stop=3500, fill=False)