import pandas as pd
from Scripts.utils_model.utils_metrics import metrics_ir_peaks, metrics_spectra, make_conv_matrix, rescale, \
    keep_peaks_prom
from Scripts.utils_model.utils_data_processing import post_processing_pred

# Load Dataframe number of peaks predicted and IR spectrum predicted
df_predictions_sp = pd.read_parquet(r'data/predictions/pred_IR_spectra_predictions_fingerprint_300_3500_prova.parquet')
df_reaL_sp = pd.read_pickle(r'data/raw/test_dtf_data_IR_smile_no_dup_no_conv.pickle')
result_df = pd.merge(df_predictions_sp, df_reaL_sp[['SMILE', 'IR_SPECTRUM']], left_on='smile', right_on='SMILE', how='inner')
result_df = result_df.drop('SMILE', axis=1)

# Rescale IR_pred and IR_true
result_df['IR_pred'] = result_df['IR_pred'].apply(lambda row: rescale(range(0, 1600), row))
result_df['IR_true'] = result_df['IR_true'].apply(lambda row: rescale(range(0, 1600), row))

# Apply keep_peaks_prom
result_df['IR_pred'] = result_df.apply(lambda row: keep_peaks_prom(row.IR_pred, round(row.pred_num_peak)), axis=1)

# Parameters for Metrics calculations
leng = list(range(301, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

result_df = post_processing_pred(result_df)

result_df = metrics_ir_peaks(result_df)

result_df = metrics_spectra(result_df, conv, leng, true_col='IR_SPECTRUM')

# Save the IR dataframe
# result_df.to_pickle(r'data/results/pred_spectra_mol2IR_IR.pickle')

print('Mean F1_20cm^-1 score value:', result_df['F1_20cm^-1'].mean())
print('Mean F1_30cm^-1 score value:', result_df['F1_30cm^-1'].mean())
print('Mean sis value:', result_df.sis.mean())
print('Mean cosine sim value:', result_df.cos_sim.mean())
