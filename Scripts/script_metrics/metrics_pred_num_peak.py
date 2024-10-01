from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datas = pd.read_pickle(r'data/predictions/pred_pred_num_peak_IR.pickle')
true_values = datas['test']['TRUE_NUM_PEAK'].astype(int)
predicted_values = datas['test']['PRED_NUM_PEAK'].astype(int)

# Calcola il Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
print("RMSE:", rmse)

# Calcola il coefficiente di determinazione (R2)
r2 = r2_score(true_values, predicted_values)
print("R^2:", r2)

# Calcola l'accuracy
accuracy = accuracy_score(true_values, predicted_values)
print("accuracy:", accuracy)

plt.scatter(true_values, predicted_values, color='blue', label='Predizioni')

# Linea di riferimento
min_val = min(min(true_values), min(predicted_values))
max_val = max(max(true_values), max(predicted_values))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Aggiungi etichette e titolo
plt.xlabel('Valori Reali')
plt.ylabel('Valori Predetti')
plt.title('Scatter Plot delle Predizioni vs Valori Reali')
plt.legend()
plt.grid(True)
plt.show()