import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

model = joblib.load('models/gbr_model.pkl')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

predictions = model.predict(X_test)
pd.DataFrame(predictions, columns=['prediction']).to_csv('data/prediction.csv', index=False)

metrics = {
    'mse': mean_squared_error(y_test, predictions),
    'r2': r2_score(y_test, predictions)
}

with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)
