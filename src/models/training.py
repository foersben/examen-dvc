import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

best_params = joblib.load('models/best_params.pkl')
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'models/gbr_model.pkl')
