import pandas as pd
import yaml
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel() # converting to 1D numpy array

gbr = GradientBoostingRegressor(random_state=42)
# maximizing => neg_mean_squared_error
grid_search = GridSearchCV(estimator=gbr, param_grid=params['gridsearch']['param_grid'], cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_params_, 'models/best_params.pkl')

