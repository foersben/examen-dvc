import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

df = pd.read_csv('data/raw/raw.csv')

drop_cols = ['silica_concentrate']
if 'date' in df.columns:
    drop_cols.append('date')

X = df.drop(columns=drop_cols)
y = df['silica_concentrate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['split']['test_size'], 
    random_state=params['split']['random_state']
)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

