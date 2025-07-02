import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path='data/synthetic_mmorpg_survey.csv'):
    return pd.read_csv(path)

def encode_categoricals(df, exclude=None):
    if exclude is None:
        exclude = []
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col not in exclude:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler
