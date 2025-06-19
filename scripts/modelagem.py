import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def treinar_modelos(df):
    df = df.sort_values('data')
    X = df[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']]
    y = df['risco_obito']

    for col in X.select_dtypes(include='category').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    y = LabelEncoder().fit_transform(y)

    tscv = TimeSeriesSplit(n_splits=5)
    modelo = RandomForestClassifier()

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\nFold {i+1}")
        print(classification_report(y_test, y_pred))
