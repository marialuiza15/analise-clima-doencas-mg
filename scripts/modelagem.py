import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def treinar_modelos(df):
    df = df.sort_values('data').copy()
    X = df[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y = df['risco_obito']

    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            X[col] = X[col].astype(str)  # Converta para string primeiro
            X[col] = LabelEncoder().fit_transform(X[col])

    y = LabelEncoder().fit_transform(y.astype(str))

    tscv = TimeSeriesSplit(n_splits=5)
    modelo = RandomForestClassifier()

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\nFold {i+1}")
        print(classification_report(y_test, y_pred))

def treinar_modelo_por_doenca(df_treino, df_teste, top_n=5):
    # Top N classes mais comuns no treino
    top_classes = df_treino['capitulo_cid_causa_basica'].value_counts().nlargest(top_n).index

    # Filtra treino e teste para manter só essas classes
    df_treino = df_treino[df_treino['capitulo_cid_causa_basica'].isin(top_classes)]
    df_teste = df_teste[df_teste['capitulo_cid_causa_basica'].isin(top_classes)]

    # Features e target
    X_train = df_treino[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y_train = df_treino['capitulo_cid_causa_basica']

    X_test = df_teste[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y_test = df_teste['capitulo_cid_causa_basica']

    # Codificação
    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train.astype(str))
    y_test_enc = le_y.transform(y_test.astype(str))

    for col in ['faixa_etaria', 'clima_extremo']:
        le_col = LabelEncoder()
        X_train[col] = le_col.fit_transform(X_train[col].astype(str))
        X_test[col] = le_col.transform(X_test[col].astype(str))

    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train_enc)
    y_pred = modelo.predict(X_test)

    relatorio = classification_report(y_test_enc, y_pred, target_names=le_y.classes_)

    return modelo, relatorio, X_test, y_test_enc, le_y, top_classes
