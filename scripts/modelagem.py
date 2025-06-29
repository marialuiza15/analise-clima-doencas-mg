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

def testar_com_dados_futuros(modelo_treinado, df_teste):
    df_teste = df_teste.sort_values('data').copy()

    X_test = df_teste[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y_test = df_teste['risco_obito'].astype(str)

    # Codificação das variáveis categóricas
    for col in X_test.columns:
        if X_test[col].dtype == 'object' or str(X_test[col].dtype).startswith('category'):
            X_test[col] = X_test[col].astype(str)
            X_test[col] = LabelEncoder().fit_transform(X_test[col])  # Para produção real, você deveria usar o mesmo encoder do treino

    y_test_enc = LabelEncoder().fit_transform(y_test)  # Mesma observação: idealmente, use o mesmo encoder do treino

    # Previsão
    y_pred = modelo_treinado.predict(X_test)

    # Avaliação
    print("\n📊 Avaliação com dados de 2023:")
    print(classification_report(y_test_enc, y_pred))
