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

def treinar_modelo_final(df_treino, df_teste):
    # Ordena por data
    df_treino = df_treino.sort_values('data')
    df_teste = df_teste.sort_values('data')

    # Seleção de features e alvo
    X_train = df_treino[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y_train = df_treino['risco_obito']

    X_test = df_teste[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    y_test = df_teste['risco_obito']

    # Encoding das features categóricas
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category'):
            le_feat = LabelEncoder()
            todos_valores = pd.concat([X_train[col], X_test[col]]).astype(str)
            le_feat.fit(todos_valores)
            X_train[col] = le_feat.transform(X_train[col].astype(str))
            X_test[col] = le_feat.transform(X_test[col].astype(str))

    # Encoding do target
    le_y = LabelEncoder()
    todos_riscos = pd.concat([y_train, y_test]).astype(str)
    le_y.fit(todos_riscos)
    y_train = le_y.transform(y_train.astype(str))
    y_test = le_y.transform(y_test.astype(str))

    # Treinamento
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Relatório com todas as classes previstas
    relatorio = classification_report(
        y_test, y_pred,
        labels=range(len(le_y.classes_)),
        target_names=le_y.classes_
    )
    return modelo, relatorio
