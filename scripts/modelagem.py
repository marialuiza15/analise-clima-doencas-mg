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

def treinar_modelo_por_doenca(df_treino, df_teste, top_n=3):
    # Define target
    target_col = 'capitulo_cid_causa_basica'

    # Mantém apenas as N classes mais frequentes
    top_categorias = df_treino[target_col].value_counts().nlargest(top_n).index
    df_treino = df_treino[df_treino[target_col].isin(top_categorias)]
    df_teste = df_teste[df_teste[target_col].isin(top_categorias)]

    # Features
    X_train = df_treino[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
    X_test = df_teste[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()

    # Alvo
    y_train = df_treino[target_col].astype(str)
    y_test = df_teste[target_col].astype(str)

    # LabelEncoder para as features categóricas
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category'):
            le_feat = LabelEncoder()
            todos_valores = pd.concat([X_train[col], X_test[col]]).astype(str)
            le_feat.fit(todos_valores)
            X_train[col] = le_feat.transform(X_train[col].astype(str))
            X_test[col] = le_feat.transform(X_test[col].astype(str))

    # LabelEncoder do target
    le_y = LabelEncoder()
    le_y.fit(pd.concat([y_train, y_test]))
    y_train_enc = le_y.transform(y_train)
    y_test_enc = le_y.transform(y_test)

    # Modelo
    modelo = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo.fit(X_train, y_train_enc)
    y_pred = modelo.predict(X_test)

    # Relatório
    relatorio = classification_report(
        y_test_enc, y_pred,
        labels=range(len(le_y.classes_)),
        target_names=le_y.classes_,
        zero_division=0
    )

    return modelo, relatorio, X_test, y_test_enc, le_y
