import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

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

def treinar_modelo_com_smote_gridsearch(df_treino, df_teste, target_col='capitulo_cid_causa_basica', top_n=2):
    # Foco nas top N classes
    top_classes = df_treino[target_col].value_counts().nlargest(top_n).index
    df_treino = df_treino[df_treino[target_col].isin(top_classes)]
    df_teste = df_teste[df_teste[target_col].isin(top_classes)]

    # Features disponíveis
    colunas_features = [
        'TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo',
        'temperatura_classe', 'umidade_classe', 'sexo', 'raca_cor',
        'mes', 'estacao', 'municipio'
    ]
    colunas_features = [col for col in colunas_features if col in df_treino.columns]

    # Separa X e y
    X_train = df_treino[colunas_features].copy()
    X_test = df_teste[colunas_features].copy()
    y_train = df_treino[target_col].astype(str)
    y_test = df_teste[target_col].astype(str)

    # Codifica features categóricas
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category'):
            le_feat = LabelEncoder()
            valores_unicos = pd.concat([X_train[col], X_test[col]]).astype(str)
            le_feat.fit(valores_unicos)
            X_train[col] = le_feat.transform(X_train[col].astype(str))
            X_test[col] = le_feat.transform(X_test[col].astype(str))

    # Codifica o target
    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train)
    y_test_enc = le_y.transform(y_test)

    # Aplica SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train_enc)

    # Define grade de parâmetros
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_resampled, y_resampled)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    relatorio = classification_report(
        y_test_enc, y_pred,
        labels=range(len(le_y.classes_)),
        target_names=le_y.classes_,
        zero_division=0
    )

    return best_model, relatorio, Counter(y_resampled), grid.best_params_, le_y
