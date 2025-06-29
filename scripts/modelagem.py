import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def treinar_e_testar(df_treino, df_teste):
    df_treino = df_treino[df_treino['risco_obito'] != 'Desconhecido'].copy()
    df_teste = df_teste[df_teste['risco_obito'] != 'Desconhecido'].copy()

    features = ['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']
    X_train = df_treino[features].copy()
    y_train = df_treino['risco_obito']
    X_test = df_teste[features].copy()
    y_test = df_teste['risco_obito']

    # Encode categóricas com consistência entre treino e teste
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(X_train[col].astype(str).unique()) + list(X_test[col].astype(str).unique()))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # Encode target
    le_y = LabelEncoder()
    le_y.fit(list(y_train.astype(str).unique()) + list(y_test.astype(str).unique()))
    y_train = le_y.transform(y_train.astype(str))
    y_test = le_y.transform(y_test.astype(str))

    # Modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Relatório
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\nRelatório de Classificação (ano de teste):")
    print(classification_report(y_test, y_pred, target_names=le_y.classes_))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.show()

    return modelo
