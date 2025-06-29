import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def treinar_e_testar(df_treino, df_teste):
    df_treino = df_treino.sort_values('data')
    df_teste = df_teste.sort_values('data')

    features = ['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']
    X_train = df_treino[features].copy()
    y_train = df_treino['risco_obito']
    X_test = df_teste[features].copy()
    y_test = df_teste['risco_obito']

    # Codificação das features categóricas
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # Codifica o y
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    y_test = le_y.transform(y_test)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Relatório
    print("\nRelatório de Classificação (2023):")
    print(classification_report(y_test, y_pred, target_names=le_y.classes_))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Teste em 2023")
    plt.tight_layout()
    plt.show()

    return modelo
