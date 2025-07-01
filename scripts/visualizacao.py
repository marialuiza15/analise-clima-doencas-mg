import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_distribuicao_causas(df, titulo='Distribuição das causas de óbito'):
    plt.figure(figsize=(10, 5))
    ordem = df['capitulo_cid_causa_basica'].value_counts().head(10).index
    sns.countplot(data=df, y='capitulo_cid_causa_basica', order=ordem, hue='capitulo_cid_causa_basica', palette='magma', legend=False)
    plt.title(titulo)
    plt.xlabel('Número de óbitos')
    plt.ylabel('Causa (Capítulo CID)')
    plt.tight_layout()
    plt.show()

def plotar_matriz_confusao(y_true, y_pred, encoder):
    cm = confusion_matrix(y_true, y_pred)
    labels = encoder.classes_

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')
    plt.title('Matriz de Confusão')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_casos_por_mes(df):
    df['mes'] = pd.to_datetime(df['data']).dt.month
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='mes', hue='capitulo_cid_causa_basica', palette='Set2')
    plt.title('Óbitos por mês e causa (top 4)')
    plt.xlabel('Mês')
    plt.ylabel('Número de óbitos')
    plt.legend(title='Causa')
    plt.tight_layout()
    plt.show()
