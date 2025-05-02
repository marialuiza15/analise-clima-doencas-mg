import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('dados_cronicas_ses_2015.csv', sep=';', encoding='utf-8')

# Verificar as colunas numéricas
print(df.info())

# Selecionar colunas numéricas relevantes
numerical_cols = ['nu_idade']  # Aparentemente a única coluna numérica no dataset

# 1. Gráficos de violino para distribuição de idade
plt.figure(figsize=(8, 5))
sns.violinplot(y=df['nu_idade'], inner='quartile')
plt.title('Distribuição de Idade')
plt.ylabel('Idade (anos)')
plt.tight_layout()
plt.show()

# 2. Histogramas para visualizar distribuições
plt.figure(figsize=(8, 5))
sns.histplot(df['nu_idade'], kde=True, bins=30)
plt.title('Distribuição de Idade')
plt.xlabel('Idade (anos)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# 3. Análise por categoria (sexo, raça/cor)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='sg_sexo', y='nu_idade', data=df, ci=None)
plt.title('Média de Idade por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Idade Média (anos)')

plt.subplot(1, 2, 2)
sns.barplot(x='tp_raca_cor', y='nu_idade', data=df, ci=None)
plt.title('Média de Idade por Raça/Cor')
plt.xlabel('Raça/Cor')
plt.ylabel('Idade Média (anos)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Análise por capítulo de causa básica (gráfico de barras)
plt.figure(figsize=(12, 6))
df_grouped = df.groupby('capitulo_cid_causa_basica')['nu_idade'].mean().sort_values(ascending=False)
sns.barplot(x=df_grouped.index, y=df_grouped.values)
plt.title('Idade Média por Capítulo de Causa Básica')
plt.xlabel('Capítulo CID')
plt.ylabel('Idade Média (anos)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 5. Contagem de óbitos por capítulo de causa básica
plt.figure(figsize=(12, 6))
df['capitulo_cid_causa_basica'].value_counts().plot(kind='bar')
plt.title('Contagem de Óbitos por Capítulo de Causa Básica')
plt.xlabel('Capítulo CID')
plt.ylabel('Número de Óbitos')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 6. Gráfico de dispersão por idade e contagem (simulando distribuição)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='nu_idade', hue='sg_sexo', multiple='stack')
plt.title('Distribuição de Idade por Sexo')
plt.xlabel('Idade (anos)')
plt.ylabel('Densidade')
plt.tight_layout()
plt.show()