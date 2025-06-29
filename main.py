from scripts.limpeza import *
from scripts.features import *
from scripts.modelagem import *

CAMINHO_CLIMA = 'dados_clima_mg/'
CAMINHO_SAUDE = 'dados_saude_mg/'

clima_total, dfs_por_ano = unir_e_separar_clima(CAMINHO_CLIMA)

df_clima_2010 = dfs_por_ano[2010]
df_clima_2011 = dfs_por_ano[2011]
df_clima_2012 = dfs_por_ano[2012]
df_clima_2013 = dfs_por_ano[2013]
df_clima_2014 = dfs_por_ano[2014]
df_clima_2015 = dfs_por_ano[2015]
df_clima_2016 = dfs_por_ano[2016]
df_clima_2017 = dfs_por_ano[2017]
df_clima_2018 = dfs_por_ano[2018]
df_clima_2019 = dfs_por_ano[2019]
df_clima_2020 = dfs_por_ano[2020]
df_clima_2021 = dfs_por_ano[2021]
df_clima_2022 = dfs_por_ano[2022]
df_clima_2023 = dfs_por_ano[2023]

df_geral_2010 = unindo_clima_saude(df_clima_2010, CAMINHO_SAUDE, 2010)
df_geral_2011 = unindo_clima_saude(df_clima_2011, CAMINHO_SAUDE, 2011)
df_geral_2012 = unindo_clima_saude(df_clima_2012, CAMINHO_SAUDE, 2012)
df_geral_2013 = unindo_clima_saude(df_clima_2013, CAMINHO_SAUDE, 2013)
df_geral_2014 = unindo_clima_saude(df_clima_2014, CAMINHO_SAUDE, 2014)
df_geral_2015 = unindo_clima_saude(df_clima_2015, CAMINHO_SAUDE, 2015)
df_geral_2016 = unindo_clima_saude(df_clima_2016, CAMINHO_SAUDE, 2016)
df_geral_2017 = unindo_clima_saude(df_clima_2017, CAMINHO_SAUDE, 2017)
df_geral_2018 = unindo_clima_saude(df_clima_2018, CAMINHO_SAUDE, 2018)
df_geral_2019 = unindo_clima_saude(df_clima_2019, CAMINHO_SAUDE, 2019)
df_geral_2020 = unindo_clima_saude(df_clima_2020, CAMINHO_SAUDE, 2020)
df_geral_2021 = unindo_clima_saude(df_clima_2021, CAMINHO_SAUDE, 2021)
df_geral_2022 = unindo_clima_saude(df_clima_2022, CAMINHO_SAUDE, 2022)
df_geral_2023 = unindo_clima_saude(df_clima_2023, CAMINHO_SAUDE, 2023)

# A partir daqui, todos os dataframes estão organizados, com dados de clima e saude para cada ano.

# resultado.to_csv("resultado_uniao_2010.csv", index=False, encoding="utf-8") # cCaso precise ver o df completo

# Junta dados de treino de 2010 até 2022
df_treino_total = pd.concat([
    engenharia_de_features(df_geral_2010),
    engenharia_de_features(df_geral_2011),
    engenharia_de_features(df_geral_2012),
    engenharia_de_features(df_geral_2013),
    engenharia_de_features(df_geral_2014),
    engenharia_de_features(df_geral_2015),
    engenharia_de_features(df_geral_2016),
    engenharia_de_features(df_geral_2017),
    engenharia_de_features(df_geral_2018),
    engenharia_de_features(df_geral_2019),
    engenharia_de_features(df_geral_2020),
    engenharia_de_features(df_geral_2021),
    engenharia_de_features(df_geral_2022)
], ignore_index=True)

# Treina e obtém o modelo
modelo_final = RandomForestClassifier()
df_treino_total = df_treino_total.sort_values('data').copy()
X = df_treino_total[['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA', 'faixa_etaria', 'clima_extremo']].copy()
y = df_treino_total['risco_obito'].astype(str)

# Codifica
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
y = LabelEncoder().fit_transform(y)

modelo_final.fit(X, y)

# Prepara os dados de 2023
df_teste_2023 = engenharia_de_features(df_geral_2023)

# Testa com dados futuros
testar_com_dados_futuros(modelo_final, df_teste_2023)
