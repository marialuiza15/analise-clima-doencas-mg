from scripts.limpeza import *
from scripts.features import *
from scripts.modelagem import *
from scripts.visualizacao import plot_distribuicao_causas, mostrar_matriz_confusao, plot_casos_por_mes

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

#df_features_2010 = engenharia_de_features(df_geral_2010)
#print(df_features_2010)
#treinar_modelos(df_features_2010)

# Junta todos os anos anteriores (treino)
df_treino = pd.concat([
    engenharia_de_features(df_geral_ano)
    for df_geral_ano in [
        df_geral_2010, df_geral_2011, df_geral_2012, df_geral_2013,
        df_geral_2014, df_geral_2015, df_geral_2016, df_geral_2017,
        df_geral_2018, df_geral_2019, df_geral_2020, df_geral_2021,
        df_geral_2022
    ]
], ignore_index=True)

# Dados de teste (ano mais recente)
df_teste = engenharia_de_features(df_geral_2023)

# Remove registros sem causa válida (target ausente)
df_treino = df_treino[df_treino['capitulo_cid_causa_basica'] != '#N/D']
df_treino = df_treino.dropna(subset=['capitulo_cid_causa_basica'])
df_teste = df_teste[df_teste['capitulo_cid_causa_basica'] != '#N/D']
df_teste = df_teste.dropna(subset=['capitulo_cid_causa_basica'])

plot_distribuicao_causas(df_treino)
plot_casos_por_mes(df_teste)

# Treinar com todos os dados anteriores, testar em 2023
modelo, relatorio, X_test, y_test_enc, le_y, top_classes = treinar_modelo_por_doenca(df_treino, df_teste, top_n=6)
print(relatorio)

mostrar_matriz_confusao(modelo, X_test, y_test_enc, le_y)

print("Classes usadas (top_n):", list(top_classes))
print("Classes presentes no teste:", df_treino['capitulo_cid_causa_basica'].value_counts())

#print("Distribuição no treino:")
#print(df_treino['risco_obito'].value_counts())

#print("\nDistribuição no teste:")
#print(df_teste['risco_obito'].value_counts())
