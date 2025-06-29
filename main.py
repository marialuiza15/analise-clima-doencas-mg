from scripts.limpeza import *
from scripts.features import *
from scripts.modelagem import *

# Caminhos
CAMINHO_CLIMA = 'dados_clima_mg/'
CAMINHO_SAUDE = 'dados_saude_mg/'

# Clima
clima_total, dfs_por_ano = unir_e_separar_clima(CAMINHO_CLIMA)

# Une clima + saúde para cada ano
df_geral = {ano: unindo_clima_saude(dfs_por_ano[ano], CAMINHO_SAUDE, ano)
            for ano in range(2010, 2024)}

# Aplica features e filtra desconhecidos
def preparar(df):
    df = engenharia_de_features(df)
    return df[df['risco_obito'] != 'Desconhecido']

df_treino = pd.concat([preparar(df_geral[ano]) for ano in range(2010, 2023)], ignore_index=True)
df_teste = preparar(df_geral[2023])

# Treinamento + Teste + Visualização
modelo = treinar_e_testar(df_treino, df_teste)
