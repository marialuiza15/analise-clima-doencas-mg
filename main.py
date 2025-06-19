from scripts.limpeza import carregar_e_limpar_dados
from scripts.features import engenharia_de_features
from scripts.modelagem import treinar_modelos

CAMINHO_CLIMA = 'dados_clima_mg/'
CAMINHO_SAUDE = 'dados_saude_mg/'

dados_clima, dados_saude = carregar_e_limpar_dados(CAMINHO_CLIMA, CAMINHO_SAUDE)

df = engenharia_de_features(dados_clima, dados_saude)

treinar_modelos(df)
