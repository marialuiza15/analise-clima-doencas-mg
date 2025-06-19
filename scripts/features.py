import pandas as pd

def classificar_temperatura(temp):
    if temp < 0: return 'frio_extremo'
    elif temp < 20: return 'frio'
    elif temp < 25: return 'termico'
    elif temp < 35: return 'quente'
    else: return 'muito_quente'

def classificar_umidade(um):
    if um < 30: return 'muito_seco'
    elif um <= 80: return 'normal'
    else: return 'muito_umido'

def engenharia_de_features(clima, saude):
    clima['temperatura_classe'] = clima['TEMPERATURA_MEDIA'].apply(classificar_temperatura)
    clima['umidade_classe'] = clima['UMIDADE_MEDIA'].apply(classificar_umidade)

    clima_agg = clima.groupby(['codigo_ibge', 'data']).agg({
        'TEMPERATURA_MEDIA': 'mean',
        'UMIDADE_MEDIA': 'mean',
        'temperatura_classe': lambda x: x.mode()[0],
        'umidade_classe': lambda x: x.mode()[0]
    }).reset_index()

    saude.rename(columns={'co_municipio_ibge_residencia': 'codigo_ibge', 'dt_obito': 'data'}, inplace=True)

    df = pd.merge(saude, clima_agg, on=['codigo_ibge', 'data'], how='inner')

    df['faixa_etaria'] = pd.cut(df['nu_idade'], bins=[0, 10, 40, 65, 150], labels=['<10', '10-40', '40-65', '>65'])
    df['clima_extremo'] = ((df['TEMPERATURA_MEDIA'] > 35) | (df['TEMPERATURA_MEDIA'] < 0)).astype(int)
    df['risco_obito'] = 'alto'  # marcador inicial para simulação de modelo

    return df
