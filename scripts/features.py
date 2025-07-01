import pandas as pd

def classificar_temperatura(temp):
    if temp < 0:
        return 'frio_extremo'
    elif temp < 20:
        return 'frio'
    elif temp < 25:
        return 'termico'
    elif temp < 35:
        return 'quente'
    else:
        return 'muito_quente'

def classificar_umidade(um):
    if um < 30:
        return 'muito_seco'
    elif um < 80:
        return 'normal'
    else:
        return 'muito_umido'

def classificar_estacao(mes):
    if mes in [12, 1, 2]:
        return 'verao'
    elif mes in [3, 4, 5]:
        return 'outono'
    elif mes in [6, 7, 8]:
        return 'inverno'
    elif mes in [9, 10, 11]:
        return 'primavera'

def engenharia_de_features(df):
    df = df.copy()

    # Temperatura e umidade médias + classes
    if 'TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)' in df.columns:
        df['TEMPERATURA_MEDIA'] = df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)']
        df['temperatura_classe'] = df['TEMPERATURA_MEDIA'].apply(classificar_temperatura)

    if 'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)' in df.columns:
        df['UMIDADE_MEDIA'] = df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)']
        df['umidade_classe'] = df['UMIDADE_MEDIA'].apply(classificar_umidade)

    # Faixa etária
    if 'nu_idade' in df.columns:
        df['faixa_etaria'] = pd.cut(df['nu_idade'], bins=[0, 10, 40, 65, 150], labels=['<10', '10-40', '40-65', '>65'])

    # Clima extremo
    df['clima_extremo'] = ((df['TEMPERATURA_MEDIA'] > 35) | (df['TEMPERATURA_MEDIA'] < 0)).astype(int)

    # Mês e estação
    if 'data' in df.columns:
        df['mes'] = pd.to_datetime(df['data'], errors='coerce').dt.month
        df['estacao'] = df['mes'].apply(classificar_estacao)

    # Sexo (já está padronizado como string)
    if 'sg_sexo' in df.columns:
        df['sexo'] = df['sg_sexo'].str.lower().str.strip()

    # Raça/cor
    if 'tp_raca_cor' in df.columns:
        df['raca_cor'] = df['tp_raca_cor'].str.lower().str.strip()

    # Município (pode já estar como string ou código)
    if 'co_municipio_ibge_residencia' in df.columns:
        df['municipio'] = df['co_municipio_ibge_residencia'].str.lower().str.strip()

    # (opcional) Cálculo de risco — pode ser removido se o modelo for prever a causa da morte
    df['risco_obito'] = 'Desconhecido'

    # Heurísticas de risco baseadas em clima extremo
    cond1 = (df['faixa_etaria'].isin(['<10', '>65'])) & (df['temperatura_classe'] == 'muito_quente') & (df['umidade_classe'] == 'muito_seco')
    df.loc[cond1, 'risco_obito'] = 'Muito Alto'

    cond2 = (df['faixa_etaria'].isin(['10-40', '40-65'])) & (df['temperatura_classe'] == 'muito_quente') & (df['umidade_classe'] == 'muito_seco')
    df.loc[cond2, 'risco_obito'] = 'Alto'

    cond3 = (df['faixa_etaria'].isin(['<10', '>65'])) & (df['temperatura_classe'] == 'frio_extremo')
    df.loc[cond3, 'risco_obito'] = 'Muito Alto'

    cond4 = (df['faixa_etaria'].isin(['10-40', '40-65'])) & (df['temperatura_classe'] == 'frio_extremo')
    df.loc[cond4, 'risco_obito'] = 'Alto'

    cond5 = (df['temperatura_classe'] == 'termico') & (df['umidade_classe'] == 'normal')
    df.loc[cond5, 'risco_obito'] = 'Baixo'

    return df
