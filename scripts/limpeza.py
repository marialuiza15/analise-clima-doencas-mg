import pandas as pd
import glob
import os

def limpar_clima(df):
    df = df.dropna(subset=['TEMPERATURA_MEDIA', 'UMIDADE_MEDIA'])
    df = df[(df['UMIDADE_MEDIA'] <= 100) & (df['TEMPERATURA_MEDIA'] > -50)]
    return df

def limpar_saude(df):
    colunas_utilizadas = [
        'dt_obito', 'nu_idade', 'sg_sexo', 'tp_raca_cor',
        'co_municipio_ibge_residencia', 'co_cid_causa_basica',
        'categoria_cid_causa_basica'
    ]
    df = df[colunas_utilizadas].dropna()
    return df

def carregar_e_limpar_dados(caminho_clima, caminho_saude):
    # climáticos
    arquivos_clima = glob.glob(os.path.join(caminho_clima, '*.csv'))
    dfs_clima = []
    for arq in arquivos_clima:
        df = pd.read_csv(arq, sep=';', encoding='latin1')
        df['codigo_ibge'] = arq.split('_')[1]  # Ex: A502 → código da estação
        df.rename(columns={
            'Data Medicao': 'data',
            'TEMPERATURA MEDIA, DIARIA (AUT)(°C)': 'TEMPERATURA_MEDIA',
            'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)': 'UMIDADE_MEDIA'
        }, inplace=True)
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        dfs_clima.append(df)
    clima = pd.concat(dfs_clima, ignore_index=True)
    clima = limpar_clima(clima)

    # saúde
    arquivos_saude = glob.glob(os.path.join(caminho_saude, '*.csv'))
    saude = pd.concat([pd.read_csv(arq, sep=';', encoding='utf-8') for arq in arquivos_saude], ignore_index=True)
    saude['dt_obito'] = pd.to_datetime(saude['dt_obito'], errors='coerce')
    saude = limpar_saude(saude)

    return clima, saude
