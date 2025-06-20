import pandas as pd
import glob
import os

def limpar_clima(df):

    df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)'] = (
        df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)']
        .str.replace(',', '.', regex=False)
        .astype(float)
    )
    df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'] = (
        df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)']
        .str.replace(',', '.', regex=False)
        .astype(float)
    )
    df = df.dropna(subset=[
        'TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)', 
        'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'
    ])
    df = df[
        (df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'] <= 100) & 
        (df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)'] > -50)
    ]
    df['TEMPERATURA_MEDIA'] = df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)']
    df['UMIDADE_MEDIA'] = df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)']

    return df

def limpar_saude(df):
    colunas_utilizadas = [
        'dt_obito', 'nu_idade', 'sg_sexo', 'tp_raca_cor',
        'co_municipio_ibge_residencia', 'co_cid_causa_basica',
        'categoria_cid_causa_basica'
    ]
    df = df[colunas_utilizadas].dropna()
    return df


def unir_e_separar_clima(caminho_clima):
    dfs = []
    for i in range(502, 572):  # A502 até A571
        arquivo = os.path.join(caminho_clima, f'dados_A{i}_D_2010-01-01_2023-12-31.csv')
        if not os.path.exists(arquivo):
            continue
        with open(arquivo, encoding='latin1') as f:
            nome_lugar = f.readline().strip().split(':')[1].strip().lower()
        df = pd.read_csv(arquivo, sep=';', encoding='latin1', header=9)
        df['Região'] = nome_lugar

        if 'TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)' in df.columns:
            df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)'] = (
                df['TEMPERATURA MEDIA, DIARIA (AUT)(Â°C)'].str.replace(',', '.', regex=False).astype(float)
            )
        if 'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)' in df.columns:
            df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'] = (
                df['UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'].str.replace(',', '.', regex=False).astype(float)
            )
        df['data'] = pd.to_datetime(df['Data Medicao'], errors='coerce')
        dfs.append(df)

    clima_total = pd.concat(dfs, ignore_index=True)

    # Separa por ano
    dfs_por_ano = {}
    for ano in range(2010, 2024):
        dfs_por_ano[ano] = clima_total[clima_total['data'].dt.year == ano].copy()

    
    return clima_total, dfs_por_ano

def unindo_clima_saude(df_clima, caminho_saude, ano_especifico):
    import os
    dfs_ano = {}
    for ano in range(2010, 2024):

        arquivo_saude = f'dados_cronicas_ses_{ano}.csv'
        caminho_arquivo_saude = os.path.join(caminho_saude, arquivo_saude)
        if not os.path.exists(caminho_arquivo_saude):
            print(f"Arquivo de saúde não encontrado para {ano}")
            continue
        df_saude = pd.read_csv(caminho_arquivo_saude, sep=';', encoding='utf-8-sig')

        df_clima = df_clima.copy()
        df_clima['data'] = pd.to_datetime(df_clima['Data Medicao']).dt.date
        df_clima['Região'] = df_clima['Região'].str.strip().str.lower()
        df_saude['data'] = pd.to_datetime(df_saude['dt_obito'], dayfirst=True, errors='coerce').dt.date
        df_saude['Região'] = df_saude['co_municipio_ibge_residencia'].str.strip().str.lower()

        df_merged = pd.merge(df_clima, df_saude, on=['data', 'Região'], how='inner', suffixes=('_clima', '_saude'))

        dfs_ano[ano] = df_merged

    return dfs_ano[ano_especifico]