import pandas as pd

def classificar_umidade(umidade):
    """
    Classifica a umidade relativa do ar conforme as categorias definidas:
    - < 30% → "muito seco"
    - >= 30% e < 80% → "normal"
    - >= 80% → "muito úmido"
    """
    if umidade < 30:
        return "muito seco"
    elif 30 <= umidade < 80:
        return "normal"
    else:
        return "muito úmido"

def classificar_temperatura(temperatura):
    """
    Classifica a temperatura média diária conforme as categorias definidas:
    - < 0°C → "frio extremo"
    - >= 0°C e < 20°C → "frio"
    - >= 20°C e < 25°C → "térmico"
    - >= 25°C e < 35°C → "quente"
    - >= 35°C → "muito quente"
    """
    if temperatura < 0:
        return "frio extremo"
    elif 0 <= temperatura < 20:
        return "frio"
    elif 20 <= temperatura < 25:
        return "térmico"
    elif 25 <= temperatura < 35:
        return "quente"
    else:
        return "muito quente"

# Exemplo de aplicação em um DataFrame
def processar_dados_climaticos(df):
    """
    Processa um DataFrame com dados climáticos, adicionando colunas de classificação
    """
    # Verifica se as colunas necessárias existem
    if 'UMIDADE RELATIVA DO AR, MEDIA DIARIA [AUT][%]' in df.columns:
        df['classificacao_umidade'] = df['UMIDADE RELATIVA DO AR, MEDIA DIARIA [AUT][%]'].apply(classificar_umidade)
    
    if 'TEMPERATURA MEDIA, DIARIA [AUT][°C]' in df.columns:
        df['classificacao_temperatura'] = df['TEMPERATURA MEDIA, DIARIA [AUT][°C]'].apply(classificar_temperatura)
    
    return df

# Exemplo de uso
if __name__ == "__main__":
    # Criando um DataFrame de exemplo
    dados_exemplo = {
        'Data Medicao': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'TEMPERATURA MEDIA, DIARIA [AUT][°C]': [-2, 15, 22, 28, 38],
        'UMIDADE RELATIVA DO AR, MEDIA DIARIA [AUT][%]': [25, 45, 75, 85, 10]
    }
    
    df_exemplo = pd.DataFrame(dados_exemplo)
    
    # Processando os dados
    df_processado = processar_dados_climaticos(df_exemplo)
    
    print("Dados climáticos com classificações:")
    print(df_processado)