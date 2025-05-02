
Projeto desenvolvido para a disciplina **INF1032 - Introdução à Ciência dos Dados** (2025.1), turmas 3WA PUC-Rio.

## 🎯 Objetivo

Investigar a relação entre variáveis climáticas (temperatura, umidade relativa e precipitação) e os óbitos por doenças crônicas no estado de Minas Gerais, no período de **2010 a 2023**. O projeto visa explorar correlações e desenvolver modelos preditivos que ajudem a compreender o impacto do aquecimento global na saúde pública.

## 🧩 Base de Dados

- **Óbitos por doenças crônicas**  
  Fonte: [Portal de Dados Abertos do Estado de Minas Gerais](https://dados.mg.gov.br/dataset/dados_doencas_cronicas_ses)  
  - 14 arquivos `.csv`
  - Mais de 1 milhão de registros (2010–2023)

- **Dados Climáticos (INMET)**  
  Fonte: [Banco de Dados Meteorológicos para Ensino e Pesquisa – INMET](https://bdmep.inmet.gov.br/#)  
  - 154 estações meteorológicas em MG
  - Temperatura média diária, umidade relativa do ar
  - Período: 2010–2023

## 🛠 Técnicas Utilizadas

- **Pré-processamento e limpeza de dados**
  - Tratamento de valores ausentes (MCAR, MAR, MNAR)
  - Detecção e classificação de outliers (univariados e multivariados)
  - Padronização de formatos e datas

- **Engenharia de Features**
  - Criação de variáveis derivadas como: faixas etárias, categorias de temperatura/umidade, estação do ano
  - Agregações temporais e espaciais

- **Modelagem Preditiva**
  - Modelos supervisionados: Regressão e Classificação
  - Validação cruzada para séries temporais (`TimeSeriesSplit`)

- **Visualização de Dados**
  - Boxplots, histogramas, scatter plots e mapas

## 📊 Resultados Esperados

- Avaliar a influência de eventos climáticos extremos sobre a mortalidade por doenças crônicas

## 💻 Ferramentas e Tecnologias

- Python (Pandas, Scikit-Learn, Seaborn, Matplotlib)
- Jupyter Notebook
- Git/GitHub
