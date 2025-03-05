# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from mpl_toolkits.mplot3d import Axes3D

# Adicionando CSS customizado
st.markdown("""
    <style>
    .main {
        background-color: #303030;
    }
    .sidebar .sidebar-content {
        background-color: #424242;
    }
    </style>
    """, unsafe_allow_html=True)

# Configurações iniciais
os.environ["OMP_NUM_THREADS"] = "1"
DADOS = "dados/Mall_Customers_no_CustomerID.csv"
RANDOM_STATE = 42

# Carregando os dados
df = pd.read_csv(DADOS)

# Pré-processamento
preprocessing = ColumnTransformer(
    [
        ("standard", StandardScaler(), ["Age", "Annual Income (k$)", "Spending Score (1-100)"]),
        ("onehot", OneHotEncoder(), ["Gender"])
    ]
)

# Pipeline de PCA e clustering
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("pca", PCA(n_components=3, random_state=RANDOM_STATE)),
        ("clustering", KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10))
    ]
)

pipeline.fit(df)
df_clustered = df.copy()
df_clustered["cluster"] = pipeline["clustering"].labels_

# Salvando o modelo
joblib.dump(pipeline, "modelos/pipeline_preprocessing_pca_clustering.pkl")

# Função para visualização de clusters
def visualizar_cluster(dataframe, colunas, quantidade_cores, centroids=None, mostrar_pontos=False, coluna_clusters=None):
    if coluna_clusters not in dataframe.columns:
        st.error(f"Coluna '{coluna_clusters}' não encontrada no dataframe.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = sns.color_palette("Set2", quantidade_cores)
    for i in range(quantidade_cores):
        pontos = dataframe[dataframe[coluna_clusters] == i] if mostrar_pontos else dataframe
        ax.scatter(pontos[colunas[0]], pontos[colunas[1]], pontos[colunas[2]], color=colors[i], label=f'Cluster {i}')
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, c='black')
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    plt.legend()
    st.pyplot(fig)

# Iniciando a aplicação Streamlit
st.title("Análise de Cluster de Clientes")
st.write("## Visualize a análise de clusters em tempo real. Selecione a visualização desejada e explore os dados!")

# Seletor de visualização
opcao_visualizacao = st.sidebar.selectbox(
    "Selecione a visualização",
    ["Pairplot", "Boxplot por Cluster", "Boxplot por Gênero", "Cluster em 3D (sem dispersão)", "Cluster em 3D (com dispersão)"]
)

# Visualizando os dados
if opcao_visualizacao == "Pairplot":
    st.write("### Pairplot dos Clusters")
    st.write("Este gráfico mostra a distribuição dos clusters com base nas variáveis originais.")
    pairplot_fig = sns.pairplot(df_clustered, diag_kind="kde", hue="cluster", palette="tab10")
    st.pyplot(pairplot_fig)

elif opcao_visualizacao == "Boxplot por Cluster":
    st.write("### Boxplot por Cluster")
    st.write("Compare as distribuições das variáveis para cada cluster.")
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), tight_layout=True)
    palette = sns.color_palette("Set2", n_colors=5)
    for ax, col in zip(axs, df_clustered.select_dtypes("number")):
        sns.boxplot(data=df_clustered, x="cluster", y=col, hue="cluster", ax=ax, palette=palette)
    st.pyplot(fig)

elif opcao_visualizacao == "Boxplot por Gênero":
    st.write("### Boxplot por Gênero")
    st.write("Compare as distribuições das variáveis para cada gênero dentro dos clusters.")
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), tight_layout=True)
    palette = sns.color_palette("Set2", n_colors=2)
    for ax, col in zip(axs, df_clustered.select_dtypes("number")):
        sns.boxplot(data=df_clustered, x="cluster", y=col, hue="Gender", ax=ax, palette=palette)
    st.pyplot(fig)

elif opcao_visualizacao == "Cluster em 3D (sem dispersão)":
    st.write("### Cluster em 3D (sem dispersão)")
    st.write("Visualização dos clusters em 3D após aplicação do PCA.")
    df_pca = pd.DataFrame(
        pipeline[:-1].fit_transform(df),
        columns=[f'pca{i}' for i in range(3)]
    )
    df_pca["cluster"] = pipeline["clustering"].labels_
    visualizar_cluster(
        dataframe=df_pca,
        colunas=['pca0', 'pca1', 'pca2'],
        quantidade_cores=5,
        coluna_clusters="cluster"
    )

elif opcao_visualizacao == "Cluster em 3D (com dispersão)":
    st.write("### Cluster em 3D (com dispersão)")
    st.write("Visualização dos clusters em 3D com dispersão após aplicação do PCA.")
    df_pca = pd.DataFrame(
        pipeline[:-1].fit_transform(df),
        columns=[f'pca{i}' for i in range(3)]
    )
    df_pca["cluster"] = pipeline["clustering"].labels_
    centroids = pipeline["clustering"].cluster_centers_
    visualizar_cluster(
        dataframe=df_pca,
        colunas=['pca0', 'pca1', 'pca2'],
        quantidade_cores=5,
        centroids=centroids,
        mostrar_pontos=True,
        coluna_clusters="cluster"
    )

# Conclusão do Projeto de Clusterização de Clientes

st.header("Conclusão do Projeto de Clusterização de Clientes")

st.subheader("Resumo dos Principais Aspectos e Resultados")
st.write("""
Neste projeto, realizamos uma análise detalhada dos dados de clientes, aplicando técnicas de pré-processamento, visualização e clusterização. Utilizamos gráficos 2D e 3D para interpretar melhor a distribuição dos clusters e identificar padrões e correlações. Implementamos a redução de dimensionalidade com PCA para melhorar a qualidade dos clusters e facilitar a visualização.
""")

st.subheader("Descrição dos Clusters Antes do PCA")
st.write("""
| Pontuação de Gastos | Renda   | Idade    | Cluster |
|---------------------|---------|----------|---------|
| Alta                | Alta    | Jovem    | 0       |
| Moderada            | Moderada| Alta     | 1       |
| Alta                | Baixa   | Jovem    | 2       |
| Baixa               | Alta    | Moderada | 3       |
| Moderada            | Moderada| Jovem    | 4       |
""")

st.subheader("Análise dos Resultados")
st.write("""
Após aplicar o PCA, identificamos os seguintes padrões nos clusters:

| Cluster | Idade                      | Renda Anual | Score de Gastos         | Observações                                                                                   |
|---------|----------------------------|-------------|------------------------|----------------------------------------------------------------------------------------------|
| 4       | 30 a 40 anos               | Alta        | Alta                   | Clientes com maiores scores, rendas anuais elevadas, demonstrando maior estabilidade financeira. |
| 3       | Menos de 30 anos           | Baixa       | Alta                   | O menor cluster, com menor renda anual, mas scores elevados, quase no nível do Cluster 4.     |
| 2       | 35 a 50 anos               | Alta        | Menor                  | Clientes com alta renda anual, mas scores menores, indicando possível falta de crédito.      |
| 0       | Acima de 50 até 60 anos    | Moderada    | Moderado               | Representam a maior parte dos clientes, com rendas anuais e scores muito semelhantes ao Cluster 1. |
| 1       | Maior que 20 até 35 anos   | Moderada    | Moderado               | Representam a maior parte dos clientes, com rendas anuais e scores muito semelhantes ao Cluster 0. |
""")

st.subheader("Conclusão Final")
st.write("""
Através desta análise, conseguimos segmentar os clientes em grupos distintos, facilitando a compreensão dos diferentes perfis de consumidores. Essas informações são valiosas para estratégias de marketing, crédito e tomada de decisões empresariais. A utilização de técnicas avançadas como PCA e visualizações 3D permitiu uma análise mais precisa e detalhada dos dados, revelando padrões importantes que não seriam aparentes em uma análise superficial.

O projeto demonstrou a importância do pré-processamento, da redução de dimensionalidade e da visualização avançada na análise de dados de clientes, proporcionando insights valiosos e acionáveis.
""")
# -


