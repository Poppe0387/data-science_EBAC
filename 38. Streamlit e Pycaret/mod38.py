import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import metrics
from scipy.stats import ks_2samp
from scipy.stats import t
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# Título da aplicação
st.title("Análise de Crédito com Streamlit")

# Carregar o arquivo de dados
uploaded_file = st.file_uploader("Carregue um arquivo de dados", type=["ftr"])

if uploaded_file is not None:
    # Carregar dados
    df = pd.read_feather(uploaded_file)
    st.write("Visualização dos primeiros dados:")
    st.write(df.head())

    # Limpeza de dados
    st.subheader("Pré-processamento de Dados")
    st.write("Removendo valores ausentes...")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    st.write("Dados após remoção de valores ausentes:")
    st.write(df.head())

    # Remover coluna 'data_ref'
    st.write("Removendo a coluna 'data_ref' para análise")
    df_drop = df.drop(columns='data_ref')

    # Metadados
    metadados = pd.DataFrame({'dtype': df_drop.dtypes})
    metadados['papel'] = 'covariavel'
    metadados.loc['mau', 'papel'] = 'resposta'
    metadados['nunique'] = df_drop.nunique()
    metadados['missing'] = df.isna().sum()
    st.write("Metadados do DataFrame:")
    st.write(metadados)

    # Análise de Data
    df['data_ref'] = pd.to_datetime(df['data_ref'])
    ultima_data = pd.to_datetime('2016-03-01')
    inicio_oot = ultima_data - pd.DateOffset(months=3)
    
    # Filtrar dados OOT e treino
    oot_df = df[(df['data_ref'] > inicio_oot) & (df['data_ref'] <= ultima_data)]
    treino_df = df[df['data_ref'] <= inicio_oot]
    
    st.write("Safra de Validação OOT:")
    st.write(oot_df)
    
    st.write("Safra de Treinamento:")
    st.write(treino_df)
    
    # Análise temporal
    st.subheader("Análise Temporal")
    num_linhas_total = df.shape[0]
    st.write(f"Número total de linhas: {num_linhas_total}")
    
    linhas_por_mes = df['data_ref'].dt.to_period('M').value_counts().sort_index()
    st.write("Número de linhas por mês em 'data_ref':")
    st.write(linhas_por_mes)
    
    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    linhas_por_mes.plot(kind='bar', color='skyblue')
    plt.title('Número de Linhas por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Número de Linhas')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    # Análise de Variáveis Qualitativas e Quantitativas
    st.subheader("Análise de Variáveis")
    qualitativas = df.select_dtypes(include=['object', 'category'])
    quantitativas = df.select_dtypes(include=['number'])
    
    # Variáveis Qualitativas
    st.write("Análise descritiva das variáveis qualitativas:")
    for col in qualitativas.columns:
        st.write(f"Coluna: {col}")
        st.write(qualitativas[col].value_counts())
        st.write(f"Valores únicos: {qualitativas[col].nunique()}")
        st.write(f"Valores ausentes: {qualitativas[col].isna().sum()}")

    # Variáveis Quantitativas
    st.write("Análise descritiva das variáveis quantitativas:")
    st.write(quantitativas.describe())
    st.write("Valores ausentes nas variáveis quantitativas:")
    st.write(quantitativas.isna().sum())

    # Correlação e Visualização
    st.subheader("Análise de Correlação")
    correlacao = quantitativas.corr()
    st.write("Correlação entre variáveis quantitativas:")
    st.write(correlacao)
    
    # Mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Calor das Correlações entre Variáveis Quantitativas')
    st.pyplot(plt)

    # Scatter plot
    st.write("Gráfico de dispersão para variáveis quantitativas:")
    sns.pairplot(quantitativas)
    st.pyplot(plt)

    # Análise Bivariada de Variáveis Qualitativas
    st.subheader("Análise Bivariada de Variáveis Qualitativas")
    for col1 in qualitativas.columns:
        for col2 in qualitativas.columns:
            if col1 != col2:
                st.write(f"Tabela de contingência entre {col1} e {col2}:")
                contingencia = pd.crosstab(df[col1], df[col2])
                st.write(contingencia)
                # Gráfico de barras empilhadas
                contingencia.plot(kind='bar', stacked=True, figsize=(10, 6))
                plt.title(f'Gráfico de Barras Empilhadas de {col1} vs {col2}')
                plt.xlabel(col1)
                plt.ylabel('Frequência')
                st.pyplot(plt)
    
    # IV (Information Value)
    st.subheader("Cálculo de Information Value (IV)")
    df['mau'] = df['mau'].astype('int64')
    
    def IV(variavel, resposta):
        tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

        rótulo_evento = tab.columns[0]
        rótulo_nao_evento = tab.columns[1]

        tab['pct_evento'] = tab[rótulo_evento] / tab.loc['total', rótulo_evento]
        tab['ep'] = tab[rótulo_evento] / tab.loc['total', rótulo_evento]

        tab['pct_nao_evento'] = tab[rótulo_nao_evento] / tab.loc['total', rótulo_nao_evento]
        tab['woe'] = np.log(tab.pct_evento / tab.pct_nao_evento)
        tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento) * tab.woe
        return tab['iv_parcial'].sum()

    iv_sexo = IV(df.sexo, df.mau)
    st.write(f'IV da variável SEXO: {iv_sexo:.1%}')

    # Outros cálculos de IV e análises bivariadas podem ser adicionados aqui...

    # Treinamento de Modelo
    st.subheader("Treinamento e Avaliação de Modelos")
    formula = '''
        mau ~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + tipo_renda +
        educacao + estado_civil + tipo_residencia + idade + tempo_emprego + renda
    '''
    rl = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
    st.write(rl.summary())

    # Previsões e Avaliação
    df['score'] = rl.predict(df)
    st.write("Acurácia, AUC, GINI e KS:")
    
    acc = metrics.accuracy_score(df.mau, df.score > 0.068)
    fpr, tpr, thresholds = metrics.roc_curve(df.mau, df.score)
    auc = metrics.auc(fpr, tpr)
    gini = 2 * auc - 1
    ks = ks_2samp(df.loc[df.mau == 1, 'score'], df.loc[df.mau != 1, 'score']).statistic

    st.write(f'Acurácia: {acc:.1%}')
    st.write(f'AUC: {auc:.1%}')
    st.write(f'GINI: {gini:.1%}')
    st.write(f'KS: {ks:.1%}')

    # Finalizar análise...

else:
    st.write("Aguardando o upload de um arquivo de dados...")



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pycaret.classification import *
from pycaret.utils import check_metric

# Função para remover outliers
def outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_sem_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_sem_outliers

# Título da aplicação
st.title("Aplicação de Machine Learning com Streamlit")

# Carregando o conjunto de dados Digits
st.subheader("Conjunto de Dados Digits")
X_digits, y_digits = load_digits(return_X_y=True, as_frame=True)
st.write(X_digits.head())

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.3, random_state=10)

# Configurando componentes para o pipeline
st.subheader("Pipeline de Treinamento")
pca = PCA(n_components=20)
scaler = StandardScaler()
logistic = LogisticRegression(max_iter=200)

# Criação do pipeline
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

# Treinamento do modelo
pipe.fit(X_train, y_train)
st.write("Modelo treinado com sucesso!")

# Predições no conjunto de teste
y_pred = pipe.predict(X_test)
st.write(f"Acurácia do Modelo: {pipe.score(X_test, y_test):.2f}")

# Seção para análise de outro dataset
st.subheader("Análise com Dataset de Crédito")
uploaded_file = st.file_uploader("Carregue um arquivo de dados de crédito", type=["ftr"])

if uploaded_file is not None:
    df = pd.read_feather(uploaded_file)
    st.write(df.head())
    
    # Pré-processamento
    st.subheader("Pré-processamento")
    
    st.write("Valores ausentes por coluna:")
    st.write(df.isnull().sum())
    
    df.drop(df[df['tempo_emprego'] == 0].index, inplace=True)
    df['tempo_emprego_mean'] = df['tempo_emprego'].fillna(value=df['tempo_emprego'].mean())
    df.drop(['level_0', 'index'], axis=1, inplace=True, errors='ignore')
    
    # Convertendo tipos de colunas
    df['idade'] = df['idade'].astype(float)
    df.dropna(inplace=True)
    
    st.write("Removendo outliers...")
    for column in df.select_dtypes(include=[np.number]).columns:
        df = outliers(df, column)
    
    df['mau'] = df['mau'].astype('int64')
    st.write("Dados após pré-processamento:")
    st.write(df.head())
    
    # Configurando PyCaret
    st.subheader("Configuração e Modelagem com PyCaret")
    exp = setup(data=df, target='mau', normalize=True, normalize_method='zscore', transformation=True, transformation_method='quantile', fix_imbalance=True, silent=True)
    st.write("Setup concluído com sucesso!")
    
    # Comparando modelos
    st.subheader("Comparando Modelos")
    best_model = compare_models(fold=10, sort='AUC')
    st.write("Melhor modelo baseado na métrica AUC:")
    st.write(best_model)
    
    # Criando e afinando modelo LightGBM
    st.subheader("Criação e Afinação do Modelo LightGBM")
    lightgbm = create_model('lightgbm')
    tuned_lightgbm = tune_model(lightgbm)
    
    st.write("Modelo LightGBM afinado:")
    st.write(tuned_lightgbm)
    
    # Visualização de gráficos do modelo
    st.subheader("Visualizações do Modelo")
    plot_model(tuned_lightgbm, plot="auc")
    plot_model(tuned_lightgbm, plot='pr')
    plot_model(tuned_lightgbm, plot='feature')
    plot_model(tuned_lightgbm, plot='confusion_matrix')

    # Avaliação do modelo
    evaluate_model(tuned_lightgbm)
    
    # Finalizando o modelo
    final_model = finalize_model(tuned_lightgbm)
    st.write("Modelo finalizado:")
    st.write(final_model)
    
    # Salvando o modelo
    save_model(final_model, 'Modelo_1')
    st.write("Modelo salvo como 'Modelo_1'.")
    
    # Previsões no conjunto de dados não visto
    unseen_predictions = predict_model(final_model, data=data_unseen)
    st.write("Previsões no conjunto de dados não visto:")
    st.write(unseen_predictions.head())
    
    # Métrica de acurácia
    accuracy = check_metric(unseen_predictions['mau'].astype(str), unseen_predictions['Label'], metric='Accuracy')
    st.write(f"Acurácia das previsões no conjunto de dados não visto: {accuracy:.2f}")
    
    # Carregando o modelo salvo
    st.subheader("Carregando Modelo Salvo")
    saved_final_lightgbm = load_model('Modelo_1')
    new_prediction = predict_model(saved_final_lightgbm, data=data_unseen)
    st.write("Previsões usando o modelo carregado:")
    st.write(new_prediction.head())
