import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle


sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de renda",
     page_icon="https://cdn-icons-png.flaticon.com/512/3594/3594449.png",
     layout="centered",
)

st.write('# Análise de previsão de renda')

renda = pd.read_csv('previsao_de_renda.csv')

#plots

st.write('# Tabela de renda (base)')
st.dataframe(renda)


# Filtro por faixa etária
faixa_etaria = st.slider('Selecione a faixa etária', min_value=18, max_value=65)
renda_filtrada = renda[renda['idade'] >= faixa_etaria]

# Gráfico de barras com personalização
fig, ax = plt.subplots()
renda_filtrada['posse_de_veiculo'].value_counts().plot(kind='bar', rot=0, color=['#2196F3', '#F44336'], ax=ax)
ax.set_title('Posse de Veículo por Faixa Etária')
ax.set_ylabel('Número de Pessoas')
ax.set_xlabel('')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Não possuem veículo', 'Possuem veículo'])

# Adicionando uma linha horizontal com a média de renda
media_renda = renda_filtrada['renda'].mean()
ax.axhline(y=media_renda, color='r', linestyle='--', label='Média de Renda')
ax.legend()
st.pyplot(fig)


# Sidebar para filtros
st.sidebar.header("Filtros")
tipo_renda_filtro = st.sidebar.multiselect("Tipo de Renda", renda['tipo_renda'].unique())
educacao_filtro = st.sidebar.multiselect("Nível de Educação", renda['educacao'].unique())

# Aplicar filtros
renda_filtrada = renda
if tipo_renda_filtro:
    renda_filtrada = renda_filtrada[renda_filtrada['tipo_renda'].isin(tipo_renda_filtro)]
if educacao_filtro:
    renda_filtrada = renda_filtrada[renda_filtrada['educacao'].isin(educacao_filtro)]


st.header("Distribuição de Tipos de Renda")
fig, ax = plt.subplots()
renda_filtrada['tipo_renda'].value_counts().plot(kind='bar', rot=45, color=['#A0522D','#D2691E','#F4A460', '#FFDEAD'], ax=ax)
ax.set_ylabel("Frequência")
st.pyplot(fig)

# Histograma da idade
st.header("Distribuição da Idade")
fig, ax = plt.subplots()
sns.histplot(data=renda_filtrada, x='idade', bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Scatter plot: Renda por idade
st.header("Relação entre Renda e Idade")
fig, ax = plt.subplots()
sns.scatterplot(data=renda_filtrada, x='idade', y='renda', hue='sexo', ax=ax)
st.pyplot(fig)


# Função de transformação dummy
def dummy_transformation(dados):
    dados_prev = pd.DataFrame([dados])
    
    colunas_categoricas = dados_prev.select_dtypes(include=['object', 'category']).columns.tolist()
    dados_prev_dummy = pd.get_dummies(dados_prev, columns=colunas_categoricas, drop_first=True)
    
    dados_prev_dummy = dados_prev_dummy.reindex(columns=renda.columns, fill_value=0)
    previsao_renda = regressao.predict(dados_prev_dummy)
    
    return previsao_renda[0]

# Interface do Streamlit
st.title("Simulador de previsão de renda")

sexo = st.selectbox('Sexo', ['M', 'F'])
posse_de_veiculo = st.checkbox('Posse de Veículo')
posse_de_imovel = st.checkbox('Posse de Imóvel')
qtd_filhos = st.number_input('Quantidade de Filhos', min_value=0, max_value=10, value=0)
tipo_renda = st.selectbox('Tipo de Renda', ['Assalariado', 'Servidor Público', 'Pensionista', 'Empresário', 'Autônomo'])
educacao = st.selectbox('Educação', ['Fundamental', 'Secundário', 'Superior', 'Pós-graduado'])
estado_civil = st.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Separado', 'Divorciado', 'Viúvo'])
tipo_residencia = st.selectbox('Tipo de Residência', ['Casa', 'Apartamento', 'Com os Pais'])
idade = st.slider('Idade', 18, 70, 30)
tempo_emprego = st.slider('Tempo de Emprego (anos)', 0, 50, 5)
qt_pessoas_residencia = st.slider('Quantidade de Pessoas na Residência', 1.0, 10.0, 2.0)

dados_prev = {
    'sexo': sexo,
    'posse_de_veiculo': posse_de_veiculo,
    'posse_de_imovel': posse_de_imovel,
    'qtd_filhos': qtd_filhos,
    'tipo_renda': tipo_renda,
    'educacao': educacao,
    'estado_civil': estado_civil,
    'tipo_residencia': tipo_residencia,
    'idade': idade,
    'tempo_emprego': tempo_emprego,
    'qt_pessoas_residencia': qt_pessoas_residencia
}

if st.button('Calcular Previsão'):
    previsao = dummy_transformation(dados_prev)
    st.write(f"A previsão de renda é: R$ {previsao:.2f}")
