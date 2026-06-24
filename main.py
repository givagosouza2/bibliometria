import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# Configuração da página
st.set_page_config(page_title="Dashboard Bibliométrico Scopus", layout="wide")

st.title("📊 Dashboard de Análise Bibliométrica (Scopus)")
st.markdown("Análise interativa de publicações, citações e autores baseada em dados exportados do Scopus.")

# --- CARREGAMENTO DE DADOS ---
# Tenta carregar o arquivo local. Se não encontrar, mostra um uploader.
try:
    df = pd.read_csv('AHerculano.csv')
except FileNotFoundError:
    st.warning("Arquivo 'AHerculano.csv' não encontrado na mesma pasta. Por favor, faça o upload:")
    uploaded_file = st.file_uploader("Escolha o arquivo CSV do Scopus", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# --- LIMPEZA E PRÉ-PROCESSAMENTO ---
# Garantir que as colunas numéricas estejam corretas
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0).astype(int)

# Filtrar apenas anos válidos
df = df.dropna(subset=['Year'])

# --- SIDEBAR (FILTROS) ---
st.sidebar.header("🔍 Filtros")

# Filtro de Ano
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider("Intervalo de Anos:", min_year, max_year, (min_year, max_year))

# Filtro de Tipo de Documento
doc_types = df['Document Type'].unique()
selected_docs = st.sidebar.multiselect("Tipos de Documento:", doc_types, default=doc_types)

# Aplicar filtros
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
df_filtered = df_filtered[df_filtered['Document Type'].isin(selected_docs)]

# --- MÉTRICAS GERAIS (KPIs) ---
st.subheader("Resumo Geral")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Publicações", len(df_filtered))
col2.metric("Total de Citações", df_filtered['Cited by'].sum())
col3.metric("Média de Citações / Artigo", round(df_filtered['Cited by'].mean(), 2))
col4.metric("Artigos em Open Access", df_filtered[df_filtered['Open Access'].str.contains('Open Access', na=False)].shape[0])

st.markdown("---")

# --- GRÁFICOS ---
col_esq, col_dir = st.columns(2)

# 1. Publicações por Ano
with col_esq:
    st.subheader("📅 Produção Científica por Ano")
    pub_por_ano = df_filtered['Year'].value_counts().sort_index()
    fig_ano = px.bar(x=pub_por_ano.index, y=pub_por_ano.values, 
                     labels={'x': 'Ano', 'y': 'Número de Publicações'},
                     color=pub_por_ano.values, color_continuous_scale='Blues')
    fig_ano.update_layout(showlegend=False)
    st.plotly_chart(fig_ano, use_container_width=True)

# 2. Tipos de Documento
with col_dir:
    st.subheader("📄 Tipos de Documento")
    tipos_doc = df_filtered['Document Type'].value_counts()
    fig_tipo = px.pie(values=tipos_doc.values, names=tipos_doc.index, hole=0.4)
    fig_tipo.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_tipo, use_container_width=True)

# 3. Top 10 Artigos Mais Citados
st.subheader("🏆 Top 10 Artigos Mais Citados")
top_citados = df_filtered.nlargest(10, 'Cited by')[['Title', 'Year', 'Cited by', 'Source title']]
# Criar um rótulo curto para o gráfico
top_citados['Titulo_Curto'] = top_citados['Title'].apply(lambda x: str(x)[:60] + '...' if len(str(x)) > 60 else str(x))
fig_citados = px.bar(top_citados, x='Cited by', y='Titulo_Curto', orientation='h',
                     color='Cited by', color_continuous_scale='Viridis',
                     labels={'Titulo_Curto': 'Artigo', 'Cited by': 'Citações'},
                     hover_data=['Year', 'Source title', 'Title'])
fig_citados.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_citados, use_container_width=True)

col_esq2, col_dir2 = st.columns(2)

# 4. Top Autores mais Produtivos
with col_esq2:
    st.subheader("👥 Top 10 Autores mais Produtivos")
    # A coluna 'Authors' vem separada por '; '
    lista_autores = []
    for autores in df_filtered['Authors'].dropna():
        lista_autores.extend([a.strip() for a in str(autores).split(';')])
    
    contagem_autores = Counter(lista_autores).most_common(10)
    df_autores = pd.DataFrame(contagem_autores, columns=['Autor', 'Publicações'])
    
    fig_autores = px.bar(df_autores, x='Publicações', y='Autor', orientation='h',
                         color='Publicações', color_continuous_scale='Plasma')
    fig_autores.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_autores, use_container_width=True)

# 5. Top Periódicos (Source Titles)
with col_dir2:
    st.subheader("📚 Top 10 Periódicos (Source Titles)")
    top_periodicos = df_filtered['Source title'].value_counts().head(10).reset_index()
    top_periodicos.columns = ['Periódico', 'Publicações']
    
    fig_periodicos = px.bar(top_periodicos, x='Publicações', y='Periódico', orientation='h',
                            color='Publicações', color_continuous_scale='Magma')
    fig_periodicos.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_periodicos, use_container_width=True)

# 6. Acesso Aberto vs Fechado
st.subheader("🔓 Status de Acesso Aberto (Open Access)")
# Normalizando a coluna Open Access
df_filtered['OA_Status'] = df_filtered['Open Access'].apply(lambda x: 'Open Access' if pd.notna(x) and 'Open Access' in str(x) else 'Fechado')
oa_counts = df_filtered['OA_Status'].value_counts()
fig_oa = px.pie(values=oa_counts.values, names=oa_counts.index, hole=0.4, 
                color_discrete_map={'Open Access': '#2ca02c', 'Fechado': '#d62728'})
fig_oa.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_oa, use_container_width=True)

# --- TABELA DE DADOS BRUTOS ---
st.markdown("---")
st.subheader("📋 Dados Filtrados")
st.dataframe(df_filtered[['Title', 'Authors', 'Year', 'Source title', 'Cited by', 'Open Access']], use_container_width=True)
