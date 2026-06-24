import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(page_title="Dashboard Bibliométrico Scopus", layout="wide", page_icon="📊")

st.title(" Dashboard de Análise Bibliométrica (Scopus)")
st.markdown("Análise interativa de publicações, citações e autores baseada em dados exportados do Scopus.")

# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin-1')

try:
    df = load_data('AHerculano.csv')
    file_loaded = True
except FileNotFoundError:
    file_loaded = False

if not file_loaded:
    st.warning("Arquivo 'AHerculano.csv' não encontrado. Faça o upload:")
    uploaded_file = st.file_uploader("Escolha o arquivo CSV do Scopus", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    else:
        st.stop()

# ============================================================
# LIMPEZA E PRÉ-PROCESSAMENTO
# ============================================================
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0).astype(int)
df = df.dropna(subset=['Year'])

# Normaliza a coluna Open Access
df['OA_Status'] = df['Open Access'].apply(
    lambda x: 'Open Access' if pd.notna(x) and str(x).strip() != '' else 'Fechado'
)

# ============================================================
# SIDEBAR - FILTROS
# ============================================================
st.sidebar.header("🔍 Filtros")

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider("Intervalo de Anos:", min_year, max_year, (min_year, max_year))

doc_types = df['Document Type'].unique()
selected_docs = st.sidebar.multiselect("Tipos de Documento:", doc_types, default=doc_types)

df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
df_filtered = df_filtered[df_filtered['Document Type'].isin(selected_docs)]

# ============================================================
# KPIs - RESUMO GERAL
# ============================================================
st.subheader("📌 Resumo Geral")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Publicações", len(df_filtered))
col2.metric("Total de Citações", int(df_filtered['Cited by'].sum()))
col3.metric("Média de Citações/Artigo", round(df_filtered['Cited by'].mean(), 2))
col4.metric("Artigos Open Access", df_filtered[df_filtered['OA_Status'] == 'Open Access'].shape[0])

st.markdown("---")

# ============================================================
# GRÁFICO COMBINADO: PUBLICAÇÕES vs CITAÇÕES
# ============================================================
st.subheader("📈 Produtividade vs. Impacto (Publicações e Citações por Ano)")

df_ano_agg = df_filtered.groupby('Year').agg(
    Publicacoes=('Title', 'count'),
    Total_Citacoes=('Cited by', 'sum'),
    Media_Citacoes=('Cited by', 'mean')
).reset_index()

fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

# Barras = Publicações
fig_combo.add_trace(
    go.Bar(
        x=df_ano_agg['Year'],
        y=df_ano_agg['Publicacoes'],
        name="Número de Publicações",
        marker_color='#1f77b4',
        hovertemplate='Ano: %{x}<br>Publicações: %{y}<extra></extra>'
    ),
    secondary_y=False,
)

# Linha = Total de Citações
fig_combo.add_trace(
    go.Scatter(
        x=df_ano_agg['Year'],
        y=df_ano_agg['Total_Citacoes'],
        name="Total de Citações",
        line=dict(color='#ff7f0e', width=4),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='Ano: %{x}<br>Total de Citações: %{y}<extra></extra>'
    ),
    secondary_y=True,
)

fig_combo.update_xaxes(title_text="Ano de Publicação", dtick=1)
fig_combo.update_yaxes(title_text="<b>Quantidade</b> de Publicações", secondary_y=False,
                       title_font=dict(color="#1f77b4"))
fig_combo.update_yaxes(title_text="<b>Total</b> de Citações", secondary_y=True,
                       title_font=dict(color="#ff7f0e"))
fig_combo.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig_combo, use_container_width=True)

st.markdown("---")

# ============================================================
# GRÁFICOS SECUNDÁRIOS
# ============================================================
col_esq, col_dir = st.columns(2)

# 1. Tipos de Documento
with col_esq:
    st.subheader("📄 Tipos de Documento")
    tipos_doc = df_filtered['Document Type'].value_counts()
    fig_tipo = px.pie(values=tipos_doc.values, names=tipos_doc.index, hole=0.4)
    fig_tipo.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_tipo, use_container_width=True)

# 2. Acesso Aberto vs Fechado
with col_dir:
    st.subheader("🔓 Status de Acesso Aberto")
    oa_counts = df_filtered['OA_Status'].value_counts()
    fig_oa = px.pie(values=oa_counts.values, names=oa_counts.index, hole=0.4,
                    color_discrete_map={'Open Access': '#2ca02c', 'Fechado': '#d62728'})
    fig_oa.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_oa, use_container_width=True)

# 3. Top 10 Artigos Mais Citados
st.subheader("🏆 Top 10 Artigos Mais Citados")
top_citados = df_filtered.nlargest(10, 'Cited by')[['Title', 'Year', 'Cited by', 'Source title']].copy()
top_citados['Titulo_Curto'] = top_citados['Title'].apply(
    lambda x: str(x)[:70] + '...' if len(str(x)) > 70 else str(x)
)
fig_citados = px.bar(top_citados, x='Cited by', y='Titulo_Curto', orientation='h',
                     color='Cited by', color_continuous_scale='Viridis',
                     labels={'Titulo_Curto': 'Artigo', 'Cited by': 'Citações'},
                     hover_data=['Year', 'Source title', 'Title'])
fig_citados.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_citados, use_container_width=True)

col_esq2, col_dir2 = st.columns(2)

# 4. Top 10 Autores
with col_esq2:
    st.subheader("👥 Top 10 Autores mais Produtivos")
    lista_autores = []
    for autores in df_filtered['Authors'].dropna():
        lista_autores.extend([a.strip() for a in str(autores).split(';')])
    contagem_autores = Counter(lista_autores).most_common(10)
    df_autores = pd.DataFrame(contagem_autores, columns=['Autor', 'Publicações'])
    fig_autores = px.bar(df_autores, x='Publicações', y='Autor', orientation='h',
                         color='Publicações', color_continuous_scale='Plasma')
    fig_autores.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_autores, use_container_width=True)

# 5. Top 10 Periódicos
with col_dir2:
    st.subheader(" Top 10 Periódicos")
    top_periodicos = df_filtered['Source title'].value_counts().head(10).reset_index()
    top_periodicos.columns = ['Periódico', 'Publicações']
    fig_periodicos = px.bar(top_periodicos, x='Publicações', y='Periódico', orientation='h',
                            color='Publicações', color_continuous_scale='Magma')
    fig_periodicos.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_periodicos, use_container_width=True)

# ============================================================
# TABELA DE DADOS
# ============================================================
st.markdown("---")
st.subheader("📋 Dados Filtrados")
st.dataframe(
    df_filtered[['Title', 'Authors', 'Year', 'Source title', 'Cited by', 'Open Access']],
    use_container_width=True
)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.caption("Dashboard gerado com Streamlit + Plotly | Dados exportados do Scopus")
