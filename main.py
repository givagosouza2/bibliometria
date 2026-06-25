import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(page_title="Dashboard Cientométrico", layout="wide", page_icon="📊")

st.title("📊 Dashboard de Análise Cientométrica")
st.markdown("Análise combinada de dados **Scopus** (publicações) e **Google Scholar** (citações por ano).")

# ============================================================
# UPLOAD DE ARQUIVOS
# ============================================================
st.sidebar.header("📁 Upload de Arquivos")

scopus_file = st.sidebar.file_uploader("📄 Arquivo Scopus (CSV)", type=["csv"], key="scopus")
gsouza_file = st.sidebar.file_uploader("📄 Arquivo Citações / Scholar (CSV)", type=["csv"], key="gsouza")

if not scopus_file or not gsouza_file:
    st.info("⬅️ Faça o upload dos dois arquivos CSV na barra lateral para iniciar a análise.")
    st.stop()

# ============================================================
# CARREGAMENTO - SCOPUS
# ============================================================
@st.cache_data
def load_scopus(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding='latin-1')
    return df

scopus_df = load_scopus(scopus_file)

# Limpeza básica do Scopus
scopus_df['Year'] = pd.to_numeric(scopus_df['Year'], errors='coerce')
scopus_df['Cited by'] = pd.to_numeric(scopus_df['Cited by'], errors='coerce').fillna(0).astype(int)
scopus_df = scopus_df.dropna(subset=['Year'])
scopus_df = scopus_df.drop_duplicates(subset=['Title', 'Year', 'Cited by'], keep='first')

# ============================================================
# CARREGAMENTO - GSOUSA (Citações por ano)
# ============================================================
@st.cache_data
def load_gsouza(file):
    """
    Carrega o arquivo do tipo GSouza.csv (perfil de citações).
    O formato tem colunas com anos (2001, 2002, ..., 2026, Subtotal, Total)
    e cada linha representa um artigo com citações recebidas por ano.
    """
    try:
        raw = pd.read_csv(file, encoding='utf-8', header=None)
    except UnicodeDecodeError:
        file.seek(0)
        raw = pd.read_csv(file, encoding='latin-1', header=None)
    return raw

gsouza_raw = load_gsouza(gsouza_file)

# --- Parsing inteligente do GSouza ---
# 1. Extrair h-index da primeira linha
h_index_text = str(gsouza_raw.iloc[0].values)
h_index_match = re.search(r'h-index\s*=\s*(\d+)', h_index_text)
h_index = int(h_index_match.group(1)) if h_index_match else None

# Total de documentos no h-index
total_docs_match = re.search(r'(\d+)\s*documents', h_index_text)
total_docs_hindex = int(total_docs_match.group(1)) if total_docs_match else None

# Autor
author_match = re.search(r'Author:\s*([^"]+)', h_index_text)
author_name = author_match.group(1).strip() if author_match else "Autor"

# 2. Identificar a linha de cabeçalho (contém os anos)
header_row = None
for i, row in gsouza_raw.iterrows():
    row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])
    if 'Subtotal' in row_str or 'Total' in row_str:
        header_row = i
        break

# 3. Parsear as citações anuais
citation_years = {}
if header_row is not None:
    headers = [str(v).strip() for v in gsouza_raw.iloc[header_row].values]

    # Encontrar colunas que são anos (números entre 2000 e 2030)
    year_cols = {}
    for idx, h in enumerate(headers):
        try:
            y = int(float(h))
            if 2000 <= y <= 2030:
                year_cols[idx] = y
        except (ValueError, TypeError):
            pass

    # Dados das citações começam após o header
    data_start = header_row + 1
    yearly_citations = {y: 0 for y in year_cols.values()}

    for i in range(data_start, len(gsouza_raw)):
        row = gsouza_raw.iloc[i]
        for col_idx, year in year_cols.items():
            val = row.iloc[col_idx] if col_idx < len(row) else 0
            try:
                yearly_citations[year] += int(float(val)) if pd.notna(val) and str(val).strip() != '' else 0
            except (ValueError, TypeError):
                pass

    citation_years = yearly_citations

# Fallback: se não conseguir parsear, cria vazio
if not citation_years:
    citation_years = {}

# ============================================================
# SIDEBAR - FILTROS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros")

min_year = int(scopus_df['Year'].min())
max_year = int(scopus_df['Year'].max())
year_range = st.sidebar.slider("Intervalo de Anos:", min_year, max_year, (min_year, max_year))

doc_types = scopus_df['Document Type'].unique()
selected_docs = st.sidebar.multiselect("Tipos de Documento:", doc_types, default=doc_types)

df_filtered = scopus_df[
    (scopus_df['Year'] >= year_range[0]) &
    (scopus_df['Year'] <= year_range[1]) &
    (scopus_df['Document Type'].isin(selected_docs))
]

# ============================================================
# KPIs
# ============================================================
st.subheader(f"📌 Resumo Geral — {author_name}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Publicações (Scopus)", len(df_filtered))
col2.metric("Total de Citações (Scopus)", int(df_filtered['Cited by'].sum()))
col3.metric("Média Citações/Artigo", round(df_filtered['Cited by'].mean(), 2))
if h_index is not None:
    col4.metric("Índice h (Scholar)", h_index)
else:
    col4.metric("Índice h (Scholar)", "N/D")
if total_docs_hindex is not None:
    col5.metric("Documentos (Scholar)", total_docs_hindex)
else:
    col5.metric("Documentos (Scholar)", "N/D")

st.markdown("---")

# ============================================================
# GRÁFICO 1: PUBLICAÇÕES (Scopus) + CITAÇÕES ANUAIS (Scholar)
# ============================================================
st.subheader("📈 Publicações por Ano (Scopus) vs. Citações Recebidas por Ano (Scholar)")
st.caption("Barras = número de artigos publicados no ano (Scopus) | Linha = citações recebidas naquele ano (Scholar)")

# Publicações por ano (Scopus)
pubs_by_year = df_filtered.groupby('Year').size().reset_index(name='Publicacoes')

# Citações por ano (Scholar)
cits_df = pd.DataFrame(list(citation_years.items()), columns=['Year', 'Citacoes_Anuais'])
cits_df['Year'] = pd.to_numeric(cits_df['Year'], errors='coerce')
cits_df = cits_df.dropna(subset=['Year'])

# Merge
merged = pd.merge(pubs_by_year, cits_df, on='Year', how='outer').fillna(0)
merged = merged.sort_values('Year')
merged['Publicacoes'] = merged['Publicacoes'].astype(int)
merged['Citacoes_Anuais'] = merged['Citacoes_Anuais'].astype(int)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(
    go.Bar(
        x=merged['Year'],
        y=merged['Publicacoes'],
        name="Publicações (Scopus)",
        marker_color='#1f77b4',
        hovertemplate='Ano: %{x}<br>Publicações: %{y}<extra></extra>'
    ),
    secondary_y=False,
)

if len(cits_df) > 0:
    fig1.add_trace(
        go.Scatter(
            x=merged['Year'],
            y=merged['Citacoes_Anuais'],
            name="Citações no Ano (Scholar)",
            line=dict(color='#ff7f0e', width=3),
            mode='lines+markers',
            marker=dict(size=7),
            hovertemplate='Ano: %{x}<br>Citações no ano: %{y}<extra></extra>'
        ),
        secondary_y=True,
    )

fig1.update_xaxes(title_text="Ano", dtick=1)
fig1.update_yaxes(title_text="Publicações", secondary_y=False, title_font=dict(color="#1f77b4"))
fig1.update_yaxes(title_text="Citações Recebidas no Ano", secondary_y=True, title_font=dict(color="#ff7f0e"))
fig1.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ============================================================
# GRÁFICO 2: CUMULATIVO (Publicações + Citações)
# ============================================================
st.subheader("📈 Evolução Cumulativa de Publicações e Citações")
st.caption("Crescimento acumulado ao longo do tempo")

merged['Publicacoes_Cumulativas'] = merged['Publicacoes'].cumsum()
merged['Citacoes_Cumulativas'] = merged['Citacoes_Anuais'].cumsum()

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(
    go.Bar(
        x=merged['Year'],
        y=merged['Publicacoes_Cumulativas'],
        name="Publicações Cumulativas",
        marker_color='#2ca02c',
        hovertemplate='Ano: %{x}<br>Publicações Acumuladas: %{y}<extra></extra>'
    ),
    secondary_y=False,
)

fig2.add_trace(
    go.Scatter(
        x=merged['Year'],
        y=merged['Citacoes_Cumulativas'],
        name="Citações Cumulativas",
        line=dict(color='#d62728', width=3),
        mode='lines+markers',
        marker=dict(size=7),
        hovertemplate='Ano: %{x}<br>Citações Acumuladas: %{y}<extra></extra>'
    ),
    secondary_y=True,
)

fig2.update_xaxes(title_text="Ano", dtick=1)
fig2.update_yaxes(title_text="Publicações Cumulativas", secondary_y=False, title_font=dict(color="#2ca02c"))
fig2.update_yaxes(title_text="Citações Cumulativas", secondary_y=True, title_font=dict(color="#d62728"))
fig2.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================================
# GRÁFICO 3: TOP 10 ARTIGOS MAIS CITADOS (Scopus)
# ============================================================
st.subheader("🏆 Top 10 Artigos Mais Citados (Scopus)")
top_citados = df_filtered.nlargest(10, 'Cited by')[['Title', 'Year', 'Cited by', 'Source title']].copy()
top_citados['Titulo_Curto'] = top_citados['Title'].apply(
    lambda x: str(x)[:70] + '...' if len(str(x)) > 70 else str(x)
)
fig3 = px.bar(top_citados, x='Cited by', y='Titulo_Curto', orientation='h',
              color='Cited by', color_continuous_scale='Viridis',
              labels={'Titulo_Curto': 'Artigo', 'Cited by': 'Citações'},
              hover_data=['Year', 'Source title', 'Title'])
fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ============================================================
# GRÁFICOS SECUNDÁRIOS
# ============================================================
col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("📄 Tipos de Documento")
    tipos_doc = df_filtered['Document Type'].value_counts()
    fig_tipo = px.pie(values=tipos_doc.values, names=tipos_doc.index, hole=0.4)
    fig_tipo.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_tipo, use_container_width=True)

with col_dir:
    st.subheader("📚 Top 10 Periódicos")
    top_journals = df_filtered['Source title'].value_counts().head(10).reset_index()
    top_journals.columns = ['Periódico', 'Publicações']
    fig_journals = px.bar(top_journals, x='Publicações', y='Periódico', orientation='h',
                          color='Publicações', color_continuous_scale='Magma')
    fig_journals.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_journals, use_container_width=True)

# ============================================================
# TOP AUTORES (Coautores)
# ============================================================
st.subheader("👥 Top 10 Coautores mais Frequentes")
lista_autores = []
for autores in df_filtered['Authors'].dropna():
    lista_autores.extend([a.strip() for a in str(autores).split(';')])

# Filtrar o próprio autor para mostrar coautores
contagem_autores = Counter(lista_autores).most_common(15)
df_autores = pd.DataFrame(contagem_autores, columns=['Autor', 'Publicações'])

# Tenta remover o autor principal da lista
if author_name:
    # Busca parcial para remover variações do nome
    mask = df_autores['Autor'].apply(lambda x: author_name.split(',')[0].strip() not in str(x))
    df_coautores = df_autores[mask].head(10)
else:
    df_coautores = df_autores.head(10)

fig_autores = px.bar(df_coautores, x='Publicações', y='Autor', orientation='h',
                     color='Publicações', color_continuous_scale='Plasma')
fig_autores.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_autores, use_container_width=True)

st.markdown("---")

# ============================================================
# TABELA: CITAÇÕES ANUAIS (Scholar)
# ============================================================
if len(cits_df) > 0:
    st.subheader("📋 Tabela de Citações por Ano (Scholar)")
    st.dataframe(
        cits_df.sort_values('Year').rename(columns={
            'Year': 'Ano',
            'Citacoes_Anuais': 'Citações Recebidas no Ano'
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TABELA: DADOS FILTRADOS (Scopus)
# ============================================================
st.markdown("---")
st.subheader("📋 Publicações Filtradas (Scopus)")
st.dataframe(
    df_filtered[['Title', 'Authors', 'Year', 'Source title', 'Cited by', 'Document Type']],
    use_container_width=True
)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.caption("Dashboard Cientométrico | Dados: Scopus + Google Scholar | Streamlit + Plotly")
