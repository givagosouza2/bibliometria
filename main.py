import io
import re
import math
import unicodedata
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="Análise Cientométrica — Scopus",
    page_icon="📚",
    layout="wide",
)

REQUIRED_COLUMNS = {
    "Authors", "Title", "Year", "Source title", "Cited by",
    "Document Type", "Affiliations", "Author Keywords",
    "Index Keywords", "Abstract", "References"
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value)


def split_semicolon(value):
    text = normalize_text(value)
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def clean_author_name(name):
    name = re.sub(r"\s*\([^)]*\)\s*$", "", normalize_text(name))
    return name.strip(" ;,")


def parse_authors(row):
    full = split_semicolon(row.get("Author full names", ""))
    if full:
        return [clean_author_name(x) for x in full]
    return [clean_author_name(x) for x in split_semicolon(row.get("Authors", ""))]


def parse_keywords(row):
    values = split_semicolon(row.get("Author Keywords", ""))
    if not values:
        values = split_semicolon(row.get("Index Keywords", ""))
    return [x.lower().strip() for x in values if x.strip()]


def parse_affiliations(value):
    return split_semicolon(value)


def load_scopus(uploaded_file):
    raw = uploaded_file.getvalue()
    attempts = [
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(io.BytesIO(raw), **kwargs)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Não foi possível ler o CSV: {last_error}")


def prepare_data(df):
    df = df.copy()
    df.columns = [normalize_text(c) for c in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "O arquivo não contém todos os campos esperados da Scopus. "
            f"Campos ausentes: {', '.join(sorted(missing))}"
        )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Year", "Title"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Cited by"] = df["Cited by"].astype(float)

    df["Authors_list"] = df.apply(parse_authors, axis=1)
    df["Keywords_list"] = df.apply(parse_keywords, axis=1)
    df["Affiliations_list"] = df["Affiliations"].apply(parse_affiliations)
    df["N_authors"] = df["Authors_list"].apply(len)
    df["N_keywords"] = df["Keywords_list"].apply(len)

    current_year = pd.Timestamp.now().year
    df["Publication_age"] = (current_year - df["Year"] + 1).clip(lower=1)
    df["Citations_per_year"] = df["Cited by"] / df["Publication_age"]

    text_cols = ["Title", "Abstract", "Author Keywords", "Index Keywords"]
    for col in text_cols:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["Text_for_topics"] = (
        df["Title"] + ". " + df["Abstract"] + ". " +
        df["Author Keywords"] + "; " + df["Index Keywords"]
    )

    return df


def h_index(citations):
    values = sorted([float(x) for x in citations if pd.notna(x)], reverse=True)
    return sum(c >= i for i, c in enumerate(values, start=1))


def g_index(citations):
    values = sorted([float(x) for x in citations if pd.notna(x)], reverse=True)
    cumulative = 0
    g = 0
    for i, value in enumerate(values, start=1):
        cumulative += value
        if cumulative >= i * i:
            g = i
    return g


def m_index(citations, first_year):
    h = h_index(citations)
    career_years = max(pd.Timestamp.now().year - int(first_year) + 1, 1)
    return h / career_years


def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def plot_network(edges, node_weights, title, max_nodes=40):
    if not edges:
        st.info("Não há conexões suficientes com os filtros atuais.")
        return

    strongest = Counter()
    for a, b, w in edges:
        strongest[a] += w
        strongest[b] += w

    selected = {n for n, _ in strongest.most_common(max_nodes)}
    graph = nx.Graph()
    for a, b, weight in edges:
        if a in selected and b in selected:
            graph.add_edge(a, b, weight=weight)

    if graph.number_of_nodes() < 2:
        st.info("A rede ficou pequena demais após a filtragem.")
        return

    pos = nx.spring_layout(graph, seed=42, weight="weight", k=1.2 / math.sqrt(graph.number_of_nodes()))

    edge_x, edge_y = [], []
    for a, b in graph.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color="rgba(130,130,130,0.45)"),
        hoverinfo="none"
    )

    node_x, node_y, labels, sizes, hover = [], [], [], [], []
    degrees = dict(graph.degree(weight="weight"))
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        sizes.append(10 + 3.2 * math.sqrt(max(node_weights.get(node, 1), 1)))
        hover.append(
            f"<b>{node}</b><br>Documentos: {node_weights.get(node, 0)}"
            f"<br>Força das conexões: {degrees.get(node, 0):.0f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=labels, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(
            size=sizes,
            color=[degrees.get(n, 0) for n in graph.nodes()],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Força"),
            line=dict(width=0.5, color="white")
        )
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=55, b=10),
        height=690,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def author_network(df):
    edge_counter = Counter()
    node_counter = Counter()
    for authors in df["Authors_list"]:
        unique = list(dict.fromkeys(authors))
        node_counter.update(unique)
        for a, b in combinations(sorted(unique), 2):
            edge_counter[(a, b)] += 1
    edges = [(a, b, w) for (a, b), w in edge_counter.items()]
    return edges, node_counter


def keyword_network(df, min_occurrence=2):
    node_counter = Counter()
    edge_counter = Counter()
    for keywords in df["Keywords_list"]:
        unique = list(dict.fromkeys(keywords))
        node_counter.update(unique)

    allowed = {k for k, n in node_counter.items() if n >= min_occurrence}
    for keywords in df["Keywords_list"]:
        unique = sorted(set(keywords).intersection(allowed))
        for a, b in combinations(unique, 2):
            edge_counter[(a, b)] += 1

    edges = [(a, b, w) for (a, b), w in edge_counter.items()]
    filtered_nodes = Counter({k: v for k, v in node_counter.items() if k in allowed})
    return edges, filtered_nodes


def thematic_clusters(df, n_clusters):
    docs = df[df["Text_for_topics"].str.len() > 40].copy()
    if len(docs) < max(8, n_clusters * 2):
        return None, None, "Há poucos documentos com texto suficiente para a análise temática."

    max_df = 0.90 if len(docs) >= 20 else 1.0
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2 if len(docs) >= 20 else 1,
        max_df=max_df,
        ngram_range=(1, 2),
        max_features=2500,
        strip_accents="unicode"
    )
    try:
        matrix = vectorizer.fit_transform(docs["Text_for_topics"])
    except ValueError as exc:
        return None, None, str(exc)

    n_clusters = min(n_clusters, matrix.shape[0] - 1)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)
    docs["Cluster"] = labels + 1

    terms = np.array(vectorizer.get_feature_names_out())
    cluster_terms = []
    for idx, center in enumerate(model.cluster_centers_):
        top = terms[center.argsort()[::-1][:10]]
        cluster_terms.append({
            "Cluster": idx + 1,
            "Termos característicos": "; ".join(top),
            "Documentos": int((labels == idx).sum()),
            "Citações": int(docs.loc[labels == idx, "Cited by"].sum())
        })

    if matrix.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix.toarray())
        docs["Dimensão 1"] = coords[:, 0]
        docs["Dimensão 2"] = coords[:, 1]
    else:
        docs["Dimensão 1"] = np.arange(len(docs))
        docs["Dimensão 2"] = 0

    return docs, pd.DataFrame(cluster_terms), None


st.title("📚 Análise Cientométrica de Exportações da Scopus")
st.caption(
    "Aplicativo para indicadores bibliométricos, produtividade, impacto, "
    "redes de colaboração e estrutura temática."
)

with st.sidebar:
    st.header("1. Arquivo")
    uploaded = st.file_uploader("Carregue o CSV exportado da Scopus", type=["csv"])
    st.markdown(
        "Na Scopus, prefira **Export → CSV → All available information**."
    )

if uploaded is None:
    st.info("Carregue um arquivo CSV da Scopus para iniciar.")
    st.stop()

try:
    raw_df = load_scopus(uploaded)
    data = prepare_data(raw_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("2. Filtros")
    min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
    year_range = st.slider(
        "Período de publicação",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    document_types = sorted(data["Document Type"].dropna().astype(str).unique())
    selected_types = st.multiselect(
        "Tipos de documento",
        document_types,
        default=document_types
    )

    sources = sorted(data["Source title"].dropna().astype(str).unique())
    selected_sources = st.multiselect(
        "Periódicos/fontes",
        sources,
        default=[]
    )

filtered = data[
    data["Year"].between(year_range[0], year_range[1]) &
    data["Document Type"].astype(str).isin(selected_types)
].copy()

if selected_sources:
    filtered = filtered[filtered["Source title"].astype(str).isin(selected_sources)]

if filtered.empty:
    st.warning("Nenhum documento corresponde aos filtros selecionados.")
    st.stop()

tabs = st.tabs([
    "Visão geral", "Produção e impacto", "Periódicos e documentos",
    "Coautoria", "Palavras-chave", "Instituições", "Temas", "Dados"
])

with tabs[0]:
    citations = filtered["Cited by"]
    first_year = int(filtered["Year"].min())
    total_docs = len(filtered)
    total_citations = int(citations.sum())
    h = h_index(citations)
    g = g_index(citations)
    m = m_index(citations, first_year)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Documentos", f"{total_docs}")
    c2.metric("Citações", f"{total_citations:,}".replace(",", "."))
    c3.metric("Citações/documento", f"{citations.mean():.2f}")
    c4.metric("Índice h", f"{h}")
    c5.metric("Índice g", f"{g}")
    c6.metric("Índice m", f"{m:.2f}")

    zero = int((citations == 0).sum())
    international_proxy = int((filtered["Affiliations"].fillna("").str.count("Brazil") <
                               filtered["Affiliations"].fillna("").str.count(";")).sum())

    st.markdown(
        f"**Período analisado:** {first_year}–{int(filtered['Year'].max())}  \n"
        f"**Documentos sem citações:** {zero}  \n"
        f"**Mediana de citações:** {citations.median():.1f}  \n"
        f"**Média de autores por documento:** {filtered['N_authors'].mean():.2f}"
    )

    annual = filtered.groupby("Year").agg(
        Documentos=("Title", "count"),
        Citacoes_dos_documentos=("Cited by", "sum"),
        Media_citacoes=("Cited by", "mean")
    ).reset_index()

    fig = px.bar(
        annual, x="Year", y="Documentos",
        title="Produção científica anual",
        labels={"Year": "Ano", "Documentos": "Número de documentos"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Baixar resumo anual",
        dataframe_to_csv(annual),
        "resumo_anual.csv",
        "text/csv"
    )

with tabs[1]:
    annual = filtered.groupby("Year").agg(
        Documentos=("Title", "count"),
        Citacoes=("Cited by", "sum"),
        Citacoes_medias=("Cited by", "mean"),
        Autores_medios=("N_authors", "mean")
    ).reset_index()

    fig1 = px.line(
        annual, x="Year", y="Documentos", markers=True,
        title="Evolução da produção",
        labels={"Year": "Ano", "Documentos": "Documentos"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(
        annual, x="Year", y="Citacoes", markers=True,
        title="Citações acumuladas pelos documentos de cada ano de publicação",
        labels={"Year": "Ano de publicação", "Citacoes": "Citações atuais"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        filtered,
        x="Year", y="Cited by",
        size="Citations_per_year",
        hover_name="Title",
        hover_data=["Source title", "Document Type", "Citations_per_year"],
        title="Impacto por documento",
        labels={
            "Year": "Ano",
            "Cited by": "Citações",
            "Citations_per_year": "Citações/ano"
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

    distribution = px.histogram(
        filtered, x="Cited by", nbins=25,
        title="Distribuição das citações",
        labels={"Cited by": "Citações", "count": "Documentos"}
    )
    st.plotly_chart(distribution, use_container_width=True)

    impact_table = filtered[[
        "Title", "Year", "Source title", "Cited by",
        "Citations_per_year", "DOI", "Document Type"
    ]].sort_values(["Citations_per_year", "Cited by"], ascending=False)
    st.subheader("Impacto normalizado pelo tempo")
    st.dataframe(impact_table, use_container_width=True, hide_index=True)

with tabs[2]:
    source_table = filtered.groupby("Source title").agg(
        Documentos=("Title", "count"),
        Citacoes=("Cited by", "sum"),
        Media_citacoes=("Cited by", "mean"),
        Primeiro_ano=("Year", "min"),
        Ultimo_ano=("Year", "max")
    ).reset_index().sort_values(
        ["Documentos", "Citacoes"], ascending=False
    )

    top_n = st.slider("Número de fontes no gráfico", 5, 30, 15)
    fig = px.bar(
        source_table.head(top_n).sort_values("Documentos"),
        x="Documentos", y="Source title", orientation="h",
        title="Periódicos e fontes mais produtivos",
        labels={"Source title": "Fonte"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(source_table, use_container_width=True, hide_index=True)

    st.subheader("Documentos mais citados")
    top_docs = filtered[[
        "Title", "Authors", "Year", "Source title", "Cited by",
        "Citations_per_year", "DOI", "Document Type"
    ]].sort_values("Cited by", ascending=False)
    st.dataframe(top_docs.head(30), use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar tabela de documentos",
        dataframe_to_csv(top_docs),
        "documentos_ordenados_por_citacoes.csv",
        "text/csv"
    )

with tabs[3]:
    edges, nodes = author_network(filtered)
    author_table = pd.DataFrame(
        nodes.most_common(),
        columns=["Autor", "Documentos"]
    )
    st.subheader("Autores mais frequentes")
    st.dataframe(author_table.head(50), use_container_width=True, hide_index=True)

    max_nodes = st.slider("Autores exibidos na rede", 10, 80, 40)
    plot_network(
        edges, nodes,
        "Rede de coautoria — tamanho do nó representa número de documentos",
        max_nodes=max_nodes
    )

    edge_table = pd.DataFrame(edges, columns=["Autor 1", "Autor 2", "Colaborações"])
    edge_table = edge_table.sort_values("Colaborações", ascending=False)
    st.download_button(
        "Baixar relações de coautoria",
        dataframe_to_csv(edge_table),
        "rede_coautoria.csv",
        "text/csv"
    )

with tabs[4]:
    keyword_counter = Counter()
    for items in filtered["Keywords_list"]:
        keyword_counter.update(set(items))

    keyword_table = pd.DataFrame(
        keyword_counter.most_common(),
        columns=["Palavra-chave", "Documentos"]
    )

    st.subheader("Palavras-chave mais frequentes")
    st.dataframe(keyword_table.head(60), use_container_width=True, hide_index=True)

    min_occ = st.slider("Ocorrência mínima na rede", 1, 10, 2)
    max_kw_nodes = st.slider("Palavras-chave exibidas", 10, 80, 40)
    kw_edges, kw_nodes = keyword_network(filtered, min_occurrence=min_occ)
    plot_network(
        kw_edges, kw_nodes,
        "Rede de coocorrência de palavras-chave",
        max_nodes=max_kw_nodes
    )

    kw_edge_table = pd.DataFrame(
        kw_edges, columns=["Palavra-chave 1", "Palavra-chave 2", "Coocorrências"]
    ).sort_values("Coocorrências", ascending=False)
    st.download_button(
        "Baixar rede de palavras-chave",
        dataframe_to_csv(kw_edge_table),
        "rede_palavras_chave.csv",
        "text/csv"
    )

with tabs[5]:
    affiliation_counter = Counter()
    affiliation_citations = Counter()
    for _, row in filtered.iterrows():
        affiliations = set(row["Affiliations_list"])
        for affiliation in affiliations:
            affiliation_counter[affiliation] += 1
            affiliation_citations[affiliation] += row["Cited by"]

    aff_table = pd.DataFrame([
        {
            "Afiliação": aff,
            "Documentos": count,
            "Citações associadas": int(affiliation_citations[aff])
        }
        for aff, count in affiliation_counter.items()
    ]).sort_values(["Documentos", "Citações associadas"], ascending=False)

    st.subheader("Afiliações presentes nos documentos")
    st.caption(
        "A exportação da Scopus pode reunir departamento, laboratório e universidade "
        "como afiliações separadas; recomenda-se padronização manual para estudos formais."
    )
    st.dataframe(aff_table.head(100), use_container_width=True, hide_index=True)

    fig = px.bar(
        aff_table.head(20).sort_values("Documentos"),
        x="Documentos", y="Afiliação", orientation="h",
        title="Afiliações mais frequentes"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Baixar afiliações",
        dataframe_to_csv(aff_table),
        "afiliacoes.csv",
        "text/csv"
    )

with tabs[6]:
    st.subheader("Agrupamento temático exploratório")
    st.caption(
        "O agrupamento usa TF–IDF e K-means sobre títulos, resumos e palavras-chave. "
        "É uma análise exploratória; os rótulos precisam de interpretação científica."
    )
    max_clusters = min(10, max(2, len(filtered) // 4))
    n_clusters = st.slider("Número de grupos temáticos", 2, max_clusters, min(5, max_clusters))

    topic_docs, topic_summary, error = thematic_clusters(filtered, n_clusters)
    if error:
        st.warning(error)
    else:
        st.dataframe(topic_summary, use_container_width=True, hide_index=True)

        fig = px.scatter(
            topic_docs,
            x="Dimensão 1", y="Dimensão 2",
            color=topic_docs["Cluster"].astype(str),
            size="Cited by",
            hover_name="Title",
            hover_data=["Year", "Source title"],
            title="Mapa bidimensional dos grupos temáticos",
            labels={"color": "Grupo"}
        )
        st.plotly_chart(fig, use_container_width=True)

        cluster_year = topic_docs.groupby(["Year", "Cluster"]).size().reset_index(name="Documentos")
        cluster_year["Cluster"] = cluster_year["Cluster"].astype(str)
        evolution = px.area(
            cluster_year, x="Year", y="Documentos", color="Cluster",
            title="Evolução temporal dos grupos temáticos"
        )
        st.plotly_chart(evolution, use_container_width=True)

        export_topics = topic_docs[[
            "Title", "Year", "Source title", "Cited by", "Cluster"
        ]].sort_values(["Cluster", "Cited by"], ascending=[True, False])
        st.download_button(
            "Baixar classificação temática",
            dataframe_to_csv(export_topics),
            "classificacao_tematica.csv",
            "text/csv"
        )

with tabs[7]:
    st.subheader("Base filtrada")
    display_cols = [
        "Authors", "Title", "Year", "Source title", "Cited by", "DOI",
        "Document Type", "Author Keywords", "Index Keywords", "Affiliations"
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar base filtrada",
        dataframe_to_csv(filtered.drop(columns=[
            "Authors_list", "Keywords_list", "Affiliations_list"
        ], errors="ignore")),
        "base_scopus_filtrada.csv",
        "text/csv"
    )

st.divider()
st.caption(
    "A contagem de citações corresponde ao valor registrado no momento da exportação "
    "da Scopus. Redes institucionais e internacionais exigem padronização das afiliações."
)
