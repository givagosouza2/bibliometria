import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Análise Cientométrica - Givago Silva Souza",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Análise Cientométrica Completa")
st.markdown("### Autor: Givago Silva Souza")
st.markdown("---")

# Carregamento dos dados
@st.cache_data
def load_data():
    try:
        # Carregar GSouza.csv
        gsouza_df = pd.read_csv('GSouza.csv', encoding='utf-8')
        
        # Carregar scopus.csv
        scopus_df = pd.read_csv('scopus.csv', encoding='utf-8')
        
        return gsouza_df, scopus_df
    except Exception as e:
        st.error(f"Erro ao carregar os arquivos: {e}")
        return None, None

gsouza_df, scopus_df = load_data()

if gsouza_df is not None and scopus_df is not None:
    
    # Sidebar com filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro por ano
    if 'Year' in scopus_df.columns:
        years = sorted(scopus_df['Year'].dropna().unique())
        year_range = st.sidebar.slider(
            "Período de Publicação",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
    
    # Filtro por periódico (top 10)
    if 'Source title' in scopus_df.columns:
        top_journals = scopus_df['Source title'].value_counts().head(10).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Periódicos (Top 10)",
            options=top_journals,
            default=top_journals
        )
    
    # Aplicar filtros
    filtered_scopus = scopus_df.copy()
    if 'Year' in filtered_scopus.columns:
        filtered_scopus = filtered_scopus[
            (filtered_scopus['Year'] >= year_range[0]) & 
            (filtered_scopus['Year'] <= year_range[1])
        ]
    
    if selected_journals and 'Source title' in filtered_scopus.columns:
        filtered_scopus = filtered_scopus[filtered_scopus['Source title'].isin(selected_journals)]
    
    # Métricas principais
    st.subheader("📈 Métricas Gerais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_publications = len(filtered_scopus)
        st.metric("Total de Publicações", total_publications)
    
    with col2:
        if 'Cited by' in filtered_scopus.columns:
            total_citations = filtered_scopus['Cited by'].sum()
            st.metric("Total de Citações", int(total_citations))
    
    with col3:
        if 'Cited by' in filtered_scopus.columns and total_publications > 0:
            avg_citations = filtered_scopus['Cited by'].mean()
            st.metric("Média de Citações", f"{avg_citations:.2f}")
    
    with col4:
        # h-index (do arquivo GSouza)
        if 'h-index' in str(gsouza_df.iloc[0, 0]):
            h_index = 18  # Valor mencionado no arquivo
            st.metric("h-index", h_index)
    
    st.markdown("---")
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Evolução Temporal",
        "📚 Periódicos",
        "🔑 Palavras-chave",
        "👥 Coautores",
        "📄 Publicações"
    ])
    
    with tab1:
        st.subheader("Evolução das Publicações por Ano")
        
        if 'Year' in filtered_scopus.columns:
            publications_by_year = filtered_scopus['Year'].value_counts().sort_index()
            
            fig = px.bar(
                x=publications_by_year.index,
                y=publications_by_year.values,
                labels={'x': 'Ano', 'y': 'Número de Publicações'},
                title='Publicações por Ano'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Citações por ano
        if 'Year' in filtered_scopus.columns and 'Cited by' in filtered_scopus.columns:
            st.subheader("Citações Acumuladas por Ano")
            
            citations_by_year = filtered_scopus.groupby('Year')['Cited by'].sum().sort_index()
            cumulative_citations = citations_by_year.cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_citations.index,
                y=cumulative_citations.values,
                mode='lines+markers',
                name='Citações Acumuladas',
                line=dict(color='red', width=3)
            ))
            fig.update_layout(
                title='Evolução das Citações Acumuladas',
                xaxis_title='Ano',
                yaxis_title='Total de Citações'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Principais Periódicos")
        
        if 'Source title' in filtered_scopus.columns:
            top_journals = filtered_scopus['Source title'].value_counts().head(15)
            
            fig = px.bar(
                x=top_journals.values,
                y=top_journals.index,
                orientation='h',
                labels={'x': 'Número de Publicações', 'y': 'Periódico'},
                title='Top 15 Periódicos'
            )
            fig.update_layout(showlegend=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("Detalhes dos Periódicos")
            journal_stats = filtered_scopus.groupby('Source title').agg({
                'Title': 'count',
                'Cited by': ['sum', 'mean']
            }).round(2)
            journal_stats.columns = ['Publicações', 'Citações Totais', 'Citações Médias']
            journal_stats = journal_stats.sort_values('Publicações', ascending=False)
            
            st.dataframe(journal_stats.head(20), use_container_width=True)
    
    with tab3:
        st.subheader("Análise de Palavras-chave")
        
        if 'Author Keywords' in filtered_scopus.columns:
            # Extrair palavras-chave
            all_keywords = []
            for keywords in filtered_scopus['Author Keywords'].dropna():
                if isinstance(keywords, str):
                    # Separar por ponto e vírgula
                    kws = [kw.strip() for kw in keywords.split(';')]
                    all_keywords.extend(kws)
            
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                top_keywords = keyword_counts.most_common(20)
                
                # Gráfico de palavras-chave
                fig = px.bar(
                    x=[count for word, count in top_keywords],
                    y=[word for word, count in top_keywords],
                    orientation='h',
                    labels={'x': 'Frequência', 'y': 'Palavra-chave'},
                    title='Top 20 Palavras-chave'
                )
                fig.update_layout(showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Word Cloud
                st.subheader("Nuvem de Palavras")
                wordcloud_text = ' '.join(all_keywords)
                
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis'
                ).generate(wordcloud_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        # Palavras-chave dos Index Keywords
        if 'Index Keywords' in filtered_scopus.columns:
            st.subheader("Termos de Indexação")
            all_index_keywords = []
            for keywords in filtered_scopus['Index Keywords'].dropna():
                if isinstance(keywords, str):
                    kws = [kw.strip() for kw in keywords.split(';')]
                    all_index_keywords.extend(kws)
            
            if all_index_keywords:
                index_keyword_counts = Counter(all_index_keywords)
                top_index_keywords = index_keyword_counts.most_common(15)
                
                fig = px.bar(
                    x=[count for word, count in top_index_keywords],
                    y=[word for word, count in top_index_keywords],
                    orientation='h',
                    labels={'x': 'Frequência', 'y': 'Termo'},
                    title='Top 15 Termos de Indexação'
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Análise de Coautores")
        
        if 'Authors' in filtered_scopus.columns:
            # Extrair coautores
            all_authors = []
            for authors in filtered_scopus['Authors'].dropna():
                if isinstance(authors, str):
                    # Separar por ponto e vírgula
                    author_list = [author.strip() for author in authors.split(';')]
                    all_authors.extend(author_list)
            
            # Remover o autor principal
            main_author_variants = ['Souza G.S.', 'Souza Givago', 'da Silva Souza G.', 'Souza G.']
            all_authors = [a for a in all_authors if not any(variant in a for variant in main_author_variants)]
            
            if all_authors:
                author_counts = Counter(all_authors)
                top_coauthors = author_counts.most_common(20)
                
                fig = px.bar(
                    x=[count for author, count in top_coauthors],
                    y=[author for author, count in top_coauthors],
                    orientation='h',
                    labels={'x': 'Número de Colaborações', 'y': 'Coautor'},
                    title='Top 20 Coautores'
                )
                fig.update_layout(showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela de coautores
                st.subheader("Lista Completa de Coautores")
                coauthor_df = pd.DataFrame(top_coauthors, columns=['Coautor', 'Colaborações'])
                st.dataframe(coauthor_df, use_container_width=True)
        
        # Análise de afiliações
        if 'Affiliations' in filtered_scopus.columns:
            st.subheader("Afiliações Institucionais")
            
            all_affiliations = []
            for affiliations in filtered_scopus['Affiliations'].dropna():
                if isinstance(affiliations, str):
                    # Separar por ponto e vírgula
                    aff_list = [aff.strip() for aff in affiliations.split(';')]
                    all_affiliations.extend(aff_list)
            
            if all_affiliations:
                affiliation_counts = Counter(all_affiliations)
                top_affiliations = affiliation_counts.most_common(15)
                
                fig = px.bar(
                    x=[count for aff, count in top_affiliations],
                    y=[aff for aff, count in top_affiliations],
                    orientation='h',
                    labels={'x': 'Número de Publicações', 'y': 'Afiliação'},
                    title='Top 15 Afiliações Institucionais'
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Lista de Publicações")
        
        # Selecionar colunas para exibir
        display_columns = ['Title', 'Year', 'Source title', 'Cited by', 'DOI']
        available_columns = [col for col in display_columns if col in filtered_scopus.columns]
        
        if available_columns:
            publications_display = filtered_scopus[available_columns].copy()
            
            # Ordenar por ano (mais recente primeiro) e citações
            if 'Year' in publications_display.columns:
                publications_display = publications_display.sort_values(
                    by=['Year', 'Cited by'] if 'Cited by' in publications_display.columns else ['Year'],
                    ascending=[False, False] if 'Cited by' in publications_display.columns else [False]
                )
            
            # Adicionar link para DOI
            if 'DOI' in publications_display.columns:
                publications_display['Link'] = publications_display['DOI'].apply(
                    lambda x: f"https://doi.org/{x}" if pd.notna(x) else ""
                )
            
            st.dataframe(
                publications_display,
                use_container_width=True,
                height=600
            )
            
            # Download da lista
            csv = publications_display.to_csv(index=False)
            st.download_button(
                label="📥 Download da Lista de Publicações (CSV)",
                data=csv,
                file_name="publicacoes_givago_souza.csv",
                mime="text/csv"
            )
    
    # Resumo final
    st.markdown("---")
    st.subheader("📋 Resumo da Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Principais Indicadores:**
        - Total de publicações: **{total_publications}**
        - Total de citações: **{int(total_citations)}**
        - Média de citações por artigo: **{avg_citations:.2f}**
        - h-index: **18**
        """)
    
    with col2:
        if 'Source title' in filtered_scopus.columns:
            most_productive_journal = filtered_scopus['Source title'].value_counts().index[0]
            st.markdown(f"""
            **Destaques:**
            - Periódico mais produtivo: **{most_productive_journal}**
            - Ano com mais publicações: **{publications_by_year.idxmax() if 'Year' in filtered_scopus.columns else 'N/A'}** ({publications_by_year.max() if 'Year' in filtered_scopus.columns else 0} artigos)
            """)
    
    st.markdown("---")
    st.markdown("*Análise gerada automaticamente a partir dos dados do Scopus e GSouza.csv*")

else:
    st.error("Não foi possível carregar os dados. Verifique se os arquivos estão no formato correto.")
