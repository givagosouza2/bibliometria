import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Análise Cientométrica - Givago Silva Souza",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para processar o GSouza.csv (que tem metadados no topo)
def parse_gsouza_csv(file):
    content = file.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    # Extraindo metadados
    author_name = lines[0].split(',')[1].strip().replace('"', '') if len(lines) > 0 and ',' in lines[0] else "Autor Desconhecido"
    h_index_str = lines[2].split(',')[1].strip().replace('"', '') if len(lines) > 2 and ',' in lines[2] else ""
    
    h_index_match = re.search(r'h-index\s*=\s*(\d+)', h_index_str)
    h_index = int(h_index_match.group(1)) if h_index_match else 0
    
    # Extraindo a tabela de dados (começa na linha 4)
    data_lines = lines[4:]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    
    return df, author_name, h_index

# Título principal
st.title("📊 Dashboard de Análise Cientométrica")
st.markdown("Faça o upload dos arquivos exportados do Scopus na barra lateral para iniciar a análise.")
st.markdown("---")

# Sidebar para Upload de Arquivos
st.sidebar.header("📂 Carregamento de Dados")
gsouza_file = st.sidebar.file_uploader("1. Carregar GSouza.csv (Perfil do Autor)", type=['csv'])
scopus_file = st.sidebar.file_uploader("2. Carregar scopus.csv (Exportação Scopus)", type=['csv'])

if gsouza_file and scopus_file:
    try:
        # Carregando e processando os dados
        gsouza_df, author_name, h_index = parse_gsouza_csv(gsouza_file)
        scopus_df = pd.read_csv(scopus_file)
        
        st.sidebar.success("✅ Arquivos carregados com sucesso!")
        st.sidebar.info(f"**Autor:** {author_name}\n**h-index:** {h_index}")
        
        # --- INÍCIO DO DASHBOARD ---
        
        # Métricas principais
        st.subheader("📈 Métricas Gerais")
        col1, col2, col3, col4 = st.columns(4)
        
        total_publications = len(scopus_df)
        total_citations = scopus_df['Cited by'].sum() if 'Cited by' in scopus_df.columns else 0
        avg_citations = scopus_df['Cited by'].mean() if 'Cited by' in scopus_df.columns and total_publications > 0 else 0
        
        with col1:
            st.metric("Total de Publicações", total_publications)
        with col2:
            st.metric("Total de Citações", int(total_citations))
        with col3:
            st.metric("Média de Citações", f"{avg_citations:.2f}")
        with col4:
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
            st.subheader("Evolução das Publicações e Citações por Ano")
            if 'Year' in scopus_df.columns:
                col_t1_1, col_t1_2 = st.columns(2)
                
                with col_t1_1:
                    pub_by_year = scopus_df['Year'].value_counts().sort_index()
                    fig_pub = px.bar(x=pub_by_year.index, y=pub_by_year.values, 
                                     labels={'x': 'Ano', 'y': 'Publicações'},
                                     title='Número de Publicações por Ano',
                                     color_discrete_sequence=['#1f77b4'])
                    fig_pub.update_layout(showlegend=False)
                    st.plotly_chart(fig_pub, use_container_width=True)
                
                with col_t1_2:
                    cit_by_year = scopus_df.groupby('Year')['Cited by'].sum().sort_index()
                    fig_cit = go.Figure()
                    fig_cit.add_trace(go.Scatter(x=cit_by_year.index, y=cit_by_year.values, 
                                                 mode='lines+markers', name='Citações',
                                                 line=dict(color='#ff7f0e', width=3)))
                    fig_cit.update_layout(title='Evolução das Citações por Ano', 
                                          xaxis_title='Ano', yaxis_title='Total de Citações')
                    st.plotly_chart(fig_cit, use_container_width=True)
        
        with tab2:
            st.subheader("Principais Periódicos de Publicação")
            if 'Source title' in scopus_df.columns:
                top_journals = scopus_df['Source title'].value_counts().head(15)
                fig_journals = px.bar(x=top_journals.values, y=top_journals.index, 
                                      orientation='h', 
                                      labels={'x': 'Número de Publicações', 'y': 'Periódico'},
                                      title='Top 15 Periódicos',
                                      color_discrete_sequence=['#2ca02c'])
                fig_journals.update_layout(showlegend=False, height=600)
                st.plotly_chart(fig_journals, use_container_width=True)
                
                st.markdown("**Detalhes dos Periódicos (Top 10):**")
                journal_stats = scopus_df.groupby('Source title').agg({
                    'Title': 'count',
                    'Cited by': ['sum', 'mean']
                }).round(2)
                journal_stats.columns = ['Publicações', 'Citações Totais', 'Citações Médias']
                journal_stats = journal_stats.sort_values('Publicações', ascending=False).head(10)
                st.dataframe(journal_stats, use_container_width=True)
        
        with tab3:
            st.subheader("Análise de Palavras-chave")
            if 'Author Keywords' in scopus_df.columns:
                all_keywords = []
                for keywords in scopus_df['Author Keywords'].dropna():
                    if isinstance(keywords, str):
                        kws = [kw.strip().lower() for kw in keywords.split(';')]
                        all_keywords.extend(kws)
                
                if all_keywords:
                    keyword_counts = Counter(all_keywords)
                    top_keywords = keyword_counts.most_common(20)
                    
                    col_t3_1, col_t3_2 = st.columns([1, 1])
                    
                    with col_t3_1:
                        fig_kw = px.bar(x=[count for word, count in top_keywords], 
                                        y=[word for word, count in top_keywords], 
                                        orientation='h',
                                        labels={'x': 'Frequência', 'y': 'Palavra-chave'},
                                        title='Top 20 Palavras-chave dos Autores',
                                        color_discrete_sequence=['#d62728'])
                        fig_kw.update_layout(showlegend=False, height=600)
                        st.plotly_chart(fig_kw, use_container_width=True)
                    
                    with col_t3_2:
                        st.markdown("**Nuvem de Palavras**")
                        wordcloud_text = ' '.join(all_keywords)
                        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                              colormap='viridis', max_words=100).generate(wordcloud_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
        
        with tab4:
            st.subheader("Rede de Coautores")
            if 'Authors' in scopus_df.columns:
                all_authors = []
                for authors in scopus_df['Authors'].dropna():
                    if isinstance(authors, str):
                        author_list = [author.strip() for author in authors.split(';')]
                        # Filtrando o autor principal para focar na rede de colaboração
                        author_list = [a for a in author_list if '57560194800' not in a and 'Souza, Givago' not in a and 'da Silva Souza' not in a]
                        all_authors.extend(author_list)
                
                if all_authors:
                    author_counts = Counter(all_authors)
                    top_coauthors = author_counts.most_common(20)
                    
                    fig_coauth = px.bar(x=[count for author, count in top_coauthors], 
                                        y=[author for author, count in top_coauthors], 
                                        orientation='h',
                                        labels={'x': 'Número de Colaborações', 'y': 'Coautor'},
                                        title='Top 20 Coautores mais Frequentes',
                                        color_discrete_sequence=['#9467bd'])
                    fig_coauth.update_layout(showlegend=False, height=600)
                    st.plotly_chart(fig_coauth, use_container_width=True)
                    
                    st.markdown("**Lista de Colaboradores:**")
                    coauthor_df = pd.DataFrame(top_coauthors, columns=['Coautor', 'Colaborações'])
                    st.dataframe(coauthor_df, use_container_width=True, height=300)
        
        with tab5:
            st.subheader("Lista de Publicações")
            display_columns = ['Title', 'Year', 'Source title', 'Cited by', 'DOI']
            available_columns = [col for col in display_columns if col in scopus_df.columns]
            
            if available_columns:
                publications_display = scopus_df[available_columns].copy()
                publications_display = publications_display.sort_values(by=['Year', 'Cited by'], ascending=[False, False])
                
                if 'DOI' in publications_display.columns:
                    publications_display['Link'] = publications_display['DOI'].apply(
                        lambda x: f"https://doi.org/{x}" if pd.notna(x) else ""
                    )
                
                st.dataframe(publications_display, use_container_width=True, height=500)
                
                csv = publications_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download da Lista de Publicações (CSV)",
                    data=csv,
                    file_name="publicacoes_givago_souza.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar os arquivos: {e}")
        st.info("Certifique-se de que os arquivos CSV estão no formato padrão de exportação do Scopus.")

else:
    st.info("👈 Por favor, faça o upload de **ambos** os arquivos CSV na barra lateral para gerar o dashboard.")
    
    st.markdown("""
    ### 📌 Como exportar os dados do Scopus:
    1. **GSouza.csv**: Vá no perfil do autor no Scopus -> Aba "Publication Years" ou "Co-authors" -> Exportar para CSV.
    2. **scopus.csv**: Vá na busca de documentos do autor -> Selecione todos -> Exportar -> CSV (selecione todas as informações, incluindo Citações, Palavras-chave e Afilições).
    """)
