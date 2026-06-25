import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_data(gsouza_path: str, scopus_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os arquivos de dados do Google Scholar e Scopus.
    
    Args:
        gsouza_path: Caminho para o arquivo CSV do Google Scholar
        scopus_path: Caminho para o arquivo CSV do Scopus
    
    Returns:
        Tuple com dois DataFrames (gsouza_df, scopus_df)
    """
    # Verificar se os arquivos existem
    if not Path(gsouza_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {gsouza_path}")
    if not Path(scopus_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {scopus_path}")
    
    # Carregar dados
    gsouza_df = pd.read_csv(gsouza_path, encoding='utf-8')
    scopus_df = pd.read_csv(scopus_path, encoding='utf-8')
    
    print(f"✓ Arquivo Google Scholar carregado: {gsouza_df.shape[0]} linhas")
    print(f"✓ Arquivo Scopus carregado: {scopus_df.shape[0]} linhas")
    
    return gsouza_df, scopus_df

def analyze_citations(gsouza_df: pd.DataFrame) -> dict:
    """
    Analisa os dados de citações do arquivo Google Scholar.
    """
    # Extrair métricas principais
    total_citations = gsouza_df['Citações'].sum()
    
    # Calcular h-index
    sorted_citations = sorted(gsouza_df['Citações'], reverse=True)
    h_index = 0
    for i, citations in enumerate(sorted_citations, 1):
        if citations >= i:
            h_index = i
        else:
            break
    
    # Publicações por ano
    publications_by_year = gsouza_df.groupby('Ano').size()
    
    # Citações por ano
    citations_by_year = gsouza_df.groupby('Ano')['Citações'].sum()
    
    # Top 10 artigos mais citados
    top_cited = gsouza_df.nlargest(10, 'Citações')[['Título', 'Ano', 'Citações']]
    
    return {
        'total_citations': total_citations,
        'h_index': h_index,
        'total_publications': len(gsouza_df),
        'publications_by_year': publications_by_year,
        'citations_by_year': citations_by_year,
        'top_cited': top_cited
    }

def analyze_publications(scopus_df: pd.DataFrame) -> dict:
    """
    Analisa os dados de publicações do arquivo Scopus.
    """
    # Total de publicações
    total_pubs = len(scopus_df)
    
    # Publicações por ano
    pubs_by_year = scopus_df['Year'].value_counts().sort_index()
    
    # Tipos de documento
    doc_types = scopus_df['Document Type'].value_counts()
    
    # Periódicos mais frequentes
    top_journals = scopus_df['Source title'].value_counts().head(10)
    
    # Autores mais frequentes
    all_authors = []
    for authors in scopus_df['Authors'].dropna():
        all_authors.extend([a.strip() for a in authors.split(';')])
    author_counts = pd.Series(all_authors).value_counts().head(10)
    
    return {
        'total_publications': total_pubs,
        'publications_by_year': pubs_by_year,
        'document_types': doc_types,
        'top_journals': top_journals,
        'top_authors': author_counts
    }

def create_visualizations(citation_data: dict, publication_data: dict):
    """
    Cria visualizações da análise cientométrica.
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Evolução de publicações e citações ao longo do tempo
    ax1 = plt.subplot(3, 2, 1)
    years = sorted(set(citation_data['publications_by_year'].index) | 
                   set(publication_data['publications_by_year'].index))
    
    pubs_scopus = [publication_data['publications_by_year'].get(y, 0) for y in years]
    pubs_gs = [citation_data['publications_by_year'].get(y, 0) for y in years]
    cits_gs = [citation_data['citations_by_year'].get(y, 0) for y in years]
    
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pubs_scopus, width, label='Scopus', alpha=0.7)
    bars2 = ax1.bar(x + width/2, pubs_gs, width, label='Google Scholar', alpha=0.7)
    
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Número de Publicações')
    ax1.set_title('Publicações por Ano (Scopus vs Google Scholar)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45)
    ax1.legend()
    
    # 2. Citações por ano
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(years, cits_gs, marker='o', linewidth=2, markersize=8)
    ax2.fill_between(years, cits_gs, alpha=0.3)
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Número de Citações')
    ax2.set_title('Evolução de Citações ao Longo do Tempo')
    ax2.set_xticks(years[::2])
    ax2.grid(True, alpha=0.3)
    
    # 3. Tipos de documento
    ax3 = plt.subplot(3, 2, 3)
    publication_data['document_types'].head(8).plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_xlabel('Número de Publicações')
    ax3.set_title('Tipos de Documento (Scopus)')
    ax3.invert_yaxis()
    
    # 4. Top 10 periódicos
    ax4 = plt.subplot(3, 2, 4)
    publication_data['top_journals'].plot(kind='barh', ax=ax4, color='coral')
    ax4.set_xlabel('Número de Publicações')
    ax4.set_title('Top 10 Periódicos (Scopus)')
    ax4.invert_yaxis()
    
    # 5. Top 10 autores
    ax5 = plt.subplot(3, 2, 5)
    publication_data['top_authors'].plot(kind='barh', ax=ax5, color='green')
    ax5.set_xlabel('Número de Publicações')
    ax5.set_title('Top 10 Autores (Scopus)')
    ax5.invert_yaxis()
    
    # 6. Top 10 artigos mais citados
    ax6 = plt.subplot(3, 2, 6)
    top_cited = citation_data['top_cited']
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_cited)))
    ax6.barh(range(len(top_cited)), top_cited['Citações'], color=colors)
    ax6.set_yticks(range(len(top_cited)))
    ax6.set_yticklabels([f"{row['Título'][:50]}..." if len(str(row['Título'])) > 50 
                         else row['Título'] for _, row in top_cited.iterrows()])
    ax6.set_xlabel('Número de Citações')
    ax6.set_title('Top 10 Artigos Mais Citados (Google Scholar)')
    ax6.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('analise_cientometrica.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo como 'analise_cientometrica.png'")
    plt.show()

def generate_report(citation_data: dict, publication_data: dict):
    """
    Gera um relatório textual da análise cientométrica.
    """
    print("\n" + "="*70)
    print("RELATÓRIO DE ANÁLISE CIENTOMÉTRICA")
    print("="*70)
    
    print("\n📊 MÉTRICAS DO GOOGLE SCHOLAR:")
    print(f"   • Total de publicações: {citation_data['total_publications']}")
    print(f"   • Total de citações: {citation_data['total_citations']:,}")
    print(f"   • Índice h: {citation_data['h_index']}")
    print(f"   • Média de citações por artigo: {citation_data['total_citations']/citation_data['total_publications']:.1f}")
    
    print("\n📚 MÉTRICAS DO SCOPUS:")
    print(f"   • Total de publicações: {publication_data['total_publications']}")
    print(f"   • Período analisado: {publication_data['publications_by_year'].index.min()} - {publication_data['publications_by_year'].index.max()}")
    
    print("\n📈 PUBLICAÇÕES POR ANO (Scopus):")
    for year, count in publication_data['publications_by_year'].items():
        print(f"   {year}: {count}")
    
    print("\n🏆 TOP 5 PERIÓDICOS:")
    for i, (journal, count) in enumerate(publication_data['top_journals'].head(5).items(), 1):
        print(f"   {i}. {journal}: {count} publicações")
    
    print("\n👥 TOP 5 AUTORES:")
    for i, (author, count) in enumerate(publication_data['top_authors'].head(5).items(), 1):
        print(f"   {i}. {author}: {count} publicações")
    
    print("\n📖 TOP 5 ARTIGOS MAIS CITADOS:")
    for i, row in citation_data['top_cited'].head(5).iterrows():
        title = str(row['Título'])[:60] + "..." if len(str(row['Título'])) > 60 else row['Título']
        print(f"   {i}. [{row['Ano']}] {title}")
        print(f"      Citações: {row['Citações']}")
    
    print("\n" + "="*70)

def main():
    """
    Função principal para executar a análise cientométrica.
    """
    # Caminhos dos arquivos
    GSOUZA_PATH = 'GSouza.csv'
    SCOPUS_PATH = 'scopus.csv'
    
    try:
        # Carregar dados
        print("Carregando dados...")
        gsouza_df, scopus_df = load_data(GSOUZA_PATH, SCOPUS_PATH)
        
        # Analisar dados
        print("\nAnalisando dados de citações...")
        citation_data = analyze_citations(gsouza_df)
        
        print("Analisando dados de publicações...")
        publication_data = analyze_publications(scopus_df)
        
        # Gerar visualizações
        print("\nGerando visualizações...")
        create_visualizations(citation_data, publication_data)
        
        # Gerar relatório
        generate_report(citation_data, publication_data)
        
        print("\n✓ Análise concluída com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a análise: {e}")
        raise

if __name__ == '__main__':
    main()
