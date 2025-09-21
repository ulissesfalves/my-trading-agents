import datetime
from pathlib import Path
from markdown_it import MarkdownIt
from weasyprint import HTML, CSS
from rich.console import Console

console = Console()

# --- Estilo CSS para o Relatório em PDF ---
CSS_STYLE = """
@page {
    size: A4;
    margin: 1.5cm;
}
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}
h1, h2, h3 {
    font-family: 'Georgia', serif;
    color: #005a9c;
    font-weight: bold;
}
h1 {
    font-size: 28pt;
    text-align: center;
    margin-bottom: 2cm;
    page-break-after: avoid;
}
h2 {
    font-size: 18pt;
    border-bottom: 2px solid #005a9c;
    padding-bottom: 5px;
    margin-top: 1.5cm;
}
h3 {
    font-size: 14pt;
    color: #333;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3px;
    margin-top: 1cm;
}
code {
    background-color: #f0f0f0;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: 'Courier New', Courier, monospace;
}
pre {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 5px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
hr {
    border: 0;
    height: 1px;
    background: #ccc;
    margin: 2cm 0;
}
"""

def compile_reports_to_pdf(ticker: str, date: str):
    """
    Combina relatórios Markdown de uma análise em um único arquivo PDF.
    """
    base_dir = Path("results") / ticker / date
    reports_dir = base_dir / "reports"
    output_pdf_path = base_dir / f"Relatorio_Compilado_{ticker}_{date}.pdf"

    if not reports_dir.is_dir():
        console.print(f"[red]Erro: Diretório de relatórios não encontrado em '{reports_dir}'[/red]")
        return

    report_order = [
        ("market_report.md", "I. Relatório de Análise de Mercado"),
        ("sentiment_report.md", "II. Relatório de Sentimento Social"),
        ("news_report.md", "III. Relatório de Notícias"),
        ("fundamentals_report.md", "IV. Relatório de Fundamentos"),
        ("investment_plan.md", "V. Decisão da Equipe de Pesquisa"),
        ("trader_investment_plan.md", "VI. Plano da Equipe de Negociação"),
        ("final_trade_decision.md", "VII. Decisão Final da Gestão de Portfólio"),
    ]

    combined_md_content = [f"# Relatório de Análise de Investimento\n\n**Ativo:** `{ticker}`\n\n**Data:** `{date}`"]

    for filename, title in report_order:
        report_path = reports_dir / filename
        if report_path.exists():
            console.print(f"Adicionando relatório: {filename}")
            content = report_path.read_text(encoding="utf-8")
            combined_md_content.append(f"## {title}\n\n{content}")
        else:
            console.print(f"[yellow]Aviso: Relatório '{filename}' não encontrado. Pulando.[/yellow]")

    if len(combined_md_content) <= 1:
        console.print("[red]Nenhum relatório encontrado para compilar.[/red]")
        return

    console.print("\n[bold blue]Convertendo relatórios para PDF...[/bold blue]")
    md = MarkdownIt()
    # Adiciona uma linha horizontal entre as seções para melhor separação visual
    html_body = md.render("\n\n---\n\n".join(combined_md_content))
    
    full_html = f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Relatório - {ticker}</title></head><body>{html_body}</body></html>'
    
    css = CSS(string=CSS_STYLE)
    HTML(string=full_html, base_url=str(reports_dir)).write_pdf(output_pdf_path, stylesheets=[css])

    console.print(f"\n[bold green]Sucesso! O relatório foi salvo em:[/bold green] [underline]{output_pdf_path}[/underline]")
