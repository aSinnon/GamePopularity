"""Aplicacao Flask para enviar CSV, analisar dados e prever popularidade."""

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from dados import (
    BASE_DIR, COLUNAS_OBRIGATORIAS,
    gerar_analise_e_graficos, preparar_base_analise,
)
from modelo import obter_opcoes_formulario, prever_popularidade

app = Flask(__name__)
app.secret_key = "steam-analyzer-local"

PASTA_UPLOADS = BASE_DIR / "uploads"
CAMINHO_UPLOAD = PASTA_UPLOADS / "dataset_atual.csv"
PASTA_DADOS = BASE_DIR / "data"
CAMINHO_TRATADO_UPLOAD = PASTA_DADOS / "dataset_analisado.csv"
PASTA_STATIC = BASE_DIR / "static"
LIMITE_UPLOAD_MB = 100
app.config["MAX_CONTENT_LENGTH"] = LIMITE_UPLOAD_MB * 1024 * 1024
PASTA_UPLOADS.mkdir(parents=True, exist_ok=True)
PASTA_DADOS.mkdir(parents=True, exist_ok=True)

def ler_csv_enviado(caminho):
    """Tenta identificar o separador e a codificacao do arquivo enviado."""
    try:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="cp1252")

def validar_csv(df):
    """Verifica se o arquivo possui estrutura e valores basicos esperados."""
    if df.empty:
        raise ValueError("O arquivo nao possui linhas de dados.")

    df.columns = [str(coluna).strip() for coluna in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("O CSV possui nomes de colunas repetidos.")

    faltantes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in df.columns]
    if faltantes:
        raise ValueError("Estao faltando colunas obrigatorias: " + ", ".join(faltantes))

    precos = pd.to_numeric(df["Price"], errors="coerce")
    if not precos.notna().any():
        raise ValueError("A coluna Price precisa ter pelo menos um valor numerico.")

    target_valido = df["Target"].dropna().astype(str).str.replace(",", "", regex=False)
    target_valido = target_valido.str.fullmatch(r"\s*\d+\s*-\s*\d+\s*")
    if not target_valido.any():
        raise ValueError("A coluna Target precisa conter faixas como 0 - 20000.")

    for coluna in ["Text", "Audio", "Mac", "Linux", "Developers", "Publishers", "Categories"]:
        valores = df[coluna].dropna().astype(str).str.strip()
        if valores.empty or valores.eq("").all():
            raise ValueError(f"A coluna {coluna} esta vazia.")

    return len(df), len(df.columns)

@app.route("/")
def inicio():
    arquivo_recebido = CAMINHO_UPLOAD.exists()
    tamanho_upload = None
    if arquivo_recebido:
        tamanho_upload = round(CAMINHO_UPLOAD.stat().st_size / (1024 * 1024), 2)

    return render_template(
        "index.html",
        arquivo_recebido=arquivo_recebido,
        tamanho_upload=tamanho_upload,
    )

@app.route("/upload", methods=["POST"])
def upload_csv():
    arquivo = request.files.get("arquivo")
    if arquivo is None or arquivo.filename == "":
        flash("Selecione um arquivo CSV.", "erro")
        return redirect(url_for("inicio"))
    if not arquivo.filename.lower().endswith(".csv"):
        flash("Envie um arquivo com extensao .csv.", "erro")
        return redirect(url_for("inicio"))

    temporario = PASTA_UPLOADS / "upload_temporario.csv"
    
    try:
        arquivo.save(temporario)
        if not temporario.exists() or temporario.stat().st_size == 0:
            raise ValueError("O arquivo enviado esta vazio.")

        df = ler_csv_enviado(temporario)
        linhas, colunas = validar_csv(df)
        temporario.replace(CAMINHO_UPLOAD)

        flash(
            f"CSV validado: {linhas:,} linhas e {colunas} colunas. "
            "O arquivo original foi salvo sem alteracoes.",
            "sucesso",
        )
    except (ValueError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as erro:
        flash(f"Nao foi possivel validar o CSV: {erro}", "erro")
    except Exception:
        app.logger.exception("Erro durante o upload do CSV")
        flash("Ocorreu um erro ao receber o arquivo. Confira o CSV e tente novamente.", "erro")
    finally:
        if temporario.exists():
            temporario.unlink()

    return redirect(url_for("inicio"))


@app.errorhandler(413)
def arquivo_muito_grande(_erro):
    flash(f"O arquivo ultrapassa o limite de {LIMITE_UPLOAD_MB} MB.", "erro")
    return redirect(url_for("inicio"))

@app.route("/analise")
def analise():
    if not CAMINHO_UPLOAD.exists():
        return render_template(
            "analise.html",
            erro="Envie um CSV na pagina inicial antes de abrir a analise.",
            resumo=None,
            relatorio=None,
        )
    try:
        df_original = ler_csv_enviado(CAMINHO_UPLOAD)
        df_tratado, resumo = preparar_base_analise(df_original)

        CAMINHO_TRATADO_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
        df_tratado.to_csv(CAMINHO_TRATADO_UPLOAD, index=False, encoding="utf-8-sig")
        relatorio = gerar_analise_e_graficos(df_tratado, PASTA_STATIC)

        return render_template(
            "analise.html",
            erro=None,
            resumo=resumo,
            relatorio=relatorio,
            caminho_tratado="data/dataset_analisado.csv",
            versao_grafico=CAMINHO_UPLOAD.stat().st_mtime_ns,
        )
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as erro:
        return render_template(
            "analise.html",
            erro=f"Nao foi possivel analisar o CSV: {erro}",
            resumo=None,
            relatorio=None,
        )
    except Exception:
        app.logger.exception("Erro durante a analise do CSV")
        return render_template(
            "analise.html",
            erro="Ocorreu um erro inesperado ao gerar a analise.",
            resumo=None,
            relatorio=None,
        )

@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    resultado = None
    erro = None
    valores = {}
    categorias_selecionadas = []
    
    try:
        opcoes = obter_opcoes_formulario()
    except FileNotFoundError as problema:
        opcoes = {"categories": []}
        erro = str(problema)

    if request.method == "POST" and erro is None:
        categorias_selecionadas = request.form.getlist("Categories")
        valores = {
            "Price": request.form.get("Price", ""),
            "Text": request.form.get("Text", ""),
            "Audio": request.form.get("Audio", ""),
            "Mac": request.form.get("Mac", ""),
            "Linux": request.form.get("Linux", ""),
        }
        valores["Categories"] = categorias_selecionadas

        try:
            resultado = prever_popularidade(valores)
        except (ValueError, TypeError) as problema:
            erro = str(problema)

    return render_template(
        "previsao.html",
        opcoes=opcoes,
        resultado=resultado,
        erro=erro,
        valores=valores,
        categorias_selecionadas=categorias_selecionadas,
    )

if __name__ == "__main__":
    app.run(debug=True)