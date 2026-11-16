"""Leitura, limpeza e analise dos dados do projeto."""

from pathlib import Path
import ast
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Caminhos usados pelos outros arquivos do projeto.
BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = BASE_DIR / "static" / "dadostreinamento.csv"
ARQUIVO_TRATADO = BASE_DIR / "data" / "dadostreinamento_tratado.csv"
# O CSV mantem estas dez colunas.
COLUNAS_OBRIGATORIAS = [
    "Name", "Target", "Price", "Text", "Audio", "Mac", "Linux",
    "Developers", "Publishers", "Categories",
]
# Estas sao as informacoes usadas pelo modelo.
COLUNAS_NUMERICAS = ["Price", "Text", "Audio", "Mac", "Linux"]
COLUNA_ALVO = "popularidade"
COLUNA_OWNERS = "owners_estimados"
CLASSES = ["Baixa", "Média", "Alta"]
LIMITE_BAIXA_MAX = 50_000
LIMITE_MEDIA_MAX = 500_000

def contar_idiomas(valor):
    """Conta quantos idiomas existem em uma celula do CSV."""
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()
    
    try:
        lista = ast.literal_eval(texto)
        if isinstance(lista, (list, tuple, set)):
            return len(lista)
    except (SyntaxError, ValueError):
        pass
    
    # Tratamento para algumas listas que nao podem ser lidas pelo literal_eval.
    if texto == "[]":
        return 0
    if texto.startswith("[") and texto.endswith("]"):
        separadores = re.findall(r"',\s*'", texto)
        if separadores:
            return len(separadores) + 1

    return np.nan

def estimar_owners(target):
    """Usa o ponto medio da faixa Target como estimativa de owners."""
    if pd.isna(target):
        return np.nan

    texto = str(target).strip().replace(",", "")
    resultado = re.fullmatch(r"(\d+)\s*-\s*(\d+)", texto)
    if not resultado:
        return np.nan

    minimo, maximo = map(int, resultado.groups())
    if maximo < minimo:
        return np.nan

    return (minimo + maximo) / 2

def converter_booleano(valor):
    """Converte valores True/False ou Sim/Nao para 1/0."""
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (bool, np.bool_)):
        return int(valor)

    texto = str(valor).strip().casefold()
    if texto in {"true", "1", "sim", "yes", "y"}:
        return 1
    if texto in {"false", "0", "nao", "não", "no", "n"}:
        return 0

    return np.nan

def classificar_popularidade(owners):
    """Transforma a estimativa de owners em Baixa, Media ou Alta."""
    if owners < LIMITE_BAIXA_MAX:
        return "Baixa"
    if owners < LIMITE_MEDIA_MAX:
        return "Média"
    return "Alta"

def ler_csv(caminho=ARQUIVO_ENTRADA):
    """Le o arquivo CSV."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {caminho}")

    return pd.read_csv(caminho, encoding="utf-8-sig", low_memory=False)

def preparar_base(df_original):
    """Limpa uma copia dos dados e devolve um resumo das alteracoes."""
    if not isinstance(df_original, pd.DataFrame):
        raise TypeError("Os dados precisam estar em uma tabela Pandas.")

    df = df_original.copy()
    df.columns = [str(coluna).strip() for coluna in df.columns]

    faltantes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in df.columns]
    if faltantes:
        raise ValueError("Colunas obrigatorias ausentes: " + ", ".join(faltantes))

    linhas_originais = len(df)
    colunas_originais = len(df.columns)
    # Remove espacos extras e transforma textos vazios em valores ausentes.
    for coluna in df.select_dtypes(include=["object", "string"]).columns:
        df[coluna] = df[coluna].map(
            lambda valor: valor.strip() if isinstance(valor, str) else valor
        )
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    ausentes = df[COLUNAS_OBRIGATORIAS].isna().sum()

    # Remove linhas duplicadas e linhas que nao possuem todos os campos obrigatorios.
    antes = len(df)
    df = df.drop_duplicates().copy()
    duplicadas_removidas = antes - len(df)
    
    antes = len(df)
    df = df.dropna(subset=COLUNAS_OBRIGATORIAS).copy()
    vazias_removidas = antes - len(df)

    # Padroniza o Target e remove os jogos sem owners estimados (0 - 0).
    df["Target"] = (
        df["Target"].astype("string").str.strip()
        .str.replace(r"\s*-\s*", " - ", regex=True)
    )
    antes = len(df)
    df = df[df["Target"] != "0 - 0"].copy()
    target_zero_removidos = antes - len(df)

    # Converte os dados que serao analisados ou usados pelo modelo.
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Text"] = df["Text"].map(contar_idiomas)
    df["Audio"] = df["Audio"].map(contar_idiomas)
    df["Mac"] = df["Mac"].map(converter_booleano)
    df["Linux"] = df["Linux"].map(converter_booleano)
    df[COLUNA_OWNERS] = df["Target"].map(estimar_owners)

    # Remove precos negativos, valores impossiveis e conversoes que falharam.
    colunas_validar = COLUNAS_NUMERICAS + [COLUNA_OWNERS]
    mascara_valida = df[colunas_validar].notna().all(axis=1)
    mascara_valida &= (df["Price"] >= 0)
    mascara_valida &= (df["Text"] >= 0)
    mascara_valida &= (df["Audio"] >= 0)
    mascara_valida &= (df[COLUNA_OWNERS] >= 0)

    antes = len(df)
    df = df[mascara_valida].copy()
    invalidas_removidas = antes - len(df)

    if df.empty:
        raise ValueError("Nenhuma linha valida restou depois da limpeza dos dados.")

    for coluna in ["Text", "Audio", "Mac", "Linux"]:
        df[coluna] = df[coluna].astype(int)

    df["Price"] = df["Price"].astype(float)
    df[COLUNA_OWNERS] = df[COLUNA_OWNERS].astype(float)
    df[COLUNA_ALVO] = df[COLUNA_OWNERS].map(classificar_popularidade)
    df = df.reset_index(drop=True)

    resumo = {
        "linhas_originais": linhas_originais,
        "colunas_originais": colunas_originais,
        "duplicadas_removidas": int(duplicadas_removidas),
        "vazias_removidas": int(vazias_removidas),
        "target_zero_removidos": int(target_zero_removidos),
        "invalidas_removidas": int(invalidas_removidas),
        "linhas_finais": len(df),
        "ausentes_por_coluna": {
            coluna: int(quantidade)
            for coluna, quantidade in ausentes.items()
            if quantidade > 0
        },
    }

    return df, resumo

def preparar_dados(dados=None):
    """Prepara a base original usada no treinamento."""
    if isinstance(dados, pd.DataFrame):
        df_original = dados
    else:
        caminho = ARQUIVO_ENTRADA if dados is None else dados
        df_original = ler_csv(caminho)

    df, resumo = preparar_base(df_original)
    ARQUIVO_TRATADO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ARQUIVO_TRATADO, index=False, encoding="utf-8-sig")

    return df, resumo

def separar_categorias(valor):
    """Transforma o conteudo da coluna Categories em uma lista."""
    if isinstance(valor, (list, tuple, set)):
        return [str(item).strip() for item in valor if str(item).strip()]

    return [item.strip() for item in str(valor).split(",") if item.strip()]

def calcular_opcoes_categories(df_treino):
    """Retorna as categorias encontradas somente nos dados de treinamento."""
    categorias = set()

    for valor in df_treino["Categories"]:
        categorias.update(separar_categorias(valor))

    return sorted(categorias, key=str.casefold)

def construir_X(df, opcoes_categories):
    """Monta as variaveis de entrada numericas e as categorias em colunas 0/1."""
    X = df[COLUNAS_NUMERICAS].copy()
    mapa_categorias = {
        categoria: indice for indice, categoria in enumerate(opcoes_categories)
    }

    matriz = np.zeros((len(df), len(opcoes_categories)), dtype=np.int8)

    for linha, valor in enumerate(df["Categories"]):
        for categoria in separar_categorias(valor):
            if categoria in mapa_categorias:
                matriz[linha, mapa_categorias[categoria]] = 1

    nomes_colunas = [f"Category_{categoria}" for categoria in opcoes_categories]
    categorias_df = pd.DataFrame(matriz, index=df.index, columns=nomes_colunas)

    return pd.concat([X, categorias_df], axis=1)

def preparar_X_formulario(entrada, opcoes_categories):
    """Prepara os valores do formulario com as mesmas colunas do treinamento."""
    df_linha = pd.DataFrame([entrada])
    return construir_X(df_linha, opcoes_categories)

def preparar_base_analise(df_original):
    """Aplica a mesma limpeza do treinamento ao CSV enviado pelo usuario."""
    return preparar_base(df_original)

def gerar_analise_e_graficos(df, pasta_static):
    """Gera tres graficos simples para a pagina de analise."""
    pasta_static = Path(pasta_static)
    pasta_static.mkdir(parents=True, exist_ok=True)

    # 1. Distribuicao das classes de popularidade.
    contagem_classes = df[COLUNA_ALVO].value_counts().reindex(CLASSES, fill_value=0)
    total = len(df)
    distribuicao_classes = [
        {
            "classe": classe,
            "quantidade": int(contagem_classes[classe]),
            "percentual": round(contagem_classes[classe] / total * 100, 1),
        }
        for classe in CLASSES
    ]

    classe_mais_comum = contagem_classes.idxmax()
    percentual_mais_comum = contagem_classes.max() / total * 100

    # 2. Conta em quantos jogos cada categoria aparece.
    contagem_categorias = {}
    for valor in df["Categories"]:
        for categoria in set(separar_categorias(valor)):
            contagem_categorias[categoria] = contagem_categorias.get(categoria, 0) + 1

    categorias = sorted(
        [
            {"categoria": categoria, "jogos": quantidade}
            for categoria, quantidade in contagem_categorias.items()
        ],
        key=lambda item: item["jogos"],
        reverse=True,
    )[:10]

    if categorias:
        categoria_mais_comum = categorias[0]["categoria"]
        quantidade_categoria_mais_comum = categorias[0]["jogos"]
    else:
        categoria_mais_comum = "Nenhuma"
        quantidade_categoria_mais_comum = 0

    # 3. Agrupa os jogos por faixa de preco e classe de popularidade.
    faixas_preco = ["Gratis", "Ate 10", "De 10 a 30", "Acima de 30"]
    df_grafico = df.copy()
    df_grafico["Faixa de preco"] = pd.cut(
        df_grafico["Price"],
        bins=[-0.001, 0, 10, 30, np.inf],
        labels=faixas_preco,
        include_lowest=True,
    )

    tabela_faixas = pd.crosstab(
        df_grafico["Faixa de preco"], df_grafico[COLUNA_ALVO]
    ).reindex(index=faixas_preco, columns=CLASSES, fill_value=0)

    # Grafico 1: histograma dos precos.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    limite_visual = df["Price"].quantile(0.99)
    precos = df.loc[df["Price"] <= limite_visual, "Price"]
    ax.hist(precos, bins=30, edgecolor="black")
    ax.set_title("Distribuicao dos precos dos jogos")
    ax.set_xlabel("Preco (unidade informada no CSV)")
    ax.set_ylabel("Quantidade de jogos")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pasta_static / "grafico_distribuicao.png", dpi=140)
    plt.close(fig)

    # Grafico 2: dez categorias que aparecem em mais jogos.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    categorias_plot = list(reversed(categorias))
    ax.barh(
        [item["categoria"] for item in categorias_plot],
        [item["jogos"] for item in categorias_plot],
    )
    ax.set_title("10 categorias mais presentes nos jogos")
    ax.set_xlabel("Quantidade de jogos")
    ax.set_ylabel("Categoria")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pasta_static / "grafico_comparacao.png", dpi=140)
    plt.close(fig)

    # Grafico 3: compara a proporcao de cada classe em cada faixa de preco.
    totais_faixas = tabela_faixas.sum(axis=1).replace(0, 1)
    tabela_percentual = tabela_faixas.div(totais_faixas, axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4.8))
    tabela_percentual.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Popularidade dos jogos por faixa de preco")
    ax.set_xlabel("Faixa de preco")
    ax.set_ylabel("Percentual de jogos na faixa (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(title="Popularidade", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pasta_static / "grafico_relacao.png", dpi=140)
    plt.close(fig)

    proporcao_media_alta = (
        tabela_faixas["Média"] + tabela_faixas["Alta"]
    ) / totais_faixas * 100
    faixa_maior_proporcao = proporcao_media_alta.idxmax()
    maior_proporcao = proporcao_media_alta.max()

    interpretacoes = {
        "popularidade": (
            f"A classe mais comum e {classe_mais_comum}, com "
            f"{percentual_mais_comum:.1f}% dos jogos analisados."
        ),
        "preco": (
            f"O histograma mostra como os precos estao distribuidos. "
            f"Os valores acima do percentil 99 foram deixados fora apenas deste grafico "
            f"para facilitar a visualizacao."
        ),
        "categorias": (
            f"A categoria mais comum e {categoria_mais_comum}, presente em "
            f"{quantidade_categoria_mais_comum:,} jogos. Um mesmo jogo pode ter varias categorias."
        ),
        "relacao": (
            "Cada barra representa 100% dos jogos daquela faixa de preco, divididos nas tres classes. "
            f"A maior proporcao de jogos com popularidade Media ou Alta aparece na faixa "
            f"'{faixa_maior_proporcao}' ({maior_proporcao:.1f}%). "
            "A comparacao mostra uma associacao, mas nao prova que o preco causa popularidade."
        ),
    }

    return {
        "distribuicao_classes": distribuicao_classes,
        "categorias": categorias,
        "interpretacoes": interpretacoes,
        "amostra": df[["Name", "Target", "Price", "Categories", COLUNA_ALVO]]
        .head(5)
        .to_dict(orient="records"),
    }