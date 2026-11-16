"""Carrega o modelo salvo e faz previsoes a partir do formulario."""

import math
import joblib
from dados import BASE_DIR, preparar_X_formulario

CAMINHO_MODELO = BASE_DIR / "models" / "modelo_popularidade.joblib"
_pacote = None

def carregar_pacote():
    """Carrega o modelo apenas uma vez."""
    global _pacote

    if _pacote is None:
        if not CAMINHO_MODELO.exists():
            raise FileNotFoundError(
                "Modelo nao encontrado. Execute 'python treinar.py' primeiro."
            )
        _pacote = joblib.load(CAMINHO_MODELO)

    return _pacote

def obter_opcoes_formulario():
    """Retorna as categorias que o formulario pode oferecer."""
    pacote = carregar_pacote()
    return {"categories": pacote["opcoes_categories"]}

def prever_popularidade(entrada):
    """Valida os campos do formulario e retorna a classe prevista."""
    pacote = carregar_pacote()
    try:
        preco = float(entrada.get("Price", ""))
        quantidade_texto = int(entrada.get("Text", ""))
        quantidade_audio = int(entrada.get("Audio", ""))
    except (TypeError, ValueError):
        raise ValueError("Preencha o preco e as quantidades de idiomas com numeros validos.")
    if not math.isfinite(preco) or preco < 0:
        raise ValueError("O preco precisa ser um numero igual ou maior que zero.")
    if quantidade_texto < 0 or quantidade_audio < 0:
        raise ValueError("As quantidades de idiomas nao podem ser negativas.")

    mac = str(entrada.get("Mac", ""))
    linux = str(entrada.get("Linux", ""))
    if mac not in {"0", "1"} or linux not in {"0", "1"}:
        raise ValueError("Selecione Sim ou Nao para o suporte a Mac e Linux.")

    categorias = entrada.get("Categories", [])
    if isinstance(categorias, str):
        categorias = [categorias] if categorias else []

    opcoes_categories = pacote["opcoes_categories"]
    for categoria in categorias:
        if categoria not in opcoes_categories:
            raise ValueError("Selecione apenas categorias disponiveis na lista.")

    dados_jogo = {
        "Price": preco,
        "Text": quantidade_texto,
        "Audio": quantidade_audio,
        "Mac": int(mac),
        "Linux": int(linux),
        "Categories": categorias,
    }

    X = preparar_X_formulario(dados_jogo, opcoes_categories)
    X = X.reindex(columns=pacote["colunas_X"], fill_value=0)

    return str(pacote["modelo"].predict(X)[0])