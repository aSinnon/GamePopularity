"""Treina e compara dois modelos de classificacao."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dados import (
    ARQUIVO_TRATADO, BASE_DIR, CLASSES, COLUNA_ALVO,
    construir_X, calcular_opcoes_categories, ler_csv, preparar_dados,
)

TAMANHO_TESTE = 0.20
RANDOM_STATE = 42
CAMINHO_MODELO = BASE_DIR / "models" / "modelo_popularidade.joblib"

def main():
    # 1. Le e limpa os dados.
    print("Lendo e preparando os dados...")
    df_original = ler_csv()
    df, resumo = preparar_dados(df_original)

    print(f"Linhas antes da limpeza: {resumo['linhas_originais']:,}")
    print(f"Linhas depois da limpeza: {resumo['linhas_finais']:,}")
    print(f"Linhas removidas por duplicidade: {resumo['duplicadas_removidas']:,}")
    print(f"Linhas removidas por campos vazios: {resumo['vazias_removidas']:,}")
    print(f"Linhas removidas por Target 0 - 0: {resumo['target_zero_removidos']:,}")
    print(f"Linhas removidas por valores invalidos: {resumo['invalidas_removidas']:,}")
    print("\nDistribuicao das classes:")
    print(df[COLUNA_ALVO].value_counts().reindex(CLASSES, fill_value=0))

    # 2. Separa os dados em treino e teste.
    df_treino, df_teste = train_test_split(
        df,
        test_size=TAMANHO_TESTE,
        random_state=RANDOM_STATE,
        stratify=df[COLUNA_ALVO],
    )

    # 3. Aprende as categorias apenas com os dados de treino.
    opcoes_categories = calcular_opcoes_categories(df_treino)
    X_treino = construir_X(df_treino, opcoes_categories)
    X_teste = construir_X(df_teste, opcoes_categories)
    y_treino = df_treino[COLUNA_ALVO]
    y_teste = df_teste[COLUNA_ALVO]

    print(f"\nRegistros para treino: {len(X_treino):,}")
    print(f"Registros para teste: {len(X_teste):,}")
    print(f"Variaveis de entrada: {len(X_treino.columns)}")
    print("Distribuicao das classes no teste:")
    print(y_teste.value_counts().reindex(CLASSES, fill_value=0))

    # 4. Define os dois modelos que serao comparados.
    modelos = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    resultados = {}
    melhor_nome = None
    melhor_modelo = None
    melhor_f1 = -1

    # 5. Treina, testa e compara os modelos com os mesmos dados.
    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        accuracy = accuracy_score(y_teste, previsoes)
        f1_macro = f1_score(
            y_teste,
            previsoes,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
        matriz = confusion_matrix(y_teste, previsoes, labels=CLASSES)
        resultados[nome] = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "confusion_matrix": matriz.tolist(),
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print("Matriz de confusao (linhas = real; colunas = previsto):")
        print(pd.DataFrame(matriz, index=CLASSES, columns=CLASSES))
        print(classification_report(
            y_teste, previsoes,
            labels=CLASSES, zero_division=0,
        ))

        if f1_macro > melhor_f1:
            melhor_f1 = f1_macro
            melhor_nome = nome
            melhor_modelo = modelo

    # 6. Salva o melhor modelo e as categorias usadas pelo formulario.
    pacote = {
        "modelo": melhor_modelo,
        "nome_modelo": melhor_nome,
        "opcoes_categories": opcoes_categories,
        "classes": CLASSES,
        "metricas": resultados,
        "colunas_X": list(X_treino.columns),
        "base_treinamento": "static/dadostreinamento.csv",
        "tamanho_teste": TAMANHO_TESTE,
        "random_state": RANDOM_STATE,
    }

    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, CAMINHO_MODELO)

    print("\n[ RESULTADO FINAL ]\n")
    print("Melhor modelo pelo F1 Macro:", melhor_nome)
    print("Modelo salvo em:", CAMINHO_MODELO)
    print("CSV tratado salvo em:", ARQUIVO_TRATADO)

if __name__ == "__main__":
    main()