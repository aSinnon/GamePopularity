# Popularidade de Jogos da Steam

## 1. Sobre o projeto

Este projeto foi desenvolvido em Python com Flask e Pandas. Ele recebe um arquivo CSV com informações de jogos da Steam, faz a validação e a limpeza dos dados, apresenta uma análise com três gráficos e usa modelos de Machine Learning para prever a classe de popularidade de um jogo.

**Problema estudado:** Um jogo da Steam terá popularidade baixa, média ou alta?

A aplicação possui três páginas:

- **Início:** envio de um arquivo CSV.
- **Análise:** resumo da limpeza, amostra dos dados e três gráficos.
- **Previsão:** formulário que recebe as características de um jogo e apresenta a classe prevista pelo modelo treinado.

## 2. Dados utilizados

O CSV de exemplo usado no treinamento está em `static/dadostreinamento.csv`. A versão atual contém dez colunas obrigatórias:

- **Name:** Nome do jogo. É mantido nos dados e aparece na amostra da análise, mas não é usado como entrada do modelo.
- **Target:** Faixa estimada de proprietários do jogo (owners). É usada para estimar a quantidade de owners e criar a classe de popularidade.
- **Price:** Preço do jogo. É uma variável de entrada do modelo e também é usada nos gráficos.
- **Text:** Lista de idiomas disponíveis em texto. É convertida na quantidade de idiomas e usada pelo modelo.
- **Audio:** Lista de idiomas disponíveis em áudio. É convertida na quantidade de idiomas e usada pelo modelo.
- **Mac:** Indica se o jogo oferece suporte a Mac. É convertida para 1 (sim) ou 0 (não).
- **Linux:** Indica se o jogo oferece suporte a Linux. É convertida para 1 (sim) ou 0 (não).
- **Developers:** Desenvolvedores do jogo. A coluna é mantida no CSV, mas não é usada pelo modelo.
- **Publishers:** Publicadoras do jogo. A coluna é mantida no CSV, mas não é usada pelo modelo.
- **Categories:** Categorias associadas ao jogo. Representa as categorias/gêneros considerados no projeto e é usada pelo modelo.

Os nomes das colunas devem estar escritos exatamente como na tabela. O arquivo enviado precisa conter as dez colunas, mesmo que algumas delas não sejam entradas do modelo.

### Decisões sobre as variáveis

- **`Required age` não é usada:** os valores disponíveis nessa coluna foram considerados duvidosos para representar a idade mínima real do jogo.
- **`Genres` é representada por `Categories`:** no conjunto de dados utilizado, as informações consideradas como gênero foram consolidadas na coluna `Categories`.
- **`Developers` e `Publishers` foram retiradas das entradas do modelo:** elas continuam presentes no CSV e na limpeza, mas não participam da previsão. O cálculo de Top N dessas empresas foi removido por não ser necessário para o objetivo atual.
- **Plataformas:** são representadas pelas colunas `Mac` e `Linux` disponíveis no conjunto de dados.

## 3. Upload e validação do CSV

Na página inicial, o usuário pode enviar um arquivo `.csv` de até 100 MB. O sistema tenta identificar o separador do arquivo automaticamente e lê arquivos com codificação UTF-8; quando necessário, tenta também a codificação Windows-1252.

A validação verifica:

1. se um arquivo foi selecionado e se sua extensão é `.csv`;
2. se o arquivo não está vazio;
3. se o CSV pode ser lido;
4. se não existem nomes de colunas duplicados;
5. se as dez colunas obrigatórias estão presentes;
6. se `Price` possui valores numéricos e `Target` possui valores no formato de faixa, como `0 - 20000`;
7. se as colunas de idiomas, plataformas, desenvolvedores, publicadoras e categorias possuem valores.

Quando o arquivo não passa pela validação, a aplicação apresenta uma mensagem indicando o problema encontrado. A validação inicial verifica a estrutura e alguns valores básicos; as conversões e verificações linha a linha são feitas na etapa de limpeza.

**O CSV original não é alterado.** O arquivo enviado é guardado em `uploads/dataset_atual.csv`. A análise é feita em uma cópia tratada, salva em `data/dataset_analisado.csv`.

## 4. Leitura e limpeza com Pandas

A função de limpeza trabalha sobre uma cópia do DataFrame, preservando o arquivo original. As principais etapas são:

1. Retirar espaços extras dos nomes das colunas e dos textos.
2. Transformar textos vazios em valores ausentes.
3. Contar os idiomas listados em `Text` e `Audio`.
4. Converter os valores de `Mac` e `Linux` para 1 (sim) ou 0 (não).
5. Converter `Price` para número.
6. Interpretar `Target` como uma faixa de owners e calcular seu ponto médio para obter uma estimativa.
7. Remover linhas duplicadas, linhas com campos obrigatórios ausentes e registros com `Target` igual a `0 - 0`.
8. Remover registros em que as conversões necessárias falharam ou que possuem valores negativos em preço, quantidade de idiomas ou owners estimados.
9. Criar a coluna `popularidade` a partir da estimativa de owners.

**Não é feito preenchimento automático de valores ausentes.** Quando falta uma informação obrigatória ou uma conversão necessária não pode ser realizada, a linha é removida. Essa decisão evita inventar valores para os jogos.

### Resultado da limpeza do CSV de treinamento

Na execução realizada com o arquivo `static/dadostreinamento.csv`, foram obtidos estes números:

- **Linhas antes da limpeza:** 125.855
- **Linhas depois da limpeza:** 100.570
- **Linhas removidas por duplicidade:** 82
- **Linhas removidas por campos vazios:** 9.098
- **Linhas removidas por `Target` igual a `0 - 0`:** 16.105
- **Linhas removidas por valores inválidos após as conversões:** 0

Esses números correspondem ao CSV de exemplo incluído nesta versão. Ao enviar outro CSV, os totais podem ser diferentes. A página de análise mostra o tamanho da base antes e depois da limpeza, as quantidades removidas e os valores ausentes encontrados nas colunas obrigatórias.

## 5. Perguntas analisadas e gráficos

A análise usa estatísticas e agrupamentos feitos com Pandas para responder a três perguntas relacionadas à popularidade dos jogos:

- **Como os preços dos jogos estão distribuídos?** O Histograma dos preços mostra onde os preços se concentram e como se espalham. Para facilitar a leitura, o gráfico não exibe os valores acima do percentil 99; esses registros continuam na base analisada.
- **Quais categorias aparecem em mais jogos?** O Gráfico de barras com as dez categorias mais frequentes permite comparar quais categorias estão presentes em mais jogos. Um mesmo jogo pode possuir várias categorias.
- **Como a popularidade se distribui entre diferentes faixas de preço?** As Barras empilhadas com o percentual de jogos de popularidade baixa, média e alta em cada faixa: grátis, até 10, de 10 a 30 e acima de 30. Cada barra representa 100% dos jogos daquela faixa de preço. O gráfico permite comparar a composição das classes entre as faixas. Ele mostra uma associação, mas não prova que o preço causa popularidade.

A página também mostra uma amostra de cinco registros tratados e a quantidade e o percentual de jogos em cada classe de popularidade. Todos os gráficos possuem título, eixos identificados e um texto de interpretação na página.

## 6. Classificação da popularidade: variável y

A coluna `Target` apresenta uma faixa estimada de owners. Para criar a classe que o modelo tentará prever, o projeto calcula o ponto médio dessa faixa e usa a estimativa resultante para classificar cada jogo:

- **Baixa:** Menos de 50.000
- **Média:** De 50.000 até menos de 500.000
- **Alta:** 500.000 ou mais

A variável **y** é a coluna `popularidade`, com os valores `Baixa`, `Média` ou `Alta`. Essas classes representam, respectivamente, as classes baixa (-1), média (0) e alta (1) usadas como exemplo no enunciado. No código, os nomes são armazenados como texto para facilitar a leitura dos resultados e da previsão.

A classificação é uma estimativa: o ponto médio de uma faixa de owners não é necessariamente a quantidade real de proprietários de um jogo.

## 7. Variáveis de entrada: X

O modelo recebe as seguintes informações:

- `Price`: preço do jogo;
- `Text`: quantidade de idiomas disponíveis em texto;
- `Audio`: quantidade de idiomas disponíveis em áudio;
- `Mac`: suporte para Mac, convertido para 0 ou 1;
- `Linux`: suporte para Linux, convertido para 0 ou 1;
- `Categories`: categorias do jogo, transformadas em colunas numéricas com valores 0 ou 1.

Cada categoria conhecida no conjunto de treinamento recebe uma coluna própria. O valor 1 indica que o jogo possui aquela categoria; 0 indica que não possui. No CSV de treinamento incluído nesta versão, são criadas 33 colunas de categoria, além das cinco variáveis numéricas, totalizando 38 entradas para o modelo.

`Name`, `Target`, `Developers` e `Publishers` não são entradas de X. `Target` é usado somente para criar y; `Name` é identificador; `Developers` e `Publishers` foram excluídos da previsão para manter o modelo mais direto.

## 8. Treinamento, teste e prevenção de vazamento

O treinamento fica separado da aplicação Flask, no arquivo `treinar.py`. A divisão é feita com `train_test_split`:

- **80%** dos registros para treinamento;
- **20%** dos registros para teste;
- `random_state=42`, para permitir repetir a mesma divisão;
- `stratify` pela classe de popularidade, para manter uma proporção semelhante das classes nos dois grupos.

Na execução com o CSV de exemplo, foram utilizados:

- **80.456 registros** para treinamento;
- **20.114 registros** para teste.

As categorias disponíveis para criar as colunas de X são identificadas somente nos dados de treinamento. Depois, a mesma lista de categorias é usada para montar X de treinamento e X de teste. Assim, o conjunto de teste não é usado para descobrir previamente as categorias do modelo.

As conversões de preço, idiomas, plataformas e `Target` são regras definidas para os campos do projeto. O código não calcula médias ou outros valores estatísticos com o conjunto completo para preencher dados ausentes: as linhas incompletas ou inválidas são removidas.

## 9. Modelos comparados

A versão final compara dois algoritmos de classificação:

### Decision Tree (Árvore de Decisão)

É um modelo que aprende regras divididas em condições. Por exemplo, pode separar jogos conforme o preço, o suporte às plataformas ou a presença de determinadas categorias. O resultado percorre as divisões da árvore até chegar a uma classe.

### Random Forest (Floresta Aleatória)

É um conjunto de várias árvores de decisão. Cada árvore realiza uma previsão e o conjunto combina os resultados para chegar à classe final. A ideia é não depender de uma única árvore.

Os dois modelos são treinados e avaliados com a mesma divisão de treino e teste. O código utiliza `class_weight="balanced"` para dar mais peso às classes que aparecem menos vezes. A profundidade das árvores também é limitada para controlar o tamanho e a complexidade dos modelos.

**Observação sobre o enunciado:** o texto inicial do trabalho também cita Logistic Regression. Por decisão do grupo, essa técnica foi retirada da versão final. A comparação implementada contém Decision Tree e Random Forest, ou seja, dois modelos diferentes, conforme o critério de avaliação que pede a comparação de dois ou mais modelos.

## 10. Avaliação dos modelos e escolha final

O script apresenta as seguintes métricas:

- **Accuracy:** proporção de previsões corretas em relação ao total de registros do teste.
- **F1 Macro:** calcula o F1 de cada classe e faz a média dando o mesmo peso às três classes. É útil neste projeto porque a classe Baixa aparece muito mais que as classes Média e Alta.
- **Matriz de confusão:** compara a classe real com a classe prevista. As linhas representam as classes reais e as colunas representam as classes previstas.
- **Relatório de classificação:** apresenta precisão, recall e F1 para cada classe.

### Resultados obtidos com o CSV de exemplo

- Decision Tree: 62,19% | 0,3927
- Random Forest: **71,82%** | **0,4534**

### Matrizes de confusão

**Decision Tree** — linhas = classe real; colunas = classe prevista:

Real Baixa: 11.410 - 3.921 - 1.980
Real Média: 857    - 759   - 692
Real Alta : 68     - 87    - 340
Previsão Baixa: 11.410 - 857 - 68
Previsão Média: 3.921  - 759 - 87
Previsão Alta : 1.980  - 692 - 340

**Random Forest** — linhas = classe real; colunas = classe prevista:

Real Baixa: 13.341 - 2.966 - 1.004
Real Média: 1.041  - 815   - 452
Real Alta : 95     - 111   - 289
Previsão Baixa: 13.341 - 1.041 - 95
Previsão Média: 2.966  - 815   - 111
Previsão Alta : 1.004  - 452   - 289

### Interpretação e escolha

O Random Forest obteve o maior F1 Macro (0,4534) e a maior Accuracy (71,82%) nesta execução. Por isso, ele é escolhido para a aplicação final e salvo no arquivo `models/modelo_popularidade.joblib`.

A base é desequilibrada: no conjunto de teste havia 17.311 jogos Baixa, 2.308 Média e 495 Alta. Isso significa que a classe Baixa representa a maior parte dos registros. A Accuracy sozinha poderia esconder dificuldades em reconhecer as classes menores; por isso, o grupo também considera o F1 Macro e a matriz de confusão.

O Random Forest acertou 13.341 dos 17.311 jogos de classe Baixa, 815 dos 2.308 jogos de classe Média e 289 dos 495 jogos de classe Alta. Ainda há erros, principalmente entre as classes Média e Alta. Por exemplo, alguns jogos realmente Médios foram previstos como Baixa ou Alta. Esses erros importam porque podem fazer um usuário acreditar que um jogo tem uma popularidade diferente da estimada.

O resultado não significa que o modelo prevê a popularidade com certeza. O F1 Macro de 0,4534 mostra que ainda há espaço para melhorar, especialmente nas classes com menos exemplos. A escolha do Random Forest significa apenas que ele apresentou o melhor equilíbrio entre as classes entre os dois modelos testados nesta base e nesta divisão.

## 11. Formulário e previsão na aplicação

A página **Previsão** possui campos para preço, quantidade de idiomas em texto, quantidade de idiomas em áudio, suporte para Mac, suporte para Linux e categorias.

Ao enviar o formulário, o arquivo `modelo.py`:

1. verifica se os campos numéricos são válidos e não negativos;
2. verifica se Mac e Linux foram selecionados como Sim ou Não;
3. confere se as categorias escolhidas existem na lista aprendida durante o treinamento;
4. monta uma linha com as mesmas variáveis e colunas usadas no treinamento;
5. chama `predict()` no modelo salvo;
6. apresenta `Baixa`, `Média` ou `Alta` no navegador.

Entradas inválidas geram mensagens de erro. O modelo é carregado do arquivo `.joblib` e não é treinado novamente ao preencher o formulário.

## 12. Relação entre o CSV enviado e o modelo

O upload de um CSV atualiza a análise e os três gráficos. Ele **não treina novamente** o modelo.

O modelo é treinado com o CSV fixo `static/dadostreinamento.csv`. Para atualizar o modelo com uma nova versão dessa base, é necessário substituir o CSV de treinamento e executar novamente:

```bash
python treinar.py
```

Depois disso, a aplicação carregará o novo modelo salvo em `models/modelo_popularidade.joblib`. Assim, a base usada no treinamento fica identificada no pacote salvo junto com o modelo.

## 13. Organização dos arquivos

```text
ProjetoSteamSimplificado/
│
├── app.py                       # Rotas Flask, upload e integração das páginas
├── dados.py                     # Leitura, limpeza, análise e gráficos
├── modelo.py                    # Carregamento do modelo e previsão do formulário
├── treinar.py                   # Divisão dos dados, treino e avaliação dos modelos
├── requirements.txt             # Bibliotecas necessárias
├── README.md                    # Documentação do projeto
│
├── templates/
│   ├── shared_layout.html       # Estrutura comum das páginas
│   ├── index.html               # Upload do CSV
│   ├── analise.html             # Resumo da análise e gráficos
│   └── previsao.html            # Formulário e resultado da previsão
│
├── static/
│   └── dadostreinamento.csv     # CSV de exemplo usado no treinamento
                                 # Também recebe os três gráficos gerados pela análise
│
├── uploads/                     # CSV original enviado pelo usuário
├── data/                        # Cópias tratadas dos dados
└── models/                      # Modelo treinado (.joblib)
```

As pastas `uploads/`, `data/` e `models/` podem ser criadas automaticamente quando necessário. Arquivos gerados, como o modelo treinado e as cópias tratadas, podem não estar no repositório ou no ZIP inicial; eles são criados ao executar o projeto.

## 14. Instalação e execução

É necessário ter Python instalado.

1. Abra o terminal na pasta `ProjetoSteamSimplificado`.
2. Instale as bibliotecas:

   ```bash
   pip install -r requirements.txt
   ```

3. Treine e compare os modelos. Esse passo cria o modelo que será usado no formulário:

   ```bash
   python treinar.py
   ```

4. Inicie a aplicação Flask:

   ```bash
   python app.py
   ```

5. Abra o endereço abaixo no navegador:

   ```text
   http://127.0.0.1:5000
   ```

6. Envie um CSV para atualizar a análise ou abra a página de previsão para testar o modelo treinado.

Para treinar novamente, execute `python treinar.py` com o CSV de treinamento atualizado. Para iniciar somente a aplicação, depois de já ter um modelo salvo, execute `python app.py`.

## 15. Rotas da aplicação

- `/`         GET  : Mostra a página inicial e o formulário de upload.
- `/upload`   POST : Recebe e valida o arquivo CSV.
- `/analise`  GET  : Limpa uma cópia do CSV enviado, atualiza os gráficos e mostra os resultados.
- `/previsao` GET  : Mostra o formulário de previsão.
- `/previsao` POST : Valida os campos e apresenta a previsão do modelo.

## 16. Demonstração e defesa do projeto

Na apresentação, a demonstração pode seguir esta ordem:

1. Explicar o problema: classificar a popularidade de jogos da Steam.
2. Mostrar as dez colunas exigidas no CSV e enviar o arquivo pela página inicial.
3. Mostrar uma mensagem de validação e explicar que o original é preservado.
4. Abrir a análise, mostrar o tamanho da base antes e depois da limpeza e explicar por que algumas linhas são removidas.
5. Apresentar os três gráficos e responder às três perguntas da análise.
6. Executar `treinar.py` e explicar a divisão 80/20, as variáveis X e y e a comparação dos dois modelos.
7. Mostrar Accuracy, F1 Macro e a matriz de confusão, explicando ao menos um exemplo de erro.
8. Abrir o formulário de previsão, preencher os campos e mostrar a classe prevista.

Na defesa individual, cada integrante deve conseguir explicar, com suas próprias palavras:

- o que cada arquivo principal faz;
- quais dados entram em X e como y é criado;
- como funciona a divisão treino/teste;
- a diferença entre Decision Tree e Random Forest;
- o significado de Accuracy, F1 Macro e matriz de confusão;
- por que o Random Forest foi escolhido e quais são as limitações do resultado.

## 17. Participação dos integrantes

Nosso grupo participou da criação do projeto do início ao fim de acordo com suas horas livres, ou seja, todos participaram (pelo menos em parte) de todo o desenvolvimento da aplicação.

## 18. Fontes e ferramentas

Referências indicadas pela equipe:

- https://github.com/RenaL27/steam-popularity-analysis
- https://github.com/Hamza-hi80/steam-popularity-prediction
- OpenAI ChatGPT — ferramenta de apoio utilizada durante o desenvolvimento para discutir, explicar, revisar e organizar partes do projeto, assim como responsável por realizar o README.md.

As referências devem ser mantidas e complementadas conforme o uso efetivo. Caso algum trecho de código, função ou solução tenha sido copiado ou adaptado de uma fonte específica, indicar a fonte e a parte correspondente. Também devem ser citadas outras ferramentas ou materiais utilizados no desenvolvimento.

## 19. Limitações conhecidas

- A coluna `Target` contém faixas estimadas de owners; o ponto médio é uma aproximação, não um valor real exato.
- A base possui muito mais jogos de popularidade Baixa do que Média ou Alta, o que dificulta a classificação das classes menores.
- O preço e a popularidade podem apresentar associação nos dados, mas isso não demonstra que um cause o outro.
- O formulário prevê a classe com base nas variáveis disponíveis e no padrão aprendido no CSV de treinamento; o resultado não é uma garantia da popularidade futura do jogo.
- O modelo permanece fixo quando um usuário envia um CSV para análise. É necessário executar `treinar.py` separadamente para treinar uma nova versão.