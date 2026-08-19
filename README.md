# Classificação de Pneumonia em Radiografias de Tórax com Deep Learning

Projeto acadêmico desenvolvido na disciplina **CCM-109 — Tópicos Especiais em Inteligência Artificial / Aprendizado Profundo em Prática**, da **Universidade Federal do ABC (UFABC)**, sob orientação do **Prof. Dr. Ronaldo Cristiano Prati**.

O projeto investiga a classificação binária de radiografias de tórax nas classes **NORMAL** e **PNEUMONIA**, utilizando o dataset público *Chest X-Ray Images (Pneumonia)* disponibilizado no Kaggle por Paul Mooney.

A abordagem experimental compara duas estratégias principais:

* uma **CNN simples treinada do zero**, utilizada como baseline;
* uma **ResNet50 com Transfer Learning**, utilizando pesos pré-treinados no ImageNet.

Além da avaliação quantitativa, foi utilizada a técnica **Grad-CAM** para investigar qualitativamente as regiões que contribuíram para as decisões dos modelos e discutir possíveis atalhos associados a características técnicas do dataset.

> **Aviso:** este é um projeto exclusivamente acadêmico e experimental. Os modelos desenvolvidos não foram validados para uso clínico e não devem ser utilizados para diagnóstico ou tomada de decisão médica.

## Objetivo

Avaliar e comparar duas abordagens de Deep Learning para classificação binária de radiografias de tórax em **NORMAL** e **PNEUMONIA**, utilizando um protocolo experimental comum e controles para reduzir risco de vazamento de dados.

Os objetivos específicos são:

- auditar o dataset quanto a duplicatas exatas e possíveis fontes de data leakage;
- construir uma divisão experimental reproduzível de treino, validação e teste;
- estabelecer uma CNN treinada do zero como baseline;
- avaliar uma ResNet50 com Transfer Learning sob o mesmo protocolo experimental;
- comparar os modelos por Accuracy, Precision, Recall, F1-score, ROC-AUC e matriz de confusão;
- investigar qualitativamente, com Grad-CAM, quais regiões das imagens influenciam as decisões dos modelos;
- discutir criticamente limitações, possíveis fatores de confusão e riscos de aprendizado de atalhos técnicos.

## Dataset e divisão experimental

Foi utilizado o dataset público **Chest X-Ray Images (Pneumonia)**, disponibilizado no Kaggle por **Paul Mooney** (`paultimothymooney/chest-xray-pneumonia`).

O conjunto original contém **5.856 imagens**:

* **NORMAL:** 1.583 imagens (27,03%);
* **PNEUMONIA:** 4.273 imagens (72,97%).

A divisão original do dataset não foi utilizada na modelagem, pois o conjunto de validação continha apenas 16 imagens. Antes da criação de uma nova divisão experimental, foi realizada uma auditoria com hashes **SHA-256**, que identificou **32 cópias excedentes** entre imagens duplicadas exatas.

Após a deduplicação, a base de modelagem passou a conter **5.824 imagens únicas**:

* **NORMAL:** 1.579;
* **PNEUMONIA:** 4.245.

A divisão experimental oficial foi criada com `StratifiedGroupKFold`, utilizando `n_splits=10`, `shuffle=True` e `random_state=42`. Os grupos utilizados foram inferidos a partir da nomenclatura dos arquivos e não devem ser interpretados como identificadores oficiais de pacientes.

| Conjunto  | Total | NORMAL | PNEUMONIA |
| --------- | ----: | -----: | --------: |
| Treino    | 4.682 |  1.259 |     3.423 |
| Validação |   561 |    154 |       407 |
| Teste     |   581 |    166 |       415 |

A auditoria final da divisão confirmou:

* **0 hashes SHA-256** compartilhados entre treino, validação e teste;
* **0 grupos candidatos** compartilhados entre os conjuntos;
* nenhuma nova divisão dos dados foi realizada durante os experimentos posteriores.

O dataset original **não é redistribuído neste repositório**. Para reproduzir os experimentos, ele deve ser obtido diretamente da fonte original no Kaggle.
## Metodologia e modelos

Todas as imagens foram padronizadas para **224 × 224 × 3**. O redimensionamento foi realizado diretamente para o formato quadrado, reconhecendo-se como limitação a possibilidade de deformação geométrica em imagens com diferentes razões de aspecto.

A codificação das classes utilizada em todos os experimentos foi:

* `NORMAL = 0`;
* `PNEUMONIA = 1`.

A classe **PNEUMONIA** foi considerada a classe positiva para Precision, Recall, F1-score e ROC-AUC.

### CNN baseline

A baseline consiste em uma CNN treinada do zero com quatro blocos convolucionais:

`Conv2D → MaxPooling2D`

com **32, 64, 128 e 256 filtros**, respectivamente, seguidos por:

`GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)`

A arquitetura possui **421.441 parâmetros treináveis**.

Para a CNN, os pixels foram convertidos para `float32` e normalizados para a faixa **0–1**.

### ResNet50 com Transfer Learning

O segundo modelo utiliza **ResNet50** com pesos pré-treinados no **ImageNet**, `include_top=False` e base convolucional integralmente congelada.

Sobre a base foram adicionadas as camadas:

`GlobalAveragePooling2D → Dropout(0.5) → Dense(1, sigmoid)`

O modelo possui **23.589.761 parâmetros**, dos quais apenas **2.049 são treináveis** nesta configuração.

Para a ResNet50 foi utilizado `tf.keras.applications.resnet50.preprocess_input`, em vez da normalização 0–1 utilizada pela CNN.

### Protocolo de treinamento

Os dois modelos utilizaram os mesmos conjuntos oficiais de treino, validação e teste, com:

* `batch_size = 32`;
* otimizador Adam;
* `learning_rate = 0.001`;
* Binary Cross-Entropy como função de perda;
* máximo de 30 épocas;
* Early Stopping monitorando `val_loss`, com `patience=5`;
* Model Checkpoint para preservação do melhor modelo.

O conjunto de teste não foi utilizado para selecionar épocas, arquiteturas ou hiperparâmetros.

Nesta comparação principal, não foram utilizados **Data Augmentation**, **class weights** ou **fine-tuning**, preservando um protocolo experimental controlado entre as duas abordagens.
## Resultados

Os dois modelos foram avaliados no mesmo conjunto de teste, composto por **581 imagens**, sem utilização desse conjunto durante o treinamento ou para seleção da melhor época.

| Métrica   | CNN baseline |   ResNet50 |
| --------- | -----------: | ---------: |
| Loss      |     0,127318 |   0,090117 |
| Accuracy  |       94,66% | **96,56%** |
| Precision |       95,28% | **97,36%** |
| Recall    |       97,35% | **97,83%** |
| F1-score  |       96,31% | **97,60%** |
| ROC-AUC   |       98,92% | **99,43%** |

### Matrizes de confusão

**CNN baseline**

|                | Pred. NORMAL | Pred. PNEUMONIA |
| -------------- | -----------: | --------------: |
| Real NORMAL    |          146 |              20 |
| Real PNEUMONIA |           11 |             404 |

Total de erros: **31**.

**ResNet50**

|                | Pred. NORMAL | Pred. PNEUMONIA |
| -------------- | -----------: | --------------: |
| Real NORMAL    |          155 |              11 |
| Real PNEUMONIA |            9 |             406 |

Total de erros: **20**.

### Comparação entre os modelos

Em relação à CNN baseline, a ResNet50 apresentou:

* ganho de **1,893 ponto percentual** em Accuracy;
* ganho de **2,079 pontos percentuais** em Precision;
* ganho de **0,482 ponto percentual** em Recall;
* ganho de **1,291 ponto percentual** em F1-score;
* ganho de **0,514 ponto percentual** em ROC-AUC;
* redução relativa de **35,48%** no total de erros;
* redução relativa de **45,00%** nos falsos positivos;
* redução relativa de **18,18%** nos falsos negativos.

A ResNet50 apresentou melhor desempenho neste protocolo experimental e neste dataset, mas esse resultado **não implica superioridade universal da arquitetura** em outros conjuntos de dados ou cenários.
## Grad-CAM e análise crítica

Para complementar a avaliação quantitativa, foi utilizada a técnica **Grad-CAM** para investigar qualitativamente quais regiões das radiografias contribuíram para as decisões da CNN baseline e da ResNet50.

Foram analisados sistematicamente:

* os **11 erros compartilhados** entre os dois modelos;
* **1 verdadeiro negativo** corretamente classificado por ambos;
* **1 verdadeiro positivo** corretamente classificado por ambos.

No total, foram inspecionadas **13 imagens e 26 mapas Grad-CAM**.

### CNN baseline

Nos 13 casos inspecionados:

* **13/13** mapas apresentaram foco predominantemente periférico;
* **11/13** apresentaram saliência associada a marcador ou texto;
* a saliência periférica também ocorreu nos controles corretamente classificados.

Portanto, não é adequado concluir que a CNN erra simplesmente porque utiliza marcadores ou regiões periféricas.

### ResNet50

Nos 13 casos inspecionados:

* **5/13** mapas foram classificados como predominantemente periféricos;
* **5/13** como predominantemente torácicos;
* **3/13** como mistos.

A ResNet50 apresentou padrões espaciais mais heterogêneos do que a CNN na amostra analisada.

### Interpretação

Os mapas são compatíveis com a hipótese de que características técnicas, enquadramento, bordas, marcadores ou outras regiões periféricas possam participar das representações aprendidas pelos modelos.

Entretanto:

* **Grad-CAM não demonstra causalidade**;
* não é possível afirmar que um marcador ou uma borda tenha causado uma classificação;
* saliência periférica também foi observada em classificações corretas;
* regiões torácicas destacadas não devem ser interpretadas como lesões;
* Grad-CAM não constitui segmentação médica;
* os percentuais acima descrevem apenas a amostra de 13 imagens inspecionadas e não devem ser generalizados para todo o conjunto de teste.

As quatro figuras comparativas utilizadas nesta análise estão disponíveis em `resultados/figuras/`.
## Limitações

Os resultados devem ser interpretados considerando as limitações do protocolo experimental e do próprio dataset:

* o conjunto apresenta **desbalanceamento entre NORMAL e PNEUMONIA**;
* foram observadas diferenças técnicas entre as classes em resolução, razão de aspecto, enquadramento e presença de artefatos;
* o redimensionamento direto para **224 × 224** pode introduzir deformação geométrica;
* os grupos utilizados para evitar vazamento foram **inferidos a partir da nomenclatura dos arquivos** e não correspondem a identificadores oficiais confirmados de pacientes;
* não foi realizada validação em um dataset externo;
* não foram realizados Data Augmentation, class weights ou fine-tuning na comparação principal;
* a análise Grad-CAM é qualitativa e foi realizada em uma amostra deliberadamente selecionada de **13 imagens**;
* o desempenho observado é específico deste dataset e deste protocolo experimental.

Os resultados, portanto, não devem ser interpretados como evidência de desempenho clínico ou de capacidade de generalização para outros contextos.

## Reprodutibilidade

Os notebooks preservados neste repositório correspondem às versões oficiais executadas e versionadas no Kaggle:

| Notebook                                        | Versão oficial          |
| ----------------------------------------------- | ----------------------- |
| `01_verificacao_dataset_pneumonia.ipynb`        | Version #4 — Successful |
| `02_analise_preparacao_imagens_pneumonia.ipynb` | Version #2 — Successful |
| `03_cnn_baseline_pneumonia.ipynb`               | Version #1 — Successful |
| `04_resnet50_transfer_learning_pneumonia.ipynb` | Version #3 — Successful |
| `05_gradcam_explicabilidade_pneumonia.ipynb`    | Version #2 — Successful |

As versões principais do ambiente utilizadas no projeto foram:

* Python **3.12.13**;
* TensorFlow **2.20.0**;
* pandas **2.3.3**;
* NumPy **2.0.2**;
* scikit-learn **1.6.1**;
* Pillow **11.3.0**;
* Matplotlib **3.10.0**.

As dependências Python também estão registradas no arquivo `requirements.txt`.

A CNN baseline utilizou `SEED = 42`, mas sua execução oficial não habilitou determinismo explícito das operações de GPU. Por esse motivo, uma execução interativa anterior apresentou pequena variação em relação à versão preservada no Kaggle. Para evitar seleção retrospectiva do melhor resultado, apenas a **Version #1 Successful** foi adotada como referência oficial.

Na ResNet50, além de `SEED = 42` e `tf.keras.utils.set_random_seed(42)`, foi utilizado `tf.config.experimental.enable_op_determinism()` antes da construção e do treinamento do modelo.

Mesmo com controle de aleatoriedade, reprodução numérica absolutamente idêntica não é garantida entre diferentes versões de software, hardware ou ambientes de execução.
## Estrutura do repositório

```text
deep-learning-pneumonia-ufabc/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_verificacao_dataset_pneumonia.ipynb
│   ├── 02_analise_preparacao_imagens_pneumonia.ipynb
│   ├── 03_cnn_baseline_pneumonia.ipynb
│   ├── 04_resnet50_transfer_learning_pneumonia.ipynb
│   └── 05_gradcam_explicabilidade_pneumonia.ipynb
│
├── resultados/
│   ├── csv/
│   └── figuras/
│
└── documentacao/
```

* `notebooks/`: contém os cinco notebooks oficiais do projeto;
* `resultados/csv/`: contém artefatos derivados utilizados para auditoria, avaliação e comparação dos experimentos;
* `resultados/figuras/`: contém as figuras finais selecionadas da análise Grad-CAM;
* `documentacao/`: reservada para documentação acadêmica do projeto;
* `requirements.txt`: registra as principais versões das bibliotecas utilizadas;
* `.gitignore`: impede o versionamento de datasets, modelos treinados e arquivos temporários.

## Como reproduzir no Kaggle

O ambiente principal utilizado no desenvolvimento foi o **Kaggle Notebook**.

### 1. Obter o dataset

Adicionar ao ambiente Kaggle o dataset:

**Chest X-Ray Images (Pneumonia)** — Paul Mooney

Identificador no Kaggle:

```text
paultimothymooney/chest-xray-pneumonia
```

O dataset original não está incluído neste repositório.

### 2. Executar os notebooks na ordem

A sequência experimental é:

1. `01_verificacao_dataset_pneumonia.ipynb`
   Auditoria do dataset, hashes SHA-256, deduplicação e geração da divisão experimental oficial.

2. `02_analise_preparacao_imagens_pneumonia.ipynb`
   Análise exploratória, caracterização técnica das imagens e validação do pipeline TensorFlow.

3. `03_cnn_baseline_pneumonia.ipynb`
   Treinamento e avaliação da CNN baseline.

4. `04_resnet50_transfer_learning_pneumonia.ipynb`
   Treinamento e avaliação da ResNet50 com Transfer Learning e comparação com a CNN.

5. `05_gradcam_explicabilidade_pneumonia.ipynb`
   Análise Grad-CAM dos modelos oficiais e investigação qualitativa dos erros e controles.

### 3. Dependências entre notebooks

Alguns notebooks utilizam artefatos produzidos por etapas anteriores, como:

* `base_modelagem_divisao_80_10_10.csv`;
* modelos `.keras` gerados pelos treinamentos;
* CSVs com previsões e comparações entre os modelos.

No Kaggle, esses artefatos foram reutilizados como **outputs versionados de notebooks anteriores**.

Ao reproduzir o projeto em outra conta ou ambiente Kaggle, pode ser necessário adicionar esses outputs como inputs e ajustar os caminhos de leitura para refletir o novo usuário ou identificador do notebook.

### 4. Hardware

A etapa exploratória foi executada sem acelerador. Os treinamentos oficiais da CNN e da ResNet50 e a etapa final de Grad-CAM utilizaram **GPU NVIDIA P100** no Kaggle.

Diferenças de hardware, versões de bibliotecas e operações numéricas podem produzir pequenas variações entre novas execuções e os resultados oficiais preservados neste repositório.
## Referências principais

* MOONEY, Paul. **Chest X-Ray Images (Pneumonia)**. Kaggle. Dataset: `paultimothymooney/chest-xray-pneumonia`.
* HE, Kaiming; ZHANG, Xiangyu; REN, Shaoqing; SUN, Jian. **Deep Residual Learning for Image Recognition**. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
* SELVARAJU, Ramprasaath R.; COGSWELL, Michael; DAS, Abhishek; VEDANTAM, Ramakrishna; PARIKH, Devi; BATRA, Dhruv. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization**. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.

## Autoria

**Monica Lourenço dos Anjos**

Projeto individual desenvolvido para a disciplina **CCM-109 — Tópicos Especiais em Inteligência Artificial / Aprendizado Profundo em Prática**, da **Universidade Federal do ABC (UFABC)**.

Professor: **Prof. Dr. Ronaldo Cristiano Prati**.

---

Este repositório reúne os notebooks oficiais, artefatos derivados, resultados e documentação necessários para registrar o protocolo experimental adotado e apoiar a reprodutibilidade do projeto.
