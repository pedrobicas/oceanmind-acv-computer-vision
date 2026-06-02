# OceanMind - Applied Computer Vision

Projeto da Global Solution 2026 da FIAP, disciplina Applied Computer Vision.

## Contexto

O OceanMind e uma proposta de plataforma para inteligencia oceanica e previsao climatica apoiada por dados espaciais, climaticos e oceanicos. A parte de Visao Computacional classifica imagens oceanicas/satelitais em categorias ambientais uteis para monitoramento maritimo.

A conexao com a Industria Espacial esta no uso de imagens de observacao da Terra. Nesta versao, o projeto trabalha com imagens reais baixadas do NASA GIBS.

## Problema de visao computacional

O objetivo e classificar imagens em cinco categorias:

- `oceano_normal`
- `anomalia_termica`
- `tempestade`
- `nuvens`
- `algas`

As CNNs sao treinadas do zero. O projeto nao usa modelos pre-treinados, transfer learning ou pesos externos.

## Integrantes

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassao | 98607 |
| Pedro Bicas | 99534 |

## Estrutura

```text
oceanmind-acv-computer-vision/
|-- app.py
|-- requirements.txt
|-- src/
|   |-- download_nasa_gibs_dataset.py
|   |-- make_dataset_contact_sheet.py
|   |-- models.py
|   |-- train.py
|   |-- predict_image.py
|   `-- generate_final_report.py
|-- dataset_real/
|-- models/
|-- notebooks/
`-- reports/
```

## Dataset real NASA GIBS

O script `src/download_nasa_gibs_dataset.py` baixa imagens observacionais do NASA Global Imagery Browse Services (GIBS) via WMS. Ele usa combinacoes de camada, regiao e data para montar amostras aproximadas das classes.

Importante: esses rotulos sao fracos. Eles nao equivalem a uma anotacao manual perfeita. Por exemplo, uma regiao escolhida para `tempestade` pode conter apenas nuvens comuns em algumas datas. Por isso, o fluxo inclui uma folha de contato para revisao visual antes do treino.

Camadas usadas:

- True Color para `oceano_normal`, `nuvens` e `tempestade`.
- Chlorophyll-a para `algas`.
- Sea Surface Temperature para `anomalia_termica`.

## Como executar com imagens reais

Crie e ative a venv:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Baixe e organize o dataset real:

```bash
python src/download_nasa_gibs_dataset.py --output dataset_real --images-per-class 80 --size 160
```

O comando cria:

```text
dataset_real/
|-- train/
|-- validation/
|-- test/
`-- dataset_metadata.json
```

Na versao final da entrega, o dataset foi revisado e limpo. Foram removidos recortes quase vazios, imagens com faixas pretas grandes e amostras com continente dominante ou fora da categoria. A classe `oceano_normal` foi substituida por recortes NASA BlueMarble de oceano aberto, sem nuvens. O dataset final ficou com 208 imagens:

| Split | algas | anomalia_termica | nuvens | oceano_normal | tempestade |
|---|---:|---:|---:|---:|---:|
| Treino | 29 | 37 | 27 | 31 | 22 |
| Validacao | 6 | 8 | 3 | 7 | 6 |
| Teste | 6 | 9 | 5 | 7 | 5 |

Gere uma folha de contato para revisar as imagens:

```bash
python src/make_dataset_contact_sheet.py --dataset dataset_real --output reports/dataset_real_contact_sheet.jpg
```

Abra `reports/dataset_real_contact_sheet.jpg` e remova manualmente imagens claramente erradas das pastas `dataset_real/train`, `dataset_real/validation` ou `dataset_real/test` antes de treinar.

Treine as duas CNNs com o dataset real:

```bash
python src/train.py --dataset dataset_real --epochs 20 --image-size 160
```

Execute a demonstracao:

```bash
streamlit run app.py
```

## Compatibilidade com o comando antigo

O dataset principal da entrega e `dataset_real`, baixado do NASA GIBS e organizado em treino, validacao e teste.

## Arquiteturas CNN

A `CNN v1` e a baseline. Ela usa tres blocos `Conv2D` + `MaxPooling2D`, seguidos de `Flatten`, `Dense`, `Dropout` e camada final `softmax`.

A `CNN v2` e mais profunda e regularizada. Ela usa `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, `GlobalAveragePooling2D` e camadas densas finais.

Ambas sao criadas do zero em `src/models.py`.

## Resultados e artefatos

Na execucao atual com `dataset_real` limpo:

| Modelo | Test accuracy | Test loss |
|---|---:|---:|
| CNN v1 | 90.62% | 0.238404 |
| CNN v2 | 46.88% | 1.578710 |

O melhor modelo foi a CNN v1, superando 88% de acuracia no conjunto de teste. Mesmo assim, o resultado deve ser interpretado com cuidado porque o dataset real ainda usa rotulos fracos, gerados por camada/regiao/data, sem anotacao manual imagem a imagem. As classes mais ambiguas continuam sendo `nuvens` e `tempestade`.

Para melhorar o resultado, o proximo passo e aumentar o dataset, revisar manualmente as imagens baixadas, remover exemplos errados e treinar novamente.

Depois do treino, os arquivos principais ficam em:

- `models/oceanmind_cnn_v1.keras`
- `models/oceanmind_cnn_v2.keras`
- `models/class_names.json`
- `reports/training_history_cnn_v1.png`
- `reports/training_history_cnn_v2.png`
- `reports/confusion_matrix_cnn_v1.png`
- `reports/confusion_matrix_cnn_v2.png`
- `reports/classification_report_cnn_v1.json`
- `reports/classification_report_cnn_v2.json`
- `reports/model_comparison.json`
- `reports/training_run_metadata.json`

O `app.py` usa `models/class_names.json` para manter a ordem correta das classes do modelo treinado.

## Relatorio

Para gerar novamente o relatorio:

```bash
python src/generate_final_report.py
```

Saidas:

- `reports/relatorio_acv.pdf`
- `reports/relatorio_acv.docx`

Se voce treinar novamente com `dataset_real`, rode o gerador do relatorio depois do treino para atualizar os resultados.
