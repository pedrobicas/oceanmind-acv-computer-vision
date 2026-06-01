# OceanMind — Applied Computer Vision (ACV)

Módulo de Visão Computacional da Global Solution 2026 da FIAP.

## Tema

**OceanMind — Inteligência Oceânica e Previsão Climática via Dados Espaciais**

Este módulo classifica imagens oceânicas/satelitais em categorias ambientais relacionadas ao monitoramento de oceanos. A proposta simula uma etapa de análise visual da plataforma OceanMind, auxiliando a identificação de padrões como anomalia térmica, tempestade, nuvens, algas e oceano normal.

## Integrantes

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassão | 98607 |
| Pedro Bicas | 99534 |

## Classes

- `oceano_normal`
- `anomalia_termica`
- `tempestade`
- `nuvens`
- `algas`

## O que esta entrega contém

- Dataset sintético gerável por script.
- Duas arquiteturas de CNN criadas do zero.
- Treinamento com divisão treino/validação/teste.
- Gráficos de accuracy/loss.
- Matriz de confusão.
- Relatórios de classificação.
- App Streamlit para demonstração com upload de imagem.

## Estrutura

```text
acv-oceanmind/
├── app.py
├── requirements.txt
├── src/
│   ├── generate_synthetic_ocean_dataset.py
│   ├── models.py
│   ├── train.py
│   └── predict_image.py
├── notebooks/
│   └── oceanmind_acv_pipeline.ipynb
├── models/
├── reports/
└── dataset/
```

## Como executar

### 1. Criar ambiente

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Gerar dataset sintético

```bash
python src/generate_synthetic_ocean_dataset.py --output dataset --images-per-class 220 --size 96
```

O comando gera 1.100 imagens, com 220 imagens por classe.

### 4. Treinar as duas CNNs

```bash
python src/train.py --dataset dataset --epochs 12
```

Ao final, serão gerados:

- `models/oceanmind_cnn_v1.keras`
- `models/oceanmind_cnn_v2.keras`
- gráficos em `reports/`
- matrizes de confusão em `reports/`
- relatórios de classificação em `reports/`

### 5. Rodar demonstração

```bash
streamlit run app.py
```

## Justificativa técnica

A primeira arquitetura, CNN v1, é mais simples e serve como baseline. A segunda arquitetura, CNN v2, inclui mais camadas convolucionais, batch normalization, global average pooling e dropout, aumentando a capacidade de generalização e reduzindo risco de overfitting.

Nenhuma arquitetura utiliza modelos pré-treinados. As CNNs são construídas do zero com TensorFlow/Keras.

## Observação sobre o dataset

O dataset sintético foi criado para fins acadêmicos e demonstrativos. Para um produto real, o ideal seria substituir ou complementar essas imagens por imagens satelitais reais de fontes como NASA, NOAA, ESA, INPE ou Copernicus.

## Entrega esperada

- Repositório GitHub público.
- Notebook `.ipynb`.
- Scripts Python.
- Pesos do melhor modelo.
- Imagens de exemplo/teste.
- README.
- Aplicação de demonstração.
- Vídeo de até 3 minutos demonstrando a solução.
