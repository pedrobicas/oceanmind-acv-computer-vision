# Relatório Técnico — Applied Computer Vision

## 1. Identificação

**Projeto:** OceanMind — Inteligência Oceânica e Previsão Climática via Dados Espaciais  
**Disciplina:** Applied Computer Vision  
**Curso:** Engenharia de Software — 4º ano  

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassão | 98607 |
| Pedro Bicas | 99534 |

## 2. Definição do problema

O OceanMind propõe uma solução de inteligência oceânica baseada em dados espaciais e ambientais. O módulo de visão computacional tem como objetivo classificar imagens oceânicas/satelitais em padrões visuais relacionados ao monitoramento ambiental.

As classes utilizadas no protótipo são:

- oceano normal;
- anomalia térmica;
- tempestade;
- nuvens;
- algas.

## 3. Conexão com a Indústria Espacial

A solução está conectada à Indústria Espacial porque utiliza o conceito de observação terrestre por satélites para monitoramento oceânico em escala ampla. O módulo simula a classificação de imagens provenientes de sensores orbitais ou fontes satelitais.

## 4. Dataset utilizado

O dataset foi gerado de forma sintética por meio do script `src/generate_synthetic_ocean_dataset.py`, com imagens RGB de 96x96 pixels. Cada classe possui imagens com padrões visuais característicos.

Quantidade sugerida:

- 5 classes;
- 220 imagens por classe;
- total de 1.100 imagens.

Divisão usada no treinamento:

- 70% treino;
- 15% validação;
- 15% teste.

## 5. Pré-processamento

As imagens são redimensionadas para 96x96 pixels e normalizadas dentro da própria arquitetura por meio da camada `Rescaling(1./255)`.

## 6. Arquiteturas treinadas

### CNN v1

Arquitetura baseline com camadas Conv2D, MaxPooling2D, Flatten, Dense e Dropout.

### CNN v2

Arquitetura mais robusta com blocos convolucionais adicionais, BatchNormalization, Dropout e GlobalAveragePooling2D.

Ambas foram criadas do zero, sem uso de modelos pré-treinados.

## 7. Avaliação

A avaliação deve apresentar:

- accuracy de treino e validação;
- loss de treino e validação;
- matriz de confusão;
- relatório de classificação;
- comparação entre as duas arquiteturas.

Inserir prints dos arquivos gerados em `reports/` após o treinamento.

## 8. Comparação técnica

A CNN v1 serve como referência inicial por possuir uma estrutura menor. A CNN v2 tende a apresentar maior capacidade de generalização por usar normalização, maior profundidade e regularização.

## 9. Demonstração funcional

A demonstração é realizada pelo app Streamlit `app.py`, no qual o usuário envia uma imagem e o modelo retorna a classe prevista e a confiança da predição.

## 10. Conclusão

O módulo ACV do OceanMind demonstra como redes neurais convolucionais treinadas do zero podem ser aplicadas à classificação de imagens oceânicas, contribuindo para a análise automatizada de padrões ambientais dentro de uma solução integrada de monitoramento espacial.
