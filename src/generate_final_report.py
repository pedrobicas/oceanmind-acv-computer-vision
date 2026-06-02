"""Gera o relatorio final em PDF e DOCX sem dependencias extras."""
from __future__ import annotations

import json
import textwrap
import zipfile
from datetime import date
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PDF_PATH = REPORTS / "relatorio_acv_final.pdf"
DOCX_PATH = REPORTS / "relatorio_acv_final.docx"

TITLE = "OceanMind - Visao Computacional Aplicada ao Monitoramento Oceanico via Dados Espaciais"
SUBTITLE = "FIAP - Global Solution 2026\nApplied Computer Vision"
STUDENTS = [
    ("Bryan Willian", "551305"),
    ("Felipe Terra", "99405"),
    ("Gabriel Doms", "98630"),
    ("Lucas Vassao", "98607"),
    ("Pedro Bicas", "99534"),
]
FIGURES = [
    ("Amostras do dataset real", REPORTS / "dataset_real_contact_sheet.jpg"),
    ("Historico de treinamento - CNN v1", REPORTS / "training_history_cnn_v1.png"),
    ("Matriz de confusao - CNN v1", REPORTS / "confusion_matrix_cnn_v1.png"),
    ("Historico de treinamento - CNN v2", REPORTS / "training_history_cnn_v2.png"),
    ("Matriz de confusao - CNN v2", REPORTS / "confusion_matrix_cnn_v2.png"),
]


def load_results() -> dict:
    path = REPORTS / "model_comparison.json"
    if not path.exists():
        return {"classes": [], "results": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_sections(results: dict) -> list[tuple[str, list[str]]]:
    run_metadata = load_json(REPORTS / "training_run_metadata.json")
    dataset_metadata = load_json(REPORTS / "dataset_metadata.json")
    classes = results.get("classes") or ["algas", "anomalia_termica", "nuvens", "oceano_normal", "tempestade"]
    dataset_name = results.get("dataset") or run_metadata.get("dataset") or "dataset"
    dataset_type = dataset_metadata.get("dataset_type", "real_nasa_gibs_weak_labels")
    cnn_v1 = results.get("results", {}).get("cnn_v1", {})
    cnn_v2 = results.get("results", {}).get("cnn_v2", {})
    v1_acc = cnn_v1.get("test_accuracy")
    v2_acc = cnn_v2.get("test_accuracy")
    v1_loss = cnn_v1.get("test_loss")
    v2_loss = cnn_v2.get("test_loss")
    best_name = "CNN v1" if (v1_acc or 0) >= (v2_acc or 0) else "CNN v2"
    best_acc = max(v1_acc or 0, v2_acc or 0)

    def pct(value):
        return "nao disponivel" if value is None else f"{value * 100:.2f}%"

    def num(value):
        return "nao disponivel" if value is None else f"{value:.6f}"

    if best_acc >= 0.88:
        performance_note = (
            f"O criterio de 88% foi atendido pelo melhor modelo ({best_name}). Ainda assim, o resultado deve ser interpretado com cautela, "
            "porque parte dos rotulos foi definida por camada, regiao e data, e nao por anotacao manual imagem a imagem."
        )
    else:
        performance_note = (
            "A acuracia minima de 88% nao foi atingida com o dataset real. A principal razao tecnica e a qualidade dos rotulos: "
            "as imagens foram coletadas por camada, regiao e data, sem anotacao manual imagem a imagem."
        )

    return [
        (
            "Introducao",
            [
                "O OceanMind e uma proposta de plataforma para inteligencia oceanica e previsao climatica baseada em dados espaciais, climaticos e oceanicos. Neste modulo da disciplina Applied Computer Vision, o foco esta na classificacao de imagens oceanicas/satelitais para apoiar a leitura automatizada de padroes ambientais.",
                "A conexao com a Industria Espacial ocorre pelo uso conceitual de observacao da Terra: imagens produzidas por sensores orbitais podem ser usadas para monitorar areas maritimas, regioes costeiras, formacoes de nuvens, tempestades, concentracoes de algas e possiveis anomalias termicas.",
            ],
        ),
        (
            "Definicao do problema",
            [
                "O problema de visao computacional foi definido como uma tarefa de classificacao multiclasse. A entrada do modelo e uma imagem RGB de uma cena oceanica/satelital e a saida e uma das categorias ambientais usadas pelo projeto.",
                "As classes do dataset sao: " + ", ".join(classes) + ". Essa classificacao representa uma etapa inicial de triagem visual, nao uma previsao climatica completa.",
            ],
        ),
        (
            "Dataset",
            [
                f"O dataset usado na ultima execucao registrada foi {dataset_name}. Tipo registrado: {dataset_type}.",
                "Quando usado o fluxo real, as imagens sao baixadas do NASA GIBS via WMS. As classes sao montadas por camada, regiao e data: true color para oceano, nuvens e tempestade; clorofila-a para algas; temperatura da superficie do mar para anomalia_termica.",
                "Esses rotulos reais sao aproximados e precisam de revisao visual, pois a origem via regiao/camada nao garante anotacao perfeita de cada imagem. O projeto inclui uma folha de contato para apoiar essa revisao.",
                "A entrega final foi mantida com o dataset real para evitar mistura com dados artificiais.",
            ],
        ),
        (
            "Pre-processamento",
            [
                "As imagens sao organizadas em subpastas por classe, o que permite o uso de image_dataset_from_directory. O script de download real ja cria pastas train, validation e test.",
                "Durante o carregamento, todas sao redimensionadas para o tamanho informado no treinamento. Na execucao final, foi usado 160x160 pixels.",
                "A normalizacao dos pixels e feita dentro das arquiteturas por meio da camada Rescaling(1./255). Assim, as imagens entram no modelo em escala padronizada sem alterar os arquivos originais do dataset.",
            ],
        ),
        (
            "Arquiteturas CNN",
            [
                "As duas redes foram criadas do zero com TensorFlow/Keras. Nenhuma usa transfer learning, pesos externos ou arquitetura pre-treinada.",
                "A CNN v1 funciona como baseline. Ela usa tres blocos Conv2D e MaxPooling2D, seguidos de Flatten, Dense, Dropout e camada final Dense com softmax.",
                "A CNN v2 e mais profunda e regularizada. Ela usa blocos com Conv2D, BatchNormalization, MaxPooling2D e Dropout, alem de GlobalAveragePooling2D antes das camadas densas finais. A principal diferenca esta na maior regularizacao e no uso de pooling global em vez de Flatten.",
            ],
        ),
        (
            "Treinamento",
            [
                "O treinamento foi configurado para ate 12 epocas, otimizador Adam, funcao de perda categorical_crossentropy e metrica accuracy.",
                "Foram usados EarlyStopping com restauracao dos melhores pesos e ReduceLROnPlateau para reduzir a taxa de aprendizado quando a loss de validacao parasse de melhorar.",
            ],
        ),
        (
            "Resultados",
            [
                f"A CNN v1 obteve test accuracy de {pct(v1_acc)} e test loss de {num(v1_loss)}. A CNN v2 obteve test accuracy de {pct(v2_acc)} e test loss de {num(v2_loss)}.",
                "Os graficos de accuracy/loss, as matrizes de confusao e os relatórios de classificacao foram salvos automaticamente na pasta reports.",
                "O classification report salvo detalha precision, recall e f1-score por classe. No dataset real atual, o desempenho ficou melhor em algas e anomalia_termica, mas houve confusao relevante entre nuvens, oceano_normal e tempestade.",
            ],
        ),
        (
            "Comparacao entre modelos",
            [
                f"Na execucao atual, o melhor modelo foi {best_name}, com acuracia de teste de {pct(best_acc)}.",
                performance_note,
                "Tambem ha limitacoes de escala: o dataset final ficou com 208 imagens apos limpeza, o que ainda e pouco para imagens satelitais reais. A variacao de sensor, data, cobertura de nuvens, paleta de camada cientifica e composicao visual torna o problema dificil.",
                "Para melhorar, o caminho mais consistente e ampliar o numero de imagens, revisar manualmente amostras incorretas, balancear melhor regioes e datas, usar data augmentation e treinar por mais epocas apos a limpeza dos rotulos.",
            ],
        ),
        (
            "Demonstracao funcional",
            [
                "A demonstracao foi implementada em Streamlit no arquivo app.py. O comando de execucao e: streamlit run app.py.",
                "O app permite selecionar a CNN treinada, enviar uma imagem e visualizar a classe prevista, a confianca da predicao e as probabilidades por classe.",
                "[INSERIR PRINT DO APP STREAMLIT AQUI, SE O PROFESSOR EXIGIR EVIDENCIA VISUAL DA EXECUCAO]",
            ],
        ),
        (
            "Conclusao",
            [
                "A entrega demonstra um pipeline completo de classificacao de imagens aplicado ao contexto espacial/oceanico: geracao de dataset, pre-processamento, treino de duas CNNs do zero, avaliacao quantitativa, comparacao entre modelos e demonstracao funcional.",
                "O projeto atende ao objetivo academico da disciplina. Como continuidade, o principal passo tecnico seria ampliar o dataset real e revisar manualmente os rotulos para aumentar a robustez em condicoes menos controladas.",
            ],
        ),
    ]


def add_wrapped_text(draw, text, xy, font, fill, max_width, line_gap=8):
    x, y = xy
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def new_page():
    return Image.new("RGB", (1240, 1754), "white")


def render_pdf(sections: list[tuple[str, list[str]]]):
    REPORTS.mkdir(exist_ok=True)
    title_font = load_font(42, bold=True)
    h_font = load_font(28, bold=True)
    body_font = load_font(22)
    small_font = load_font(18)
    pages = []

    page = new_page()
    draw = ImageDraw.Draw(page)
    y = 210
    y = add_wrapped_text(draw, TITLE, (110, y), title_font, "#14324a", 1020, line_gap=12)
    y += 45
    for line in SUBTITLE.splitlines():
        draw.text((110, y), line, font=h_font, fill="#1d1d1d")
        y += 42
    y += 60
    draw.text((110, y), "Integrantes", font=h_font, fill="#1d1d1d")
    y += 48
    for name, rm in STUDENTS:
        draw.text((130, y), f"{name} - RM {rm}", font=body_font, fill="#1d1d1d")
        y += 34
    draw.text((110, 1580), f"Gerado em {date.today().isoformat()}", font=small_font, fill="#555555")
    pages.append(page)

    page = new_page()
    draw = ImageDraw.Draw(page)
    y = 90
    for heading, paragraphs in sections:
        if y > 1460:
            pages.append(page)
            page = new_page()
            draw = ImageDraw.Draw(page)
            y = 90
        draw.text((90, y), heading, font=h_font, fill="#14324a")
        y += 46
        for paragraph in paragraphs:
            y = add_wrapped_text(draw, paragraph, (110, y), body_font, "#1d1d1d", 1020)
            y += 20
            if y > 1510:
                pages.append(page)
                page = new_page()
                draw = ImageDraw.Draw(page)
                y = 90
        y += 20
    pages.append(page)

    for caption, path in FIGURES:
        page = new_page()
        draw = ImageDraw.Draw(page)
        draw.text((90, 80), caption, font=h_font, fill="#14324a")
        if path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((1030, 1280))
            x = (1240 - img.width) // 2
            page.paste(img, (x, 170))
            draw.text((90, 1515), str(path.relative_to(ROOT)), font=small_font, fill="#555555")
        else:
            draw.text((110, 220), f"[INSERIR IMAGEM MANUALMENTE: {path.name}]", font=body_font, fill="#1d1d1d")
        pages.append(page)

    first, rest = pages[0], pages[1:]
    first.save(PDF_PATH, "PDF", resolution=150.0, save_all=True, append_images=rest)


def paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"


def image_paragraph(rel_id: str, name: str, width_px: int, height_px: int) -> str:
    max_width_emu = 5486400
    ratio = height_px / max(width_px, 1)
    cx = max_width_emu
    cy = int(max_width_emu * ratio)
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="1" name="{escape(name)}"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr><pic:cNvPr id="0" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rel_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def render_docx(sections: list[tuple[str, list[str]]]):
    rels = []
    body = [
        paragraph(TITLE, "Title"),
        paragraph("FIAP - Global Solution 2026", "Subtitle"),
        paragraph("Applied Computer Vision", "Subtitle"),
        paragraph("Integrantes", "Heading1"),
    ]
    for name, rm in STUDENTS:
        body.append(paragraph(f"{name} - RM {rm}"))

    for heading, paragraphs in sections:
        body.append(paragraph(heading, "Heading1"))
        for text in paragraphs:
            for part in textwrap.wrap(text, 500):
                body.append(paragraph(part))

    media_files = []
    for idx, (caption, path) in enumerate(FIGURES, start=1):
        body.append(paragraph(caption, "Heading1"))
        if path.exists():
            rel_id = f"rId{idx + 1}"
            media_name = f"image{idx}.png"
            with Image.open(path) as img:
                width_px, height_px = img.size
            rels.append(
                f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
            )
            body.append(image_paragraph(rel_id, media_name, width_px, height_px))
            media_files.append((media_name, path))
        else:
            body.append(paragraph(f"[INSERIR IMAGEM MANUALMENTE: {path.name}]"))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body)}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>
  </w:body>
</w:document>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
</w:styles>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  {''.join(rels)}
</Relationships>
"""
    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        for media_name, path in media_files:
            docx.write(path, f"word/media/{media_name}")


def main():
    REPORTS.mkdir(exist_ok=True)
    sections = build_sections(load_results())
    render_pdf(sections)
    render_docx(sections)
    print(f"PDF gerado: {PDF_PATH}")
    print(f"DOCX gerado: {DOCX_PATH}")


if __name__ == "__main__":
    main()
