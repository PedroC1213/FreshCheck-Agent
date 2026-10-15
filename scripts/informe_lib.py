"""Utilidades comunes para generar los PDF del informe y la bitácora (reportlab)."""
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

FUENTES = "/usr/share/fonts/truetype/crosextra/"
pdfmetrics.registerFont(TTFont("Carlito", FUENTES + "Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Bold", FUENTES + "Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Italic", FUENTES + "Carlito-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-BoldItalic", FUENTES + "Carlito-BoldItalic.ttf"))
pdfmetrics.registerFontFamily("Carlito", normal="Carlito", bold="Carlito-Bold",
                              italic="Carlito-Italic", boldItalic="Carlito-BoldItalic")

AZUL = colors.HexColor("#1F3864")
AZUL_CL = colors.HexColor("#E8EEF7")
GRIS = colors.HexColor("#555555")
VERDE = colors.HexColor("#1E6B3A")
ROJO = colors.HexColor("#9C1C1C")

ESTILOS = {
    "cuerpo": ParagraphStyle("cuerpo", fontName="Carlito", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6),
    "cuerpo_izq": ParagraphStyle("cuerpo_izq", fontName="Carlito", fontSize=10.5, leading=14.2, alignment=TA_LEFT, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Carlito-Bold", fontSize=16, leading=20, textColor=AZUL, spaceBefore=14, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Carlito-Bold", fontSize=12.5, leading=16, textColor=AZUL, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Carlito-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#2E4B6B"), spaceBefore=6, spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="Carlito", fontSize=10.5, leading=14, alignment=TA_JUSTIFY, leftIndent=16, bulletIndent=4, spaceAfter=3),
    "celda": ParagraphStyle("celda", fontName="Carlito", fontSize=8.6, leading=10.6, alignment=TA_LEFT),
    "celda_j": ParagraphStyle("celda_j", fontName="Carlito", fontSize=8.6, leading=10.6, alignment=TA_JUSTIFY),
    "celda_b": ParagraphStyle("celda_b", fontName="Carlito-Bold", fontSize=8.6, leading=10.6, alignment=TA_LEFT),
    "celda_h": ParagraphStyle("celda_h", fontName="Carlito-Bold", fontSize=8.8, leading=10.8, textColor=colors.white, alignment=TA_LEFT),
    "leyenda": ParagraphStyle("leyenda", fontName="Carlito-Italic", fontSize=9, leading=11, textColor=GRIS, alignment=TA_CENTER, spaceAfter=8),
    "nota": ParagraphStyle("nota", fontName="Carlito-Italic", fontSize=9.6, leading=12.6, textColor=colors.HexColor("#33475B"),
                           backColor=colors.HexColor("#F3F6FA"), borderPadding=5, spaceBefore=4, spaceAfter=8, alignment=TA_JUSTIFY),
    "portada_t": ParagraphStyle("portada_t", fontName="Carlito-Bold", fontSize=22, leading=27, alignment=TA_CENTER, textColor=AZUL),
    "portada_s": ParagraphStyle("portada_s", fontName="Carlito-Italic", fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#33475B")),
    "portada": ParagraphStyle("portada", fontName="Carlito", fontSize=12, leading=17, alignment=TA_CENTER),
    "portada_b": ParagraphStyle("portada_b", fontName="Carlito-Bold", fontSize=12.5, leading=17, alignment=TA_CENTER),
}


def _esc(texto):
    """Escapa los '&' sueltos (R&N, Russell & Norvig) sin tocar las entidades ya escritas."""
    return re.sub(r"&(?!amp;|lt;|gt;|#)", "&amp;", texto)


def P(texto, estilo="cuerpo"):
    return Paragraph(_esc(texto), ESTILOS[estilo])


def B(texto):
    return Paragraph(_esc(texto), ESTILOS["bullet"], bulletText="•")


def tabla(filas, anchos_cm, encabezado=True, zebra=True, estilo_celda="celda", negrita_primera=False, extra=None):
    """Tabla con celdas Paragraph (ajuste de texto automático) y fila de encabezado repetida."""
    datos = []
    for i, fila in enumerate(filas):
        nueva = []
        for j, c in enumerate(fila):
            if isinstance(c, str):
                if encabezado and i == 0:
                    est = "celda_h"
                elif negrita_primera and j == 0:
                    est = "celda_b"
                else:
                    est = estilo_celda
                nueva.append(Paragraph(_esc(c), ESTILOS[est]))
            else:
                nueva.append(c)
        datos.append(nueva)
    t = Table(datos, colWidths=[a * cm for a in anchos_cm], repeatRows=1 if encabezado else 0)
    est = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA5B5")),
           ("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if encabezado:
        est.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
    if zebra:
        for i in range(1 if encabezado else 0, len(datos)):
            if i % 2 == 0:
                est.append(("BACKGROUND", (0, i), (-1, i), AZUL_CL))
    if extra:
        est.extend(extra)
    t.setStyle(TableStyle(est))
    return t


def figura(ruta, ancho_cm, leyenda=None):
    w, h = PILImage.open(ruta).size
    img = Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w)
    elementos = [img]
    if leyenda:
        elementos.append(P(leyenda, "leyenda"))
    return KeepTogether(elementos)


def crear_documento(ruta_pdf, titulo_encabezado, autor="Pedro José Castro Blanquicett", titulo="Informe", inicial="portrait"):
    ancho_v, alto_v = letter
    ancho_h, alto_h = landscape(letter)

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("Carlito", 8.5)
        canvas.setFillColor(GRIS)
        w, h = canvas._pagesize
        canvas.drawCentredString(w / 2, 1.1 * cm, f"{doc.page}")
        if doc.page > 1:
            canvas.drawRightString(w - 2.0 * cm, h - 1.2 * cm, titulo_encabezado)
        canvas.restoreState()

    doc = BaseDocTemplate(str(ruta_pdf), pagesize=letter, author=autor, title=titulo,
                          leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.0 * cm)
    f_v = Frame(2.2 * cm, 2.0 * cm, ancho_v - 4.4 * cm, alto_v - 4.4 * cm, id="v")
    f_h = Frame(2.0 * cm, 1.8 * cm, ancho_h - 4.0 * cm, alto_h - 4.0 * cm, id="h")
    plantillas = [PageTemplate(id="portrait", frames=[f_v], onPage=pie, pagesize=letter),
                  PageTemplate(id="landscape", frames=[f_h], onPage=pie, pagesize=landscape(letter))]
    if inicial == "landscape":
        plantillas.reverse()
    doc.addPageTemplates(plantillas)
    return doc


def a_horizontal():
    return [NextPageTemplate("landscape"), PageBreak()]


def a_vertical():
    return [NextPageTemplate("portrait"), PageBreak()]


def num(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return num(100 * x, d) + " %"
