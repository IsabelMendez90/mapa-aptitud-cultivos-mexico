# -*- coding: utf-8 -*-

import json
import re
import hashlib
import colorsys
import unicodedata
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# CONEXIÓN CON LLM
# ============================================================

try:
    from llm_openrouter import (
        interpretar_pregunta_usuario,
        buscar_requerimientos_web_cultivo,
        llamar_llm,
    )
    LLM_DISPONIBLE = True
    LLM_ERROR_IMPORT = None
except Exception as e:
    LLM_DISPONIBLE = False
    LLM_ERROR_IMPORT = str(e)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Mapa integrado de cultivos",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

CARPETA_INTEGRADOS = BASE_DIR / "data/mapas_integrados"
INDICE_INTEGRADO = CARPETA_INTEGRADOS / "indice_mapas_integrados.csv"

CARPETA_PDF_CRUDO = BASE_DIR / "data/mapas_pdf_crudo"
INDICE_PDF = CARPETA_PDF_CRUDO / "indice_cumplimiento_pdf.csv"

CARPETA_MAPAS_MUNICIPIOS = BASE_DIR / "data/mapas_municipios"
GEOJSON_DEFAULT = CARPETA_MAPAS_MUNICIPIOS / "mexico_municipios_mejor_sistema.geojson"


# ============================================================
# IDENTIDAD Y FUENTE BASE
# ============================================================

APP_NOMBRE = "Asistente de mapas de aptitud de cultivos"
APP_DESARROLLADORA = "Dra. Juana Isabel Méndez"
APP_CONTACTO = "isabelmendez@tec.mx"

PDF_CITA_BASE = (
    "Ruiz Corral, J., García, G., Acuña, I., Flores López, H., & Ojeda, G. (2020). "
    "Requerimientos agroecológicos de cultivos (2da ed.)."
)

APP_IDENTIDAD = f"""
Soy un asistente de apoyo para interpretar mapas de cultivos, aptitud agroecológica y escenarios de cultivo urbano en México.

Esta herramienta fue desarrollada por la {APP_DESARROLLADORA} para facilitar la exploración de cultivos por municipio, comparar escenarios como azotea, invernadero o cultivo interior controlado, y explicar los resultados del mapa en lenguaje claro.

Puedo ayudarte a interpretar el índice de aptitud, el cumplimiento base del PDF, los factores limitantes y la comparación entre sistemas de cultivo urbano.

La fuente documental base de los requerimientos agroecológicos es:

{PDF_CITA_BASE}

Si necesitas más apoyo, puedes escribir a: {APP_CONTACTO}
"""


# ============================================================
# OPCIONES
# ============================================================

LECTURAS = {
    "pdf": "Cumplimiento base del PDF",
    "aptitud": "Índice de aptitud",
}

VISTAS = {
    "un_cultivo": "Un cultivo",
    "todos_cultivos": "Todos los cultivos",
}

ESCENARIOS = {
    "solo": "Solo condiciones del cultivo",
    "todos_sistemas": "Todos los sistemas / mejor escenario",
    "cielo_abierto": "Exterior horizontal / cielo abierto",
    "azotea_extensiva": "Azotea extensiva",
    "azotea_intensiva": "Azotea intensiva",
    "huerto_vertical": "Huerto vertical exterior",
    "invernadero": "Invernadero",
    "interior_led": "Cultivo interior controlado con luz artificial",
}

SISTEMAS_URBANOS = [
    "cielo_abierto",
    "azotea_extensiva",
    "azotea_intensiva",
    "huerto_vertical",
    "invernadero",
    "interior_led",
]

ALIAS_SISTEMAS = {
    "solo": ["solo", "sin sistema", "base", "aptitud del cultivo"],
    "todos_sistemas": [
        "todos",
        "todos los sistemas",
        "mejor sistema",
        "sistema recomendado",
        "recomendado",
        "comparacion de sistemas",
        "comparación de sistemas",
        "que sistema conviene",
        "qué sistema conviene",
    ],
    "cielo_abierto": ["cielo abierto", "exterior", "campo abierto", "horizontal"],
    "azotea_extensiva": ["azotea extensiva", "techo extensivo"],
    "azotea_intensiva": ["azotea", "azotea intensiva", "techo", "techo intensivo", "departamento"],
    "huerto_vertical": ["vertical", "huerto vertical", "muro verde"],
    "invernadero": ["invernadero", "greenhouse"],
    "interior_led": ["interior", "led", "luz artificial", "indoor", "cultivo interior controlado"],
}

COLORES_FACTOR = {
    "temperatura": "#e41a1c",
    "calor": "#ff7f00",
    "frio": "#377eb8",
    "precipitacion": "#4daf4a",
    "riego_drenaje": "#41ab5d",
    "manejo_hidrico_controlado": "#238b45",
    "control_termico": "#fb6a4a",
    "calor_microclima": "#fd8d3c",
    "luz": "#ffd92f",
    "luz_artificial": "#fdd835",
    "sustrato": "#8c510a",
    "sustrato_soporte": "#bf812d",
    "altitud": "#a65628",
    "latitud": "#f781bf",
    "ph": "#999999",
    "salinidad": "#b3cde3",
    "compatibilidad_urbana": "#6a3d9a",
    "ambiente_controlado": "#80cdc1",
    "sin_limitante": "#1a9850",
    "sin_dato": "#bdbdbd",
}

ESTADOS_MEXICO = [
    "Aguascalientes",
    "Baja California",
    "Baja California Sur",
    "Campeche",
    "Chiapas",
    "Chihuahua",
    "Ciudad de México",
    "Coahuila",
    "Colima",
    "Durango",
    "Guanajuato",
    "Guerrero",
    "Hidalgo",
    "Jalisco",
    "México",
    "Michoacán",
    "Morelos",
    "Nayarit",
    "Nuevo León",
    "Oaxaca",
    "Puebla",
    "Querétaro",
    "Quintana Roo",
    "San Luis Potosí",
    "Sinaloa",
    "Sonora",
    "Tabasco",
    "Tamaulipas",
    "Tlaxcala",
    "Veracruz",
    "Yucatán",
    "Zacatecas",
]


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s_-]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_columna(texto):
    texto = normalizar_texto(texto)
    texto = texto.replace(" ", "_")
    texto = re.sub(r"_+", "_", texto)
    return texto.strip("_")


def valor_no_vacio(valor):
    if valor is None:
        return False

    try:
        if pd.isna(valor):
            return False
    except Exception:
        pass

    if str(valor).strip() in ["", "NA", "nan", "None"]:
        return False

    return True


def hex_to_rgb(hex_color):
    hex_color = str(hex_color).replace("#", "")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgba_from_hex(hex_color, alpha=170):
    r, g, b = hex_to_rgb(hex_color)
    return [r, g, b, alpha]


def color_escala(valor):
    try:
        valor = float(valor)
    except Exception:
        valor = 0

    if valor >= 90:
        return "#1a9850"
    elif valor >= 75:
        return "#66bd63"
    elif valor >= 50:
        return "#fee08b"
    elif valor >= 25:
        return "#fdae61"
    else:
        return "#d73027"


def clase_valor(valor):
    try:
        valor = float(valor)
    except Exception:
        valor = 0

    if valor >= 90:
        return "Muy alta"
    elif valor >= 75:
        return "Alta"
    elif valor >= 50:
        return "Media"
    elif valor >= 25:
        return "Baja"
    else:
        return "Muy baja"


def color_categoria(texto):
    texto = str(texto)
    digest = hashlib.md5(texto.encode("utf-8")).hexdigest()

    hue = (int(digest[:8], 16) % 360) / 360.0
    saturation = 0.55
    value = 0.85

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )


def nombre_factor_legible(factor):
    dic = {
        "temperatura": "Temperatura",
        "calor": "Calor extremo",
        "frio": "Frío / helada",
        "precipitacion": "Precipitación",
        "riego_drenaje": "Riego / drenaje",
        "manejo_hidrico_controlado": "Manejo hídrico controlado",
        "control_termico": "Control térmico",
        "calor_microclima": "Calor en microclima urbano",
        "luz": "Luz",
        "luz_artificial": "Luz artificial",
        "sustrato": "Sustrato",
        "sustrato_soporte": "Sustrato / soporte",
        "altitud": "Altitud",
        "latitud": "Latitud",
        "ph": "pH",
        "salinidad": "Salinidad",
        "compatibilidad_urbana": "Compatibilidad urbana",
        "ambiente_controlado": "Ambiente controlado",
        "sin_limitante": "Sin limitante principal",
        "sin_dato": "Sin dato",
    }
    return dic.get(str(factor), str(factor))


def formato_valor(valor, decimales=1):
    try:
        return round(float(valor), decimales)
    except Exception:
        return "NA"


def detectar_factor_desde_texto(texto):
    texto = normalizar_columna(texto)

    if not texto or texto in ["na", "nan", "none"]:
        return "sin_limitante"

    if "calor" in texto or "max" in texto:
        return "calor"

    if "frio" in texto or "helada" in texto or "min" in texto:
        return "frio"

    if "temperatura" in texto or "temp" in texto:
        return "temperatura"

    if "precipitacion" in texto or "lluvia" in texto or "agua" in texto:
        return "precipitacion"

    if "luz" in texto or "radiacion" in texto:
        return "luz"

    if "sustrato" in texto or "suelo" in texto:
        return "sustrato"

    if "altitud" in texto:
        return "altitud"

    if "latitud" in texto:
        return "latitud"

    if "ph" in texto:
        return "ph"

    if "salinidad" in texto or "sal" in texto:
        return "salinidad"

    if "compat" in texto or "urbana" in texto or "sistema" in texto:
        return "compatibilidad_urbana"

    return texto


def obtener_gid_fila(row):
    for campo in ["gid_municipio", "GID_2", "GID2", "gid", "id"]:
        if campo in row.index and valor_no_vacio(row[campo]):
            return str(row[campo])

    estado = row.get("estado", "")
    municipio = row.get("municipio", "")

    return f"{estado}|{municipio}"


def deduplicar_municipios_para_mapa(df, ordenar_por="_valor_mapa"):
    """
    Devuelve una sola fila por municipio para métricas, leyenda, mapa y contexto LLM.

    Algunas vistas integradas pueden traer más de una fila por municipio porque
    combinan escenarios o provienen de joins/intermedios con varias combinaciones.
    Para visualización municipal, se conserva la fila con mayor valor de mapa.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    df["_gid_join"] = df.apply(obtener_gid_fila, axis=1)

    if ordenar_por in df.columns:
        df[ordenar_por] = pd.to_numeric(df[ordenar_por], errors="coerce").fillna(0)
        df = df.sort_values(ordenar_por, ascending=False)

    df = df.drop_duplicates("_gid_join", keep="first")

    return df.drop(columns=["_gid_join"], errors="ignore")


def contar_municipios_unicos(df):
    """Cuenta municipios únicos con la misma llave usada para unir al GeoJSON."""
    if df is None or df.empty:
        return 0

    gids = df.apply(obtener_gid_fila, axis=1)
    return int(gids.nunique())


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(show_spinner=False)
def cargar_indice_integrado():
    if not INDICE_INTEGRADO.exists():
        return pd.DataFrame()

    return pd.read_csv(INDICE_INTEGRADO)


@st.cache_data(show_spinner=False)
def cargar_indice_pdf():
    if not INDICE_PDF.exists():
        return pd.DataFrame()

    df = pd.read_csv(INDICE_PDF)

    if "nombre_cultivo" not in df.columns or "archivo_csv" not in df.columns:
        return pd.DataFrame()

    return (
        df[["nombre_cultivo", "archivo_csv"]]
        .drop_duplicates("nombre_cultivo")
        .sort_values("nombre_cultivo")
        .reset_index(drop=True)
    )


def resolver_ruta(ruta_str):
    # Los índices históricos se generaron en Windows y algunos contienen
    # separadores "\". Normalizarlos permite usar los mismos índices en Linux,
    # que es el entorno de Streamlit Community Cloud.
    ruta = Path(str(ruta_str).replace("\\", "/"))

    if ruta.exists():
        return ruta

    ruta2 = BASE_DIR / ruta

    if ruta2.exists():
        return ruta2

    return ruta


@st.cache_data(show_spinner=False)
def cargar_csv(ruta_csv_str):
    ruta = resolver_ruta(ruta_csv_str)

    if not ruta.exists():
        return pd.DataFrame()

    if ruta.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(ruta)

    return pd.read_csv(ruta)


def resolver_geojson():
    if GEOJSON_DEFAULT.exists():
        return GEOJSON_DEFAULT

    candidatos = sorted(CARPETA_MAPAS_MUNICIPIOS.glob("*mejor_sistema*.geojson"))
    if candidatos:
        return candidatos[0]

    candidatos = sorted(CARPETA_MAPAS_MUNICIPIOS.glob("*.geojson"))
    if candidatos:
        return candidatos[0]

    return None


@st.cache_data(show_spinner=False)
def cargar_geojson(ruta_geojson_str):
    with open(ruta_geojson_str, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ADAPTACIÓN DE DATOS
# ============================================================

def adaptar_integrado_a_app(df):
    df = df.copy()

    columnas_base = {
        "nombre_cultivo": "_cultivo_mapa",
        "valor": "_valor_mapa",
        "clase": "_clase_mapa",
        "escenario": "_escenario_mapa",
        "escenario_key": "_escenario_key",
        "valor_base": "_valor_base_mapa",
        "compatibilidad": "_compatibilidad_mapa",
        "factor_escenario": "_factor_mapa",
        "factor_base": "_factor_base",
    }

    for col_origen, col_destino in columnas_base.items():
        if col_origen in df.columns:
            df[col_destino] = df[col_origen]
        else:
            df[col_destino] = "NA"

    df["_valor_mapa"] = pd.to_numeric(df["_valor_mapa"], errors="coerce").fillna(0)

    df["_valor_base_mapa"] = pd.to_numeric(
        df["_valor_base_mapa"],
        errors="coerce"
    ).fillna(df["_valor_mapa"])

    if "_clase_mapa" not in df.columns or df["_clase_mapa"].isna().all():
        df["_clase_mapa"] = df["_valor_mapa"].apply(clase_valor)

    if "gid_municipio" not in df.columns:
        df["gid_municipio"] = "NA"

    if "estado" not in df.columns:
        df["estado"] = "NA"

    if "municipio" not in df.columns:
        df["municipio"] = "NA"

    return df


def preparar_pdf_un_cultivo_original(df, nombre_cultivo):
    df = df.copy()

    if "porcentaje_cumplimiento_pdf" not in df.columns:
        return pd.DataFrame()

    if "gid_municipio" not in df.columns:
        df["gid_municipio"] = "NA"

    if "estado" not in df.columns:
        df["estado"] = "NA"

    if "municipio" not in df.columns:
        df["municipio"] = "NA"

    df["_cultivo_mapa"] = nombre_cultivo
    df["_valor_mapa"] = pd.to_numeric(
        df["porcentaje_cumplimiento_pdf"],
        errors="coerce"
    ).fillna(0)

    df["_valor_base_mapa"] = df["_valor_mapa"]
    df["_clase_mapa"] = df["_valor_mapa"].apply(clase_valor)
    df["_escenario_key"] = "pdf"
    df["_escenario_mapa"] = "Cumplimiento base del PDF"
    df["_compatibilidad_mapa"] = "NA"

    if "factores_limitantes_pdf" in df.columns:
        df["_factor_base"] = df["factores_limitantes_pdf"].fillna("NA").apply(
            lambda x: detectar_factor_desde_texto(str(x).split("|")[0])
        )
    else:
        df["_factor_base"] = "sin_dato"

    df["_factor_mapa"] = df["_factor_base"]

    return df


@st.cache_data(show_spinner=False)
def cargar_pdf_un_cultivo_original(ruta_csv_str, nombre_cultivo):
    df = cargar_csv(ruta_csv_str)
    return preparar_pdf_un_cultivo_original(df, nombre_cultivo)


def obtener_archivo_integrado(indice_integrado, lectura, vista, escenario_key=None, cultivo=None):
    if indice_integrado.empty:
        return None

    df = indice_integrado.copy()

    df = df[
        (df["lectura"] == lectura) &
        (df["vista"] == vista)
    ]

    if escenario_key is not None:
        df = df[df["escenario_key"] == escenario_key]

    if cultivo is not None:
        df = df[df["nombre_cultivo"] == cultivo]

    if df.empty:
        return None

    return df.iloc[0]["archivo_csv"]


def obtener_cultivos_disponibles(indice_integrado, indice_pdf, lectura):
    if lectura == "pdf":
        if not indice_pdf.empty:
            return (
                indice_pdf["nombre_cultivo"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
        return []

    df = indice_integrado[
        (indice_integrado["lectura"] == "aptitud") &
        (indice_integrado["vista"] == "un_cultivo")
    ]

    if df.empty:
        return []

    return (
        df["nombre_cultivo"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


# ============================================================
# CARGA DE MAPA
# ============================================================

def comparar_sistemas_desde_precalculo(df_todos_escenarios):
    if df_todos_escenarios.empty:
        return pd.DataFrame()

    df = df_todos_escenarios.copy()
    df = df[df["escenario_key"].isin(SISTEMAS_URBANOS)]

    if df.empty:
        return pd.DataFrame()

    registros = []

    for escenario_key, grupo in df.groupby("escenario_key"):
        # Las estadísticas por sistema deben calcularse a nivel municipal,
        # no por número de filas. Esto evita sobrecontar municipios duplicados.
        grupo = deduplicar_municipios_para_mapa(grupo)

        valor = pd.to_numeric(grupo["_valor_mapa"], errors="coerce").fillna(0)
        compat = pd.to_numeric(grupo["_compatibilidad_mapa"], errors="coerce")

        registros.append({
            "escenario_key": escenario_key,
            "sistema": ESCENARIOS.get(escenario_key, escenario_key),
            "aptitud_promedio": round(valor.mean(), 1),
            "aptitud_mediana": round(valor.median(), 1),
            "municipios_75_o_mas": int((valor >= 75).sum()),
            "municipios_90_o_mas": int((valor >= 90).sum()),
            "compatibilidad_promedio": round(compat.mean(), 1) if compat.notna().sum() else None,
        })

    salida = pd.DataFrame(registros)

    if salida.empty:
        return salida

    return salida.sort_values("aptitud_promedio", ascending=False)


def cargar_dataframe_mapa(lectura, vista, cultivo_label, escenario_key, indice_integrado, indice_pdf):
    if lectura == "pdf" and vista == "todos_cultivos":
        archivo = obtener_archivo_integrado(
            indice_integrado=indice_integrado,
            lectura="pdf",
            vista="todos_cultivos",
            escenario_key="pdf",
            cultivo=None,
        )

        if archivo is None:
            return pd.DataFrame(), pd.DataFrame()

        df = cargar_csv(archivo)
        return adaptar_integrado_a_app(df), pd.DataFrame()

    if lectura == "pdf" and vista == "un_cultivo":
        if indice_pdf.empty:
            return pd.DataFrame(), pd.DataFrame()

        fila = indice_pdf[indice_pdf["nombre_cultivo"] == cultivo_label]

        if fila.empty:
            return pd.DataFrame(), pd.DataFrame()

        archivo = fila.iloc[0]["archivo_csv"]
        df = cargar_pdf_un_cultivo_original(archivo, cultivo_label)
        return df, pd.DataFrame()

    if lectura == "aptitud" and vista == "todos_cultivos":
        archivo = obtener_archivo_integrado(
            indice_integrado=indice_integrado,
            lectura="aptitud",
            vista="todos_cultivos",
            escenario_key=escenario_key,
            cultivo=None,
        )

        if archivo is None:
            return pd.DataFrame(), pd.DataFrame()

        df = cargar_csv(archivo)
        return adaptar_integrado_a_app(df), pd.DataFrame()

    if lectura == "aptitud" and vista == "un_cultivo":
        archivo = obtener_archivo_integrado(
            indice_integrado=indice_integrado,
            lectura="aptitud",
            vista="un_cultivo",
            escenario_key="todos_en_archivo",
            cultivo=cultivo_label,
        )

        if archivo is None:
            return pd.DataFrame(), pd.DataFrame()

        df_todos_escenarios = adaptar_integrado_a_app(cargar_csv(archivo))

        if df_todos_escenarios.empty:
            return pd.DataFrame(), pd.DataFrame()

        comparacion = comparar_sistemas_desde_precalculo(df_todos_escenarios)

        df = df_todos_escenarios[
            df_todos_escenarios["escenario_key"] == escenario_key
        ].copy()

        if df.empty and escenario_key == "todos_sistemas":
            df_sistemas = df_todos_escenarios[
                df_todos_escenarios["escenario_key"].isin(SISTEMAS_URBANOS)
            ].copy()

            if not df_sistemas.empty:
                df_sistemas["_gid_join"] = df_sistemas.apply(obtener_gid_fila, axis=1)
                df_sistemas = df_sistemas.sort_values("_valor_mapa", ascending=False)
                df = df_sistemas.drop_duplicates("_gid_join", keep="first").copy()
                df["_escenario_key"] = "todos_sistemas"
                df["_escenario_mapa"] = "Todos los sistemas / mejor escenario"

        return df, comparacion

    return pd.DataFrame(), pd.DataFrame()


# ============================================================
# UBICACIÓN Y RANKING LOCAL
# ============================================================

def encontrar_opcion_por_texto(opciones, texto):
    if texto is None:
        return None

    texto_norm = normalizar_texto(texto)

    if not texto_norm:
        return None

    opciones = [str(o) for o in opciones if valor_no_vacio(o)]

    for opcion in opciones:
        if normalizar_texto(opcion) == texto_norm:
            return opcion

    for opcion in opciones:
        opcion_norm = normalizar_texto(opcion)

        if texto_norm in opcion_norm or opcion_norm in texto_norm:
            return opcion

    return None


def detectar_ubicacion_basica_desde_texto(pregunta):
    pregunta_norm = normalizar_texto(pregunta)

    estado = None
    municipio = None
    lugar_mencionado = None

    for estado_mx in ESTADOS_MEXICO:
        if normalizar_texto(estado_mx) in pregunta_norm:
            estado = estado_mx
            break

    if "campus queretaro" in pregunta_norm or (
        "tec" in pregunta_norm and "queretaro" in pregunta_norm
    ):
        estado = "Querétaro"
        municipio = "Querétaro"
        lugar_mencionado = "Tec de Monterrey, Campus Querétaro"

    elif "queretaro" in pregunta_norm:
        estado = estado or "Querétaro"
        municipio = municipio or "Querétaro"
        lugar_mencionado = "Querétaro"

    if "monterrey" in pregunta_norm:
        estado = estado or "Nuevo León"
        municipio = municipio or "Monterrey"
        lugar_mencionado = lugar_mencionado or "Monterrey"

    if "guadalajara" in pregunta_norm:
        estado = estado or "Jalisco"
        municipio = municipio or "Guadalajara"
        lugar_mencionado = lugar_mencionado or "Guadalajara"

    if "zapopan" in pregunta_norm:
        estado = estado or "Jalisco"
        municipio = municipio or "Zapopan"
        lugar_mencionado = lugar_mencionado or "Zapopan"

    if "merida" in pregunta_norm or "mérida" in pregunta_norm:
        estado = estado or "Yucatán"
        municipio = municipio or "Mérida"
        lugar_mencionado = lugar_mencionado or "Mérida"

    return {
        "estado": estado,
        "municipio": municipio,
        "lugar_mencionado": lugar_mencionado,
    }


def consulta_pide_recomendacion_local(pregunta, intencion=None):
    pregunta_norm = normalizar_texto(pregunta)

    patrones = [
        "que puede cultivarse",
        "qué puede cultivarse",
        "que puedo cultivar",
        "qué puedo cultivar",
        "que se puede cultivar",
        "qué se puede cultivar",
        "que cultivos",
        "qué cultivos",
        "cultivos recomiendas",
        "cultivo recomiendas",
        "donde me recomiendas cultivarla",
        "dónde me recomiendas cultivarla",
        "quiero cultivar",
    ]

    if any(normalizar_texto(p) in pregunta_norm for p in patrones):
        return True

    if isinstance(intencion, dict):
        accion = intencion.get("accion", "")
        if accion in ["mapa_todos_cultivos", "recomendar_cultivo_sitio", "recomendar_cultivo_hogar"]:
            return True

    return False


def consulta_sitio_sin_ubicacion(pregunta, intencion=None):
    pregunta_norm = normalizar_texto(pregunta)

    if isinstance(intencion, dict):
        if intencion.get("requiere_ubicacion_especifica", False):
            return True

    patrones_sitio = [
        "mi casa",
        "mi hogar",
        "mi departamento",
        "mi campus",
        "mi escuela",
        "mi edificio",
        "mi localidad",
        "donde vivo",
        "en mi zona",
        "cerca de mi",
        "cerca de mí",
        "aqui",
        "aquí",
        "en este predio",
        "en este proyecto",
    ]

    menciona_sitio = any(normalizar_texto(p) in pregunta_norm for p in patrones_sitio)
    ubicacion = detectar_ubicacion_basica_desde_texto(pregunta)

    tiene_ubicacion = bool(ubicacion.get("estado")) or bool(ubicacion.get("municipio"))

    if isinstance(intencion, dict):
        tiene_ubicacion = tiene_ubicacion or bool(intencion.get("estado")) or bool(intencion.get("municipio"))

    return menciona_sitio and not tiene_ubicacion


def guardar_ubicacion_desde_intencion(pregunta, intencion):
    estado = None
    municipio = None
    lugar_mencionado = None

    if isinstance(intencion, dict):
        estado = intencion.get("estado")
        municipio = intencion.get("municipio")
        lugar_mencionado = (
            intencion.get("lugar_mencionado")
            or intencion.get("ubicacion_usuario")
        )

    respaldo = detectar_ubicacion_basica_desde_texto(pregunta)

    if not estado:
        estado = respaldo.get("estado")

    if not municipio:
        municipio = respaldo.get("municipio")

    if not lugar_mencionado:
        lugar_mencionado = respaldo.get("lugar_mencionado")

    if estado or municipio:
        st.session_state["usar_filtro_ubicacion_integrada"] = True
        st.session_state["estado_objetivo_integrado"] = estado
        st.session_state["municipio_objetivo_integrado"] = municipio
        st.session_state["lugar_mencionado_integrado"] = lugar_mencionado
        return True

    return False


def aplicar_filtro_ubicacion_chat(df):
    if df.empty:
        return df, None

    if not st.session_state.get("usar_filtro_ubicacion_integrada", False):
        return df, None

    estado_obj = st.session_state.get("estado_objetivo_integrado")
    municipio_obj = st.session_state.get("municipio_objetivo_integrado")
    lugar_obj = st.session_state.get("lugar_mencionado_integrado")

    df_filtrado = df.copy()

    estado_match = None
    municipio_match = None

    if estado_obj and "estado" in df_filtrado.columns:
        estado_match = encontrar_opcion_por_texto(
            df_filtrado["estado"].dropna().unique().tolist(),
            estado_obj
        )

        if estado_match:
            df_filtrado = df_filtrado[df_filtrado["estado"] == estado_match]

    if municipio_obj and "municipio" in df_filtrado.columns:
        municipio_match = encontrar_opcion_por_texto(
            df_filtrado["municipio"].dropna().unique().tolist(),
            municipio_obj
        )

        if municipio_match:
            df_filtrado = df_filtrado[df_filtrado["municipio"] == municipio_match]

    info = {
        "estado_objetivo": estado_obj,
        "municipio_objetivo": municipio_obj,
        "estado_aplicado": estado_match,
        "municipio_aplicado": municipio_match,
        "lugar_mencionado": lugar_obj,
        "filas_resultantes": len(df_filtrado),
    }

    return df_filtrado, info


@st.cache_data(show_spinner=False)
def calcular_ranking_local_cacheado(indice_json, estado_obj, municipio_obj, escenario_key, top_n=12):
    indice = pd.DataFrame(json.loads(indice_json))

    if indice.empty:
        return pd.DataFrame()

    filas_indice = indice[
        (indice["lectura"] == "aptitud") &
        (indice["vista"] == "un_cultivo") &
        (indice["escenario_key"] == "todos_en_archivo")
    ].copy()

    if filas_indice.empty:
        return pd.DataFrame()

    registros = []

    for _, fila_indice in filas_indice.iterrows():
        cultivo = fila_indice.get("nombre_cultivo")
        archivo = fila_indice.get("archivo_csv")

        if not archivo:
            continue

        df = adaptar_integrado_a_app(cargar_csv(archivo))

        if df.empty:
            continue

        df["_gid_join"] = df.apply(obtener_gid_fila, axis=1)

        if estado_obj:
            estado_match = encontrar_opcion_por_texto(
                df["estado"].dropna().unique().tolist(),
                estado_obj
            )

            if estado_match:
                df = df[df["estado"] == estado_match]
            else:
                continue

        if municipio_obj:
            municipio_match = encontrar_opcion_por_texto(
                df["municipio"].dropna().unique().tolist(),
                municipio_obj
            )

            if municipio_match:
                df = df[df["municipio"] == municipio_match]
            else:
                continue

        if df.empty:
            continue

        if escenario_key == "todos_sistemas":
            df_esc = df[df["escenario_key"].isin(SISTEMAS_URBANOS)].copy()

            if df_esc.empty:
                continue

            if municipio_obj:
                df_esc = df_esc.sort_values("_valor_mapa", ascending=False)
                mejor = df_esc.iloc[0]

                registros.append({
                    "cultivo": cultivo,
                    "valor": round(float(mejor.get("_valor_mapa", 0)), 1),
                    "clase": mejor.get("_clase_mapa", clase_valor(mejor.get("_valor_mapa", 0))),
                    "escenario": mejor.get("_escenario_mapa", ""),
                    "escenario_key": mejor.get("_escenario_key", ""),
                    "estado": mejor.get("estado", ""),
                    "municipio": mejor.get("municipio", ""),
                    "factor": mejor.get("_factor_mapa", "sin_dato"),
                    "factor_legible": nombre_factor_legible(mejor.get("_factor_mapa", "sin_dato")),
                    "valor_base": formato_valor(mejor.get("_valor_base_mapa", 0)),
                    "compatibilidad": formato_valor(mejor.get("_compatibilidad_mapa", "NA")),
                    "municipios_evaluados": 1,
                    "municipios_75_o_mas": int(float(mejor.get("_valor_mapa", 0)) >= 75),
                })

            else:
                df_esc = df_esc.sort_values("_valor_mapa", ascending=False)
                df_best_mun = df_esc.drop_duplicates("_gid_join", keep="first").copy()
                valores = pd.to_numeric(df_best_mun["_valor_mapa"], errors="coerce").fillna(0)

                mejor = df_best_mun.sort_values("_valor_mapa", ascending=False).iloc[0]

                registros.append({
                    "cultivo": cultivo,
                    "valor": round(float(valores.mean()), 1),
                    "clase": clase_valor(valores.mean()),
                    "escenario": "Todos los sistemas / mejor escenario",
                    "escenario_key": "todos_sistemas",
                    "estado": estado_obj or "",
                    "municipio": "",
                    "factor": mejor.get("_factor_mapa", "sin_dato"),
                    "factor_legible": nombre_factor_legible(mejor.get("_factor_mapa", "sin_dato")),
                    "valor_base": formato_valor(mejor.get("_valor_base_mapa", 0)),
                    "compatibilidad": formato_valor(mejor.get("_compatibilidad_mapa", "NA")),
                    "municipios_evaluados": int(len(df_best_mun)),
                    "municipios_75_o_mas": int((valores >= 75).sum()),
                })

        else:
            df_esc = df[df["escenario_key"] == escenario_key].copy()

            if df_esc.empty:
                continue

            if municipio_obj:
                mejor = df_esc.sort_values("_valor_mapa", ascending=False).iloc[0]

                registros.append({
                    "cultivo": cultivo,
                    "valor": round(float(mejor.get("_valor_mapa", 0)), 1),
                    "clase": mejor.get("_clase_mapa", clase_valor(mejor.get("_valor_mapa", 0))),
                    "escenario": mejor.get("_escenario_mapa", ""),
                    "escenario_key": mejor.get("_escenario_key", ""),
                    "estado": mejor.get("estado", ""),
                    "municipio": mejor.get("municipio", ""),
                    "factor": mejor.get("_factor_mapa", "sin_dato"),
                    "factor_legible": nombre_factor_legible(mejor.get("_factor_mapa", "sin_dato")),
                    "valor_base": formato_valor(mejor.get("_valor_base_mapa", 0)),
                    "compatibilidad": formato_valor(mejor.get("_compatibilidad_mapa", "NA")),
                    "municipios_evaluados": 1,
                    "municipios_75_o_mas": int(float(mejor.get("_valor_mapa", 0)) >= 75),
                })

            else:
                valores = pd.to_numeric(df_esc["_valor_mapa"], errors="coerce").fillna(0)
                mejor = df_esc.sort_values("_valor_mapa", ascending=False).iloc[0]

                registros.append({
                    "cultivo": cultivo,
                    "valor": round(float(valores.mean()), 1),
                    "clase": clase_valor(valores.mean()),
                    "escenario": mejor.get("_escenario_mapa", ""),
                    "escenario_key": mejor.get("_escenario_key", ""),
                    "estado": estado_obj or "",
                    "municipio": "",
                    "factor": mejor.get("_factor_mapa", "sin_dato"),
                    "factor_legible": nombre_factor_legible(mejor.get("_factor_mapa", "sin_dato")),
                    "valor_base": formato_valor(mejor.get("_valor_base_mapa", 0)),
                    "compatibilidad": formato_valor(mejor.get("_compatibilidad_mapa", "NA")),
                    "municipios_evaluados": int(len(df_esc)),
                    "municipios_75_o_mas": int((valores >= 75).sum()),
                })

    salida = pd.DataFrame(registros)

    if salida.empty:
        return salida

    salida = salida.sort_values(
        ["valor", "municipios_75_o_mas", "cultivo"],
        ascending=[False, False, True]
    )

    return salida.head(top_n).reset_index(drop=True)


def calcular_ranking_local(indice_integrado, info_ubicacion, escenario_key, top_n=12):
    if info_ubicacion is None:
        return pd.DataFrame()

    estado = info_ubicacion.get("estado_aplicado") or info_ubicacion.get("estado_objetivo")
    municipio = info_ubicacion.get("municipio_aplicado") or info_ubicacion.get("municipio_objetivo")

    if not estado and not municipio:
        return pd.DataFrame()

    indice_json = indice_integrado.to_json(orient="records")

    return calcular_ranking_local_cacheado(
        indice_json=indice_json,
        estado_obj=estado,
        municipio_obj=municipio,
        escenario_key=escenario_key,
        top_n=top_n
    )


# ============================================================
# GEOJSON JOIN
# ============================================================

def obtener_gid_feature(feature):
    props = feature.get("properties", {})

    for campo in ["gid_municipio", "GID_2", "GID2", "gid", "id"]:
        if campo in props and valor_no_vacio(props[campo]):
            return str(props[campo])

    estado = props.get("estado", props.get("NAME_1", ""))
    municipio = props.get("municipio", props.get("NAME_2", ""))

    return f"{estado}|{municipio}"


def color_por_fila(fila, modo_color, alpha):
    if modo_color in ["Porcentaje de cumplimiento", "Índice de aptitud"]:
        color_hex = color_escala(fila.get("_valor_mapa", 0))
        return rgba_from_hex(color_hex, alpha)

    if modo_color == "Factor limitante":
        factor = fila.get("_factor_mapa", "sin_dato")
        color_hex = COLORES_FACTOR.get(factor, "#bdbdbd")
        return rgba_from_hex(color_hex, alpha)

    if modo_color == "Mejor opción estimada":
        cultivo = fila.get("_cultivo_mapa", "NA")
        color_hex = color_categoria(cultivo)
        return rgba_from_hex(color_hex, alpha)

    return rgba_from_hex("#bdbdbd", alpha)


@st.cache_data(show_spinner=False)
def construir_geojson_cacheado(geojson_base, df_json, modo_color, alpha):
    df = pd.DataFrame(json.loads(df_json))
    df["_gid_join"] = df.apply(obtener_gid_fila, axis=1)

    lookup = {
        str(row["_gid_join"]): row.to_dict()
        for _, row in df.iterrows()
    }

    features = []

    for feature in geojson_base.get("features", []):
        gid = obtener_gid_feature(feature)

        if gid not in lookup:
            continue

        fila = lookup[gid]
        nueva = dict(feature)
        props = dict(feature.get("properties", {}))

        color = color_por_fila(fila, modo_color, alpha)

        props.update({
            "gid_municipio": fila.get("gid_municipio", props.get("gid_municipio", "NA")),
            "estado": fila.get("estado", props.get("estado", props.get("NAME_1", "NA"))),
            "municipio": fila.get("municipio", props.get("municipio", props.get("NAME_2", "NA"))),
            "cultivo_mapa": fila.get("_cultivo_mapa", "NA"),
            "valor_mapa": formato_valor(fila.get("_valor_mapa", 0)),
            "valor_base_mapa": formato_valor(fila.get("_valor_base_mapa", 0)),
            "clase_mapa": fila.get("_clase_mapa", "NA"),
            "escenario_mapa": fila.get("_escenario_mapa", "NA"),
            "compatibilidad_mapa": formato_valor(fila.get("_compatibilidad_mapa", "NA")),
            "factor_mapa": nombre_factor_legible(fila.get("_factor_mapa", "sin_dato")),
            "factor_base": nombre_factor_legible(fila.get("_factor_base", "sin_dato")),
            "temp_media_c": fila.get("temp_media_c", "NA"),
            "temp_min_c": fila.get("temp_min_c", "NA"),
            "temp_max_c": fila.get("temp_max_c", "NA"),
            "precipitacion_mm": fila.get("precipitacion_mm", "NA"),
            "altitud_m": fila.get("altitud_m", "NA"),
            "fill_color": color,
        })

        nueva["properties"] = props
        features.append(nueva)

    return {
        "type": "FeatureCollection",
        "features": features
    }


# ============================================================
# LEYENDA
# ============================================================

def obtener_items_leyenda(modo_color, df_mapa):
    if df_mapa.empty:
        return []

    items = []

    if modo_color == "Mejor opción estimada":
        conteo = (
            df_mapa["_cultivo_mapa"]
            .fillna("NA")
            .astype(str)
            .value_counts()
        )

        for clave, n in conteo.items():
            items.append({
                "clave": str(clave),
                "etiqueta": str(clave),
                "color": color_categoria(clave),
                "municipios": int(n),
                "tipo": "cultivo"
            })

    elif modo_color == "Factor limitante":
        conteo = (
            df_mapa["_factor_mapa"]
            .fillna("sin_dato")
            .astype(str)
            .value_counts()
        )

        for clave, n in conteo.items():
            items.append({
                "clave": str(clave),
                "etiqueta": nombre_factor_legible(clave),
                "color": COLORES_FACTOR.get(clave, "#bdbdbd"),
                "municipios": int(n),
                "tipo": "factor"
            })

    elif modo_color in ["Porcentaje de cumplimiento", "Índice de aptitud"]:
        valor = pd.to_numeric(df_mapa["_valor_mapa"], errors="coerce").fillna(0)

        items = [
            {
                "clave": "90_100",
                "etiqueta": "90–100: Muy alto",
                "color": "#1a9850",
                "municipios": int((valor >= 90).sum()),
                "tipo": "rango"
            },
            {
                "clave": "75_89",
                "etiqueta": "75–89: Alto",
                "color": "#66bd63",
                "municipios": int(((valor >= 75) & (valor < 90)).sum()),
                "tipo": "rango"
            },
            {
                "clave": "50_74",
                "etiqueta": "50–74: Medio",
                "color": "#fee08b",
                "municipios": int(((valor >= 50) & (valor < 75)).sum()),
                "tipo": "rango"
            },
            {
                "clave": "25_49",
                "etiqueta": "25–49: Bajo",
                "color": "#fdae61",
                "municipios": int(((valor >= 25) & (valor < 50)).sum()),
                "tipo": "rango"
            },
            {
                "clave": "0_24",
                "etiqueta": "0–24: Muy bajo",
                "color": "#d73027",
                "municipios": int((valor < 25).sum()),
                "tipo": "rango"
            },
        ]

    return items


def aplicar_filtro_leyenda(df, modo_color, clave_activa):
    if df.empty:
        return df

    if clave_activa in [None, "", "TODOS"]:
        return df

    df = df.copy()

    if modo_color == "Mejor opción estimada":
        return df[df["_cultivo_mapa"].astype(str) == str(clave_activa)]

    if modo_color == "Factor limitante":
        return df[df["_factor_mapa"].astype(str) == str(clave_activa)]

    if modo_color in ["Porcentaje de cumplimiento", "Índice de aptitud"]:
        valor = pd.to_numeric(df["_valor_mapa"], errors="coerce").fillna(0)

        if clave_activa == "90_100":
            return df[valor >= 90]

        if clave_activa == "75_89":
            return df[(valor >= 75) & (valor < 90)]

        if clave_activa == "50_74":
            return df[(valor >= 50) & (valor < 75)]

        if clave_activa == "25_49":
            return df[(valor >= 25) & (valor < 50)]

        if clave_activa == "0_24":
            return df[valor < 25]

    return df


def render_leyenda_interactiva(modo_color, df_mapa, key_prefix="leyenda"):
    estado_key = f"{key_prefix}_categoria_activa"

    if estado_key not in st.session_state:
        st.session_state[estado_key] = "TODOS"

    items = obtener_items_leyenda(modo_color, df_mapa)

    claves_validas = ["TODOS"] + [item["clave"] for item in items]

    if st.session_state[estado_key] not in claves_validas:
        st.session_state[estado_key] = "TODOS"

    st.markdown("### Leyenda")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button("Mostrar todo", key=f"{key_prefix}_mostrar_todo"):
            st.session_state[estado_key] = "TODOS"
            safe_rerun()

    with col_b:
        activo = st.session_state.get(estado_key, "TODOS")

        if activo == "TODOS":
            st.caption("Filtro: todos")
        else:
            etiqueta_activa = activo

            for item in items:
                if item["clave"] == activo:
                    etiqueta_activa = item["etiqueta"]
                    break

            st.caption(f"Filtro: {etiqueta_activa}")

    try:
        contenedor = st.container(height=430, border=True)
    except TypeError:
        contenedor = st.expander("Ver leyenda completa", expanded=True)

    with contenedor:
        for idx, item in enumerate(items):
            clave = item["clave"]
            etiqueta = item["etiqueta"]
            color = item["color"]
            municipios = item["municipios"]

            activo = st.session_state.get(estado_key, "TODOS") == clave
            borde = "3px solid #111" if activo else "1px solid #999"

            c1, c2 = st.columns([0.16, 0.84])

            with c1:
                st.markdown(
                    f"""
                    <div style="
                        width:20px;
                        height:20px;
                        background:{color};
                        border:{borde};
                        margin-top:8px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                texto_boton = f"{etiqueta} · {municipios}"

                if st.button(
                    texto_boton,
                    key=f"{key_prefix}_item_{idx}_{normalizar_columna(clave)}",
                    width="stretch"
                ):
                    if st.session_state[estado_key] == clave:
                        st.session_state[estado_key] = "TODOS"
                    else:
                        st.session_state[estado_key] = clave

                    safe_rerun()

    return st.session_state.get(estado_key, "TODOS")


# ============================================================
# COMPARACIÓN DE SISTEMAS
# ============================================================

def render_matriz_sistemas(df_comp):
    if df_comp is None or df_comp.empty:
        st.info("No hay datos suficientes para comparar sistemas en este cultivo.")
        return

    st.markdown("### Comparación de sistemas")

    for _, row in df_comp.iterrows():
        sistema = row["sistema"]
        valor = float(row["aptitud_promedio"])
        color = color_escala(valor)

        st.markdown(
            f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                    <span>{sistema}</span>
                    <b>{valor:.1f}</b>
                </div>
                <div style="height:14px; background:#eeeeee; border-radius:7px; overflow:hidden;">
                    <div style="width:{max(0, min(valor, 100))}%;
                                background:{color};
                                height:14px;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def es_pregunta_comparacion_sistemas(pregunta):
    pregunta_norm = normalizar_texto(pregunta)

    patrones = [
        "comparacion de sistemas",
        "comparación de sistemas",
        "como interpreto la comparacion",
        "cómo interpreto la comparación",
        "explica la comparacion",
        "explica la comparación",
        "que significa aptitud promedio",
        "qué significa aptitud promedio",
        "que significa compatibilidad promedio",
        "qué significa compatibilidad promedio",
        "municipios 75",
        "barra de sistemas",
        "barras de sistemas",
        "como leo los sistemas",
        "cómo leo los sistemas",
        "que sistema conviene",
        "qué sistema conviene",
        "mejor sistema",
        "sistema recomendado",
    ]

    return any(normalizar_texto(p) in pregunta_norm for p in patrones)


def respuesta_comparacion_sistemas(comparacion_sistemas, cultivo_label):
    cultivo_txt = cultivo_label or "el cultivo seleccionado"

    if comparacion_sistemas is None or comparacion_sistemas.empty:
        return (
            "La comparación de sistemas se interpreta en la vista de **Índice de aptitud + Un cultivo**. "
            "No pertenece al cumplimiento base del PDF, porque el PDF solo contiene requerimientos "
            "agroecológicos del cultivo, no escenarios urbanos como azotea, invernadero o cultivo interior controlado con luz artificial.\n\n"
            "Para verla correctamente, selecciona:\n\n"
            "- **Tipo de lectura:** Índice de aptitud\n"
            "- **Vista:** Un cultivo\n"
            "- **Escenario:** Todos los sistemas / mejor escenario\n\n"
            "Cuando esté disponible, debes leerla así:\n\n"
            "- **Aptitud promedio:** desempeño general del cultivo bajo cada sistema.\n"
            "- **Municipios con 75 o más:** cuántos municipios alcanzan aptitud alta.\n"
            "- **Compatibilidad promedio:** qué tan adecuado es el sistema para el porte, raíz, peso, "
            "sustrato, luz, riego o control ambiental que requiere el cultivo."
        )

    df = comparacion_sistemas.copy()
    df = df.sort_values("aptitud_promedio", ascending=False)

    mejor = df.iloc[0]
    mejor_sistema = mejor["sistema"]
    mejor_valor = mejor["aptitud_promedio"]
    mejor_m75 = mejor.get("municipios_75_o_mas", "NA")
    mejor_compat = mejor.get("compatibilidad_promedio", "NA")

    texto = (
        f"La comparación de sistemas te ayuda a entender qué escenario de cultivo urbano funciona mejor "
        f"para **{cultivo_txt}**.\n\n"
        f"En la selección actual, el sistema mejor posicionado es **{mejor_sistema}**, con una aptitud "
        f"promedio de **{mejor_valor}**. También tiene **{mejor_m75} municipios** con valor de 75 o más"
    )

    try:
        if pd.notna(mejor_compat):
            texto += f" y una compatibilidad promedio de **{mejor_compat}**."
        else:
            texto += "."
    except Exception:
        texto += "."

    texto += (
        "\n\nCómo leer la comparación:\n\n"
        "- **Aptitud promedio:** desempeño general del cultivo bajo cada sistema.\n"
        "- **Municipios con 75 o más:** cobertura territorial alta.\n"
        "- **Compatibilidad promedio:** adecuación del sistema para porte, raíz, peso, sustrato, luz, riego o control ambiental.\n\n"
        "El mejor sistema promedio no siempre es el mejor para todos los municipios. Para una decisión local, revisa el municipio específico."
    )

    return texto


# ============================================================
# RESPUESTAS FIJAS
# ============================================================

def es_pregunta_identidad(pregunta):
    pregunta_norm = normalizar_texto(pregunta)

    patrones = [
        "quien eres",
        "quién eres",
        "que eres",
        "qué eres",
        "como te llamas",
        "cómo te llamas",
        "quien te hizo",
        "quién te hizo",
        "quien te desarrollo",
        "quién te desarrolló",
        "quien desarrollo esta app",
        "quién desarrolló esta app",
        "quien creo esta herramienta",
        "quién creó esta herramienta",
        "quien hizo este mapa",
        "quién hizo este mapa",
        "about you",
        "who are you",
    ]

    return any(normalizar_texto(p) in pregunta_norm for p in patrones)


def respuesta_identidad_app():
    return APP_IDENTIDAD.strip()


def es_pregunta_azotea_intensiva_extensiva(pregunta):
    pregunta_norm = normalizar_texto(pregunta)

    return (
        "azotea intensiva" in pregunta_norm
        and "azotea extensiva" in pregunta_norm
    ) or (
        "intensiva" in pregunta_norm
        and "extensiva" in pregunta_norm
        and "azotea" in pregunta_norm
    )


def respuesta_diferencia_azoteas(cultivo_label=None):
    cultivo_txt = cultivo_label or "cultivos comestibles"

    return (
        "**Azotea extensiva** y **azotea intensiva** no son lo mismo.\n\n"
        "**Azotea extensiva:** es más ligera, usa poco sustrato, requiere menos mantenimiento "
        "y normalmente funciona mejor para vegetación resistente, cubresuelos o plantas de bajo porte. "
        "Tiene menos profundidad para raíces y menor capacidad de retener agua.\n\n"
        "**Azotea intensiva:** se parece más a un huerto en contenedores, jardineras o camas de cultivo. "
        "Usa más sustrato, requiere más riego y mantenimiento, pero permite cultivar mejor plantas "
        "comestibles porque hay más profundidad, más control y mejor soporte.\n\n"
        f"Para **{cultivo_txt}**, normalmente conviene más un sistema tipo **azotea intensiva** "
        "o contenedores con buen drenaje. Antes de instalar camas pesadas en una azotea, conviene revisar la capacidad estructural."
    )


def respuesta_pedir_ubicacion(cultivo_label=None):
    cultivo_txt = cultivo_label or "ese cultivo"

    return (
        f"Sí puedo ayudarte con **{cultivo_txt}**, pero para darte una recomendación local necesito saber "
        "el **municipio y estado** del lugar donde quieres cultivar.\n\n"
        "No necesitas conocer la altitud, temperatura o precipitación: la app traduce esos datos automáticamente "
        "a partir del municipio.\n\n"
        "Puedes responder, por ejemplo:\n\n"
        "- `Estoy en Monterrey, Nuevo León`\n"
        "- `Vivo en Zapopan, Jalisco`\n"
        "- `Es cerca del Campus Querétaro, Querétaro`\n\n"
        "Con eso puedo decirte si la aptitud aparece alta, media o baja y qué sistema conviene más."
    )


def respuesta_recomendacion_local(ranking_local, info_ubicacion, escenario_key, pregunta=None):
    if ranking_local is None or ranking_local.empty:
        return (
            "Detecté una ubicación, pero no pude construir un ranking local con los datos disponibles. "
            "Puedes probar cambiando el escenario o revisando que los archivos precalculados estén actualizados."
        )

    estado = info_ubicacion.get("estado_aplicado") or info_ubicacion.get("estado_objetivo") or ""
    municipio = info_ubicacion.get("municipio_aplicado") or info_ubicacion.get("municipio_objetivo") or ""
    lugar = info_ubicacion.get("lugar_mencionado") or ""

    partes = []

    if municipio:
        partes.append(str(municipio))

    if estado:
        partes.append(str(estado))

    ubicacion_txt = ", ".join(partes) if partes else "la ubicación indicada"

    if lugar:
        encabezado = f"Para **{lugar}** ({ubicacion_txt}), el ranking local del modelo muestra estas opciones mejor posicionadas:"
    else:
        encabezado = f"Para **{ubicacion_txt}**, el ranking local del modelo muestra estas opciones mejor posicionadas:"

    top = ranking_local.head(6).copy()

    lineas = []

    for i, row in top.iterrows():
        cultivo = row.get("cultivo", "NA")
        valor = row.get("valor", "NA")
        clase = row.get("clase", clase_valor(valor))
        escenario = row.get("escenario", ESCENARIOS.get(escenario_key, escenario_key))
        factor = row.get("factor_legible", nombre_factor_legible(row.get("factor", "sin_dato")))

        lineas.append(
            f"{i + 1}. **{cultivo}** — aptitud {valor} ({clase}); sistema/escenario: {escenario}; limitante principal: {factor}."
        )

    texto = encabezado + "\n\n" + "\n".join(lineas)

    texto += (
        "\n\nEsto no significa que sean los únicos cultivos posibles. Significa que, dentro del modelo, "
        "son los mejor posicionados para esa ubicación y escenario. La lectura es municipal, así que para una decisión puntual "
        "todavía conviene revisar sombra, riego, suelo o sustrato, exposición solar y, si es azotea, capacidad estructural.\n\n"
        f"Fuente base de requerimientos agroecológicos: {PDF_CITA_BASE}"
    )

    if pregunta and es_pregunta_azotea_intensiva_extensiva(pregunta):
        texto = respuesta_diferencia_azoteas(top.iloc[0].get("cultivo", None)) + "\n\n" + texto

    return texto


# ============================================================
# CHAT / LLM
# ============================================================

def inicializar_estado():
    defaults = {
        "chat_integrado_historial": [],
        "ultima_intencion_integrada": None,
        "ultima_pregunta_integrada": None,
        "respuesta_pendiente_integrada": False,
        "lectura_integrada": "aptitud",
        "vista_integrada": "un_cultivo",
        "cultivo_integrado": None,
        "escenario_integrado": "solo",
        "color_integrado": "Índice de aptitud",
        "fuentes_web_ultima": [],

        "usar_filtro_ubicacion_integrada": False,
        "estado_objetivo_integrado": None,
        "municipio_objetivo_integrado": None,
        "lugar_mencionado_integrado": None,
        "requiere_ubicacion_integrada": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def buscar_cultivo_en_lista(cultivos, texto_busqueda):
    if texto_busqueda is None:
        return None

    texto_norm = normalizar_texto(texto_busqueda)

    if not texto_norm:
        return None

    for nombre in cultivos:
        if normalizar_texto(nombre) == texto_norm:
            return nombre

    for nombre in cultivos:
        nombre_norm = normalizar_texto(nombre)
        if texto_norm in nombre_norm:
            return nombre

    for nombre in cultivos:
        nombre_norm = normalizar_texto(nombre)
        if nombre_norm and nombre_norm in texto_norm:
            return nombre

    return None


def detectar_lectura_pregunta(pregunta, intencion=None):
    pregunta_norm = normalizar_texto(pregunta)

    if any(p in pregunta_norm for p in ["pdf", "crudo", "cumplimiento", "rango original", "rangos originales"]):
        return "pdf"

    if intencion:
        modelo = intencion.get("modelo", "")

        if modelo == "pdf_crudo":
            return "pdf"

        if modelo in ["fuzzy_base", "urbano"]:
            return "aptitud"

    return "aptitud"


def detectar_vista_pregunta(pregunta, cultivo_detectado):
    pregunta_norm = normalizar_texto(pregunta)

    if any(p in pregunta_norm for p in [
        "que puedo cultivar",
        "qué puedo cultivar",
        "que puede cultivarse",
        "qué puede cultivarse",
        "que se puede cultivar",
        "qué se puede cultivar",
        "que cultivos",
        "qué cultivos",
        "cultivos recomiendas",
        "cultivo recomiendas",
        "mejor cultivo",
        "todos los cultivos",
        "cultivos por municipio",
        "opciones de cultivo",
    ]):
        return "todos_cultivos"

    if cultivo_detectado:
        return "un_cultivo"

    return "todos_cultivos"


def detectar_escenario_pregunta(pregunta, intencion=None):
    if intencion:
        sistema = intencion.get("sistema")

        if sistema in ESCENARIOS:
            if sistema == "ninguno":
                return "solo"
            return sistema

    pregunta_norm = normalizar_texto(pregunta)

    if "sistema" in pregunta_norm and any(p in pregunta_norm for p in ["recomienda", "recomiendas", "mejor", "conviene"]):
        return "todos_sistemas"

    for escenario_key, alias in ALIAS_SISTEMAS.items():
        for palabra in alias:
            if normalizar_texto(palabra) in pregunta_norm:
                return escenario_key

    return "solo"


def detectar_color_pregunta(pregunta, lectura, vista):
    pregunta_norm = normalizar_texto(pregunta)

    if "limitante" in pregunta_norm or "factor" in pregunta_norm:
        return "Factor limitante"

    if vista == "todos_cultivos" and any(p in pregunta_norm for p in [
        "que puedo cultivar",
        "qué puedo cultivar",
        "que puede cultivarse",
        "qué puede cultivarse",
        "que se puede cultivar",
        "qué se puede cultivar",
        "mejor cultivo",
        "recomienda",
    ]):
        return "Mejor opción estimada"

    if lectura == "pdf":
        return "Porcentaje de cumplimiento"

    return "Índice de aptitud"


def procesar_pregunta_chat(pregunta, cultivos_disponibles):
    st.session_state["requiere_ubicacion_integrada"] = False

    if es_pregunta_identidad(pregunta):
        intencion = {
            "accion": "pregunta_identidad",
            "cultivo": None,
            "cultivo_2": None,
            "estado": None,
            "municipio": None,
            "modelo": "desconocido",
            "sistema": "ninguno",
            "usar_ia_complementaria": False,
            "necesita_mapa": False,
            "necesita_explicacion": True,
        }

        st.session_state["ultima_intencion_integrada"] = intencion
        st.session_state["ultima_pregunta_integrada"] = pregunta
        st.session_state["respuesta_pendiente_integrada"] = True
        st.session_state["chat_integrado_historial"].append({"role": "user", "content": pregunta})
        return

    if LLM_DISPONIBLE:
        try:
            intencion = interpretar_pregunta_usuario(pregunta)
        except Exception as e:
            intencion = {}
            intencion["error_llm"] = str(e)
    else:
        intencion = {}

    guardar_ubicacion_desde_intencion(pregunta, intencion)

    if consulta_sitio_sin_ubicacion(pregunta, intencion):
        st.session_state["requiere_ubicacion_integrada"] = True

    cultivo_detectado = intencion.get("cultivo")
    cultivo_encontrado = buscar_cultivo_en_lista(cultivos_disponibles, cultivo_detectado)

    if cultivo_encontrado is None:
        cultivo_encontrado = buscar_cultivo_en_lista(cultivos_disponibles, pregunta)

    if es_pregunta_comparacion_sistemas(pregunta):
        lectura = "aptitud"
        vista = "un_cultivo"
        escenario = "todos_sistemas"
        color = "Índice de aptitud"

        if cultivo_encontrado is None:
            cultivo_actual = st.session_state.get("cultivo_integrado", None)

            if cultivo_actual in cultivos_disponibles:
                cultivo_encontrado = cultivo_actual

        intencion["accion"] = "comparar_sistemas_cultivo"

    else:
        lectura = detectar_lectura_pregunta(pregunta, intencion)
        vista = detectar_vista_pregunta(pregunta, cultivo_encontrado)
        escenario = detectar_escenario_pregunta(pregunta, intencion)
        color = detectar_color_pregunta(pregunta, lectura, vista)

        if consulta_pide_recomendacion_local(pregunta, intencion):
            lectura = "aptitud"
            vista = "todos_cultivos"
            color = "Mejor opción estimada"

            if escenario == "solo":
                escenario = "todos_sistemas"

    intencion.update({
        "lectura": lectura,
        "vista": vista,
        "sistema": escenario,
        "modo_color": color,
        "cultivo": cultivo_encontrado,
    })

    st.session_state["lectura_integrada"] = lectura
    st.session_state["vista_integrada"] = vista
    st.session_state["escenario_integrado"] = escenario
    st.session_state["color_integrado"] = color

    if cultivo_encontrado is not None:
        st.session_state["cultivo_integrado"] = cultivo_encontrado

    pregunta_norm = normalizar_texto(pregunta)

    if any(p in pregunta_norm for p in ["internet", "web", "busca", "enriquece", "complementa", "ia", "fuente", "fuentes"]):
        intencion["usar_ia_complementaria"] = True

    st.session_state["ultima_intencion_integrada"] = intencion
    st.session_state["ultima_pregunta_integrada"] = pregunta
    st.session_state["respuesta_pendiente_integrada"] = True
    st.session_state["chat_integrado_historial"].append({"role": "user", "content": pregunta})


def factores_frecuentes(df):
    if df.empty or "_factor_mapa" not in df.columns:
        return []

    conteo = (
        df["_factor_mapa"]
        .fillna("sin_dato")
        .value_counts()
        .head(5)
    )

    return [
        {
            "factor": str(factor),
            "factor_legible": nombre_factor_legible(str(factor)),
            "municipios": int(valor),
        }
        for factor, valor in conteo.items()
    ]


def cultivos_frecuentes(df):
    if df.empty or "_cultivo_mapa" not in df.columns:
        return []

    conteo = (
        df["_cultivo_mapa"]
        .fillna("NA")
        .value_counts()
        .head(6)
    )

    return [
        {
            "cultivo": str(cultivo),
            "municipios": int(valor),
        }
        for cultivo, valor in conteo.items()
    ]


def extraer_fuentes_web(info_web):
    fuentes = []

    if not isinstance(info_web, dict):
        return fuentes

    for f in info_web.get("fuentes", []):
        if not isinstance(f, dict):
            continue

        titulo = f.get("titulo") or f.get("title") or "Fuente"
        url = f.get("url") or f.get("link") or ""
        dato = f.get("dato_usado") or f.get("dato") or ""
        institucion = f.get("institucion") or ""
        pais = f.get("pais") or ""
        tipo = f.get("tipo_fuente") or ""

        if url:
            fuentes.append({
                "titulo": titulo,
                "url": url,
                "dato": dato,
                "institucion": institucion,
                "pais": pais,
                "tipo": tipo,
            })

    annotations = info_web.get("_openrouter_annotations", [])

    if isinstance(annotations, list):
        for a in annotations:
            if not isinstance(a, dict):
                continue

            url = a.get("url") or a.get("source_url") or ""
            titulo = a.get("title") or a.get("source_title") or "Fuente consultada"

            if url:
                fuentes.append({
                    "titulo": titulo,
                    "url": url,
                    "dato": "",
                    "institucion": "",
                    "pais": "",
                    "tipo": "",
                })

    salida = []
    vistos = set()

    for f in fuentes:
        url = f.get("url", "")

        if url and url not in vistos:
            salida.append(f)
            vistos.add(url)

    return salida


def construir_contexto_llm(
    pregunta,
    intencion,
    lectura,
    vista,
    cultivo_label,
    escenario_key,
    modo_color,
    df_mapa,
    df_filtrado,
    comparacion_sistemas=None,
    ranking_local=None,
    info_ubicacion=None,
    usar_web_ia=False,
):
    df_contexto = df_filtrado.copy()

    if df_contexto.empty:
        df_contexto = df_mapa.copy()

    df_contexto = deduplicar_municipios_para_mapa(df_contexto)
    df_filtrado_unico = deduplicar_municipios_para_mapa(df_filtrado)

    total = len(df_contexto)
    promedio = round(df_contexto["_valor_mapa"].mean(), 1) if total else 0
    n75 = int((df_contexto["_valor_mapa"] >= 75).sum()) if total else 0
    n90 = int((df_contexto["_valor_mapa"] >= 90).sum()) if total else 0

    df_top = df_contexto.sort_values("_valor_mapa", ascending=False).head(8)

    columnas_top = [
        "estado",
        "municipio",
        "_cultivo_mapa",
        "_valor_mapa",
        "_valor_base_mapa",
        "_compatibilidad_mapa",
        "_escenario_mapa",
        "_clase_mapa",
        "_factor_mapa",
        "_factor_base",
        "temp_media_c",
        "temp_min_c",
        "temp_max_c",
        "precipitacion_mm",
        "altitud_m",
    ]

    columnas_top = [c for c in columnas_top if c in df_top.columns]
    top = df_top[columnas_top].copy()

    if "_factor_mapa" in top.columns:
        top["_factor_mapa"] = top["_factor_mapa"].apply(nombre_factor_legible)

    if "_factor_base" in top.columns:
        top["_factor_base"] = top["_factor_base"].apply(nombre_factor_legible)

    contexto = {
        "pregunta_usuario": pregunta,
        "lectura_actual": LECTURAS.get(lectura, lectura),
        "vista_actual": VISTAS.get(vista, vista),
        "cultivo_mostrado": cultivo_label if vista == "un_cultivo" else "Todos los cultivos",
        "escenario_mostrado": ESCENARIOS.get(escenario_key, escenario_key),
        "modo_color": modo_color,
        "fuente_documental_base": PDF_CITA_BASE,
        "accion_detectada": intencion.get("accion") if intencion else None,
        "ubicacion_aplicada": info_ubicacion or None,
        "ranking_local": (
            ranking_local.to_dict(orient="records")
            if ranking_local is not None and not ranking_local.empty
            else []
        ),
        "resumen": {
            "municipios_evaluados": total,
            "valor_promedio": promedio,
            "municipios_con_75_o_mas": n75,
            "municipios_con_90_o_mas": n90,
            "municipios_mostrados_con_filtros": len(df_filtrado_unico),
        },
        "top_municipios": top.to_dict(orient="records"),
        "factores_mas_frecuentes": factores_frecuentes(df_contexto),
        "cultivos_mas_recomendados_por_municipio": cultivos_frecuentes(df_contexto),
        "comparacion_sistemas": (
            comparacion_sistemas.to_dict(orient="records")
            if comparacion_sistemas is not None and not comparacion_sistemas.empty
            else []
        ),
        "reglas_de_respuesta": [
            "Si existe ranking_local, responder usando ranking_local y no sustituir esos cultivos por otros.",
            "No agregar cultivos que no estén en ranking_local cuando la pregunta sea local.",
            "No escribir una sección llamada Nota metodológica.",
            "No usar la palabra fuzzy.",
            "No decir 'de acuerdo al PDF' al inicio.",
            "Cuando hables de la fuente base, mencionar de forma natural que los requerimientos agroecológicos base provienen de Ruiz Corral et al. (2020).",
            "No presentar el resultado como garantía de producción o rendimiento.",
        ],
    }

    fuentes = []

    if usar_web_ia and LLM_DISPONIBLE and vista == "un_cultivo" and cultivo_label:
        try:
            info_web = buscar_requerimientos_web_cultivo(cultivo_label)
            contexto["ia_web_complementaria"] = info_web
            fuentes = extraer_fuentes_web(info_web)
        except Exception as e:
            contexto["ia_web_complementaria"] = {"error": str(e)}

    return contexto, fuentes


def redactar_respuesta_integrada(pregunta_usuario, contexto):
    prompt = """
Eres una asistente dentro de una app de mapas agrícolas urbanos.

Responde en español natural, humano y práctico.
Sólo si te preguntan en inglés, responde en inglés.
No uses una sección llamada "Nota metodológica".
No uses la palabra "fuzzy".
No empieces con "De acuerdo al PDF" ni "Según el dataset".

Primero contesta la pregunta del usuario.
Después explica brevemente qué está mostrando el mapa.

Regla obligatoria:
- Si el contexto incluye ranking_local y la pregunta es local, responde SOLO con los cultivos de ranking_local.
- No agregues cultivos que no estén en ranking_local.
- No reemplaces el ranking local por un resumen nacional.

Si el usuario pregunta quién eres, responde que eres un asistente de apoyo para interpretar mapas de cultivos, aptitud agroecológica y escenarios de cultivo urbano en México. Di que la herramienta fue desarrollada por la Dra. Juana Isabel Méndez. Si necesitan más apoyo, pueden escribir a isabelmendez@tec.mx.

Cuando menciones la fuente documental base, cita de manera breve: Ruiz Corral et al. (2020), Requerimientos agroecológicos de cultivos, 2da edición.

Si el usuario pregunta por azotea, cultivo interior controlado con luz artificial o invernadero, no trates precipitación como lluvia directa del sitio: interprétala como manejo hídrico, riego o condición controlada cuando el contexto lo indique.
No inventes datos que no estén en el contexto.
No presentes el resultado como rendimiento garantizado.
"""

    if not LLM_DISPONIBLE:
        raise RuntimeError("LLM no disponible.")

    msg = llamar_llm(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Pregunta del usuario:\n{pregunta_usuario}\n\n"
                    "Contexto calculado por la app:\n"
                    f"{json.dumps(contexto, ensure_ascii=False, indent=2)}"
                )
            }
        ],
        reasoning=False,
        temperature=0.25,
        usar_web=False,
        json_mode=False,
    )

    return msg.get("content", "").strip()


def respuesta_fallback(contexto):
    lectura = contexto.get("lectura_actual", "la lectura actual")
    vista = contexto.get("vista_actual", "")
    cultivo = contexto.get("cultivo_mostrado", "")
    escenario = contexto.get("escenario_mostrado", "")
    resumen = contexto.get("resumen", {})
    top = contexto.get("top_municipios", [])
    factores = contexto.get("factores_mas_frecuentes", [])
    cultivos = contexto.get("cultivos_mas_recomendados_por_municipio", [])
    ranking_local = contexto.get("ranking_local", [])

    if ranking_local:
        lineas = []

        for i, row in enumerate(ranking_local[:6]):
            lineas.append(
                f"{i + 1}. **{row.get('cultivo', 'NA')}** — aptitud {row.get('valor', 'NA')} "
                f"({row.get('clase', 'NA')}); escenario: {row.get('escenario', 'NA')}."
            )

        return (
            "Para la ubicación indicada, las opciones mejor posicionadas en el modelo son:\n\n"
            + "\n".join(lineas)
            + f"\n\nFuente base de requerimientos agroecológicos: {PDF_CITA_BASE}"
        )

    promedio = resumen.get("valor_promedio", 0)
    n75 = resumen.get("municipios_con_75_o_mas", 0)
    total = resumen.get("municipios_evaluados", 0)

    texto = (
        f"El mapa muestra {lectura.lower()} para {cultivo}, "
        f"bajo el escenario: {escenario}. El valor promedio es {promedio}, "
        f"y {n75} de {total} municipios alcanzan 75 o más.\n\n"
    )

    if vista == "Todos los cultivos" and cultivos:
        opciones = ", ".join([c["cultivo"] for c in cultivos[:5]])
        texto += f"Las opciones que más aparecen como mejor estimación son: {opciones}.\n\n"

    if top:
        mejores = [
            f"{m.get('municipio', 'NA')}, {m.get('estado', 'NA')} ({m.get('_valor_mapa', 'NA')})"
            for m in top[:5]
        ]
        texto += "Los municipios mejor posicionados son: " + "; ".join(mejores) + ".\n\n"

    if factores:
        factores_txt = ", ".join([f["factor_legible"] for f in factores[:4]])
        texto += f"Los factores que más limitan el resultado son: {factores_txt}."

    texto += f"\n\nFuente base de requerimientos agroecológicos: {PDF_CITA_BASE}"

    return texto


def generar_respuesta_chat_si_pendiente(
    lectura,
    vista,
    cultivo_label,
    escenario_key,
    modo_color,
    df_mapa,
    df_filtrado,
    comparacion_sistemas,
    ranking_local,
    info_ubicacion,
    usar_web_ia=False,
):
    if not st.session_state.get("respuesta_pendiente_integrada", False):
        return

    pregunta = st.session_state.get("ultima_pregunta_integrada")
    intencion = st.session_state.get("ultima_intencion_integrada") or {}

    if es_pregunta_identidad(pregunta):
        st.session_state["chat_integrado_historial"].append({
            "role": "assistant",
            "content": respuesta_identidad_app()
        })
        st.session_state["fuentes_web_ultima"] = []
        st.session_state["respuesta_pendiente_integrada"] = False
        return

    if st.session_state.get("requiere_ubicacion_integrada", False):
        cultivo_detectado = cultivo_label or intencion.get("cultivo") or "ese cultivo"
        respuesta = ""

        if es_pregunta_azotea_intensiva_extensiva(pregunta):
            respuesta += respuesta_diferencia_azoteas(cultivo_detectado)
            respuesta += "\n\n"

        respuesta += respuesta_pedir_ubicacion(cultivo_detectado)

        st.session_state["chat_integrado_historial"].append({
            "role": "assistant",
            "content": respuesta
        })
        st.session_state["fuentes_web_ultima"] = []
        st.session_state["respuesta_pendiente_integrada"] = False
        return

    if es_pregunta_comparacion_sistemas(pregunta):
        st.session_state["chat_integrado_historial"].append({
            "role": "assistant",
            "content": respuesta_comparacion_sistemas(comparacion_sistemas, cultivo_label)
        })
        st.session_state["fuentes_web_ultima"] = []
        st.session_state["respuesta_pendiente_integrada"] = False
        return

    if info_ubicacion is not None and ranking_local is not None and not ranking_local.empty and consulta_pide_recomendacion_local(pregunta, intencion):
        respuesta = respuesta_recomendacion_local(
            ranking_local=ranking_local,
            info_ubicacion=info_ubicacion,
            escenario_key=escenario_key,
            pregunta=pregunta,
        )

        st.session_state["chat_integrado_historial"].append({
            "role": "assistant",
            "content": respuesta
        })
        st.session_state["fuentes_web_ultima"] = []
        st.session_state["respuesta_pendiente_integrada"] = False
        return

    contexto, fuentes = construir_contexto_llm(
        pregunta=pregunta,
        intencion=intencion,
        lectura=lectura,
        vista=vista,
        cultivo_label=cultivo_label,
        escenario_key=escenario_key,
        modo_color=modo_color,
        df_mapa=df_mapa,
        df_filtrado=df_filtrado,
        comparacion_sistemas=comparacion_sistemas,
        ranking_local=ranking_local,
        info_ubicacion=info_ubicacion,
        usar_web_ia=usar_web_ia,
    )

    try:
        respuesta = redactar_respuesta_integrada(
            pregunta_usuario=pregunta,
            contexto=contexto
        )
    except Exception as e:
        respuesta = respuesta_fallback(contexto)
        respuesta += f"\n\nNo pude usar la redacción con IA en esta ejecución: {e}"

    st.session_state["chat_integrado_historial"].append({
        "role": "assistant",
        "content": respuesta
    })

    st.session_state["fuentes_web_ultima"] = fuentes
    st.session_state["respuesta_pendiente_integrada"] = False


def render_chat_historial():
    historial = st.session_state.get("chat_integrado_historial", [])

    if not historial:
        return

    st.markdown("### Respuesta del mapa")

    for mensaje in historial[-4:]:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    fuentes = st.session_state.get("fuentes_web_ultima", [])

    if fuentes:
        with st.expander("Fuentes consultadas", expanded=True):
            for fuente in fuentes:
                titulo = fuente.get("titulo", "Fuente")
                url = fuente.get("url", "")
                dato = fuente.get("dato", "")
                institucion = fuente.get("institucion", "")
                pais = fuente.get("pais", "")
                tipo = fuente.get("tipo", "")
                detalle = " — ".join([x for x in [institucion, pais, tipo, dato] if x])

                if url:
                    if detalle:
                        st.markdown(f"- [{titulo}]({url}) — {detalle}")
                    else:
                        st.markdown(f"- [{titulo}]({url})")


# ============================================================
# INICIO
# ============================================================

inicializar_estado()

st.title("Mapa integrado de cultivos")

st.markdown(
    f"""
    <div style="
        margin-top:-8px;
        margin-bottom:8px;
        color:#5f6368;
        font-size:0.95rem;">
        Desarrollado por <b>{APP_DESARROLLADORA}</b> · 
        <a href="mailto:{APP_CONTACTO}" style="text-decoration:none;">
            {APP_CONTACTO}
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Herramienta para explorar el cumplimiento base de requerimientos agroecológicos "
    " conforme a lo reportado por Ruiz Corral et al. (2020) "
    "y el índice de aptitud de cultivos por municipio y sistema de cultivo urbano."
)

indice_integrado = cargar_indice_integrado()
indice_pdf = cargar_indice_pdf()

if indice_integrado.empty:
    st.error(
        "No encontré `data/mapas_integrados/indice_mapas_integrados.csv`. "
        "Primero corre `python A_precalculo_cultivos.py --limpiar`."
    )
    st.stop()

ruta_geojson = resolver_geojson()

if ruta_geojson is None:
    st.error(
        "No encontré un GeoJSON municipal en `data/mapas_municipios`. "
        "Primero corre `07b_precalcular_mapa_municipios_mexico.py --sistema todos`."
    )
    st.stop()

geojson_base = cargar_geojson(str(ruta_geojson))

cultivos_pdf = obtener_cultivos_disponibles(indice_integrado, indice_pdf, "pdf")
cultivos_aptitud = obtener_cultivos_disponibles(indice_integrado, indice_pdf, "aptitud")
cultivos_todos = sorted(list(set(cultivos_pdf + cultivos_aptitud)))


# ============================================================
# CHAT
# ============================================================

pregunta_usuario = st.chat_input(
    "Pregúntale al mapa. Ejemplo: ¿Qué puedo cultivar en una azotea intensiva? / Compara sistemas para lenteja"
)

if pregunta_usuario:
    with st.spinner("Interpretando tu pregunta..."):
        procesar_pregunta_chat(
            pregunta=pregunta_usuario,
            cultivos_disponibles=cultivos_todos,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuración del mapa")

if not LLM_DISPONIBLE:
    with st.sidebar.expander("Estado IA", expanded=True):
        st.warning(
            "La app funciona, pero la IA no está conectada. "
            "Revisa `llm_openrouter.py` y `.streamlit/secrets.toml`."
        )
        st.code(LLM_ERROR_IMPORT or "Sin detalle.")
else:
    with st.sidebar.expander("Estado IA", expanded=False):
        st.success("LLM conectado.")
        st.write("La IA interpreta la pregunta y redacta respuestas usando los datos del mapa.")

usar_web_ia = st.sidebar.checkbox(
    "Buscar información complementaria en web",
    value=False,
    help=(
        "Si está activo, la IA puede buscar fuentes externas sobre el cultivo. "
        "Esto no cambia el mapa; solo complementa la respuesta y muestra fuentes."
    )
)

with st.sidebar.expander("Fuente base", expanded=False):
    st.markdown(PDF_CITA_BASE)
    st.caption(
        "Los rangos agroecológicos normalizados de los cultivos se basan en esta fuente documental. "
        "El índice de aptitud, la comparación de sistemas y los escenarios de cultivo urbano "
        "son capas de interpretación calculadas posteriormente."
    )

with st.sidebar.expander("Acerca de la herramienta", expanded=False):
    st.markdown(f"**Desarrollado por:** {APP_DESARROLLADORA}")
    st.markdown(f"**Contacto:** {APP_CONTACTO}")
    st.markdown(
        "Esta app integra datos municipales, requerimientos agroecológicos de cultivos "
        "y escenarios de cultivo urbano para apoyar la interpretación territorial de aptitud."
    )
    st.markdown(
        "Los requerimientos agroecológicos base de los cultivos fueron normalizados "
        "a partir de Ruiz Corral et al. (2020)."
    )
lecturas_disponibles = []

if not indice_pdf.empty:
    lecturas_disponibles.append("pdf")

if not indice_integrado[indice_integrado["lectura"] == "aptitud"].empty:
    lecturas_disponibles.append("aptitud")

if st.session_state["lectura_integrada"] not in lecturas_disponibles:
    st.session_state["lectura_integrada"] = lecturas_disponibles[0]

lectura = st.sidebar.radio(
    "Tipo de lectura",
    lecturas_disponibles,
    format_func=lambda k: LECTURAS[k],
    key="lectura_integrada",
)

vista = st.sidebar.radio(
    "Vista",
    ["un_cultivo", "todos_cultivos"],
    format_func=lambda k: VISTAS[k],
    key="vista_integrada",
)

cultivos_labels = obtener_cultivos_disponibles(indice_integrado, indice_pdf, lectura)

if not cultivos_labels and vista == "un_cultivo":
    st.error("No hay cultivos disponibles para esta lectura.")
    st.stop()

if vista == "un_cultivo":
    if st.session_state["cultivo_integrado"] not in cultivos_labels:
        st.session_state["cultivo_integrado"] = cultivos_labels[0]

    cultivo_label = st.sidebar.selectbox(
        "Cultivo",
        cultivos_labels,
        key="cultivo_integrado",
    )
else:
    cultivo_label = None
    st.sidebar.info("La vista mostrará la mejor opción estimada por municipio.")

if lectura == "aptitud":
    opciones_escenario = list(ESCENARIOS.keys())

    if st.session_state["escenario_integrado"] not in opciones_escenario:
        st.session_state["escenario_integrado"] = "solo"

    escenario_key = st.sidebar.selectbox(
        "Escenario de cultivo",
        opciones_escenario,
        format_func=lambda k: ESCENARIOS[k],
        key="escenario_integrado",
    )
else:
    escenario_key = "pdf"

if lectura == "pdf":
    opciones_color = ["Porcentaje de cumplimiento", "Factor limitante"]

    if vista == "todos_cultivos":
        opciones_color.append("Mejor opción estimada")
else:
    opciones_color = ["Índice de aptitud", "Factor limitante"]

    if vista == "todos_cultivos":
        opciones_color.append("Mejor opción estimada")

if st.session_state["color_integrado"] not in opciones_color:
    st.session_state["color_integrado"] = opciones_color[0]

modo_color = st.sidebar.radio(
    "Colorear por",
    opciones_color,
    key="color_integrado",
)

alpha = st.sidebar.slider(
    "Opacidad",
    min_value=60,
    max_value=230,
    value=170,
    step=10
)

st.sidebar.markdown("---")

with st.sidebar.expander("Consulta detectada por IA", expanded=False):
    if st.session_state.get("ultima_intencion_integrada"):
        st.json(st.session_state["ultima_intencion_integrada"])
    else:
        st.write("Todavía no hay consulta.")

with st.sidebar.expander("Ubicación detectada", expanded=False):
    if st.session_state.get("usar_filtro_ubicacion_integrada", False):
        st.write("Estado:", st.session_state.get("estado_objetivo_integrado"))
        st.write("Municipio:", st.session_state.get("municipio_objetivo_integrado"))
        st.write("Lugar:", st.session_state.get("lugar_mencionado_integrado"))

        if st.button("Quitar filtro de ubicación"):
            st.session_state["usar_filtro_ubicacion_integrada"] = False
            st.session_state["estado_objetivo_integrado"] = None
            st.session_state["municipio_objetivo_integrado"] = None
            st.session_state["lugar_mencionado_integrado"] = None
            safe_rerun()
    else:
        st.write("No hay ubicación activa.")


# ============================================================
# DATAFRAME DEL MAPA
# ============================================================

df_mapa, comparacion_sistemas = cargar_dataframe_mapa(
    lectura=lectura,
    vista=vista,
    cultivo_label=cultivo_label,
    escenario_key=escenario_key,
    indice_integrado=indice_integrado,
    indice_pdf=indice_pdf,
)

if df_mapa.empty:
    st.warning(
        "No encontré datos suficientes para esta combinación. "
        "Prueba otra lectura, cultivo o escenario."
    )
    st.stop()


# ============================================================
# FILTRO DE UBICACIÓN DETECTADA POR CHAT
# ============================================================

df_mapa_ubicacion, info_ubicacion = aplicar_filtro_ubicacion_chat(df_mapa)

if info_ubicacion is not None:
    if info_ubicacion["filas_resultantes"] > 0:
        lugar = info_ubicacion.get("lugar_mencionado") or ""
        estado_aplicado = info_ubicacion.get("estado_aplicado")
        municipio_aplicado = info_ubicacion.get("municipio_aplicado")

        partes = []

        if municipio_aplicado:
            partes.append(str(municipio_aplicado))

        if estado_aplicado:
            partes.append(str(estado_aplicado))

        ubicacion_txt = ", ".join(partes)

        if lugar:
            st.info(f"Mostrando resultados para **{lugar}** ({ubicacion_txt}).")
        else:
            st.info(f"Mostrando resultados para **{ubicacion_txt}**.")
    else:
        st.warning(
            "Detecté una ubicación en tu pregunta, pero no encontré coincidencias "
            "en los municipios del mapa. Revisa si escribiste el municipio y estado."
        )
        df_mapa_ubicacion = df_mapa.copy()
else:
    df_mapa_ubicacion = df_mapa.copy()

# A partir de aquí, la vista de mapa trabaja a escala municipal:
# una sola fila por municipio para evitar conteos inflados por duplicados.
df_mapa_ubicacion = deduplicar_municipios_para_mapa(df_mapa_ubicacion)


ranking_local = pd.DataFrame()

if info_ubicacion is not None and info_ubicacion.get("filas_resultantes", 0) > 0 and lectura == "aptitud":
    ranking_local = calcular_ranking_local(
        indice_integrado=indice_integrado,
        info_ubicacion=info_ubicacion,
        escenario_key=escenario_key,
        top_n=12
    )


# ============================================================
# FILTROS MANUALES
# ============================================================

with st.sidebar.expander("Filtros", expanded=False):
    estados = sorted(df_mapa_ubicacion["estado"].dropna().unique().tolist()) if "estado" in df_mapa_ubicacion.columns else []

    estados_sel = st.multiselect(
        "Estados",
        estados,
        default=[],
    )

    min_valor = st.slider(
        "Valor mínimo",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    factores_disponibles = sorted(
        df_mapa_ubicacion["_factor_mapa"]
        .fillna("sin_dato")
        .astype(str)
        .unique()
        .tolist()
    )

    factores_sel = st.multiselect(
        "Factores",
        factores_disponibles,
        default=[],
        format_func=nombre_factor_legible,
    )

df_filtrado = df_mapa_ubicacion.copy()

if estados_sel and "estado" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["estado"].isin(estados_sel)]

df_filtrado = df_filtrado[df_filtrado["_valor_mapa"] >= min_valor]

if factores_sel:
    df_filtrado = df_filtrado[df_filtrado["_factor_mapa"].isin(factores_sel)]

# Mantener una fila por municipio también después de filtros manuales.
df_filtrado = deduplicar_municipios_para_mapa(df_filtrado)


# ============================================================
# RESPUESTA IA / RESPUESTA LOCAL
# ============================================================

generar_respuesta_chat_si_pendiente(
    lectura=lectura,
    vista=vista,
    cultivo_label=cultivo_label,
    escenario_key=escenario_key,
    modo_color=modo_color,
    df_mapa=df_mapa_ubicacion,
    df_filtrado=df_filtrado,
    comparacion_sistemas=comparacion_sistemas,
    ranking_local=ranking_local,
    info_ubicacion=info_ubicacion,
    usar_web_ia=usar_web_ia,
)

render_chat_historial()


# ============================================================
# MÉTRICAS
# ============================================================

# Las métricas deben contarse a nivel de municipio único, no por filas.
df_metricas = deduplicar_municipios_para_mapa(df_mapa_ubicacion)

total = len(df_metricas)
promedio = round(df_metricas["_valor_mapa"].mean(), 1) if total else 0
n75 = int((df_metricas["_valor_mapa"] >= 75).sum()) if total else 0

factores_resumen = factores_frecuentes(df_metricas)
factor_mas_frecuente = factores_resumen[0]["factor_legible"] if factores_resumen else "NA"

m1, m2, m3, m4 = st.columns(4)

m1.metric("Municipios evaluados", f"{total:,}")
m2.metric("Valor promedio", f"{promedio}")
m3.metric("Municipios ≥75", f"{n75:,}")
m4.metric("Limitante frecuente", factor_mas_frecuente)

st.markdown("---")


# ============================================================
# RANKING LOCAL
# ============================================================

if ranking_local is not None and not ranking_local.empty:
    with st.expander("Ranking local de cultivos", expanded=True):
        st.caption(
            "Este ranking se calcula para la ubicación detectada. No reemplaza el mapa nacional; "
            "solo traduce la consulta a una lectura local."
        )

        st.dataframe(
            ranking_local.rename(columns={
                "cultivo": "Cultivo",
                "valor": "Aptitud",
                "clase": "Clase",
                "escenario": "Sistema / escenario",
                "factor_legible": "Limitante principal",
                "compatibilidad": "Compatibilidad",
                "municipios_evaluados": "Municipios evaluados",
                "municipios_75_o_mas": "Municipios ≥75",
            }),
            width="stretch",
            hide_index=True
        )


# ============================================================
# MAPA + LEYENDA
# ============================================================

col_mapa, col_panel = st.columns([3.25, 1.05])

with col_panel:
    clave_leyenda_activa = render_leyenda_interactiva(
        modo_color=modo_color,
        df_mapa=df_filtrado,
        key_prefix=f"leyenda_{lectura}_{vista}_{escenario_key}_{normalizar_columna(modo_color)}"
    )

    df_filtrado_leyenda = aplicar_filtro_leyenda(
        df=df_filtrado,
        modo_color=modo_color,
        clave_activa=clave_leyenda_activa
    )

    st.markdown("### Lectura rápida")

    if lectura == "pdf":
        st.write(
            "Esta vista muestra qué tanto se aproximan los municipios a los rangos "
            "normalizados del cultivo."
        )
    else:
        st.write(
            "Esta vista muestra un índice de aptitud de 0 a 100. "
            "Si eliges un sistema urbano, el resultado se ajusta al escenario seleccionado."
        )

    if vista == "todos_cultivos":
        top_cultivos = cultivos_frecuentes(df_filtrado_leyenda)

        if top_cultivos:
            st.markdown("### Opciones visibles")

            for item in top_cultivos[:8]:
                st.write(f"**{item['cultivo']}** · {item['municipios']} municipios")

    if lectura == "aptitud" and vista == "un_cultivo":
        with st.expander("Comparación de sistemas", expanded=True):
            render_matriz_sistemas(comparacion_sistemas)


with col_mapa:
    if vista == "un_cultivo":
        titulo_mapa = cultivo_label
    else:
        titulo_mapa = "Mejor opción estimada por municipio"

    st.subheader(titulo_mapa)

    if lectura == "aptitud":
        st.caption(f"Escenario: {ESCENARIOS.get(escenario_key, escenario_key)}")
    else:
        st.caption("Lectura: cumplimiento base del PDF")

    if df_filtrado_leyenda.empty:
        st.warning("No hay municipios con los filtros seleccionados.")
    else:
        df_json = df_filtrado_leyenda.to_json(orient="records")

        geojson_filtrado = construir_geojson_cacheado(
            geojson_base=geojson_base,
            df_json=df_json,
            modo_color=modo_color,
            alpha=alpha
        )

        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_filtrado,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[70, 70, 70, 120],
            line_width_min_pixels=0.4,
        )

        view_state = pdk.ViewState(
            latitude=23.7,
            longitude=-102.0,
            zoom=4.2,
            pitch=0,
        )

        tooltip = {
            "html": """
            <b>{municipio}, {estado}</b><br/>
            <b>Cultivo:</b> {cultivo_mapa}<br/>
            <b>Escenario:</b> {escenario_mapa}<br/>
            <b>Valor:</b> {valor_mapa}<br/>
            <b>Clase:</b> {clase_mapa}<br/>
            <b>Valor base:</b> {valor_base_mapa}<br/>
            <b>Compatibilidad:</b> {compatibilidad_mapa}<br/>
            <b>Limitante del escenario:</b> {factor_mapa}<br/>
            <b>Limitante base:</b> {factor_base}<br/>
            <hr/>
            <b>Temp media:</b> {temp_media_c} °C<br/>
            <b>Temp min:</b> {temp_min_c} °C<br/>
            <b>Temp max:</b> {temp_max_c} °C<br/>
            <b>Precipitación:</b> {precipitacion_mm} mm<br/>
            <b>Altitud:</b> {altitud_m} m
            """,
            "style": {
                "backgroundColor": "rgba(30, 30, 30, 0.88)",
                "color": "white",
                "fontSize": "12px",
            },
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip=tooltip,
        )

        st.pydeck_chart(deck, width="stretch")


# ============================================================
# DATOS TÉCNICOS
# ============================================================

with st.expander("Ver datos técnicos", expanded=False):
    columnas = [
        "estado",
        "municipio",
        "_cultivo_mapa",
        "_valor_mapa",
        "_clase_mapa",
        "_escenario_mapa",
        "_valor_base_mapa",
        "_compatibilidad_mapa",
        "_factor_mapa",
        "_factor_base",
        "temp_media_c",
        "temp_min_c",
        "temp_max_c",
        "precipitacion_mm",
        "altitud_m",
    ]

    columnas_existentes = [c for c in columnas if c in df_filtrado_leyenda.columns]

    df_tabla = df_filtrado_leyenda[columnas_existentes].copy()

    rename_cols = {
        "_cultivo_mapa": "cultivo",
        "_valor_mapa": "valor",
        "_clase_mapa": "clase",
        "_escenario_mapa": "escenario",
        "_valor_base_mapa": "valor_base",
        "_compatibilidad_mapa": "compatibilidad",
        "_factor_mapa": "limitante_escenario",
        "_factor_base": "limitante_base",
    }

    df_tabla = df_tabla.rename(columns=rename_cols)

    if "limitante_escenario" in df_tabla.columns:
        df_tabla["limitante_escenario"] = df_tabla["limitante_escenario"].apply(nombre_factor_legible)

    if "limitante_base" in df_tabla.columns:
        df_tabla["limitante_base"] = df_tabla["limitante_base"].apply(nombre_factor_legible)

    if "valor" in df_tabla.columns:
        df_tabla = df_tabla.sort_values(
            ["valor", "estado", "municipio"],
            ascending=[False, True, True]
        )

    st.dataframe(
        df_tabla,
        width="stretch",
        hide_index=True
    )

    csv_descarga = df_filtrado_leyenda.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Descargar datos filtrados",
        data=csv_descarga,
        file_name="mapa_integrado_cultivos.csv",
        mime="text/csv"
    )
