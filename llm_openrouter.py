# -*- coding: utf-8 -*-

import json
import re
import unicodedata

import streamlit as st
from openai import OpenAI


# ============================================================
# CLIENTE OPENROUTER
# ============================================================

@st.cache_resource(show_spinner=False)
def get_openrouter_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY", None)

    if not api_key:
        raise RuntimeError(
            "No encontré OPENROUTER_API_KEY en .streamlit/secrets.toml"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_model_name():
    return st.secrets.get("OPENROUTER_MODEL", "openrouter/free")


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


def extraer_json_desde_texto(texto):
    """
    Intenta extraer JSON aunque el modelo lo devuelva con ```json ... ```.
    """

    if texto is None:
        raise ValueError("Respuesta vacía del modelo.")

    texto = str(texto).strip()

    # Intento directo
    try:
        return json.loads(texto)
    except Exception:
        pass

    # Quitar fences markdown
    texto_limpio = texto.replace("```json", "```")
    texto_limpio = texto_limpio.replace("```JSON", "```")

    if "```" in texto_limpio:
        partes = texto_limpio.split("```")

        for parte in partes:
            parte = parte.strip()

            if parte.startswith("{") and parte.endswith("}"):
                try:
                    return json.loads(parte)
                except Exception:
                    pass

    # Buscar primer objeto JSON
    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio != -1 and fin != -1 and fin > inicio:
        posible_json = texto[inicio:fin + 1]
        return json.loads(posible_json)

    raise ValueError(f"No pude extraer JSON de la respuesta:\n{texto}")


def message_to_dict(msg):
    """
    Convierte el mensaje del SDK a dict y preserva campos extra cuando existan.
    """

    try:
        return msg.model_dump(exclude_none=True)
    except Exception:
        salida = {
            "role": "assistant",
            "content": getattr(msg, "content", "")
        }

        if hasattr(msg, "reasoning_details"):
            salida["reasoning_details"] = msg.reasoning_details

        if hasattr(msg, "annotations"):
            salida["annotations"] = msg.annotations

        return salida


def extraer_fuentes_desde_annotations(annotations):
    """
    Convierte anotaciones devueltas por OpenRouter en una lista simple de fuentes.
    """

    if not isinstance(annotations, list):
        return []

    fuentes = []

    for item in annotations:
        if not isinstance(item, dict):
            continue

        url = (
            item.get("url")
            or item.get("source_url")
            or item.get("link")
            or ""
        )

        titulo = (
            item.get("title")
            or item.get("source_title")
            or item.get("name")
            or "Fuente consultada"
        )

        if url:
            fuentes.append({
                "titulo": titulo,
                "institucion": "",
                "pais": "",
                "url": url,
                "dato_usado": "",
                "tipo_fuente": "fuente_web",
                "confianza": "media"
            })

    salida = []
    vistos = set()

    for fuente in fuentes:
        url = fuente.get("url", "")

        if url and url not in vistos:
            salida.append(fuente)
            vistos.add(url)

    return salida


def asegurar_fuentes_en_respuesta(datos, annotations=None):
    """
    Asegura que el JSON tenga campo fuentes y que también preserve annotations.
    """

    if not isinstance(datos, dict):
        datos = {}

    if annotations:
        datos["_openrouter_annotations"] = annotations

    fuentes = datos.get("fuentes", [])

    if not isinstance(fuentes, list):
        fuentes = []

    if not fuentes and annotations:
        fuentes = extraer_fuentes_desde_annotations(annotations)

    datos["fuentes"] = fuentes

    return datos


# ============================================================
# LLAMADA BASE A OPENROUTER
# ============================================================

def llamar_llm(
    messages,
    reasoning=False,
    temperature=0.2,
    usar_web=False,
    web_max_results=5,
    json_mode=False,
):
    """
    Llamada general a OpenRouter.

    reasoning=False por default para que la app responda más rápido.

    usar_web=True:
    agrega el server tool openrouter:web_search para que el modelo pueda buscar internet.

    json_mode=True:
    solicita salida JSON cuando el modelo/proveedor lo soporte.
    """

    client = get_openrouter_client()

    extra_body = {}

    if reasoning:
        extra_body["reasoning"] = {"enabled": True}

    if usar_web:
        extra_body["tools"] = [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "max_results": web_max_results,
                    "max_total_results": web_max_results,
                    "search_context_size": "medium",
                }
            }
        ]

    kwargs = {
        "model": get_model_name(),
        "messages": messages,
        "temperature": temperature,
    }

    if extra_body:
        kwargs["extra_body"] = extra_body

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        # Algunos modelos gratuitos o proveedores no aceptan response_format.
        # Si falla con json_mode, reintentamos sin response_format.
        if json_mode:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    msg = response.choices[0].message
    salida = message_to_dict(msg)

    try:
        salida["_raw_response"] = response.model_dump(exclude_none=True)
    except Exception:
        salida["_raw_response"] = None

    return salida


# ============================================================
# 1) INTERPRETAR PREGUNTA DEL USUARIO
# ============================================================

PROMPT_INTENCION = """
Eres un parser para una app de mapas agrícolas urbanos en México.

Tu trabajo NO es contestar al usuario.
Tu trabajo es convertir la pregunta en JSON válido.

La app trabaja con datos municipales de México. Por eso, si el usuario pregunta por un sitio específico, campus, casa, escuela, edificio, localidad, predio o proyecto, debes identificar si ya dio municipio y estado. Si no los dio, marca que se requiere ubicación específica.

La app puede mostrar:
1. Cumplimiento base del PDF.
2. Índice de aptitud.
3. Escenarios de cultivo urbano:
   - cielo_abierto
   - azotea_extensiva
   - azotea_intensiva
   - huerto_vertical
   - invernadero
   - interior_led
4. Vista de un cultivo.
5. Vista de todos los cultivos.
6. Comparación de sistemas para un cultivo.
7. Ranking local de cultivos para una ubicación específica.

Acciones posibles:
- mapa_por_cultivo
- mapa_todos_cultivos
- explicar_municipio_cultivo
- comparar_cultivos
- comparar_sistemas_cultivo
- explicar_concepto
- recomendar_cultivo_sitio
- recomendar_cultivo_hogar
- pregunta_identidad
- pregunta_general

Modelos posibles:
- pdf_crudo: si el usuario pide rangos originales, PDF, cumplimiento directo o lectura base.
- fuzzy_base: si pregunta dónde es apto, dónde puede cultivar, mejores lugares o aptitud del cultivo sin sistema urbano.
- urbano: si menciona azotea, vertical, invernadero, interior, LED, casa, campus, escuela, edificio o sistema de cultivo urbano.
- desconocido: si no queda claro.

Sistemas posibles:
- solo
- todos_sistemas
- cielo_abierto
- azotea_extensiva
- azotea_intensiva
- huerto_vertical
- invernadero
- interior_led
- ninguno

Escala de consulta:
- nacional: pregunta por México en general.
- estatal: menciona un estado.
- municipal: menciona un municipio.
- localidad: menciona ciudad, colonia, localidad o zona sin suficiente precisión municipal.
- sitio: menciona casa, campus, escuela, edificio, azotea, patio, predio, huerto comunitario o proyecto.
- desconocida: no queda claro.

Tipo de lugar:
- hogar
- campus
- escuela
- edificio
- azotea
- patio
- terraza
- huerto_comunitario
- espacio_publico
- proyecto
- predio
- localidad
- desconocido

Devuelve SOLO JSON válido con esta estructura exacta:

{
  "accion": "mapa_por_cultivo",
  "cultivo": null,
  "cultivo_2": null,
  "estado": null,
  "municipio": null,
  "modelo": "fuzzy_base",
  "sistema": "ninguno",
  "escala_consulta": "nacional",
  "tipo_lugar": "desconocido",
  "lugar_mencionado": null,
  "requiere_ubicacion_especifica": false,
  "ubicacion_minima_requerida": null,
  "ubicacion_usuario": null,
  "pregunta_conceptual": false,
  "conceptos_a_explicar": [],
  "usar_ia_complementaria": false,
  "necesita_mapa": true,
  "necesita_explicacion": true
}

Reglas de intención:
- Si el usuario pregunta "dónde puedo cultivar X", usa accion="mapa_por_cultivo".
- Si pregunta "puedo cultivar X en Y", usa accion="mapa_por_cultivo".
- Si pregunta "qué puedo cultivar", "qué puede cultivarse", "qué se puede cultivar", "qué cultivos", "qué cultivo recomiendas", "opciones" o "mejor cultivo", usa accion="mapa_todos_cultivos".
- Si pregunta por un sitio específico, campus, casa, escuela, edificio, localidad, predio o proyecto, y pide qué cultivar, usa accion="recomendar_cultivo_sitio".
- Si dice explícitamente casa, hogar o departamento propio, puedes usar accion="recomendar_cultivo_hogar".
- Si pregunta "por qué no sale X en Y", usa accion="explicar_municipio_cultivo".
- Si pregunta "comparar X y Y", usa accion="comparar_cultivos".
- Si pregunta "comparación de sistemas", "qué sistema conviene", "mejor sistema", "sistema recomendado" o "cómo interpreto los sistemas", usa accion="comparar_sistemas_cultivo".
- Si pregunta "quién eres", "quién hizo esta app", "quién desarrolló esto", usa accion="pregunta_identidad".
- Si pregunta "cuál es la diferencia entre azotea intensiva y extensiva", usa accion="explicar_concepto", pregunta_conceptual=true y conceptos_a_explicar=["azotea_intensiva", "azotea_extensiva"].
- Si pregunta eso y además quiere cultivar algo en un lugar específico, usa accion="recomendar_cultivo_sitio" y conserva conceptos_a_explicar.

Reglas de modelo:
- Si dice "según el PDF", "cumplimiento", "rango original", "rangos originales" o "crudo", usa modelo="pdf_crudo".
- Si menciona sistema urbano, azotea, techo, campus, casa, edificio, escuela, interior, LED, invernadero o huerto vertical, usa modelo="urbano".
- Si no menciona sistema urbano y pregunta por aptitud general, usa modelo="fuzzy_base".

Reglas de sistema:
- Si dice "azotea", "techo" o "departamento", usa sistema="azotea_intensiva", salvo que diga explícitamente "azotea extensiva".
- Si dice "azotea extensiva", usa sistema="azotea_extensiva".
- Si dice "azotea intensiva", usa sistema="azotea_intensiva".
- Si dice "vertical" o "muro verde", usa sistema="huerto_vertical".
- Si dice "invernadero", usa sistema="invernadero".
- Si dice "interior", "LED", "luz artificial" o "indoor", usa sistema="interior_led".
- Si pregunta por el mejor sistema, usa sistema="todos_sistemas".
- Si pregunta qué puede cultivarse en una ubicación y no especifica sistema, usa sistema="todos_sistemas".
- Si no menciona sistema, usa sistema="solo" si pregunta por aptitud del cultivo.

Reglas de ubicación:
- Si el usuario menciona estado, llena "estado".
- Si el usuario menciona municipio, llena "municipio".
- Si el usuario dice "mi casa", "mi hogar", "mi departamento", "mi campus", "mi escuela", "mi edificio", "mi localidad", "donde vivo", "aquí", "en mi zona", "cerca de mí", "en este predio", "en este proyecto", usa escala_consulta="sitio" o "localidad", según corresponda.
- Si menciona un campus con ciudad clara, intenta inferir municipio y estado. Ejemplo: "Tec de Monterrey, Campus Querétaro" corresponde a municipio="Querétaro", estado="Querétaro".
- Si menciona "Campus Monterrey", corresponde a municipio="Monterrey", estado="Nuevo León".
- Si menciona "Campus Guadalajara", corresponde normalmente a municipio="Zapopan", estado="Jalisco".
- Si menciona "Campus Estado de México", no inventes municipio si no queda claro.
- Si el usuario pide recomendación para un lugar concreto y no dio municipio ni estado, usa requiere_ubicacion_especifica=true y ubicacion_minima_requerida="municipio_estado".
- No pidas temperatura, altitud ni precipitación. La app debe pedir municipio y estado para traducir esos datos automáticamente.

Reglas de web:
- Si pide internet, fuentes, web, búsqueda externa, complementar o enriquecer, usar_ia_complementaria=true.

Reglas finales:
- No inventes cultivos.
- No inventes municipios.
- No agregues texto fuera del JSON.
"""


def interpretar_pregunta_usuario(pregunta):
    messages = [
        {"role": "system", "content": PROMPT_INTENCION},
        {"role": "user", "content": pregunta},
    ]

    msg = llamar_llm(
        messages=messages,
        reasoning=False,
        temperature=0.0,
        usar_web=False,
        json_mode=True,
    )

    contenido = msg.get("content", "")

    try:
        datos = extraer_json_desde_texto(contenido)
    except Exception:
        datos = {
            "accion": "pregunta_general",
            "cultivo": None,
            "cultivo_2": None,
            "estado": None,
            "municipio": None,
            "modelo": "desconocido",
            "sistema": "ninguno",
            "escala_consulta": "desconocida",
            "tipo_lugar": "desconocido",
            "lugar_mencionado": None,
            "requiere_ubicacion_especifica": False,
            "ubicacion_minima_requerida": None,
            "ubicacion_usuario": None,
            "pregunta_conceptual": False,
            "conceptos_a_explicar": [],
            "usar_ia_complementaria": False,
            "necesita_mapa": False,
            "necesita_explicacion": True,
            "error_parseo": contenido,
        }

    salida = {
        "accion": datos.get("accion", "pregunta_general"),
        "cultivo": datos.get("cultivo", None),
        "cultivo_2": datos.get("cultivo_2", None),
        "estado": datos.get("estado", None),
        "municipio": datos.get("municipio", None),
        "modelo": datos.get("modelo", "desconocido"),
        "sistema": datos.get("sistema", "ninguno"),
        "escala_consulta": datos.get("escala_consulta", "desconocida"),
        "tipo_lugar": datos.get("tipo_lugar", "desconocido"),
        "lugar_mencionado": datos.get("lugar_mencionado", None),
        "requiere_ubicacion_especifica": bool(datos.get("requiere_ubicacion_especifica", False)),
        "ubicacion_minima_requerida": datos.get("ubicacion_minima_requerida", None),
        "ubicacion_usuario": datos.get("ubicacion_usuario", None),
        "pregunta_conceptual": bool(datos.get("pregunta_conceptual", False)),
        "conceptos_a_explicar": datos.get("conceptos_a_explicar", []),
        "usar_ia_complementaria": bool(datos.get("usar_ia_complementaria", False)),
        "necesita_mapa": bool(datos.get("necesita_mapa", True)),
        "necesita_explicacion": bool(datos.get("necesita_explicacion", True)),
    }

    return salida


# ============================================================
# 2) BÚSQUEDA WEB COMPLEMENTARIA POR CULTIVO
# ============================================================

PROMPT_REQUERIMIENTOS_WEB = """
Eres una asistente de apoyo para una app científica de aptitud agroclimática y cultivo urbano en México.

Tu trabajo es buscar información externa confiable sobre el cultivo indicado.
No decidas dónde se puede cultivar.
No colorees mapas.
No generes ranking de municipios.
Solo recupera requerimientos técnicos complementarios que puedan servir para interpretar el mapa, el índice de aptitud o una futura capa fenológica.

PRIORIDAD DE FUENTES PARA MÉXICO:

1. Fuentes oficiales mexicanas de agricultura:
   - INIFAP: Instituto Nacional de Investigaciones Forestales, Agrícolas y Pecuarias.
   - Biblioteca Digital INIFAP.
   - Secretaría de Agricultura y Desarrollo Rural / Agricultura, Gobierno de México.
   - SIAP: Servicio de Información Agroalimentaria y Pesquera.
   - SENASICA: Servicio Nacional de Sanidad, Inocuidad y Calidad Agroalimentaria.

2. Instituciones mexicanas académicas o técnicas:
   - Universidad Autónoma Chapingo.
   - Colegio de Postgraduados.
   - Universidad Autónoma Agraria Antonio Narro.
   - UNAM.
   - IPN.
   - UAM.
   - Universidades públicas mexicanas.
   - Centros públicos de investigación mexicanos.

3. Fuentes internacionales confiables:
   - FAO.
   - CGIAR.
   - CIMMYT.
   - CIAT.
   - ICARDA.
   - USDA.
   - Extension services universitarios.
   - Universidades o agencias públicas agrícolas.

4. Fuentes secundarias:
   - Usa blogs, páginas comerciales o sitios genéricos solo si no hay fuentes técnicas disponibles.
   - Si se usa una fuente secundaria, marca la confianza como media o baja.

Busca especialmente:
- horas frío o chilling hours, si aplica
- temperatura crítica por calor
- temperatura crítica por helada
- rango térmico de desarrollo
- temperatura óptima
- requerimientos de precipitación o riego
- altitud o condiciones climáticas relevantes
- sensibilidad a salinidad o pH
- notas sobre fenología
- si el cultivo es templado, tropical, árido, anual, perenne, frutal, hortaliza, leguminosa, cereal, etc.
- manejo relevante para azotea, invernadero, interior LED o sustrato, si aplica

REGLAS:
- Prioriza fuentes mexicanas cuando existan.
- No uses fuentes de España u otros países si hay una fuente mexicana equivalente.
- Si no encuentras fuente mexicana, aclara que la información proviene de fuentes internacionales.
- No inventes URLs.
- No inventes datos técnicos.
- No mezcles datos de distintas fuentes como si fueran una sola.
- Usa null cuando no encuentres un dato.
- Si hay valores contradictorios entre fuentes, pon el rango más conservador y explica en observaciones.
- Si el dato depende de cultivar, variedad o manejo, indícalo en observaciones.
- No uses Wikipedia como fuente técnica principal.
- No incluyas texto fuera del JSON.

Devuelve SOLO JSON válido con esta estructura exacta:

{
  "nombre_cultivo": "",
  "nombre_cientifico": "",
  "tipo_fenologico": "",
  "tipo_funcional_probable": "",
  "requiere_horas_frio": null,
  "horas_frio_min": null,
  "horas_frio_optimas_min": null,
  "horas_frio_optimas_max": null,
  "temperatura_desarrollo_min_c": null,
  "temperatura_desarrollo_max_c": null,
  "temperatura_optima_min_c": null,
  "temperatura_optima_max_c": null,
  "temperatura_calor_critico_c": null,
  "temperatura_helada_critica_c": null,
  "gdd_base_c": null,
  "gdd_min": null,
  "gdd_optimo": null,
  "precipitacion_min_mm": null,
  "precipitacion_max_mm": null,
  "altitud_min_m": null,
  "altitud_max_m": null,
  "ph_min": null,
  "ph_max": null,
  "salinidad_tolerada_max_ds_m": null,
  "sensibilidad_calor": "",
  "sensibilidad_frio": "",
  "sensibilidad_salinidad": "",
  "manejo_urbano": {
    "azotea": "",
    "invernadero": "",
    "interior_led": "",
    "riego": "",
    "sustrato": ""
  },
  "observaciones": "",
  "fuentes": [
    {
      "titulo": "",
      "institucion": "",
      "pais": "",
      "url": "",
      "dato_usado": "",
      "tipo_fuente": "institucion_mexicana | universidad_mexicana | institucion_internacional | extension_universitaria | articulo_academico | fuente_secundaria",
      "confianza": "alta | media | baja"
    }
  ],
  "confianza_general": "alta | media | baja",
  "advertencias": [],
  "requiere_revision_humana": true
}
"""


def buscar_requerimientos_web_cultivo(nombre_cultivo):
    messages = [
        {"role": "system", "content": PROMPT_REQUERIMIENTOS_WEB},
        {
            "role": "user",
            "content": (
                f"Cultivo: {nombre_cultivo}\n\n"
                "Busca requerimientos fenológicos y agroclimáticos complementarios. "
                "Prioriza fuentes mexicanas oficiales o académicas. "
                "Si no existen fuentes mexicanas claras, usa fuentes internacionales confiables "
                "y aclara esa limitación en advertencias."
            )
        }
    ]

    msg = llamar_llm(
        messages=messages,
        reasoning=True,
        temperature=0.1,
        usar_web=True,
        web_max_results=6,
        json_mode=True,
    )

    contenido = msg.get("content", "")

    try:
        datos = extraer_json_desde_texto(contenido)
    except Exception:
        datos = {
            "nombre_cultivo": nombre_cultivo,
            "error_parseo": True,
            "respuesta_texto": contenido,
            "fuentes": [],
            "confianza_general": "baja",
            "advertencias": [
                "No se pudo convertir la respuesta web a JSON válido."
            ],
            "requiere_revision_humana": True
        }

    annotations = msg.get("annotations", [])

    datos = asegurar_fuentes_en_respuesta(
        datos=datos,
        annotations=annotations
    )

    return datos


# ============================================================
# 3) REDACTAR RESPUESTA HUMANA
# ============================================================

PROMPT_RESPUESTA_HUMANA = """
Eres una asistente dentro de una app de mapas agrícolas urbanos de México.

Responde en español natural, claro y humano.
Sólo si te preguntan en inglés, responde en inglés.
Primero da la conclusión práctica.
Después explica brevemente por qué.
No inventes datos.
Usa únicamente el contexto estructurado que te da la app.

Identidad:
- Si el usuario pregunta quién eres, responde que eres un asistente de apoyo para interpretar mapas de cultivos, aptitud agroecológica y escenarios de cultivo urbano en México.
- Di que la herramienta fue desarrollada por la Dra. Juana Isabel Méndez.
- Si necesitan más apoyo, pueden escribir a isabelmendez@tec.mx.

Fuente base:
- Si mencionas la fuente base, di: Ruiz Corral et al. (2020), Requerimientos agroecológicos de cultivos, 2da edición.
- No empieces con “De acuerdo al PDF” ni “Según el dataset”.
- Puedes decir: “Los requerimientos agroecológicos base provienen de Ruiz Corral et al. (2020)”.

Lenguaje:
- No uses la palabra fuzzy.
- Usa “índice de aptitud”, “aptitud estimada” o “potencial estimado”.
- No escribas una sección llamada “Nota metodológica”.
- No menciones razonamiento interno.
- No menciones que eres un parser.
- No muestres JSON.

Interpretación:
- Si el usuario pregunta por azotea, interior LED o invernadero, interpreta precipitación como manejo hídrico, riego o condición controlada, no como lluvia directa disponible.
- Si el usuario pregunta cómo interpretar la comparación de sistemas, explica:
  1. Qué significa la aptitud promedio.
  2. Qué significan los municipios con 75 o más.
  3. Qué significa la compatibilidad promedio.
  4. Qué sistema aparece mejor posicionado.
  5. Que el mejor sistema promedio no siempre es el mejor en todos los municipios.
  6. Que debe revisar el municipio específico en el mapa.
- Si el contexto incluye ranking_local, responde usando ranking_local y no agregues cultivos que no estén ahí.
- Si el contexto incluye ubicación aplicada, aclara que la lectura es municipal y que no sustituye una evaluación puntual del sitio.

Si el contexto incluye ia_web_complementaria:
- úsala para complementar la explicación;
- no la trates como resultado final del mapa;
- explica si aporta variables que la fuente base no tenía, como horas frío, GDD, calor crítico o heladas;
- si las fuentes web no son mexicanas, aclara que son fuentes internacionales o secundarias;
- menciona que las fuentes aparecen debajo de la respuesta.

No prometas rendimiento agrícola.
No presentes el resultado como garantía de producción.
"""


def redactar_respuesta_humana(pregunta_usuario, contexto_app):
    messages = [
        {"role": "system", "content": PROMPT_RESPUESTA_HUMANA},
        {
            "role": "user",
            "content": (
                "Pregunta del usuario:\n"
                f"{pregunta_usuario}\n\n"
                "Contexto calculado por la app:\n"
                f"{json.dumps(contexto_app, ensure_ascii=False, indent=2)}"
            )
        }
    ]

    msg = llamar_llm(
        messages=messages,
        reasoning=False,
        temperature=0.25,
        usar_web=False,
        json_mode=False,
    )

    return msg.get("content", "").strip()


# ============================================================
# 4) REDACTAR RESPUESTA CON WEB DIRECTO
# ============================================================

def responder_con_web_y_contexto(
    pregunta_usuario,
    contexto_app,
    nombre_cultivo=None,
    devolver_dict=False,
):
    """
    Función opcional.

    Si nombre_cultivo se proporciona:
    - busca requerimientos web estructurados;
    - los agrega al contexto;
    - redacta una respuesta usando ese contexto.

    Si nombre_cultivo no se proporciona:
    - permite que el modelo use web durante la redacción.
    """

    if nombre_cultivo:
        info_web = buscar_requerimientos_web_cultivo(nombre_cultivo)

        contexto = dict(contexto_app)
        contexto["ia_web_complementaria"] = info_web

        respuesta = redactar_respuesta_humana(
            pregunta_usuario=pregunta_usuario,
            contexto_app=contexto,
        )

        if devolver_dict:
            return {
                "respuesta": respuesta,
                "web": info_web,
            }

        return respuesta

    messages = [
        {"role": "system", "content": PROMPT_RESPUESTA_HUMANA},
        {
            "role": "user",
            "content": (
                "Pregunta del usuario:\n"
                f"{pregunta_usuario}\n\n"
                "Contexto calculado por la app:\n"
                f"{json.dumps(contexto_app, ensure_ascii=False, indent=2)}\n\n"
                "Puedes buscar en internet solo para complementar datos agronómicos "
                "o fenológicos faltantes. No uses internet para reemplazar el cálculo "
                "del mapa."
            )
        }
    ]

    msg = llamar_llm(
        messages=messages,
        reasoning=True,
        temperature=0.3,
        usar_web=True,
        web_max_results=5,
        json_mode=False,
    )

    return msg.get("content", "").strip()