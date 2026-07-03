# Mapa de aptitud de cultivos en México

Aplicación web interactiva para explorar la aptitud agroecológica de cultivos
por municipio en México y comparar distintos sistemas de producción urbana.

## Funciones principales

- Consulta de aptitud por cultivo y municipio.
- Comparación de sistemas como cielo abierto, azoteas, huertos verticales,
  invernaderos y cultivo interior con iluminación artificial.
- Identificación de factores limitantes y escenarios recomendados.
- Visualización geográfica interactiva.
- Asistente para interpretar los resultados en lenguaje claro.
- Descarga de los datos filtrados.

## Fuente de referencia

Los requerimientos agroecológicos se basan en:

> Ruiz Corral, J., García, G., Acuña, I., Flores López, H. y Ojeda, G. (2020).
> *Requerimientos agroecológicos de cultivos* (2.ª ed.).

## Tecnologías

- Python
- Streamlit
- pandas y Parquet
- PyDeck
- OpenRouter

## Ejecución local

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Crea `.streamlit/secrets.toml`:

   ```toml
   OPENROUTER_API_KEY = "tu_api_key"
   OPENROUTER_MODEL = "openrouter/free"
   ```

3. Ejecuta la aplicación:

   ```bash
   streamlit run app.py
   ```

## Despliegue en Streamlit Community Cloud

Selecciona `app.py` como archivo principal y agrega las credenciales desde
**Advanced settings > Secrets**. Los archivos `secrets.toml` no deben
publicarse en GitHub.

## Autora

**Dra. Juana Isabel Méndez**  
Tecnológico de Monterrey  
Contacto: isabelmendez@tec.mx

## Nota

Esta herramienta es de apoyo para exploración y análisis. Sus resultados no
sustituyen una evaluación agronómica detallada en campo.
