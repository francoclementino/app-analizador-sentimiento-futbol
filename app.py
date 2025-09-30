import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
from pysentimiento import create_analyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import date, timedelta

# --- Configuración de la Página y Analizador de Sentimiento ---

# Configura el layout de la página para que sea más ancho.
st.set_page_config(layout="wide", page_title="Análisis de Sentimiento de Futbolistas")

# Inicializa el analizador de sentimiento para español.
# Se utiliza 'st.cache_resource' para que el modelo se cargue una sola vez.
@st.cache_resource
def load_sentiment_analyzer():
    """Carga y cachea el modelo de análisis de sentimiento."""
    return create_analyzer(task="sentiment", lang="es")

sentiment_analyzer = load_sentiment_analyzer()

# --- Funciones Auxiliares ---

def clean_text(text):
    """Limpia el texto de los tweets para el análisis y la nube de palabras."""
    text = re.sub(r"http\S+", "", text)  # Elimina URLs
    text = re.sub(r"@[A-Za-z0-9]+", "", text)  # Elimina menciones
    text = re.sub(r"#[A-Za-z0-9]+", "", text)  # Elimina hashtags
    text = re.sub(r"\n", " ", text)  # Elimina saltos de línea
    text = re.sub(r"\s+", " ", text).strip()  # Elimina espacios extra
    return text.lower()

def get_sentiment_label(prediction):
    """Convierte la predicción del modelo en una etiqueta simple."""
    output = prediction.output
    if output == "POS":
        return "Positivo"
    elif output == "NEG":
        return "Negativo"
    else:
        return "Neutral"

def create_wordcloud(text, title):
    """Genera y muestra una nube de palabras."""
    # Lista de stopwords en español. Se pueden añadir más palabras específicas del fútbol si es necesario.
    stopwords_es = set([
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'no', 'con', 'un', 'una', 'su',
        'para', 'es', 'por', 'lo', 'las', 'como', 'más', 'pero', 'sus', 'le', 'al', 'si', 'ya',
        'me', 'ha', 'mi', 'o', 'este', 'yo', 'qué', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'son',
        'fue', 'hay', 'era', 'está', 'porque', 'todo', 'le', 'ese', 'así', 'hace', 'tiene', 'también'
    ])
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords_es,
        colormap='viridis',
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    st.pyplot(fig)

def build_search_query(player_name, team_name, start_date, end_date):
    """Construye una consulta de búsqueda optimizada para Twitter."""
    query_parts = []
    # Divide el nombre del jugador para búsquedas más específicas
    name_parts = player_name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else ""

    # 1. Búsqueda del nombre completo
    if player_name:
        query_parts.append(f'"{player_name}"')
    
    # 2. Búsqueda de nombre/apellido y equipo (si ambos existen)
    if team_name and first_name:
        query_parts.append(f'("{first_name}" AND "{team_name}")')
    if team_name and last_name and first_name != last_name:
        query_parts.append(f'("{last_name}" AND "{team_name}")')
        
    # Combina todas las partes con OR
    full_query = " OR ".join(query_parts)
    
    # Añade filtro de fecha y lenguaje
    full_query += f" since:{start_date} until:{end_date} lang:es"
    return full_query

# --- Interfaz de Usuario (Streamlit) ---

st.title("⚽ Analizador de Sentimiento de Futbolistas en X (Twitter)")
st.markdown("""
Esta herramienta es una prueba de concepto basada en el informe de viabilidad. 
Permite realizar un análisis de sentimiento bajo demanda utilizando **web scraping (sin costo de API)** y un **modelo de PLN pre-entrenado para español**.
""")

with st.form("analysis_form"):
    st.subheader("Parámetros de Búsqueda")
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Nombre del Jugador", "Ever Banega", help="Introduce el nombre completo para mejores resultados.")
        team_name = st.text_input("Equipo (Opcional)", "Newells Old Boys", help="Añadir el equipo ayuda a dar contexto y filtrar resultados irrelevantes.")
    with col2:
        # Por defecto, el rango de fechas es la última semana.
        today = date.today()
        default_start_date = today - timedelta(days=7)
        date_range = st.date_input(
            "Rango de Fechas",
            (default_start_date, today),
            max_value=today,
            help="Selecciona el período para buscar los tweets."
        )
        max_tweets = st.number_input("Cantidad Máxima de Tweets", min_value=50, max_value=2000, value=200, step=50, help="Número de tweets a recopilar y analizar. Un número mayor tomará más tiempo.")

    submit_button = st.form_submit_button("📊 Iniciar Análisis")


# --- Lógica de Análisis y Visualización ---

if submit_button:
    if not player_name:
        st.error("Por favor, introduce el nombre de un jugador.")
    else:
        start_date, end_date = date_range
        
        # Construye la consulta de búsqueda
        query = build_search_query(player_name, team_name, start_date, end_date)
        
        st.info(f"Buscando tweets con la consulta: `{query}`")
        
        tweets_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            with st.spinner(f"Recopilando hasta {max_tweets} tweets... Este proceso puede tardar unos minutos."):
                # Usa snscrape para obtener los tweets
                scraper = sntwitter.TwitterSearchScraper(query)
                for i, tweet in enumerate(scraper.get_items()):
                    if i >= max_tweets:
                        break
                    tweets_list.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.url])
                    progress_bar.progress((i + 1) / max_tweets)
                
            if not tweets_list:
                st.warning("No se encontraron tweets para los criterios de búsqueda seleccionados. Intenta con un rango de fechas más amplio o un jugador diferente.")
            else:
                tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
                st.success(f"¡Recopilación completada! Se encontraron {len(tweets_df)} tweets.")

                # --- Análisis de Sentimiento ---
                with st.spinner("Analizando el sentimiento de los tweets..."):
                    tweets_df['Clean Text'] = tweets_df['Text'].apply(clean_text)
                    
                    # Realiza la predicción de sentimiento en el lote de textos limpios
                    sentiments = sentiment_analyzer.predict(tweets_df['Clean Text'].tolist())
                    tweets_df['Sentiment'] = [get_sentiment_label(s) for s in sentiments]

                # --- Visualización de Resultados ---
                st.header("Resultados del Análisis de Sentimiento")
                
                sentiment_counts = tweets_df['Sentiment'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tweets Positivos", sentiment_counts.get("Positivo", 0))
                col2.metric("Tweets Negativos", sentiment_counts.get("Negativo", 0))
                col3.metric("Tweets Neutrales", sentiment_counts.get("Neutral", 0))
                
                # Gráfico de Torta
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#4CAF50', '#F44336', '#FFC107'])
                ax.set_ylabel('') # Oculta la etiqueta del eje y
                ax.set_title("Distribución de Sentimiento")
                st.pyplot(fig)
                
                # --- Nubes de Palabras ---
                st.header("Temas Frecuentes")
                
                col_wc1, col_wc2 = st.columns(2)
                with col_wc1:
                    st.subheader("Palabras en Tweets Positivos")
                    positive_text = " ".join(tweet for tweet in tweets_df[tweets_df['Sentiment'] == 'Positivo']['Clean Text'])
                    if positive_text:
                        create_wordcloud(positive_text, "Tweets Positivos")
                    else:
                        st.write("No hay suficientes tweets positivos para generar una nube de palabras.")

                with col_wc2:
                    st.subheader("Palabras en Tweets Negativos")
                    negative_text = " ".join(tweet for tweet in tweets_df[tweets_df['Sentiment'] == 'Negativo']['Clean Text'])
                    if negative_text:
                        create_wordcloud(negative_text, "Tweets Negativos")
                    else:
                        st.write("No hay suficientes tweets negativos para generar una nube de palabras.")
                
                # --- Muestra de Tweets ---
                st.header("Muestra de Tweets Analizados")
                
                st.subheader("💬 Tweets Más Relevantes")
                st.dataframe(tweets_df[['Username', 'Text', 'Sentiment', 'URL']], use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante la recopilación de datos: {e}")
            st.warning("La plataforma X (Twitter) cambia frecuentemente sus sistemas, lo que puede causar fallos temporales en las herramientas de scraping. Por favor, inténtalo de nuevo más tarde.")
