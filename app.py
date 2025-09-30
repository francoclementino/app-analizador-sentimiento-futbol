import streamlit as st
import pandas as pd
from ntscraper import Nitter # <-- CAMBIO: Importamos la nueva librería
from pysentimiento import create_analyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from datetime import date, timedelta, datetime

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
    text = re.sub(r"@[A-za-z0-9]+", "", text)  # Elimina menciones
    text = re.sub(r"#[A-za-z0-9]+", "", text)  # Elimina hashtags
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

def build_search_terms(player_name, team_name):
    """Construye una lista de términos de búsqueda para la nueva librería."""
    terms = []
    name_parts = player_name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else ""

    if player_name:
        terms.append(f'"{player_name}"')
    if team_name:
        if first_name:
            terms.append(f'"{first_name}" "{team_name}"')
        if last_name and first_name != last_name:
            terms.append(f'"{last_name}" "{team_name}"')
    return terms

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
        today = date.today()
        default_start_date = today - timedelta(days=7)
        date_range = st.date_input(
            "Rango de Fechas",
            (default_start_date, today),
            max_value=today,
            help="Selecciona el período para buscar los tweets."
        )
        max_tweets = st.number_input("Cantidad Máxima de Tweets", min_value=50, max_value=500, value=100, step=50, help="Número de tweets a recopilar. Un número mayor tomará más tiempo. Límite: 500.")

    submit_button = st.form_submit_button("📊 Iniciar Análisis")

# --- Lógica de Análisis y Visualización ---

if submit_button:
    if not player_name:
        st.error("Por favor, introduce el nombre de un jugador.")
    else:
        start_date, end_date = date_range
        
        search_terms = build_search_terms(player_name, team_name)
        st.info(f"Buscando tweets con los términos: `{', '.join(search_terms)}`")
        
        tweets_list = []
        
        try:
            # --- CAMBIO: Lógica de scraping actualizada ---
            with st.spinner(f"Recopilando hasta {max_tweets} tweets... Este proceso puede tardar unos minutos."):
                # --- CAMBIO: Hacemos el scraper más "paciente" aumentando el timeout ---
                scraper = Nitter(timeout=20)
                # La nueva librería busca en un único llamado
                results = scraper.get_tweets(
                    terms=search_terms,
                    since=start_date.strftime('%Y-%m-%d'),
                    until=end_date.strftime('%Y-%m-%d'),
                    number=max_tweets
                )
                
                # Procesamos los resultados obtenidos
                for tweet in results['tweets']:
                    # Convertimos la fecha de string a objeto datetime
                    tweet_date = datetime.strptime(tweet['date'], '%b %d, %Y · %I:%M %p UTC')
                    
                    tweets_list.append([
                        tweet_date, 
                        tweet['link'].split('/')[-1], # Obtenemos el ID del link
                        tweet['text'], 
                        tweet['user']['username'], 
                        tweet['link']
                    ])
            
            if not tweets_list:
                st.warning("No se encontraron tweets para los criterios de búsqueda seleccionados. Intenta con un rango de fechas más amplio o un jugador diferente.")
            else:
                tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
                st.success(f"¡Recopilación completada! Se encontraron {len(tweets_df)} tweets.")

                # --- Análisis de Sentimiento ---
                with st.spinner("Analizando el sentimiento de los tweets..."):
                    tweets_df['Clean Text'] = tweets_df['Text'].apply(clean_text)
                    sentiments = sentiment_analyzer.predict(tweets_df['Clean Text'].tolist())
                    tweets_df['Sentiment'] = [get_sentiment_label(s) for s in sentiments]

                # --- Visualización de Resultados ---
                st.header("Resultados del Análisis de Sentimiento")
                
                sentiment_counts = tweets_df['Sentiment'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tweets Positivos", sentiment_counts.get("Positivo", 0))
                col2.metric("Tweets Negativos", sentiment_counts.get("Negativo", 0))
                col3.metric("Tweets Neutrales", sentiment_counts.get("Neutral", 0))
                
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#4CAF50', '#F44336', '#FFC107'])
                ax.set_ylabel('')
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
                st.dataframe(tweets_df[['Username', 'Text', 'Sentiment', 'URL']], use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante la recopilación o el análisis de datos: {e}")
            # --- CAMBIO: Añadimos un mensaje de ayuda específico para el error común ---
            if "empty sequence" in str(e).lower():
                st.warning(
                    "**Explicación:** Este error específico usualmente significa que la librería de scraping (`ntscraper`) "
                    "no pudo encontrar un servidor público de Nitter que funcione en este momento. "
                    "Esto es algo temporal y fuera del control de la aplicación."
                )
                st.info("**¿Qué puedes hacer?** Por favor, espera unos minutos y vuelve a intentarlo.")
            else:
                st.warning(
                    "La plataforma X (Twitter) cambia frecuentemente sus sistemas. "
                    "Si el problema persiste, la herramienta puede requerir mantenimiento."
                )

