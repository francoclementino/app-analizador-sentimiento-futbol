import streamlit as st
import pandas as pd
# --- CAMBIO: Volvemos a snscrape, la librería de scraping directo ---
import snscrape.modules.twitter as sntwitter
from pysentimiento import create_analyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from datetime import date, timedelta, datetime

# --- Configuración de la Página y Analizador de Sentimiento ---

st.set_page_config(layout="wide", page_title="Análisis de Sentimiento de Futbolistas")

@st.cache_resource
def load_sentiment_analyzer():
    """Carga y cachea el modelo de análisis de sentimiento."""
    return create_analyzer(task="sentiment", lang="es")

sentiment_analyzer = load_sentiment_analyzer()

# --- Funciones Auxiliares ---

def clean_text(text):
    """Limpia el texto de los tweets para el análisis y la nube de palabras."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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

# --- CAMBIO: La función para construir la query es ahora más compleja ---
def build_snscrape_query(player_name, team_name, start_date, end_date):
    """Construye la cadena de búsqueda para snscrape."""
    terms = []
    name_parts = player_name.strip().split()
    
    # Crea combinaciones del nombre y el equipo
    if player_name:
        terms.append(f'"{player_name}"')
    if team_name and len(name_parts) > 0:
        terms.append(f'("{name_parts[0]}" AND "{team_name}")')
        if len(name_parts) > 1:
            terms.append(f'("{name_parts[-1]}" AND "{team_name}")')

    # Une los términos con OR para que busque cualquiera de las combinaciones
    query_terms = " OR ".join(terms)
    
    # Añade el rango de fechas
    query = f"({query_terms}) since:{start_date.strftime('%Y-%m-%d')} until:{end_date.strftime('%Y-%m-%d')}"
    return query

# --- Interfaz de Usuario (Streamlit) ---

st.title("⚽ Analizador de Sentimiento de Futbolistas en X (Twitter)")
st.markdown("""
Esta herramienta es una prueba de concepto basada en el informe de viabilidad. 
Permite realizar un análisis de sentimiento bajo demanda utilizando **web scraping directo (sin costo de API)** y un **modelo de PLN pre-entrenado para español**.
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
        date_range = st.date_input("Rango de Fechas", (default_start_date, today), max_value=today)
        max_tweets = st.number_input("Cantidad Máxima de Tweets", min_value=50, max_value=500, value=100, step=50)

    submit_button = st.form_submit_button("📊 Iniciar Análisis")

# --- Lógica de Análisis y Visualización ---

if submit_button:
    if not player_name:
        st.error("Por favor, introduce el nombre de un jugador.")
    else:
        start_date, end_date = date_range
        
        query = build_snscrape_query(player_name, team_name, start_date, end_date)
        st.info(f"Buscando tweets con la consulta: `{query}`")
        
        tweets_list = []
        
        try:
            with st.spinner(f"Recopilando hasta {max_tweets} tweets... Este proceso puede ser más lento y depende de la disponibilidad de X."):
                # --- CAMBIO: Nueva lógica de scraping con snscrape ---
                scraper = sntwitter.TwitterSearchScraper(query)
                for i, tweet in enumerate(scraper.get_items()):
                    if i >= max_tweets:
                        break
                    tweets_list.append([
                        tweet.date, 
                        tweet.id, 
                        tweet.rawContent, 
                        tweet.user.username, 
                        tweet.url
                    ])
            
            if not tweets_list:
                st.warning("No se encontraron tweets para los criterios de búsqueda seleccionados. Intenta con un rango de fechas más amplio o un jugador diferente.")
            else:
                tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
                st.success(f"¡Recopilación completada! Se encontraron {len(tweets_df)} tweets.")

                # El resto del código de análisis y visualización permanece igual
                with st.spinner("Analizando el sentimiento de los tweets..."):
                    tweets_df['Clean Text'] = tweets_df['Text'].apply(clean_text)
                    sentiments = sentiment_analyzer.predict(tweets_df['Clean Text'].tolist())
                    tweets_df['Sentiment'] = [get_sentiment_label(s) for s in sentiments]

                st.header("Resultados del Análisis de Sentimiento")
                sentiment_counts = tweets_df['Sentiment'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tweets Positivos", sentiment_counts.get("Positivo", 0))
                col2.metric("Tweets Negativos", sentiment_counts.get("Negativo", 0))
                col3.metric("Tweets Neutrales", sentiment_counts.get("Neutral", 0))
                
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#4CAF50', '#F44336', '#FFC107'])
                ax.set_ylabel('')
                st.pyplot(fig)
                
                st.header("Temas Frecuentes")
                col_wc1, col_wc2 = st.columns(2)
                with col_wc1:
                    st.subheader("Palabras en Tweets Positivos")
                    positive_text = " ".join(tweet for tweet in tweets_df[tweets_df['Sentiment'] == 'Positivo']['Clean Text'])
                    if positive_text: create_wordcloud(positive_text, "Tweets Positivos")
                    else: st.write("No hay suficientes datos.")
                with col_wc2:
                    st.subheader("Palabras en Tweets Negativos")
                    negative_text = " ".join(tweet for tweet in tweets_df[tweets_df['Sentiment'] == 'Negativo']['Clean Text'])
                    if negative_text: create_wordcloud(negative_text, "Tweets Negativos")
                    else: st.write("No hay suficientes datos.")
                
                st.header("Muestra de Tweets Analizados")
                st.dataframe(tweets_df[['Username', 'Text', 'Sentiment', 'URL']], use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante la recopilación de datos: {e}")
            st.warning(
                "El scraping directo de X (Twitter) es inestable porque la plataforma implementa activamente medidas anti-bots. "
                "Si este error persiste, puede que X haya bloqueado temporalmente el acceso desde el servidor de la aplicación. "
                "Por favor, inténtalo de nuevo más tarde."
            )

