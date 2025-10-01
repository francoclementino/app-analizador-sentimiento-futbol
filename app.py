import streamlit as st
import pandas as pd
from pysentimiento import create_analyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from datetime import date, timedelta
import asyncio
from twscrape import API, gather
import json

# Configuración de la página
st.set_page_config(layout="wide", page_title="Análisis de Sentimiento de Futbolistas")

# --- FUNCIÓN PARA INICIALIZAR CUENTAS DE TWITTER ---
async def initialize_twitter_accounts():
    """Inicializa las cuentas de Twitter desde los secretos de Streamlit."""
    # Usar directorio temporal para la base de datos
    db_path = "/tmp/accounts.db"
    api = API(db_path)
    
    # Verificar si ya hay cuentas configuradas
    try:
        accounts = await api.pool.accounts_info()
        if len(accounts) > 0:
            st.info("✅ Usando cuenta de Twitter ya configurada")
            return api
    except:
        pass
    
    # Configurar desde secrets
    try:
        username = st.secrets["twitter"]["username"]
        password = st.secrets["twitter"]["password"]
        email = st.secrets["twitter"]["email"]
        
        # Usar cookies (método más confiable)
        if "cookies" in st.secrets["twitter"]:
            cookies_raw = st.secrets["twitter"]["cookies"]
            
            # Parsear cookies desde JSON
            try:
                cookies_list = json.loads(cookies_raw)
                
                # Convertir lista de cookies al formato que espera twscrape
                cookies_dict = {}
                for cookie in cookies_list:
                    cookies_dict[cookie['name']] = cookie['value']
                
                # Convertir a string JSON
                cookies_str = json.dumps(cookies_dict)
                
                # Agregar cuenta con cookies
                await api.pool.add_account(username, password, email, "", cookies=cookies_str)
                st.success("✅ Cuenta configurada con cookies")
                return api
                
            except json.JSONDecodeError as e:
                st.error(f"Error al parsear cookies: {e}")
                return None
            except Exception as e:
                st.error(f"Error al configurar cookies: {e}")
                return None
        else:
            st.error("No se encontraron cookies en los secretos")
            return None
        
    except KeyError as e:
        st.error(f"⚠️ Falta configurar en los secretos: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error al conectar con Twitter: {str(e)}")
        st.info("Verifica que las cookies estén correctamente configuradas")
        return None

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

async def scrape_tweets_async(player_name, team_name, start_date, end_date, max_tweets):
    """Función asíncrona para scrapear tweets usando twscrape."""
    # Inicializar API con cuentas
    api = await initialize_twitter_accounts()
    if api is None:
        return []
    
    # Construir query de búsqueda
    query_parts = [f'"{player_name}"']
    if team_name:
        query_parts.append(f'"{team_name}"')
    
    query = f'{" ".join(query_parts)} since:{start_date.strftime("%Y-%m-%d")} until:{end_date.strftime("%Y-%m-%d")}'
    
    tweets_list = []
    try:
        async for tweet in api.search(query, limit=max_tweets):
            tweets_list.append({
                'Datetime': tweet.date,
                'Tweet Id': tweet.id,
                'Text': tweet.rawContent,
                'Username': tweet.user.username,
                'URL': tweet.url
            })
    except Exception as e:
        st.error(f"Error durante el scraping: {e}")
    
    return tweets_list

def scrape_tweets(player_name, team_name, start_date, end_date, max_tweets):
    """Wrapper sincrónico para usar en Streamlit."""
    return asyncio.run(scrape_tweets_async(player_name, team_name, start_date, end_date, max_tweets))

# --- Interfaz de Usuario (Streamlit) ---

st.title("⚽ Analizador de Sentimiento de Futbolistas en X (Twitter)")
st.markdown("""
Esta herramienta utiliza **twscrape** (2025), una librería activamente mantenida que permite 
realizar análisis de sentimiento en tiempo real de tweets sobre futbolistas.
""")

with st.form("analysis_form"):
    st.subheader("Parámetros de Búsqueda")
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Nombre del Jugador", "Lionel Messi", help="Introduce el nombre completo para mejores resultados.")
        team_name = st.text_input("Equipo (Opcional)", "", help="Añadir el equipo ayuda a dar contexto y filtrar resultados irrelevantes.")
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
        
        st.info(f"🔍 Buscando tweets sobre **{player_name}** {f'({team_name})' if team_name else ''}")
        
        try:
            with st.spinner(f"Recopilando hasta {max_tweets} tweets... Esto puede tardar 1-2 minutos."):
                tweets_list = scrape_tweets(player_name, team_name, start_date, end_date, max_tweets)
            
            if not tweets_list:
                st.warning("No se encontraron tweets para los criterios de búsqueda seleccionados. Intenta con un rango de fechas más amplio o un jugador diferente.")
            else:
                tweets_df = pd.DataFrame(tweets_list)
                st.success(f"¡Recopilación completada! Se encontraron {len(tweets_df)} tweets.")

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
                    if positive_text: 
                        create_wordcloud(positive_text, "Tweets Positivos")
                    else: 
                        st.write("No hay suficientes datos.")
                        
                with col_wc2:
                    st.subheader("Palabras en Tweets Negativos")
                    negative_text = " ".join(tweet for tweet in tweets_df[tweets_df['Sentiment'] == 'Negativo']['Clean Text'])
                    if negative_text: 
                        create_wordcloud(negative_text, "Tweets Negativos")
                    else: 
                        st.write("No hay suficientes datos.")
                
                st.header("Muestra de Tweets Analizados")
                st.dataframe(tweets_df[['Username', 'Text', 'Sentiment', 'URL']], use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante la recopilación de datos: {e}")
            st.warning("""
            **Posibles causas:**
            - Las cookies pueden haber expirado (vuelve a exportarlas desde tu navegador)
            - Twitter está bloqueando temporalmente el acceso
            - Problemas de conectividad
            """)