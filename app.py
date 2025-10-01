import asyncio
from twikit import Client
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

class TwitterScraper:
    def __init__(self):
        self.client = Client('es-ES')  # Idioma español
        self.logged_in = False
    
    async def login(self, username, email, password):
        """Login con credenciales de Twitter"""
        try:
            await self.client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password
            )
            # Guardar cookies para reutilizar
            self.client.save_cookies('twitter_cookies.json')
            self.logged_in = True
            print("✅ Login exitoso")
        except Exception as e:
            print(f"❌ Error en login: {e}")
            raise
    
    async def login_with_cookies(self, cookies_file='twitter_cookies.json'):
        """Login rápido usando cookies guardadas"""
        try:
            self.client.load_cookies(cookies_file)
            self.logged_in = True
            print("✅ Login con cookies exitoso")
        except Exception as e:
            print(f"⚠️ No se pudieron cargar cookies: {e}")
            return False
        return True
    
    async def search_player_tweets(self, player_name, team_name=None, 
                                   days_back=7, max_tweets=100):
        """
        Busca tweets sobre un jugador de fútbol
        
        Args:
            player_name: Nombre del jugador (ej: "Lionel Messi", "gallo")
            team_name: Equipo opcional (ej: "All Boys")
            days_back: Días hacia atrás para buscar
            max_tweets: Máximo de tweets a retornar
        """
        if not self.logged_in:
            raise Exception("Debes hacer login primero")
        
        # Construir query
        if team_name:
            query = f'"{player_name}" "{team_name}" lang:es -filter:retweets'
        else:
            query = f'"{player_name}" (gol OR asistencia OR partido) lang:es -filter:retweets'
        
        print(f"🔍 Buscando: {query}")
        
        tweets_data = []
        try:
            # Búsqueda con twikit
            tweets = await self.client.search_tweet(query, 'Latest')
            
            for tweet in tweets:
                # Limit de tweets
                if len(tweets_data) >= max_tweets:
                    break
                
                # Filtrar por fecha si es necesario
                tweet_date = datetime.strptime(
                    tweet.created_at, 
                    '%a %b %d %H:%M:%S %z %Y'
                )
                if (datetime.now() - tweet_date.replace(tzinfo=None)).days > days_back:
                    continue
                
                tweets_data.append({
                    'id': tweet.id,
                    'fecha': tweet.created_at,
                    'usuario': tweet.user.name,
                    'username': tweet.user.screen_name,
                    'contenido': tweet.text,
                    'likes': tweet.favorite_count,
                    'retweets': tweet.retweet_count,
                    'replies': tweet.reply_count,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                })
                
                # Delay para evitar rate limits
                await asyncio.sleep(1)
            
            print(f"✅ Encontrados {len(tweets_data)} tweets")
            return tweets_data
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    async def get_user_tweets(self, username, max_tweets=100):
        """
        Alternativa más confiable: obtener tweets directamente del usuario
        Úsalo cuando conozcas la cuenta oficial del jugador/club
        """
        if not self.logged_in:
            raise Exception("Debes hacer login primero")
        
        try:
            user = await self.client.get_user_by_screen_name(username)
            print(f"📱 Usuario: @{user.screen_name} - {user.name}")
            print(f"👥 Seguidores: {user.followers_count:,}")
            
            tweets = await self.client.get_user_tweets(user.id, 'Tweets')
            
            tweets_data = []
            for tweet in tweets:
                if len(tweets_data) >= max_tweets:
                    break
                
                tweets_data.append({
                    'id': tweet.id,
                    'fecha': tweet.created_at,
                    'contenido': tweet.text,
                    'likes': tweet.favorite_count,
                    'retweets': tweet.retweet_count,
                    'url': f"https://twitter.com/{username}/status/{tweet.id}"
                })
                
                await asyncio.sleep(1)
            
            return tweets_data
            
        except Exception as e:
            print(f"❌ Error obteniendo tweets de usuario: {e}")
            return []

class SentimentAnalyzer:
    """Análisis de sentimiento básico"""
    def __init__(self):
        # VADER funciona decentemente con español si se adapta
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Palabras clave positivas en español (fútbol)
        self.positive_keywords = [
            'gol', 'golazo', 'crack', 'genio', 'figura', 'estrella',
            'excelente', 'brillante', 'increíble', 'partidazo', 'asistencia',
            'talento', 'magia', 'fenómeno', 'bestia', 'máquina'
        ]
        
        # Palabras clave negativas
        self.negative_keywords = [
            'mal', 'pésimo', 'desastre', 'horrible', 'nefasto', 'perdió',
            'error', 'fallo', 'expulsado', 'lesión', 'lesionado', 'banco',
            'suplente', 'fracaso'
        ]
    
    def analyze_spanish(self, text):
        """Análisis básico adaptado para español"""
        text_lower = text.lower()
        
        # Contar palabras positivas y negativas
        pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
        neg_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Análisis VADER como base
        vader_scores = self.analyzer.polarity_scores(text)
        
        # Ajustar con keywords en español
        if pos_count > neg_count:
            sentiment = 'positivo'
            score = min(0.8 + (pos_count * 0.1), 1.0)
        elif neg_count > pos_count:
            sentiment = 'negativo'
            score = max(-0.8 - (neg_count * 0.1), -1.0)
        else:
            # Usar VADER por defecto
            if vader_scores['compound'] >= 0.05:
                sentiment = 'positivo'
                score = vader_scores['compound']
            elif vader_scores['compound'] <= -0.05:
                sentiment = 'negativo'
                score = vader_scores['compound']
            else:
                sentiment = 'neutral'
                score = vader_scores['compound']
        
        return {
            'sentiment': sentiment,
            'score': score,
            'pos_words': pos_count,
            'neg_words': neg_count
        }

# STREAMLIT APP COMPLETA
def main():
    st.set_page_config(
        page_title="Análisis de Sentimiento - Jugadores de Fútbol",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Análisis de Sentimiento de Jugadores de Fútbol")
    st.markdown("*Análisis de tweets en español sobre jugadores sudamericanos*")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Credenciales (usar st.secrets en producción)
        st.subheader("Credenciales Twitter")
        username = st.text_input("Username", value=st.secrets.get("TWITTER_USERNAME", ""))
        email = st.text_input("Email", value=st.secrets.get("TWITTER_EMAIL", ""))
        password = st.text_input("Password", type="password", 
                                value=st.secrets.get("TWITTER_PASSWORD", ""))
        
        # Parámetros de búsqueda
        st.subheader("Parámetros")
        days_back = st.slider("Días hacia atrás", 1, 30, 7)
        max_tweets = st.slider("Máximo de tweets", 10, 200, 50)
    
    # Input de búsqueda
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Nombre del jugador", 
                                    placeholder="Ej: Lionel Messi, gallo, Enzo Fernández")
    with col2:
        team_name = st.text_input("Equipo (opcional)", 
                                  placeholder="Ej: All Boys, River Plate")
    
    # Botón de búsqueda
    if st.button("🔍 Buscar y Analizar", type="primary"):
        if not player_name:
            st.error("Por favor ingresa el nombre de un jugador")
            return
        
        if not (username and email and password):
            st.error("Por favor configura las credenciales de Twitter en la sidebar")
            return
        
        # Inicializar scraper
        scraper = TwitterScraper()
        sentiment_analyzer = SentimentAnalyzer()
        
        with st.spinner("Iniciando sesión en Twitter..."):
            try:
                # Intentar login con cookies primero
                success = asyncio.run(scraper.login_with_cookies())
                
                if not success:
                    # Login normal si no hay cookies
                    asyncio.run(scraper.login(username, email, password))
                
            except Exception as e:
                st.error(f"Error en login: {e}")
                return
        
        with st.spinner(f"Buscando tweets sobre {player_name}..."):
            tweets = asyncio.run(
                scraper.search_player_tweets(
                    player_name, 
                    team_name, 
                    days_back, 
                    max_tweets
                )
            )
        
        if not tweets:
            st.warning(f"No se encontraron tweets para '{player_name}' en los últimos {days_back} días")
            st.info("""
            **Posibles razones:**
            - El jugador/equipo no es muy mencionado en Twitter
            - La búsqueda es muy específica
            - Twitter no tiene estos tweets indexados
            
            **Prueba:**
            - Usar solo el apellido del jugador
            - Buscar sin equipo
            - Aumentar los días hacia atrás
            - Usar la cuenta oficial del jugador (función "Buscar por Usuario")
            """)
            return
        
        # Análisis de sentimiento
        with st.spinner("Analizando sentimiento..."):
            for tweet in tweets:
                analysis = sentiment_analyzer.analyze_spanish(tweet['contenido'])
                tweet.update(analysis)
        
        # Crear DataFrame
        df = pd.DataFrame(tweets)
        
        # Métricas generales
        st.header("📊 Resumen")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", len(df))
        with col2:
            positivos = len(df[df['sentiment'] == 'positivo'])
            st.metric("Sentimiento Positivo", f"{positivos} ({positivos/len(df)*100:.1f}%)")
        with col3:
            negativos = len(df[df['sentiment'] == 'negativo'])
            st.metric("Sentimiento Negativo", f"{negativos} ({negativos/len(df)*100:.1f}%)")
        with col4:
            promedio_score = df['score'].mean()
            st.metric("Score Promedio", f"{promedio_score:.2f}")
        
        # Distribución de sentimiento
        st.subheader("Distribución de Sentimiento")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Tweets más relevantes
        st.subheader("🔥 Tweets Más Populares")
        df_sorted = df.sort_values('likes', ascending=False)
        
        for idx, tweet in df_sorted.head(10).iterrows():
            sentiment_emoji = {
                'positivo': '😊',
                'negativo': '😠',
                'neutral': '😐'
            }
            
            with st.expander(f"{sentiment_emoji[tweet['sentiment']]} @{tweet['username']} - {tweet['likes']} ❤️"):
                st.write(tweet['contenido'])
                st.caption(f"📅 {tweet['fecha']} | 🔁 {tweet['retweets']} RT | 💬 {tweet['replies']} replies")
                st.caption(f"Score: {tweet['score']:.2f}")
                st.markdown(f"[Ver en Twitter]({tweet['url']})")
        
        # Tabla completa
        st.subheader("📋 Todos los Tweets")
        st.dataframe(
            df[['fecha', 'username', 'contenido', 'sentiment', 'score', 'likes', 'retweets']],
            use_container_width=True
        )
        
        # Descarga CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos (CSV)",
            data=csv,
            file_name=f"sentiment_{player_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()