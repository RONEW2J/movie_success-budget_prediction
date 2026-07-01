import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, roc_auc_score,
                            silhouette_score, r2_score, mean_absolute_error, mean_squared_error)
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# TMDB poster base URL
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w200"

# ============================================================
# NLP FUNCTIONS FOR OVERVIEW ANALYSIS
# ============================================================

# Stop words to remove
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
    'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'once', 'if', 'because', 'until', 'while', 'although', 'though', 'after', 'before',
    'about', 'into', 'through', 'during', 'above', 'below', 'between', 'under', 'again',
    'further', 'out', 'up', 'down', 'off', 'over', 'any', 'him', 'her', 'his', 'their',
    'them', 'my', 'your', 'our', 'me', 'us', 'being', 'having', 'doing', 'get', 'gets',
    'got', 'getting', 'one', 'two', 'first', 'new', 'man', 'woman', 'find', 'finds',
    'take', 'takes', 'make', 'makes', 'go', 'goes', 'come', 'comes', 'see', 'sees',
    'know', 'knows', 'think', 'thinks', 'look', 'looks', 'want', 'wants', 'give', 'gives',
    'use', 'uses', 'try', 'tries', 'tell', 'tells', 'ask', 'asks', 'seem', 'seems',
    'feel', 'feels', 'leave', 'leaves', 'call', 'calls', 'keep', 'keeps', 'let', 'lets',
    'begin', 'begins', 'show', 'shows', 'hear', 'hears', 'play', 'plays', 'run', 'runs',
    'move', 'moves', 'live', 'lives', 'believe', 'believes', 'hold', 'holds', 'bring',
    'brings', 'happen', 'happens', 'write', 'writes', 'provide', 'provides', 'sit', 'sits',
    'stand', 'stands', 'lose', 'loses', 'pay', 'pays', 'meet', 'meets', 'include', 'includes',
    'continue', 'continues', 'set', 'sets', 'learn', 'learns', 'change', 'changes',
    'lead', 'leads', 'understand', 'understands', 'watch', 'watches', 'follow', 'follows',
    'stop', 'stops', 'create', 'creates', 'speak', 'speaks', 'read', 'reads', 'allow',
    'allows', 'add', 'adds', 'spend', 'spends', 'grow', 'grows', 'open', 'opens',
    'walk', 'walks', 'win', 'wins', 'offer', 'offers', 'remember', 'remembers',
    'love', 'loves', 'consider', 'considers', 'appear', 'appears', 'buy', 'buys',
    'wait', 'waits', 'serve', 'serves', 'die', 'dies', 'send', 'sends', 'expect',
    'expects', 'build', 'builds', 'stay', 'stays', 'fall', 'falls', 'cut', 'cuts',
    'reach', 'reaches', 'kill', 'kills', 'remain', 'remains', 'years', 'year', 'day',
    'days', 'life', 'time', 'way', 'world', 'story', 'movie', 'film', 'must', 'however'
}

# Keywords that indicate positive movie attributes
POSITIVE_KEYWORDS = {
    'action': ['action', 'battle', 'fight', 'war', 'combat', 'mission', 'hero', 'save', 'rescue', 'adventure'],
    'drama': ['emotional', 'powerful', 'moving', 'tragic', 'struggle', 'journey', 'relationship', 'family', 'love'],
    'comedy': ['funny', 'hilarious', 'comedy', 'laugh', 'humor', 'comic', 'witty', 'amusing'],
    'thriller': ['suspense', 'thriller', 'mystery', 'danger', 'escape', 'chase', 'secret', 'conspiracy', 'crime'],
    'scifi': ['space', 'future', 'alien', 'robot', 'technology', 'science', 'planet', 'universe', 'experiment'],
    'horror': ['horror', 'terror', 'scary', 'haunted', 'ghost', 'monster', 'nightmare', 'evil', 'demon'],
    'romance': ['romance', 'love', 'passion', 'heart', 'romantic', 'relationship', 'wedding', 'kiss'],
    'fantasy': ['magic', 'fantasy', 'wizard', 'dragon', 'kingdom', 'enchanted', 'mythical', 'quest'],
    'animation': ['animated', 'animation', 'cartoon', 'adventure', 'magical', 'colorful'],
    'documentary': ['documentary', 'real', 'true', 'story', 'history', 'biography', 'exploration']
}

def tokenize_overview(text):
    """Tokenize text and remove stop words"""
    if not text or pd.isna(text):
        return []
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stop words and short words
    tokens = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    
    return tokens

def analyze_overview_sentiment(text):
    """Analyze overview for genre keywords and sentiment"""
    tokens = tokenize_overview(text)
    
    if not tokens:
        return {'score': 0.5, 'keywords': [], 'genres_detected': []}
    
    found_keywords = []
    genres_detected = []
    
    for genre, keywords in POSITIVE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in tokens:
                found_keywords.append(keyword)
                if genre not in genres_detected:
                    genres_detected.append(genre)
    
    # Calculate score based on keyword density
    score = min(1.0, len(found_keywords) / 5 + 0.3)  # Base score + keyword bonus
    
    return {
        'score': score,
        'keywords': found_keywords[:10],
        'genres_detected': genres_detected,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens))
    }


class OverviewNLPAnalyzer:
    """NLP Analyzer for movie overview text"""
    
    def __init__(self):
        self.genre_keywords = POSITIVE_KEYWORDS
    
    def extract_keywords(self, text, top_n=10):
        """Extract keywords from overview text"""
        tokens = tokenize_overview(text)
        if not tokens:
            return []
        
        # Count word frequencies
        word_counts = {}
        for word in tokens:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]
    
    def detect_genre_themes(self, text):
        """Detect genre themes from overview text"""
        tokens = set(tokenize_overview(text))
        detected_themes = []
        
        for genre, keywords in self.genre_keywords.items():
            matches = len(tokens & set(keywords))
            if matches >= 1:
                detected_themes.append(genre)
        
        return detected_themes
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of overview (simple rule-based)"""
        if not text:
            return 0.0
        
        positive_words = {'great', 'amazing', 'beautiful', 'love', 'wonderful', 'brilliant',
                         'exciting', 'fantastic', 'incredible', 'outstanding', 'perfect',
                         'hero', 'save', 'triumph', 'victory', 'hope', 'dream'}
        negative_words = {'terrible', 'awful', 'bad', 'horrible', 'evil', 'death',
                         'kill', 'destroy', 'horror', 'fear', 'danger', 'dark',
                         'tragedy', 'loss', 'failure', 'nightmare'}
        
        tokens = set(tokenize_overview(text))
        positive_count = len(tokens & positive_words)
        negative_count = len(tokens & negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Return sentiment score between -1 and 1
        return (positive_count - negative_count) / max(total, 1) * 0.5

# Page configuration
st.set_page_config(
    page_title="🎬 Movie ML Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():
    """Load and preprocess the movie dataset"""
    try:
        df = pd.read_csv('./data/TMDB_all_movies.csv')
        
        # Basic preprocessing
        df = df[df['status'] == 'Released'].copy()
        df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
        df = df[(df['vote_average'] > 0) & (df['vote_count'] >= 10)].copy()
        
        # Feature engineering
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        df['success'] = (df['revenue'] > df['budget']).astype(int)
        
        # Extract primary genre
        def extract_primary_genre(genres_str):
            if pd.isna(genres_str):
                return 'Unknown'
            return str(genres_str).split(',')[0].strip()
        
        df['primary_genre'] = df['genres'].apply(extract_primary_genre)
        
        # Release date features
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month.fillna(1).astype(int)
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['release_year', 'runtime'])
        df['release_year'] = df['release_year'].astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    models = {}
    try:
        with open('models/success_prediction_model.pkl', 'rb') as f:
            models['success_model'] = pickle.load(f)
        with open('models/success_scaler.pkl', 'rb') as f:
            models['success_scaler'] = pickle.load(f)
        with open('models/success_model_metadata.pkl', 'rb') as f:
            models['success_metadata'] = pickle.load(f)
    except:
        models['success_model'] = None
        
    try:
        with open('models/kmeans_model.pkl', 'rb') as f:
            models['kmeans_model'] = pickle.load(f)
        with open('models/kmeans_scaler.pkl', 'rb') as f:
            models['kmeans_scaler'] = pickle.load(f)
        with open('models/kmeans_metadata.pkl', 'rb') as f:
            models['kmeans_metadata'] = pickle.load(f)
    except:
        models['kmeans_model'] = None
    
    # Load Budget Prediction model
    try:
        with open('models/budget_prediction_model.pkl', 'rb') as f:
            models['budget_model'] = pickle.load(f)
        with open('models/budget_scaler.pkl', 'rb') as f:
            models['budget_scaler'] = pickle.load(f)
        with open('models/budget_model_metadata.pkl', 'rb') as f:
            models['budget_metadata'] = pickle.load(f)
    except:
        models['budget_model'] = None
    
    return models

# Load data and models
df = load_data()
models = load_models()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(" Movie ML Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    " Select Page",
    [" Overview", 
     " Data Exploration", 
     " NLP Overview Analysis",
     " Success Prediction",
     " Train Gradient Boosting",
     " Budget Prediction",
     " Train Budget RF",
     " K-Means Clustering",
     " Train K-Means",
     " Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.info("Use the pages to explore data, train models, and make predictions!")


# ============================================================
# PERSONNEL REPUTATION (LOADED FROM MODEL METADATA)
# ============================================================
# Reputation scores are now pre-calculated during model training
# and loaded from metadata - much faster than runtime calculation!


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == " Overview":
    st.markdown('<h1 class="main-header"> Movie Machine Learning Dashboard</h1>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(" Total Movies", f"{len(df):,}")
        col2.metric(" Avg Budget", f"${df['budget'].mean()/1e6:.1f}M")
        col3.metric(" Success Rate", f"{df['success'].mean()*100:.1f}%")
        col4.metric(" Avg Rating", f"{df['vote_average'].mean():.2f}")
        
        st.markdown("---")
        
        st.markdown("###  Available Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("####  Success Prediction")
            if models.get('success_model'):
                st.success(" Model loaded!")
                metrics = models['success_metadata']['model_metrics']
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                st.metric("F1 Score", f"{metrics['f1']:.2%}")
            else:
                st.warning(" Not trained")
        
        with col2:
            st.markdown("####  Budget Prediction")
            if models.get('budget_model'):
                st.success(" Model loaded!")
                metrics = models['budget_metadata']['model_metrics']
                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                st.metric("MAE", f"${metrics['test_mae']/1e6:.1f}M")
            else:
                st.warning("⚠️ Not trained")
        
        with col3:
            st.markdown("####  K-Means Clustering")
            if models.get('kmeans_model'):
                st.success(" Model loaded!")
                meta = models['kmeans_metadata']
                st.metric("Clusters", meta['n_clusters'])
                st.metric("Features", len(meta['cluster_features']))
            else:
                st.warning(" Not trained")

# ============================================================
# PAGE: DATA EXPLORATION
# ============================================================

elif page == " Data Exploration":
    st.markdown('<h1 class="main-header"> Data Exploration</h1>', unsafe_allow_html=True)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Trends Over Time", "🎭 Genre Analysis", "💰 Financial Analysis"])
        
        with tab1:
            st.subheader("📈 Movie Industry Trends Over Time")
            
            # Feature selection for yearly trends
            feature_options = {
                'Budget': 'budget',
                'Revenue': 'revenue',
                'ROI': 'roi',
                'Rating': 'vote_average',
                'Runtime': 'runtime',
                'Popularity': 'popularity'
            }
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                options=list(feature_options.keys()),
                default=['Budget', 'Revenue', 'Rating']
            )
            
            if selected_features:
                # Create subplots
                n_features = len(selected_features)
                cols = min(3, n_features)
                rows = (n_features + cols - 1) // cols
                
                fig = make_subplots(rows=rows, cols=cols, 
                                   subplot_titles=selected_features)
                
                colors = ['forestgreen', 'steelblue', 'darkorange', 'purple', 'crimson', 'teal']
                
                for idx, feature_name in enumerate(selected_features):
                    feature_col = feature_options[feature_name]
                    yearly_data = df.groupby('release_year')[feature_col].mean()
                    
                    row = idx // cols + 1
                    col = idx % cols + 1
                    
                    fig.add_trace(
                        go.Scatter(x=yearly_data.index, y=yearly_data.values,
                                  mode='lines', fill='tozeroy',
                                  name=feature_name,
                                  line=dict(color=colors[idx % len(colors)])),
                        row=row, col=col
                    )
                
                fig.update_layout(height=400*rows, showlegend=False,
                                 title_text="Feature Trends Over Years")
                st.plotly_chart(fig, use_container_width=True)
            
            # Movies per year
            st.subheader(" Number of Movies Released Per Year")
            yearly_movies = df.groupby('release_year').size().reset_index(name='count')
            fig = px.area(yearly_movies, x='release_year', y='count',
                         title='Movies Released Per Year',
                         labels={'release_year': 'Year', 'count': 'Number of Movies'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader(" Genre Analysis")
            
            # Genre distribution
            genre_counts = df['primary_genre'].value_counts().head(15)
            fig = px.bar(x=genre_counts.values, y=genre_counts.index,
                        orientation='h', title='Top 15 Genres by Movie Count',
                        labels={'x': 'Number of Movies', 'y': 'Genre'},
                        color=genre_counts.values, color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Genre performance
            genre_stats = df.groupby('primary_genre').agg({
                'budget': 'mean',
                'revenue': 'mean',
                'roi': 'mean',
                'vote_average': 'mean',
                'success': 'mean'
            }).round(2)
            
            metric_choice = st.selectbox("Select metric to compare genres:",
                                        ['revenue', 'budget', 'roi', 'vote_average', 'success'])
            
            top_genres = genre_stats[metric_choice].sort_values(ascending=False).head(15)
            fig = px.bar(x=top_genres.values, y=top_genres.index,
                        orientation='h', 
                        title=f'Top 15 Genres by {metric_choice.title()}',
                        color=top_genres.values, color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(" Budget vs Revenue Analysis")
            
            # Add profitability filter
            st.markdown("###  Profitable vs Non-Profitable Movies")
            st.markdown("**Profitable** = Revenue >= 2x Budget (100%+ ROI)")
            
            # Create profitability column
            df_financial = df.copy()
            df_financial['is_profitable'] = (df_financial['revenue'] >= 2 * df_financial['budget']).astype(int)
            df_financial['profitability'] = df_financial['is_profitable'].map({1: '✅ Profitable (2x+)', 0: '❌ Not Profitable'})
            
            # Budget range filter for scatter plot
            st.markdown("###  Budget Range Filter")
            min_budget = int(df_financial['budget'].min())
            max_budget = int(df_financial['budget'].max())
            
            budget_range = st.slider(
                "Filter by Budget Range ($)",
                min_value=min_budget,
                max_value=max_budget,
                value=(min_budget, max_budget),
                format="$%d",
                help="Filter movies by their production budget"
            )
            
            # Apply budget filter
            df_budget_filtered = df_financial[
                (df_financial['budget'] >= budget_range[0]) & 
                (df_financial['budget'] <= budget_range[1])
            ]
            
            st.caption(f" Movies in budget range: {len(df_budget_filtered):,} / {len(df_financial):,}")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_profitable = st.checkbox("Show Profitable Movies", value=True)
            with col2:
                show_non_profitable = st.checkbox("Show Non-Profitable Movies", value=True)
            
            # Filter data
            if show_profitable and show_non_profitable:
                filtered_df = df_budget_filtered
            elif show_profitable:
                filtered_df = df_budget_filtered[df_budget_filtered['is_profitable'] == 1]
            elif show_non_profitable:
                filtered_df = df_budget_filtered[df_budget_filtered['is_profitable'] == 0]
            else:
                filtered_df = df_budget_filtered.head(0)
            
            # Sample for performance
            sample_size = st.slider("Number of movies to display", 100, 2000, 500)
            sample_df = filtered_df.sample(min(sample_size, len(filtered_df))) if len(filtered_df) > 0 else filtered_df
            
            # Create interactive scatter with detailed hover info
            fig = go.Figure()
            
            # Add profitable movies
            if show_profitable:
                profitable = sample_df[sample_df['is_profitable'] == 1].copy()
                if len(profitable) > 0:
                    # Add poster URL for display
                    profitable['poster_url'] = profitable.apply(
                        lambda row: f"{POSTER_BASE_URL}{row['poster_path']}" 
                        if pd.notna(row.get('poster_path')) and str(row['poster_path']).strip() 
                        else "No poster", 
                        axis=1
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=profitable['budget'],
                        y=profitable['revenue'],
                        mode='markers',
                        name=' Profitable (2x+)',
                        marker=dict(color='#51cf66', size=10, opacity=0.7,
                                   line=dict(width=1, color='darkgreen')),
                        text=profitable['title'],
                        customdata=np.column_stack((
                            profitable['title'],
                            profitable['budget'] / 1e6,
                            profitable['revenue'] / 1e6,
                            profitable['profit'] / 1e6,
                            profitable['roi'] * 100,
                            profitable['vote_average'],
                            profitable['primary_genre'],
                            profitable['release_year'].fillna(0).astype(int),
                            profitable['poster_url']
                        )),
                        hovertemplate='<b>%{customdata[0]}</b><br>' +
                                     ' Budget: $%{customdata[1]:.1f}M<br>' +
                                     ' Revenue: $%{customdata[2]:.1f}M<br>' +
                                     ' Profit: $%{customdata[3]:.1f}M<br>' +
                                     ' ROI: %{customdata[4]:.0f}%<br>' +
                                     ' Rating: %{customdata[5]:.1f}<br>' +
                                     ' Genre: %{customdata[6]}<br>' +
                                     ' Year: %{customdata[7]}<br>'
                    ))
            
            # Add non-profitable movies
            if show_non_profitable:
                non_profitable = sample_df[sample_df['is_profitable'] == 0].copy()
                if len(non_profitable) > 0:
                    # Add poster URL for display
                    non_profitable['poster_url'] = non_profitable.apply(
                        lambda row: f"{POSTER_BASE_URL}{row['poster_path']}" 
                        if pd.notna(row.get('poster_path')) and str(row['poster_path']).strip() 
                        else "No poster", 
                        axis=1
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=non_profitable['budget'],
                        y=non_profitable['revenue'],
                        mode='markers',
                        name=' Not Profitable',
                        marker=dict(color='#ff6b6b', size=10, opacity=0.7,
                                   line=dict(width=1, color='darkred')),
                        text=non_profitable['title'],
                        customdata=np.column_stack((
                            non_profitable['title'],
                            non_profitable['budget'] / 1e6,
                            non_profitable['revenue'] / 1e6,
                            non_profitable['profit'] / 1e6,
                            non_profitable['roi'] * 100,
                            non_profitable['vote_average'],
                            non_profitable['primary_genre'],
                            non_profitable['release_year'].fillna(0).astype(int),
                            non_profitable['poster_url']
                        )),
                        hovertemplate='<b>%{customdata[0]}</b><br>' +
                                     ' Budget: $%{customdata[1]:.1f}M<br>' +
                                     ' Revenue: $%{customdata[2]:.1f}M<br>' +
                                     ' Profit: $%{customdata[3]:.1f}M<br>' +
                                     ' ROI: %{customdata[4]:.0f}%<br>' +
                                     ' Rating: %{customdata[5]:.1f}<br>' +
                                     ' Genre: %{customdata[6]}<br>' +
                                     ' Year: %{customdata[7]}<br>' +
                                     ' <a href="%{customdata[8]}" target="_blank">View Poster</a>' +
                                     '<extra></extra>'
                    ))
            
            # Add break-even line (1x)
            max_val = max(sample_df['budget'].max(), sample_df['revenue'].max()) if len(sample_df) > 0 else 1e8
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', name='Break-even (1x)',
                line=dict(color='orange', dash='dash', width=2)
            ))
            
            # Add 2x profit line
            fig.add_trace(go.Scatter(
                x=[0, max_val/2], y=[0, max_val],
                mode='lines', name='2x Profit Line',
                line=dict(color='green', dash='dot', width=2)
            ))
            
            fig.update_layout(
                title=f'Budget vs Revenue (Budget: ${budget_range[0]/1e6:.1f}M - ${budget_range[1]/1e6:.1f}M)',
                xaxis_title='Budget ($)',
                yaxis_title='Revenue ($)',
                height=650,
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            profitable_count = len(df_budget_filtered[df_budget_filtered['is_profitable'] == 1])
            total_count = len(df_budget_filtered)
            
            col1.metric("Total Movies (in range)", f"{total_count:,}")
            col2.metric("Profitable (2x+)", f"{profitable_count:,}")
            col3.metric("Profitability Rate", f"{profitable_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
            
            # Show movie cards with posters - with budget filter
            st.markdown("### Sample Movies with Posters")
            
            # Budget filter for Top Profitable / Top Losses
            st.markdown("#### Filter by Budget for Rankings")
            col1, col2 = st.columns(2)
            with col1:
                min_budget_ranking = st.number_input(
                    "Minimum Budget ($)", 
                    min_value=0, 
                    max_value=500_000_000, 
                    value=1_000_000,
                    step=1_000_000,
                    format="%d",
                    help="Filter movies with at least this budget"
                )
            with col2:
                max_budget_ranking = st.number_input(
                    "Maximum Budget ($)", 
                    min_value=0, 
                    max_value=500_000_000, 
                    value=500_000_000,
                    step=10_000_000,
                    format="%d",
                    help="Filter movies with at most this budget"
                )
            
            # Apply budget filter for rankings
            df_ranking = df_financial[
                (df_financial['budget'] >= min_budget_ranking) & 
                (df_financial['budget'] <= max_budget_ranking)
            ]
            
            st.caption(f" Movies in budget range for ranking: {len(df_ranking):,}")
            
            display_type = st.radio("Show:", ["Top Profitable", "Top Losses"], horizontal=True)
            
            if display_type == "Top Profitable":
                top_movies = df_ranking.nlargest(10, 'roi')
            else:
                top_movies = df_ranking.nsmallest(10, 'profit')
            
            # Display as cards with posters
            cols = st.columns(5)
            for idx, (_, movie) in enumerate(top_movies.iterrows()):
                with cols[idx % 5]:
                    poster_path = movie.get('poster_path')
                    if pd.notna(poster_path) and poster_path:
                        # Ensure proper path format
                        if not str(poster_path).startswith('/'):
                            poster_path = '/' + str(poster_path)
                        poster_url = f"{POSTER_BASE_URL}{poster_path}"
                        try:
                            st.image(poster_url, use_container_width=True)
                        except:
                            st.caption("🎬 No poster")
                    else:
                        st.caption("🎬 No poster")
                    
                    st.markdown(f"**{movie['title'][:20]}...**" if len(str(movie['title'])) > 20 else f"**{movie['title']}**")
                    st.caption(f"💰 ${movie['budget']/1e6:.1f}M → ${movie['revenue']/1e6:.1f}M")
                    st.caption(f"📈 ROI: {movie['roi']*100:.0f}%")
            
            # ROI distribution
            st.subheader("📊 ROI Distribution")
            fig = px.histogram(df[df['roi'].between(-1, 10)], x='roi', nbins=50,
                             title='ROI Distribution (filtered -100% to 1000%)',
                             labels={'roi': 'Return on Investment'})
            fig.add_vline(x=0, line_dash="dash", line_color="red",
                         annotation_text="Break-even")
            fig.add_vline(x=1, line_dash="dash", line_color="green",
                         annotation_text="2x Profit")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: NLP OVERVIEW ANALYSIS
# ============================================================

elif page == " NLP Overview Analysis":
    st.markdown('<h1 class="main-header">🔤 NLP Analysis of Movie Overviews</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Analysis of descriptions of movies with rating-based segmentation to identify success keywords.
    
    **Success Criteria:**
    -  **Successful movie:** rating > 6.5
    -  **Unsuccessful movie:** rating <= 6.5
    """)
    
    if df is not None:
        # Filter movies with overviews
        df_nlp = df[df['overview'].notna()].copy()
        df_nlp['is_successful'] = (df_nlp['vote_average'] > 6.5).astype(int)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Movies with Overview", f"{len(df_nlp):,}")
        col2.metric(" Successful (>6.5)", f"{df_nlp['is_successful'].sum():,}")
        col3.metric(" Not Successful (<=6.5)", f"{(df_nlp['is_successful'] == 0).sum():,}")
        
        # Rating distribution
        st.subheader(" Rating Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_nlp, x='vote_average', nbins=50,
                              title='Distribution of Ratings',
                              color_discrete_sequence=['steelblue'])
            fig.add_vline(x=6.5, line_dash="dash", line_color="red",
                         annotation_text="Threshold = 6.5")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            success_counts = df_nlp['is_successful'].value_counts()
            fig = px.pie(values=success_counts.values, 
                        names=['Not Successful (<=6.5)', 'Successful (>6.5)'],
                        title='Success Distribution',
                        color_discrete_sequence=['#ff6b6b', '#51cf66'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Word analysis
        st.subheader(" Word Analysis")
        
        # Sample for performance
        sample_size = st.slider("Sample size for analysis", 1000, 20000, 5000)
        
        if st.button("🔍 Analyze Words", type="primary"):
            with st.spinner("Analyzing overviews..."):
                sample_df = df_nlp.sample(min(sample_size, len(df_nlp)))
                
                successful_overviews = sample_df[sample_df['is_successful'] == 1]['overview'].tolist()
                unsuccessful_overviews = sample_df[sample_df['is_successful'] == 0]['overview'].tolist()
                
                # Count words
                from collections import Counter
                
                def get_word_counts(texts):
                    words = []
                    for text in texts:
                        tokens = tokenize_overview(text)
                        words.extend(tokens)
                    return Counter(words)
                
                success_words = get_word_counts(successful_overviews)
                fail_words = get_word_counts(unsuccessful_overviews)
                
                # Display top words
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("###  Top Words in Successful Movies")
                    top_success = success_words.most_common(20)
                    fig = px.bar(x=[w[1] for w in top_success], 
                               y=[w[0] for w in top_success],
                               orientation='h', 
                               title='Top 20 Words (Successful)',
                               color_discrete_sequence=['#51cf66'])
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("###  Top Words in Unsuccessful Movies")
                    top_fail = fail_words.most_common(20)
                    fig = px.bar(x=[w[1] for w in top_fail], 
                               y=[w[0] for w in top_fail],
                               orientation='h', 
                               title='Top 20 Words (Unsuccessful)',
                               color_discrete_sequence=['#ff6b6b'])
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Unique words comparison
                st.subheader(" Words More Common in Each Category")
                
                # Calculate ratios
                n_success = len(successful_overviews)
                n_fail = len(unsuccessful_overviews)
                
                common_words = set(success_words.keys()) & set(fail_words.keys())
                word_ratios = {}
                
                for word in common_words:
                    if success_words[word] + fail_words[word] < 10:
                        continue
                    success_freq = success_words[word] / max(n_success, 1)
                    fail_freq = fail_words[word] / max(n_fail, 1)
                    if fail_freq > 0:
                        word_ratios[word] = {
                            'ratio': success_freq / fail_freq,
                            'success_count': success_words[word],
                            'fail_count': fail_words[word]
                        }
                
                # Sort
                sorted_ratios = sorted(word_ratios.items(), key=lambda x: x[1]['ratio'], reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("** More Common in SUCCESSFUL movies:**")
                    for word, data in sorted_ratios[:10]:
                        st.caption(f"**{word}** - {data['ratio']:.2f}x more likely")
                
                with col2:
                    st.markdown("** More Common in UNSUCCESSFUL movies:**")
                    for word, data in sorted_ratios[-10:][::-1]:
                        st.caption(f"**{word}** - {1/data['ratio']:.2f}x more likely")
        
        # Test your own overview
        st.subheader(" Test Your Own Overview")
        
        test_overview = st.text_area("Enter a movie overview to analyze:", height=150)
        
        if test_overview and len(test_overview) > 20:
            nlp_analyzer = OverviewNLPAnalyzer()
            
            keywords = nlp_analyzer.extract_keywords(test_overview, top_n=10)
            themes = nlp_analyzer.detect_genre_themes(test_overview)
            sentiment = nlp_analyzer.analyze_sentiment(test_overview)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("** Keywords:**")
                st.write(", ".join(keywords))
            
            with col2:
                st.markdown("** Detected Themes:**")
                st.write(", ".join(themes) if themes else "General / Mixed")
            
            with col3:
                st.markdown("** Sentiment:**")
                if sentiment > 0.1:
                    st.success(f"Positive ({sentiment:.2f})")
                elif sentiment < -0.1:
                    st.error(f"Negative ({sentiment:.2f})")
                else:
                    st.info(f"Neutral ({sentiment:.2f})")

# ============================================================
# PAGE: SUCCESS PREDICTION
# ============================================================

elif page ==  " Success Prediction":
    st.markdown('<h1 class="main-header"> Movie Success Prediction</h1>', unsafe_allow_html=True)
    
    if models.get('success_model'):
        metadata = models['success_metadata']
        
        # Extract unique personnel from dataset for selection
        @st.cache_data
        def get_personnel_lists():
            """Extract unique personnel names from dataset."""
            # Directors
            directors = set()
            for d in df['director'].dropna().unique():
                for name in str(d).split(','):
                    name = name.strip()
                    if name and len(name) > 1:
                        directors.add(name)
            
            # Composers (column is music_composer)
            composers = set()
            if 'music_composer' in df.columns:
                for c in df['music_composer'].dropna().unique():
                    for name in str(c).split(','):
                        name = name.strip()
                        if name and len(name) > 1:
                            composers.add(name)
            
            # Writers (column is writers)
            writers = set()
            if 'writers' in df.columns:
                for w in df['writers'].dropna().unique():
                    for name in str(w).split(','):
                        name = name.strip()
                        if name and len(name) > 1:
                            writers.add(name)
            
            # Producers (column is producers)
            producers = set()
            if 'producers' in df.columns:
                for p in df['producers'].dropna().unique():
                    for name in str(p).split(','):
                        name = name.strip()
                        if name and len(name) > 1:
                            producers.add(name)
            
            # Actors (from cast)
            actors = set()
            for c in df['cast'].dropna().unique():
                for name in str(c).split(',')[:10]:  # Limit per movie
                    name = name.strip()
                    if name and len(name) > 1:
                        actors.add(name)
            
            return (sorted(list(directors))[:300], 
                    sorted(list(composers))[:200],
                    sorted(list(writers))[:300],
                    sorted(list(producers))[:300],
                    sorted(list(actors))[:500])
        
        director_list, composer_list, writer_list, producer_list, actor_list = get_personnel_lists()
        all_genres = metadata.get('genre_classes', ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Movie Details")
            
            budget = st.number_input(" Budget (USD)", 
                                    min_value=100000, max_value=500000000, 
                                    value=50000000, step=1000000)
            
            runtime = st.slider("⏱ Runtime (minutes)", 30, 240, 120)
            
            release_year = st.slider("📅 Release Year", 1980, 2025, 2024)
            release_month = st.selectbox("📆 Release Month",
                                        options=list(range(1, 13)),
                                        format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
            
            # Primary and additional genres
            primary_genre = st.selectbox(" Primary Genre", options=all_genres)
            available_genres = [g for g in all_genres if g != primary_genre]
            additional_genres = st.multiselect(" Additional Genres", options=available_genres, max_selections=4)
            
            # Tagline input
            tagline = st.text_input(" Marketing Tagline", placeholder="Enter a catchy tagline...")
            
            # Overview with NLP analysis
            st.subheader(" Movie Overview")
            overview = st.text_area("Enter movie synopsis/overview:", height=150,
                                   placeholder="A young hero must save the world from destruction...")
            
            # Analyze overview with NLP
            if overview and len(overview) > 20:
                nlp_analyzer = OverviewNLPAnalyzer()
                keywords = nlp_analyzer.extract_keywords(overview, top_n=8)
                themes = nlp_analyzer.detect_genre_themes(overview)
                sentiment = nlp_analyzer.analyze_sentiment(overview)
                
                st.markdown("** NLP Analysis:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"**Keywords:** {', '.join(keywords[:5])}")
                    st.caption(f"**Detected Themes:** {', '.join(themes[:3]) if themes else 'General'}")
                with col_b:
                    sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😢"
                    st.caption(f"**Sentiment:** {sentiment_emoji} ({sentiment:.2f})")
        
        with col2:
            st.subheader(" Personnel")
            
            # Director selection
            director = st.selectbox(" Director", options=["(No specific director)"] + director_list)
            director = "" if director == "(No specific director)" else director
            
            # Composer selection
            composer = st.selectbox(" Composer", options=["(No specific composer)"] + composer_list)
            composer = "" if composer == "(No specific composer)" else composer
            
            # Writers multiselect
            selected_writers = st.multiselect(" Writers", options=writer_list, max_selections=5,
                                             help="Select screenplay/story writers")
            
            # Producers multiselect
            selected_producers = st.multiselect(" Producers", options=producer_list, max_selections=5,
                                               help="Select movie producers")
            
            # Actors multiselect
            selected_actors = st.multiselect(" Lead Actors", options=actor_list, max_selections=10,
                                            help="Select main cast members")
            
            # Show selected cast summary
            if selected_actors:
                st.info(f"**Selected Cast ({len(selected_actors)}):** {', '.join(selected_actors[:5])}{'...' if len(selected_actors) > 5 else ''}")
        
        if st.button(" Predict Success", type="primary", use_container_width=True):
            # Get reputation scores from metadata (pre-calculated during training)
            dir_rep_dict = metadata.get('director_reputation', {})
            actor_rep_dict = metadata.get('actor_reputation', {})
            composer_rep_dict = metadata.get('composer_reputation', {})
            writer_rep_dict = metadata.get('writer_reputation', {})
            producer_rep_dict = metadata.get('producer_reputation', {})
            
            dir_reputation = dir_rep_dict.get(director, 0.5) if director else 0.5
            comp_reputation = composer_rep_dict.get(composer, 0.5) if composer else 0.5
            
            # Calculate actor reputations
            actor_reputations = {a: actor_rep_dict.get(a, 0.5) for a in selected_actors}
            actor_avg_rep = np.mean(list(actor_reputations.values())) if actor_reputations else 0.5
            
            # Calculate writer and producer reputation (average)
            writer_avg_rep = np.mean([writer_rep_dict.get(w, 0.5) for w in selected_writers]) if selected_writers else 0.5
            producer_avg_rep = np.mean([producer_rep_dict.get(p, 0.5) for p in selected_producers]) if selected_producers else 0.5
            
            # Create feature vector
            genre_encoded = all_genres.index(primary_genre) if primary_genre in all_genres else 0
            total_genres = 1 + len(additional_genres)
            cast_count = len(selected_actors)
            has_tagline = 1 if tagline and len(tagline) > 5 else 0
            overview_length = len(overview) if overview else 0
            
            # NLP features
            nlp_score = 0.5
            if overview and len(overview) > 20:
                nlp_analyzer = OverviewNLPAnalyzer()
                sentiment = nlp_analyzer.analyze_sentiment(overview)
                themes = nlp_analyzer.detect_genre_themes(overview)
                # Bonus if detected themes match selected genres
                theme_match = len(set(themes) & set([primary_genre.lower()] + [g.lower() for g in additional_genres]))
                nlp_score = 0.5 + (sentiment * 0.2) + (theme_match * 0.1)
            
            # Feature vector must match training features (21 features including actor_reputation)
            features = np.array([[
                np.log1p(budget), runtime, release_year, release_month,
                ((release_month - 1) // 3) + 1, 1 if release_year >= 2010 else 0,
                genre_encoded, total_genres, max(cast_count, 1),
                has_tagline, overview_length,
                1 if director else 0, 1 if composer else 0,
                1 if selected_writers else 0, 1 if selected_producers else 0, 0,
                dir_reputation, comp_reputation, writer_avg_rep, producer_avg_rep,
                actor_avg_rep  # NEW: Actor reputation now part of model features!
            ]])
            
            # Scale and predict
            features_scaled = models['success_scaler'].transform(features)
            prediction = models['success_model'].predict(features_scaled)[0]
            probability = models['success_model'].predict_proba(features_scaled)[0][1]
            
            # Adjust probability based on NLP score (actor reputation is now in the model)
            adjusted_probability = probability * 0.95 + nlp_score * 0.05
            
            # Display results
            st.markdown("---")
            st.subheader(" Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if adjusted_probability >= 0.5:
                    st.success(f" **Predicted: LIKELY TO SUCCEED**")
                else:
                    st.error(f" **Predicted: UNLIKELY TO SUCCEED**")
            
            with col2:
                st.metric("Blockbuster Probability", f"{adjusted_probability:.1%}")
            
            # Show personnel warnings
            st.markdown("---")
            st.subheader("👥 Personnel Analysis")
            
            # Director warning
            if director:
                if dir_reputation >= 0.6:
                    st.success(f" **Director '{director}'** has a good track record ({dir_reputation:.0%} success rate)")
                elif dir_reputation >= 0.4:
                    st.info(f" **Director '{director}'** has an average track record ({dir_reputation:.0%} success rate)")
                else:
                    st.warning(f" **Director '{director}'** has a low success rate ({dir_reputation:.0%}). Consider a different director.")
            
            # Actor warnings
            if selected_actors:
                st.markdown("** Actor Analysis:**")
                
                good_actors = [(a, r) for a, r in actor_reputations.items() if r >= 0.6]
                avg_actors = [(a, r) for a, r in actor_reputations.items() if 0.4 <= r < 0.6]
                weak_actors = [(a, r) for a, r in actor_reputations.items() if r < 0.4]
                
                if good_actors:
                    for actor, rep in good_actors[:3]:
                        st.success(f" **{actor}** - Высокий рейтинг успеха ({rep:.0%})")
                
                if weak_actors:
                    for actor, rep in weak_actors[:3]:
                        st.warning(f" **{actor}** - Низкий рейтинг успеха ({rep:.0%}). Рассмотрите другого актёра.")
                
                if avg_actors and not good_actors and not weak_actors:
                    for actor, rep in avg_actors[:3]:
                        st.info(f"ℹ **{actor}** - Средний рейтинг ({rep:.0%})")
                
                st.caption(f"**Средняя репутация актёров:** {actor_avg_rep:.0%}")
            
            # Composer warning
            if composer:
                if comp_reputation >= 0.6:
                    st.success(f" **Composer '{composer}'** - Высокая репутация ({comp_reputation:.0%})")
                elif comp_reputation < 0.4:
                    st.warning(f" **Composer '{composer}'** - Низкий рейтинг ({comp_reputation:.0%})")
        
            # Show factors
            st.markdown("---")
            st.markdown("** Key Factors:**")
            factors_col1, factors_col2, factors_col3 = st.columns(3)
            with factors_col1:
                st.caption(f" Budget: ${budget/1e6:.1f}M")
                st.caption(f" Genres: {primary_genre}" + (f" + {len(additional_genres)} more" if additional_genres else ""))
            with factors_col2:
                st.caption(f" Director Rep: {dir_reputation:.0%}")
                st.caption(f" Actor Avg Rep: {actor_avg_rep:.0%}")
            with factors_col3:
                st.caption(f" Has Tagline: {'Yes' if has_tagline else 'No'}")
                st.caption(f" NLP Score: {nlp_score:.0%}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=adjusted_probability * 100,
                title={'text': "Success Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#ff6b6b"},
                        {'range': [25, 50], 'color': "#ffd93d"},
                        {'range': [50, 75], 'color': "#6bcb77"},
                        {'range': [75, 100], 'color': "#4d96ff"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(" Success prediction model not found. Please train it first!")

# ============================================================
# PAGE: TRAIN GRADIENT BOOSTING
# ============================================================

elif page == " Train Gradient Boosting":
    st.markdown('<h1 class="main-header"> Train Gradient Boosting Classifier</h1>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("###  Model Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Estimators", 50, 500, 200, 50)
            max_depth = st.slider("Max Depth", 2, 15, 5)
            learning_rate = st.select_slider("Learning Rate", 
                                            options=[0.01, 0.05, 0.1, 0.15, 0.2],
                                            value=0.1)
        
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2)
            subsample = st.slider("Subsample", 0.5, 1.0, 0.9, 0.1)
        
        with col3:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            use_random_search = st.checkbox("Use RandomizedSearchCV", value=False)
        
        if st.button(" Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                # Prepare data
                df_train = df.copy()
                
                # Remove outliers
                for col in ['budget', 'revenue', 'roi', 'runtime']:
                    Q1, Q3 = df_train[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    df_train = df_train[(df_train[col] >= Q1 - 3*IQR) & (df_train[col] <= Q3 + 3*IQR)]
                
                # Create success variable with stricter criteria
                df_train['ROI_pct'] = (df_train['revenue'] - df_train['budget']) / df_train['budget'] * 100
                df_train['is_successful'] = ((df_train['ROI_pct'] > 100) & (df_train['vote_average'] >= 6.0)).astype(int)
                
                # Feature engineering
                df_train['budget_log'] = np.log1p(df_train['budget'])
                df_train['cast_count'] = df_train['cast'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
                df_train['has_tagline'] = (df_train['tagline'].notna()).astype(int)
                df_train['overview_length'] = df_train['overview'].fillna('').apply(len)
                df_train['is_modern'] = (df_train['release_year'] >= 2010).astype(int)
                df_train['release_quarter'] = ((df_train['release_month'] - 1) // 3) + 1
                
                # Encode genre
                le = LabelEncoder()
                df_train['genre_encoded'] = le.fit_transform(df_train['primary_genre'])
                
                # Select features
                feature_cols = ['budget_log', 'runtime', 'release_year', 'release_month',
                               'release_quarter', 'is_modern', 'genre_encoded',
                               'cast_count', 'has_tagline', 'overview_length']
                
                X = df_train[feature_cols].fillna(0)
                y = df_train['is_successful']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if use_random_search:
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7, 10],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    
                    gb = GradientBoostingClassifier(random_state=42)
                    random_search = RandomizedSearchCV(
                        gb, param_grid, n_iter=20, cv=cv_folds,
                        scoring='f1', random_state=42, n_jobs=-1
                    )
                    random_search.fit(X_train_scaled, y_train)
                    model = random_search.best_estimator_
                    
                    st.info(f"Best parameters found: {random_search.best_params_}")
                else:
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        subsample=subsample,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                st.success(" Model trained successfully!")
                
                # Display metrics
                st.markdown("###  Model Performance")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1:.2%}")
                col5.metric("ROC-AUC", f"{roc_auc:.2%}")
                
                # Visualizations
                st.markdown("### 📈 Model Visualizations")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", 
                                                  "Feature Importance", "Learning Curve"])
                
                with tab1:
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Not Successful', 'Successful'],
                                   y=['Not Successful', 'Successful'],
                                   text_auto=True, color_continuous_scale='Blues')
                    fig.update_layout(title='Confusion Matrix')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                           name=f'ROC (AUC = {roc_auc:.4f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                           line=dict(dash='dash'), name='Random'))
                    fig.update_layout(title='ROC Curve',
                                     xaxis_title='False Positive Rate',
                                     yaxis_title='True Positive Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title='Feature Importance',
                               color='Importance', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X_train_scaled, y_train, cv=cv_folds, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
                    )
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean,
                                           mode='lines+markers', name='Training Score',
                                           line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=train_sizes, y=test_mean,
                                           mode='lines+markers', name='CV Score',
                                           line=dict(color='orange')))
                    fig.update_layout(title='Learning Curve',
                                     xaxis_title='Training Size',
                                     yaxis_title='F1 Score')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Overfitting analysis
                    gap = train_mean[-1] - test_mean[-1]
                    if gap > 0.1:
                        st.warning(" Model shows signs of OVERFITTING")
                    elif test_mean[-1] < 0.5:
                        st.warning(" Model shows signs of UNDERFITTING")
                    else:
                        st.success(" Model is well-balanced (good generalization)")

# ============================================================
# PAGE: BUDGET PREDICTION
# ============================================================

elif page == " Budget Prediction":
    st.markdown('<h1 class="main-header"> Movie Budget Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Предсказание оптимального бюджета для фильма на основе его характеристик.
    
    **Модель:** Random Forest Regressor с регуляризацией (NO DATA LEAKAGE)
    
     **Важно:** Модель использует только признаки, известные **ДО** релиза фильма!
    """)
    
    if models.get('budget_model'):
        metadata = models['budget_metadata']
        
        # Show model info
        st.markdown("### 📊 Model Performance")
        metrics = metadata['model_metrics']
        gap = metrics.get('gap', metrics['train_r2'] - metrics['test_r2'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Train R²", f"{metrics['train_r2']:.4f}")
        col2.metric("Test R²", f"{metrics['test_r2']:.4f}")
        col3.metric("Gap", f"{gap:.4f}", delta="✅ Good" if gap < 0.05 else "⚠️")
        col4.metric("Test MAE", f"${metrics['test_mae']/1e6:.2f}M")
        col5.metric("Test RMSE", f"${metrics['test_rmse']/1e6:.2f}M")
        
        # Model comparison if available
        if 'model_comparison' in metadata:
            with st.expander(" Model Comparison (Ridge, Lasso, ElasticNet, RF, GB)"):
                comparison = metadata['model_comparison']
                comp_df = pd.DataFrame([
                    {'Model': name, 'Test R²': data['test_r2'], 'Gap': data['gap']}
                    for name, data in comparison.items()
                ]).sort_values('Gap', ascending=True)
                st.dataframe(comp_df, use_container_width=True)
                st.info(f" Selected: **{metadata.get('best_model_name', 'RandomForest')}** (lowest overfitting gap)")
        
        st.markdown("---")
        st.markdown("###  Enter Movie Details (Pre-Production)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre selection
            all_genres = metadata.get('genre_classes', ['Action', 'Comedy', 'Drama', 'Horror'])
            primary_genre = st.selectbox(" Primary Genre", options=all_genres)
            
            # High budget genre indicator
            high_budget_genres = metadata.get('high_budget_genres', 
                ['Action', 'Adventure', 'Science Fiction', 'Fantasy', 'Animation'])
            is_high_budget_genre = 1 if primary_genre in high_budget_genres else 0
            
            if is_high_budget_genre:
                st.success(f" {primary_genre} is a high-budget genre!")
            
            genre_count = st.slider(" Number of Genres", 1, 5, 2)
            
            runtime = st.slider("⏱ Runtime (minutes)", 60, 240, 120)
            
            release_year = st.slider(" Planned Release Year", 1980, 2030, 2024)
            is_modern = 1 if release_year >= 2010 else 0
            decade = ((release_year - 1900) // 10)
        
        with col2:
            vote_average = st.slider(" Expected Rating (target)", 1.0, 10.0, 7.0, 0.1)
            
            cast_count = st.slider(" Cast Size (main actors)", 1, 20, 10)
            
            has_director = st.checkbox(" Director Assigned", value=True)
            
            overview = st.text_area(" Movie Synopsis/Overview", 
                                    placeholder="Enter a brief movie synopsis...",
                                    height=100)
            overview_length = len(overview) if overview else 200
            overview_length_log = np.log1p(overview_length)
            has_overview = 1 if overview_length > 10 else 0
            
            # Production company
            company_classes = metadata.get('company_classes', [])
            if company_classes:
                production_company = st.selectbox("🏢 Production Company", 
                    options=["(Select company)"] + company_classes[:50])  # Top 50
                has_production_company = 1 if production_company != "(Select company)" else 0
            else:
                has_production_company = st.checkbox("🏢 Has Production Company", value=True)
                production_company = None
        
        if st.button(" Predict Budget", type="primary", use_container_width=True):
            # Encode genre
            genre_encoder = metadata.get('genre_encoder')
            if genre_encoder and primary_genre in genre_encoder.classes_:
                genre_encoded = list(genre_encoder.classes_).index(primary_genre)
            else:
                genre_encoded = 0
            
            # Encode company
            company_encoder = metadata.get('company_encoder')
            if company_encoder and production_company and production_company in company_encoder.classes_:
                company_encoded = list(company_encoder.classes_).index(production_company)
            else:
                company_encoded = 0
            
            # Create feature vector (EXACT order from training)
            # feature_cols = ['runtime', 'vote_average', 'genre_count', 'cast_count', 
            #                'overview_length_log', 'is_modern', 'decade', 'is_high_budget_genre',
            #                'has_director', 'has_overview', 'has_production_company',
            #                'genre_encoded', 'company_encoded']
            features = np.array([[
                runtime,
                vote_average,
                genre_count,
                cast_count,
                overview_length_log,
                is_modern,
                decade,
                is_high_budget_genre,
                1 if has_director else 0,
                has_overview,
                1 if has_production_company else 0,
                genre_encoded,
                company_encoded
            ]])
            
            # Scale features
            features_scaled = models['budget_scaler'].transform(features)
            
            # Predict (model returns LOG budget)
            predicted_log_budget = models['budget_model'].predict(features_scaled)[0]
            
            # Convert from log to original scale
            if metadata.get('uses_log_budget', True):
                predicted_budget = np.expm1(predicted_log_budget)
            else:
                predicted_budget = predicted_log_budget
            
            # Display result
            st.markdown("---")
            st.markdown("###  Predicted Budget")
            
            col1, col2, col3 = st.columns(3)
            
            with col2:
                st.metric("Recommended Budget", f"${predicted_budget/1e6:.1f}M")
            
            # Budget range suggestion
            budget_range = metadata.get('budget_range', {})
            min_budget = budget_range.get('min', 1000000)
            max_budget = budget_range.get('max', 70000000)
            
            # Confidence interval (rough estimate based on MAE)
            mae = metrics['test_mae']
            lower_bound = max(min_budget, predicted_budget - mae)
            upper_bound = min(max_budget * 2, predicted_budget + mae)
            
            st.info(f" **Budget Range Estimate:** ${lower_bound/1e6:.1f}M - ${upper_bound/1e6:.1f}M")
            
            # Show input summary
            with st.expander(" Input Summary"):
                st.write(f"- **Genre:** {primary_genre} {'(High Budget)' if is_high_budget_genre else ''}")
                st.write(f"- **Runtime:** {runtime} min")
                st.write(f"- **Release Year:** {release_year} ({'Modern' if is_modern else 'Classic'})")
                st.write(f"- **Expected Rating:** {vote_average}")
                st.write(f"- **Cast Size:** {cast_count}")
                st.write(f"- **Has Director:** {'Yes' if has_director else 'No'}")
                st.write(f"- **Overview Length:** {overview_length} chars")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_budget / 1e6,
                number={'suffix': 'M', 'prefix': '$'},
                title={'text': "Predicted Budget"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "#e8f5e9"},
                        {'range': [10, 25], 'color': "#c8e6c9"},
                        {'range': [25, 50], 'color': "#a5d6a7"},
                        {'range': [50, 75], 'color': "#81c784"},
                        {'range': [75, 100], 'color': "#66bb6a"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("###  Feature Impact on Budget")
            importance = metadata.get('feature_importances', {})
            if importance:
                imp_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                            title='Feature Importance (RandomForest)', color='Importance',
                            color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Top factors explanation
                top_features = imp_df.nlargest(3, 'Importance')
                st.markdown("** Top 3 Budget Drivers:**")
                for _, row in top_features.iterrows():
                    st.write(f"- **{row['Feature']}**: {row['Importance']:.2%} importance")
    else:
        st.warning("⚠️ Budget prediction model not found. Train it in the 'Train Budget RF' page!")

# ============================================================
# PAGE: TRAIN BUDGET RF
# ============================================================

elif page == " Train Budget RF":
    st.markdown('<h1 class="main-header"> Train Budget Prediction (Random Forest)</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Regularized Random Forest** для предсказания бюджета фильма.
    
     **NO DATA LEAKAGE:** Модель использует только признаки, известные ДО релиза!
    -  Убраны: vote_count, popularity (известны только после релиза)
    -  Используются: runtime, genre, cast_count, director, production_company
    """)
    
    if df is not None:
        st.markdown("###  Regularization Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Tree Structure (Regularization)**")
            n_estimators = st.slider("Number of Trees", 50, 200, 100, 25)
            max_depth = st.slider("Max Depth (lower = more regularization)", 3, 15, 7)
            min_samples_split = st.slider("Min Samples Split", 10, 100, 50)
        
        with col2:
            st.markdown("**Leaf Constraints**")
            min_samples_leaf = st.slider("Min Samples Leaf", 10, 50, 30)
            max_features = st.selectbox("Max Features", [0.3, 0.5, 'sqrt'], index=1)
            max_samples = st.slider("Bootstrap Sample Size", 0.5, 0.9, 0.7, 0.1)
        
        with col3:
            st.markdown("**Data Filters**")
            min_budget_filter = st.number_input("Min Budget ($)", 
                                                min_value=100000, value=1000000, step=500000)
            max_budget_filter = st.number_input("Max Budget ($)", 
                                                min_value=10000000, value=400000000, step=10000000)
            test_size = st.slider("Test Size (%)", 10, 30, 20) / 100
        
        # Show current parameters
        st.info(f"""
        **Regularization Summary:**
        - Max Depth: {max_depth} (shallower = less overfitting)
        - Min Samples Split: {min_samples_split} (higher = more regularization)
        - Min Samples Leaf: {min_samples_leaf} (higher = more regularization)
        - Bootstrap Samples: {max_samples:.0%} (lower = more variance)
        """)
        
        if st.button(" Train Regularized Model", type="primary", use_container_width=True):
            with st.spinner("Training Regularized Random Forest..."):
                # Prepare data
                df_train = df.copy()
                
                # Filter budget range
                df_train = df_train[(df_train['budget'] >= min_budget_filter) & 
                                   (df_train['budget'] <= max_budget_filter)]
                df_train = df_train[df_train['revenue'] > 0]
                df_train = df_train[(df_train['runtime'] >= 60) & (df_train['runtime'] <= 240)]
                
                # Remove outliers using IQR (stricter)
                for col in ['budget', 'runtime']:
                    Q1, Q3 = df_train[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    df_train = df_train[(df_train[col] >= Q1 - 1.5*IQR) & 
                                       (df_train[col] <= Q3 + 1.5*IQR)]
                
                st.info(f" Training on {len(df_train):,} movies after filtering")
                
                # Feature engineering (NO DATA LEAKAGE)
                df_train['genre_count'] = df_train['genres'].fillna('').apply(
                    lambda x: len(x.split(',')) if x else 0)
                df_train['cast_count'] = df_train['cast'].fillna('').apply(
                    lambda x: min(len(x.split(',')), 20) if x else 0)
                df_train['overview_length'] = df_train['overview'].fillna('').apply(len)
                df_train['overview_length_log'] = np.log1p(df_train['overview_length'])
                df_train['is_modern'] = (df_train['release_year'] >= 2010).astype(int)
                df_train['decade'] = ((df_train['release_year'] - 1900) // 10).astype(int)
                df_train['has_director'] = (df_train['director'].fillna('') != '').astype(int)
                df_train['has_overview'] = (df_train['overview'].fillna('').str.len() > 10).astype(int)
                
                # High budget genre
                high_budget_genres = ['Action', 'Adventure', 'Science Fiction', 'Fantasy', 'Animation']
                df_train['is_high_budget_genre'] = df_train['primary_genre'].isin(high_budget_genres).astype(int)
                
                # Production company
                if 'production_companies' in df_train.columns:
                    df_train['has_production_company'] = (df_train['production_companies'].fillna('').str.len() > 0).astype(int)
                    df_train['main_production_company'] = df_train['production_companies'].fillna('').apply(
                        lambda x: x.split(',')[0].strip() if x else 'Unknown')
                else:
                    df_train['has_production_company'] = 0
                    df_train['main_production_company'] = 'Unknown'
                
                # Log budget as target
                df_train['budget_log'] = np.log1p(df_train['budget'])
                
                # Encode genre
                genre_encoder = LabelEncoder()
                df_train['genre_encoded'] = genre_encoder.fit_transform(df_train['primary_genre'])
                
                # Encode company
                company_encoder = LabelEncoder()
                df_train['company_encoded'] = company_encoder.fit_transform(df_train['main_production_company'])
                
                # Feature columns (NO DATA LEAKAGE - no vote_count, popularity!)
                feature_cols = [
                    'runtime', 'vote_average', 'genre_count', 'cast_count',
                    'overview_length_log', 'is_modern', 'decade', 'is_high_budget_genre',
                    'has_director', 'has_overview', 'has_production_company',
                    'genre_encoded', 'company_encoded'
                ]
                
                X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                y = df_train['budget_log']  # Log budget!
                y_original = df_train['budget']
                
                # Reset indices
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
                y_original = y_original.reset_index(drop=True)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                y_train_orig = y_original.loc[X_train.index]
                y_test_orig = y_original.loc[X_test.index]
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train REGULARIZED Random Forest
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    max_samples=max_samples,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # Predictions (log space)
                y_train_pred_log = model.predict(X_train_scaled)
                y_test_pred_log = model.predict(X_test_scaled)
                
                # Convert to original scale
                y_train_pred = np.expm1(y_train_pred_log)
                y_test_pred = np.expm1(y_test_pred_log)
                
                # Metrics (log space for R²)
                train_r2 = r2_score(y_train, y_train_pred_log)
                test_r2 = r2_score(y_test, y_test_pred_log)
                
                # Metrics (original scale for interpretability)
                train_mae = mean_absolute_error(y_train_orig, y_train_pred)
                test_mae = mean_absolute_error(y_test_orig, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
                
                train_r2_orig = r2_score(y_train_orig, y_train_pred)
                test_r2_orig = r2_score(y_test_orig, y_test_pred)
                
                r2_gap = train_r2 - test_r2
                
                st.success(" Model trained successfully!")
                
                # Metrics
                st.markdown("###  Model Performance")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Train R²", f"{train_r2:.4f}")
                col2.metric("Test R²", f"{test_r2:.4f}")
                col3.metric("Gap", f"{r2_gap:.4f}", delta="✅ Good" if r2_gap < 0.05 else "⚠️")
                col4.metric("Test MAE", f"${test_mae/1e6:.2f}M")
                col5.metric("Test RMSE", f"${test_rmse/1e6:.2f}M")
                
                # Overfitting check
                if r2_gap < 0.03:
                    st.success(" EXCELLENT: No overfitting (gap < 0.03)")
                elif r2_gap < 0.05:
                    st.success(" GOOD: Minimal overfitting (gap < 0.05)")
                elif r2_gap < 0.1:
                    st.warning(" MODERATE: Some overfitting - consider stronger regularization")
                else:
                    st.error(" HIGH: Significant overfitting - reduce max_depth or increase min_samples")
                
                # Visualizations
                tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test_orig/1e6, y=y_test_pred/1e6,
                        mode='markers', opacity=0.5,
                        name='Predictions'
                    ))
                    max_val = max(y_test_orig.max(), y_test_pred.max()) / 1e6
                    fig.add_trace(go.Scatter(
                        x=[0, max_val], y=[0, max_val],
                        mode='lines', line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    ))
                    fig.update_layout(
                        title=f'Actual vs Predicted Budget (R² = {test_r2_orig:.4f})',
                        xaxis_title='Actual Budget (M$)',
                        yaxis_title='Predicted Budget (M$)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    residuals = (y_test_orig - y_test_pred) / 1e6
                    fig = px.histogram(residuals, nbins=50, 
                                      title='Residual Distribution (Original Scale)',
                                      labels={'value': 'Residual (M$)'})
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"**Mean Error:** ${residuals.mean():.2f}M")
                    st.write(f"**Median Error:** ${np.median(residuals):.2f}M")
                
                with tab3:
                    imp_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                title='Feature Importance', color='Importance',
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save model
                st.markdown("###  Save Model")
                if st.button(" Save Model", use_container_width=True):
                    os.makedirs('models', exist_ok=True)
                    
                    with open('models/budget_prediction_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    with open('models/budget_scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    budget_metadata = {
                        'best_model_name': 'RandomForest',
                        'feature_columns': feature_cols,
                        'genre_classes': list(genre_encoder.classes_),
                        'genre_encoder': genre_encoder,
                        'company_encoder': company_encoder,
                        'company_classes': list(company_encoder.classes_),
                        'uses_log_budget': True,
                        'no_data_leakage': True,
                        'high_budget_genres': high_budget_genres,
                        'regularization_applied': True,
                        'feature_importances': dict(zip(feature_cols, model.feature_importances_)),
                        'model_metrics': {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_r2_orig': train_r2_orig,
                            'test_r2_orig': test_r2_orig,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'gap': r2_gap
                        },
                        'budget_range': {
                            'min': int(y_original.min()),
                            'max': int(y_original.max()),
                            'mean': int(y_original.mean()),
                            'median': int(y_original.median())
                        },
                        'rf_params': {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_features': max_features,
                            'max_samples': max_samples
                        }
                    }
                    
                    with open('models/budget_model_metadata.pkl', 'wb') as f:
                        pickle.dump(budget_metadata, f)
                    
                    st.success(" Model saved! Reload the app to use it.")

# ============================================================
# PAGE: K-MEANS CLUSTERING
# ============================================================

elif page == " K-Means Clustering":
    st.markdown('<h1 class="main-header"> K-Means Movie Clustering</h1>', unsafe_allow_html=True)
    
    if models.get('kmeans_model'):
        metadata = models['kmeans_metadata']
        
        # Display cluster info
        st.subheader(" Cluster Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Clusters", metadata['n_clusters'])
        col2.metric("Features Used", len(metadata['cluster_features']))
        
        # Best silhouette score
        silhouettes = metadata['silhouette_scores']
        best_k = max(silhouettes, key=silhouettes.get)
        col3.metric("Best K (by Silhouette)", best_k)
        
        # Elbow and Silhouette plots
        st.subheader(" Elbow & Silhouette Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inertias = metadata['inertias']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(inertias.keys()), y=list(inertias.values()),
                                   mode='lines+markers', name='Inertia'))
            fig.update_layout(title='Elbow Method', xaxis_title='K', yaxis_title='Inertia')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(silhouettes.keys()), y=list(silhouettes.values()),
                                   mode='lines+markers', name='Silhouette Score'))
            fig.update_layout(title='Silhouette Scores', xaxis_title='K', yaxis_title='Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster centroids
        st.subheader("📍 Cluster Centroids")
        centroids_df = pd.DataFrame(metadata['centroids'])
        st.dataframe(centroids_df, use_container_width=True)
        
        # Segment statistics
        st.subheader("📊 Segment Statistics")
        
        seg_stats = metadata['segment_stats']
        stats_df = pd.DataFrame({
            'Segment': list(seg_stats['counts'].keys()),
            'Count': list(seg_stats['counts'].values()),
            'Avg ROI': [seg_stats['roi_mean'].get(k, 0) for k in seg_stats['counts'].keys()],
            'Avg Profit': [seg_stats['profit_mean'].get(k, 0) for k in seg_stats['counts'].keys()]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(stats_df, values='Count', names='Segment',
                        title='Movies per Segment', color='Segment',
                        color_discrete_map={'Best': '#51cf66', 'Average': '#ffd93d', 'Worst': '#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(stats_df, x='Segment', y='Avg ROI',
                        title='Average ROI by Segment', color='Segment',
                        color_discrete_map={'Best': '#51cf66', 'Average': '#ffd93d', 'Worst': '#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ K-Means model not found. Train it in the 'Train K-Means' page!")

# ============================================================
# PAGE: TRAIN K-MEANS
# ============================================================

elif page == " Train K-Means":
    st.markdown('<h1 class="main-header"> Train K-Means Clustering</h1>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("###  Clustering Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature selection
            all_features = ['budget', 'revenue', 'profit', 'roi', 'vote_average', 
                           'vote_count', 'popularity', 'runtime']
            
            selected_features = st.multiselect(
                "Select features for clustering:",
                options=all_features,
                default=['budget', 'revenue', 'profit', 'roi', 'vote_average', 'popularity']
            )
        
        with col2:
            k_range = st.slider("K Range for Analysis", 2, 10, (2, 7))
            final_k = st.slider("Final number of clusters (K)", 2, 10, 3)
            n_init = st.slider("Number of initializations", 5, 30, 10)
        
        if st.button(" Train K-Means", type="primary", use_container_width=True):
            if len(selected_features) < 2:
                st.error("Please select at least 2 features!")
            else:
                with st.spinner("Training K-Means model..."):
                    # Prepare data
                    df_k = df.copy()
                    df_k['profit'] = df_k['revenue'] - df_k['budget']
                    df_k['roi'] = (df_k['revenue'] - df_k['budget']) / df_k['budget']
                    
                    # Remove infinities and NaN
                    df_k = df_k.replace([np.inf, -np.inf], np.nan)
                    df_k = df_k.dropna(subset=selected_features).reset_index(drop=True)
                    
                    # Extract features
                    X = df_k[selected_features].astype(float)
                    
                    # Scale
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Find optimal K
                    st.markdown("### 📈 Finding Optimal K")
                    
                    inertias = []
                    silhouettes = []
                    K_range = range(k_range[0], k_range[1] + 1)
                    
                    progress_bar = st.progress(0)
                    for i, k in enumerate(K_range):
                        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
                        labels = km.fit_predict(X_scaled)
                        inertias.append(km.inertia_)
                        silhouettes.append(silhouette_score(X_scaled, labels))
                        progress_bar.progress((i + 1) / len(K_range))
                    
                    # Display Elbow and Silhouette
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(K_range), y=inertias,
                                               mode='lines+markers', name='Inertia'))
                        fig.add_vline(x=final_k, line_dash="dash", line_color="red",
                                     annotation_text=f"Selected K={final_k}")
                        fig.update_layout(title='Elbow Method', xaxis_title='K', yaxis_title='Inertia')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(K_range), y=silhouettes,
                                               mode='lines+markers', name='Silhouette'))
                        fig.add_vline(x=final_k, line_dash="dash", line_color="red",
                                     annotation_text=f"Selected K={final_k}")
                        fig.update_layout(title='Silhouette Scores', xaxis_title='K', yaxis_title='Score')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Train final model
                    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=n_init)
                    df_k['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Get centroids
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    centroid_df = pd.DataFrame(centroids, columns=selected_features)
                    centroid_df['cluster'] = centroid_df.index
                    
                    # Label clusters
                    if 'roi' in selected_features:
                        centroid_df['score'] = centroid_df['roi']
                    elif 'revenue' in selected_features:
                        centroid_df['score'] = centroid_df['revenue']
                    else:
                        centroid_df['score'] = centroid_df.index
                    
                    centroid_df = centroid_df.sort_values('score', ascending=False).reset_index(drop=True)
                    labels = ['Best', 'Average', 'Worst'][:final_k]
                    if len(labels) < final_k:
                        labels += [f'Cluster_{i}' for i in range(len(labels), final_k)]
                    centroid_df['label'] = labels
                    
                    cluster_to_label = centroid_df.set_index('cluster')['label'].to_dict()
                    df_k['segment'] = df_k['cluster'].map(cluster_to_label)
                    
                    st.success(f"✅ K-Means model trained with K={final_k}!")
                    
                    # Display results
                    st.markdown("### 📊 Cluster Results")
                    st.dataframe(centroid_df, use_container_width=True)
                    
                    # Scatter plot
                    st.markdown("### 🔍 Cluster Visualization")
                    
                    if 'budget' in selected_features and 'revenue' in selected_features:
                        sample = df_k.sample(min(5000, len(df_k)))
                        fig = px.scatter(sample, x='budget', y='revenue', color='segment',
                                        title='Budget vs Revenue by Segment',
                                        color_discrete_map={'Best': '#51cf66', 'Average': '#ffd93d', 'Worst': '#ff6b6b'})
                        
                        max_val = max(sample['budget'].max(), sample['revenue'].max())
                        fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                                     line=dict(color='black', dash='dash'))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ROI distribution
                    if 'roi' in df_k.columns:
                        fig = px.box(df_k[df_k['roi'].between(-1, 10)], x='segment', y='roi',
                                    title='ROI Distribution by Segment',
                                    color='segment',
                                    color_discrete_map={'Best': '#51cf66', 'Average': '#ffd93d', 'Worst': '#ff6b6b'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Segment counts
                    st.markdown("### 📈 Segment Statistics")
                    seg_counts = df_k['segment'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    for i, (seg, count) in enumerate(seg_counts.items()):
                        [col1, col2, col3][i % 3].metric(seg, f"{count:,} movies")

# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================

elif page == " Model Comparison":
    st.markdown('<h1 class="main-header"> Model Comparison</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 Success Prediction Model")
        if models.get('success_model'):
            metrics = models['success_metadata']['model_metrics']
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                'Value': [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1'], metrics['roc_auc']]
            })
            
            fig = px.bar(metrics_df, x='Metric', y='Value',
                        title='Classification Metrics',
                        color='Value', color_continuous_scale='viridis')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            importance = models['success_metadata']['feature_importances']
            imp_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                        title='Top 10 Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model not trained yet!")
    
    with col2:
        st.markdown("###  K-Means Clustering Model")
        if models.get('kmeans_model'):
            meta = models['kmeans_metadata']
            
            # Silhouette scores
            silhouettes = meta['silhouette_scores']
            sil_df = pd.DataFrame({
                'K': list(silhouettes.keys()),
                'Silhouette Score': list(silhouettes.values())
            })
            
            fig = px.line(sil_df, x='K', y='Silhouette Score',
                         title='Silhouette Score vs K', markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment distribution
            seg_counts = meta['segment_stats']['counts']
            fig = px.pie(values=list(seg_counts.values()), 
                        names=list(seg_counts.keys()),
                        title='Movie Distribution by Segment',
                        color_discrete_sequence=['#51cf66', '#ffd93d', '#ff6b6b'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model not trained yet!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🎬 Movie ML Dashboard | Built with Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
