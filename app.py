import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="SocialPulse - Brand Intelligence",
    page_icon=":bar_chart:",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .main-title { font-size: 42px; font-weight: bold; color: #00d4ff; }
    .subtitle { font-size: 16px; color: #888; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">SocialPulse - Brand Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time sentiment, clustering and crisis detection for Indian startups</p>', unsafe_allow_html=True)

st.sidebar.title("SocialPulse")
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Sentiment Analysis", 
    "Brand Clusters", 
    "Crisis Detection"
])

st.write(f"Current page: {page}")

# Load data once
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_reviews_app.csv")
    brand_features = pd.read_csv("brand_clusters.csv")
    return df, brand_features

df, brand_features = load_data()

if page == "Overview":
    st.subheader("Platform Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Brands Tracked", df['brand'].nunique())
    col3.metric("Positive %", f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
    col4.metric("Negative %", f"{(df['sentiment']=='Negative').mean()*100:.1f}%")

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     color=sentiment_counts.index,
                     color_discrete_map={'Positive':'#00cc44','Negative':'#ff4444','Neutral':'#ffaa00'},
                     title="Overall Sentiment Distribution")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        avg_rating = df.groupby('brand')['rating'].mean().sort_values(ascending=False)
        fig2 = px.bar(x=avg_rating.values, y=avg_rating.index, orientation='h',
                      title="Average Rating by Brand",
                      labels={'x':'Average Rating','y':'Brand'})
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

if page == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")

    st.markdown("### Try It Yourself")
    user_review = st.text_area("Enter a review to analyze:", 
                                 placeholder="Type something like 'delivery was late and food was cold'")

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            model = pickle.load(open('sentiment_model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
            cleaned = user_review.lower()
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]

            if prediction == "Positive":
                st.success(f"Predicted Sentiment: Positive (confidence: {max(proba)*100:.1f}%)")
            else:
                st.error(f"Predicted Sentiment: Negative (confidence: {max(proba)*100:.1f}%)")
        else:
            st.warning("Please enter a review first")

    st.markdown("---")
    st.markdown("### Sentiment Trends by Brand")

    selected_brand = st.selectbox("Select a brand", sorted(df['brand'].unique()))
    brand_df = df[df['brand'] == selected_brand]

    col1, col2 = st.columns(2)
    with col1:
        sent_counts = brand_df['sentiment'].value_counts()
        fig = px.bar(x=sent_counts.index, y=sent_counts.values,
                     color=sent_counts.index,
                     color_discrete_map={'Positive':'#00cc44','Negative':'#ff4444','Neutral':'#ffaa00'},
                     title=f"{selected_brand} - Sentiment Breakdown")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Average Rating", f"{brand_df['rating'].mean():.2f} / 5")
        st.metric("Total Reviews", f"{len(brand_df):,}")
        st.metric("Positive Ratio", f"{(brand_df['sentiment']=='Positive').mean()*100:.1f}%")

if page == "Brand Clusters":
    st.subheader("Brand Clustering Analysis")
    st.markdown("Brands grouped automatically by customer satisfaction patterns using K-Means clustering")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    for i, cluster_name in enumerate(brand_features['cluster_name'].unique()):
        brands_in_cluster = brand_features[brand_features['cluster_name'] == cluster_name]['brand'].tolist()
        with [col1, col2, col3][i % 3]:
            st.markdown(f"**{cluster_name}**")
            for b in brands_in_cluster:
                st.write(f"- {b}")

    st.markdown("---")

    fig = px.scatter(brand_features, x='positive_ratio', y='negative_ratio',
                      color='cluster_name', text='brand', size='engagement_score',
                      title="Brand Clusters - Positive vs Negative Ratio",
                      color_discrete_sequence=['#00cc44', '#ff4444', '#ffaa00'])
    fig.update_traces(textposition='top center')
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detailed Brand Metrics")
    st.dataframe(
        brand_features[['brand', 'avg_rating', 'positive_ratio', 'negative_ratio', 
                        'engagement_score', 'cluster_name']].sort_values('avg_rating', ascending=False),
        use_container_width=True
    )

if page == "Crisis Detection":
    st.subheader("Crisis Detection System")
    st.markdown("Automatically detects unusual spikes in negative sentiment using Isolation Forest")

    st.markdown("---")

    selected_brand_crisis = st.selectbox("Select a brand to monitor", sorted(df['brand'].unique()), key="crisis_select")

    from sklearn.ensemble import IsolationForest

    brand_daily = df[df['brand'] == selected_brand_crisis].copy()
    brand_daily['date'] = pd.to_datetime(brand_daily['date'])
    brand_daily['day'] = brand_daily['date'].dt.date

    daily_stats = brand_daily.groupby('day').agg(
        total_reviews=('review_id', 'count'),
        avg_rating=('rating', 'mean'),
        negative_count=('sentiment', lambda x: (x == 'Negative').sum()),
    ).reset_index()
    daily_stats['negative_ratio'] = daily_stats['negative_count'] / daily_stats['total_reviews']

    if len(daily_stats) >= 30:
        features = daily_stats[['total_reviews', 'avg_rating', 'negative_ratio']]
        iso = IsolationForest(contamination=0.1, random_state=42)
        daily_stats['anomaly'] = iso.fit_predict(features)
        daily_stats['status'] = daily_stats['anomaly'].map({1: 'Normal', -1: 'Crisis Alert'})

        latest_status = daily_stats.sort_values('day').iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Status", latest_status['status'])
        col2.metric("Latest Negative Ratio", f"{latest_status['negative_ratio']*100:.1f}%")
        col3.metric("Total Crisis Days Detected", (daily_stats['status'] == 'Crisis Alert').sum())

        normal = daily_stats[daily_stats['status'] == 'Normal']
        crisis = daily_stats[daily_stats['status'] == 'Crisis Alert']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_stats['day'], y=daily_stats['negative_ratio'],
                                  mode='lines', line=dict(color='gray', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=normal['day'], y=normal['negative_ratio'],
                                  mode='markers', marker=dict(color='#00cc44', size=6), name='Normal Day'))
        fig.add_trace(go.Scatter(x=crisis['day'], y=crisis['negative_ratio'],
                                  mode='markers', marker=dict(color='#ff4444', size=12), name='Crisis Alert'))
        fig.update_layout(title=f"{selected_brand_crisis} - Negative Sentiment Timeline",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                          xaxis_title="Date", yaxis_title="Negative Review Ratio")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Not enough historical data for {selected_brand_crisis} to run crisis detection (need 30+ days)")
