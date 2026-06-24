# SocialPulse — Brand Intelligence Platform

An end-to-end machine learning platform that analyzes real customer sentiment, clusters brands by performance, and detects sentiment crises for 10 major Indian startups — built entirely from live-scraped Google Play Store reviews.

## Live Demo
[Try the app here](https://socialpulse-bhoomi.streamlit.app/)

## Overview
SocialPulse answers a question every brand wants answered: *"What are our customers really saying about us, and is something going wrong right now?"*

The platform collects, cleans, and analyzes hundreds of thousands of real user reviews to surface sentiment trends, automatically group similar-performing brands, and flag abnormal spikes in negative sentiment in near real-time.

## Data
- **500,000 reviews** collected live via the Google Play Scraper API
- **10 brands tracked**: Zomato, Swiggy, Blinkit, CRED, Zepto, Meesho, PhonePe, Paytm, Ola, Uber
- **475,220 reviews** after cleaning and preprocessing

## Features

### Sentiment Analysis
- Naive Bayes classifier with TF-IDF vectorization
- **94% accuracy** on held-out test data
- Live text input — analyze any custom review in real time

### Brand Clustering
- K-Means clustering (k=3, selected via Elbow Method)
- Automatically groups brands into Top Performers, Middle Ground, and Struggling Brands based on rating, sentiment ratio, and engagement
- PCA visualization of cluster separation

### Crisis Detection
- Isolation Forest anomaly detection per brand
- Identifies days with abnormal negative sentiment spikes
- Live status indicator and historical crisis timeline

### Interactive Dashboard
- Built with Streamlit
- Dark-themed, multi-page interface
- Real-time brand comparison and filtering

## Tech Stack
- **Language:** Python
- **ML:** Scikit-learn (Naive Bayes, K-Means, Isolation Forest)
- **NLP:** TF-IDF Vectorization
- **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud
- **Deployment:** Streamlit Cloud
- **Data Collection:** google-play-scraper

## Project Structure
├── 1_data_collection.ipynb      # Scraping 500K reviews via API

├── 2_eda.ipynb                  # Exploratory data analysis & visualizations

├── 3_sentiment_model.ipynb      # Naive Bayes sentiment classifier

├── 4_clustering.ipynb           # K-Means brand clustering

├── 5_anomaly_detection.ipynb    # Isolation Forest crisis detection

├── app.py                       # Streamlit dashboard

├── requirements.txt

└── models/                      # Saved model files (.pkl)
## How to Run Locally
```bash
git clone https://github.com/Bhoomi-1401/socialpulse-brand-intelligence
cd socialpulse-brand-intelligence
pip install -r requirements.txt
streamlit run app.py
```

## Key Insight
Clustering revealed that fintech and payment apps (PhonePe, Paytm) consistently show higher customer satisfaction than quick-commerce and ride-hailing apps (Ola, Zepto, Swiggy) — likely due to differences in operational complexity and service reliability.
