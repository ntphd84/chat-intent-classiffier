import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import json

def analyze_chats(input_file, output_file):
    # Load data
    df = pd.read_excel(input_file)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT  # fallback if no date column

    nlp = spacy.load('en_core_web_sm')
    sia = SentimentIntensityAnalyzer()
    
    # Process chats
    results = []
    all_messages = []
    
    for i, chat in enumerate(df['Chats']):
        if not isinstance(chat, str):
            continue

        chat_date = df.loc[i, 'date'] if 'date' in df.columns else None
        
        # Extract user messages
        lines = chat.split('\n')
        user_messages = []
        idx = 0
        while idx < len(lines):
            if lines[idx].startswith('user('):
                if idx + 1 < len(lines):
                    user_messages.append(lines[idx + 1])
                idx += 3
            else:
                idx += 1
        
        if not user_messages:
            continue
            
        # Analyze sentiment
        combined_text = ' '.join(user_messages)
        sentiment = sia.polarity_scores(combined_text)
        sentiment['category'] = 'positive' if sentiment['compound'] > 0.05 else 'negative' if sentiment['compound'] < -0.05 else 'neutral'
        
        # Extract entities
        doc = nlp(combined_text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        
        all_messages.extend(user_messages)
        results.append({
            'messages': user_messages,
            'sentiment': sentiment,
            'entities': entities,
            'date': chat_date.strftime('%Y-%m-%d') if pd.notnull(chat_date) else None
        })
    
    # Topic modeling
    if len(all_messages) > 5:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        dtm = vectorizer.fit_transform(all_messages)
        
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda_output = lda.fit_transform(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append({
                'id': idx,
                'words': top_words,
                'weight': float(np.mean(lda_output[:, idx]))
            })
        
        # Clustering
        clustering = AgglomerativeClustering(n_clusters=10)
        clusters = clustering.fit_predict(dtm.toarray()).tolist()
    else:
        topics = []
        clusters = []
    
    # Save results
    analysis_results = {
        'chat_analysis': results,
        'topics': topics,
        'clusters': clusters
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

if __name__ == '__main__':
    base_dir = os.path.expanduser('~/chat_classification_files')
    input_file = os.path.join(base_dir, 'results', 'master_chat_class.xlsx')
    output_file = os.path.join(base_dir, 'results', 'analysis_results.json')
    analyze_chats(input_file, output_file)
