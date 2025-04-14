from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def get_cached_embeddings(texts, cache_path, model, batch_size=300):

    # If cache exists, load and return
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

# Load book data
df = pd.read_csv("./book-dataset/books.csv")

dfro = pd.read_csv("./book-dataset/romanian_corpus.csv")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Clean column names and fill missing values
df.columns = [col.lower() for col in df.columns]
df['title'] = df['title'].fillna('')
df['authors'] = df['authors'].fillna('')
df['description'] = df['description'].fillna('')
df['categories'] = df['categories'].fillna('')
df['combined'] = df['description'] + ' ' + df['categories']

dfro.columns = [col.lower() for col in dfro.columns]
dfro['titlu'] = dfro['titlu'].fillna('')
dfro['titlu2'] = dfro['titlu2'].fillna('')
dfro['autor'] = dfro['autor'].fillna('')
dfro['genwiki'] = dfro['genwiki'].fillna('')
dfro['combined'] = dfro['titlu'] + '' + dfro['genwiki'] + dfro['titlu2']

embeddings = get_cached_embeddings(df['combined'], './cache/embeddings_books.pkl', model)
embeddingsro = get_cached_embeddings(dfro['combined'], './cache/embeddings_ro.pkl', model)


# Set up NearestNeighbors for similarity search
nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddings)

nnro = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddingsro)

# Index for efficient searching by title
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

indicesro = pd.Series(dfro.index, index=dfro['titlu']).drop_duplicates()

# Flask app setup
app = Flask(__name__)

# Function to recommend books based on title
def recommend_books(title, num_recommendations=5):
    # Step 1: Check for books that contain the title as a substring
    matching_books = df[df['title'].str.contains(title, case=False, na=False)]

    if matching_books.empty:
        return "No books found with that title substring."
    
    # Step 2: Find the index for the title
    idx = None
    if title in indices:
        idx = indices[title]
    else:
        # If the title is not found exactly, take the first match from the substring search
        idx = matching_books.index[0]
    
    distances, neighbors = nn.kneighbors([embeddings[idx]])

    recommended_indices = neighbors[0][1:num_recommendations+1]
    
    # Step 3: Recommend books that contain the title substring and the actual recommendations
    recommended_books = df[['title', 'authors', 'categories']].iloc[recommended_indices]
    
    # Combine books containing the title substring and the actual recommendations
    final_recommendations = pd.concat([matching_books[['title', 'authors', 'categories']], recommended_books])

    return final_recommendations.drop_duplicates().reset_index(drop=True)

def recommend_books_ro(title, num_recommendations=5):
    # Step 1: Check for books that contain the title as a substring
    matching_books = dfro[dfro['titlu'].str.contains(title, case=False, na=False)]

    if matching_books.empty:
        return "No books found with that title substring."
    
    # Step 2: Find the index for the title
    idx = None
    if title in indicesro:
        idx = indicesro[title]
    else:
        # If the title is not found exactly, take the first match from the substring search
        idx = matching_books.index[0]
    
    distances, neighbors = nnro.kneighbors([embeddingsro[idx]])

    recommended_indices = neighbors[0][1:num_recommendations+1]
    
    # Step 3: Recommend books that contain the title substring and the actual recommendations
    recommended_books = dfro[['titlu', 'autor', 'genwiki']].iloc[recommended_indices]
    
    # Combine books containing the title substring and the actual recommendations
    final_recommendations = pd.concat([matching_books[['titlu', 'autor', 'genwiki']], recommended_books])

    return final_recommendations.drop_duplicates().reset_index(drop=True)


# Route for homepage to serve the frontend HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Assuming your HTML file is inside the "templates" folder

@app.route('/ro')
def home_ro():
    return render_template('ro.html')  # Assuming your HTML file is inside the "templates" folder

# Route for book recommendations
@app.route('/recommend', methods=['GET'])
def recommend_books_route():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a title parameter"}), 400
    
    recommendations = recommend_books(title)

    if isinstance(recommendations, str):  # If we got a string, it's an error message
        return jsonify({"error": recommendations}), 404
    
    # Convert the recommendations DataFrame to a list of dictionaries for JSON response
    recommendations_list = recommendations.to_dict(orient='records')
    
    return jsonify({"recommendations": recommendations_list})

@app.route('/ro/recommend', methods=['GET'])
def recommend_books_ro_route():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Please provide a title parameter"}), 400
    
    recommendations = recommend_books_ro(title)

    if isinstance(recommendations, str):  # If we got a string, it's an error message
        return jsonify({"error": recommendations}), 404
    
    # Convert the recommendations DataFrame to a list of dictionaries for JSON response
    recommendations_list = recommendations.to_dict(orient='records')
    
    return jsonify({"recommendations": recommendations_list})

if __name__ == '__main__':
    app.run(debug=True)