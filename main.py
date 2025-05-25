from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def get_cached_embeddings(texts, cache_path, model, batch_size=300):

    # Daca exista cache, încarcă și returnează.
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Salveaza in cache.
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

# Se încarcă seturile de date folosind librăria pandas.
df = pd.read_csv("./book-dataset/books.csv")

dfro = pd.read_csv("./book-dataset/romanian_corpus.csv")
# Modelul “light” antrenat și pe limba română, aspect esential pentru a găsi într-un timp cat mai scurt legături contextuale dintre coloanele tabelului de mai jos, extras din setul de date.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Se genereaza coloanele din tabel pentru care vor fi manipulate datele.
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

# Se gasesc embeddings, folosind apelarea functiei get_cached_embeddings.

embeddings = get_cached_embeddings(df['combined'], './cache/embeddings_books.pkl', model)
embeddingsro = get_cached_embeddings(dfro['combined'], './cache/embeddings_ro.pkl', model)


# NearestNeighbors pentru căutarea de similitudini 
nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddings)

nnro = NearestNeighbors(n_neighbors=6, metric='cosine').fit(embeddingsro)

# Index pentru o căutare eficienta după titlu
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

indicesro = pd.Series(dfro.index, index=dfro['titlu']).drop_duplicates()

# Inițializarea aplicației Flask
app = Flask(__name__)

# Funcția cu care se recomanda cărți după titlu
def recommend_books(title, num_recommendations=5):
# Dacă exista cărți ce conțin în componenta titlul, atunci se iau și ele în considerare
    matching_books = df[df['title'].str.contains(title, case=False, na=False)]

    if matching_books.empty:
        return "No books found with that title substring."
    
# Se cauta indexul pentru titlu
    idx = None
    if title in indices:
        idx = indices[title]
    else:
        # Dacă numele cărții nu este găsit exact, atunci se ia primul element din căutarea substringului        
        idx = matching_books.index[0]
   
    distances, neighbors = nn.kneighbors([embeddings[idx]])

    recommended_indices = neighbors[0][1:num_recommendations+1]
    
   # Se recomanda cărțile cu titlul ce conțin o parte din numele cărții căutate și cărțile cu numele exact. 
    recommended_books = df[['title', 'authors', 'categories']].iloc[recommended_indices]
    
   # Se combina toate recomandarile
    final_recommendations = pd.concat([matching_books[['title', 'authors', 'categories']], recommended_books])

    return final_recommendations.drop_duplicates().reset_index(drop=True)

def recommend_books_ro(title, num_recommendations=5):
    matching_books = dfro[dfro['titlu'].str.contains(title, case=False, na=False)]

    if matching_books.empty:
        return "No books found with that title substring."
    
    idx = None
    if title in indicesro:
        idx = indicesro[title]
    else:
        idx = matching_books.index[0]
    
    distances, neighbors = nnro.kneighbors([embeddingsro[idx]])

    recommended_indices = neighbors[0][1:num_recommendations+1]
    
    recommended_books = dfro[['titlu', 'autor', 'genwiki']].iloc[recommended_indices]
    
    final_recommendations = pd.concat([matching_books[['titlu', 'autor', 'genwiki']], recommended_books])

    return final_recommendations.drop_duplicates().reset_index(drop=True)


# Ruta catre pagina principala
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/ro')
def home_ro():
    return render_template('ro.html')

@app.route('/cartilemele')
def home_cartilemele():
    return render_template('cartilemele.html')

@app.route('/recommend', methods=['GET'])
def recommend_books_route():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Precizeaza un parametru pentru titlu"}), 400
    
    recommendations = recommend_books(title)

    if isinstance(recommendations, str):
        return jsonify({"error": recommendations}), 404
    
# Transforma recomandarea din DataFrame (structura bidimensionala) într-o listă de 
# dicționare pentru un răspuns de tipul JSON, pentru a putea fi manipulat și mai ușor de utilizat.    
    recommendations_list = recommendations.to_dict(orient='records')
    return jsonify({"recommendations": recommendations_list})

@app.route('/ro/recommend', methods=['GET'])
def recommend_books_ro_route():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Precizeaza un parametru pentru titlu"}), 400
    
    recommendations = recommend_books_ro(title)

    if isinstance(recommendations, str):
        return jsonify({"error": recommendations}), 404
    
    recommendations_list = recommendations.to_dict(orient='records')
    return jsonify({"recommendations": recommendations_list})


@app.route('/cartilemele', methods=['GET'])
def cartilemele_route():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Precizeaza un parametru pentru titlu"}), 400
    
    recommendations = recommend_books_ro(title)

    if isinstance(recommendations, str):
        return jsonify({"error": recommendations}), 404
    
    recommendations_list = recommendations.to_dict(orient='records')
    return jsonify({"recommendations": recommendations_list})

if __name__ == '__main__':
    app.run(debug=True)