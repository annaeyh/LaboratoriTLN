import nltk 
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Resource Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# File loading
df = pd.read_csv('../CSV/Dataset_definizioni.csv')
musics = df.iloc[:, 1].tolist()
ethics = df.iloc[:, 2].tolist()
trees = df.iloc[:, 3].tolist()
teapots = df.iloc[:, 4].tolist()

# Preprocessing function
def clean_up(entry):
    result = []
    for string in entry:
        words = nltk.word_tokenize(string.lower())
        filtered = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        result.append(" ".join(filtered))
    return result

# Jaccard similarity function
def jaccard(def1, def2):
    words1 = set(def1.lower().split())
    words2 = set(def2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    jaccard_score = len(intersection) / len(union) if union else 0
    return jaccard_score

# Mean Jaccard similarity for a column of definitions
def jaccard_column(column):
    values = []
    # Iterate over all unique pairs of definitions in the column
    for i in range(len(column)):
        for j in range(i+1, len(column)):
            values.append(jaccard(column[i], column[j]))
    return np.mean(values)

# Lexical similarity using Jaccard index
def lexical(trees, musics, ethics, teapots):
    overlap_tree = jaccard_column(trees)
    overlap_music = jaccard_column(musics)
    overlap_ethics = jaccard_column(ethics)
    overlap_teapot = jaccard_column(teapots)
    print(f"Similarità di Jaccard: \n Tree: {overlap_tree:.2f} \n Teapot: {overlap_teapot:.2f} \n Music: {overlap_music:.2f} \n Ethic: {overlap_ethics:.2f}")

# Cosine similarity for a column of embeddings
def cosine_similarity(column):
    values = []
    # Iterate over all unique pairs of definitions in the column
    for i in range(len(column)):
        for j in range(i+1, len(column)):
            sim = util.cos_sim(column[i], column[j])
            values.append(sim.item())
    return np.mean(values)

# Semantic similarity using cosine similarity of sentence embeddings
def semantic(trees, musics, ethics, teapots):
    trees_emb = model.encode(trees)
    musics_emb = model.encode(musics)
    ethics_emb = model.encode(ethics)
    teapots_emb = model.encode(teapots)

    cos_sim_tree = cosine_similarity(trees_emb)
    cos_sim_musics = cosine_similarity(musics_emb)
    cos_sim_ethics = cosine_similarity(ethics_emb)
    cos_sim_teapot = cosine_similarity(teapots_emb)
    print(f"Similarità Coseno: \n Tree: {cos_sim_tree:.2f} \n Teapot: {cos_sim_teapot:.2f} \n Music: {cos_sim_musics:.2f} \n Ethic: {cos_sim_ethics:.2f}")

if __name__ == "__main__":
    trees_cleaned = clean_up(trees)
    musics_cleaned = clean_up(musics)
    teapots_cleaned = clean_up(teapots)
    ethics_cleaned = clean_up(ethics)

    lexical(trees_cleaned, musics_cleaned, ethics_cleaned, teapots_cleaned)
    semantic(trees, musics, ethics, teapots)