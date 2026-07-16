import nltk
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import matplotlib.pyplot as plt

# Resources setup
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_lg")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# File loading and filtering
df = pd.read_csv('../CSV/Dataset_definizioni.csv')
mistery = df.iloc[:, 5].tolist()
mistery = [entry for entry in mistery if entry != "Unknown"]

# Preprocessing
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

# Setup Wordnet synset vectors for semantic similarity
def setup_wordnet():
    synsets_list = list(wn.all_synsets(pos='n'))
    synset_vectors = [] # Vectors
    valid_synsets = [] # Corresponding synsets

    # Iteration over all synsets
    for s in synsets_list:
        content = s.definition() + " " + " ".join(s.lemma_names())
        vec = nlp.make_doc(content)

        # Exclude synsets without valid vectors
        if vec.has_vector and vec.vector_norm > 0:
            synset_vectors.append(vec)
            valid_synsets.append(s)

    return synset_vectors, valid_synsets

# Lexical similarity using Jaccard
def synsets(string):
    # Best syntext and overlap score
    best_syn = None
    best_overlap = 0

    # Iteration over all synsets in Wordnet
    for syn in wn.all_synsets(pos='n'):
        syn_text = syn.definition()
        # Preprocess
        syn_words = nltk.word_tokenize(syn_text)
        filtered_syn = set([word for word in syn_words if word.isalpha() and word not in stop_words])
        # Calculate Jaccard similarity
        overlap = jaccard(string, " ".join(filtered_syn))

        # Update best synset if current overlap is better
        if overlap > best_overlap:
            best_overlap = overlap
            best_syn = syn

    return best_syn, best_overlap

# Semantic similarity using Spacy vectors
def synsets2(string, synset_vectors, valid_synsets):
    emb_string = nlp.make_doc(string)

    # Check if the input string has a valid vector representation
    if not emb_string.has_vector or emb_string.vector_norm == 0:
        return None, 0

    # Best synset and similarity score
    best_syn = None
    best_sim = 0

    # Iteration over the two lists
    for i in range(len(synset_vectors)):
        sim = emb_string.similarity(synset_vectors[i])
        # Update best synset if current similarity is better
        if sim > best_sim:
            best_syn = valid_synsets[i]
            best_sim = sim

    return best_syn, best_sim

if __name__ == "__main__":
    # All the results and similarity values
    jaccard_values = []
    cosine_values = []
    lex_results = []
    sem_results = []

    # The target results
    gold_standard = [
        "/",  # Cringe: too recent and not in Wordnet
        "thought.n.03",
        "computer_game.n.01",
        "shame.n.01",
        "waterworks.n.02",
        "sushi.n.01",
        "spectacles.n.01",
        "gasoline.n.01",
        "scientist.n.01",
        "/", # Pixel: too specific and not in Wordnet
        "anger.n.01",
        "ad.n.01",
        "sofa.n.01",
        "gambling.n.01",
        "screwdriver.n.01",
        "plunger.n.03",
        "bobby_pin.n.01",
        "epinephrine.n.01",
        "push_button.n.01",
        "sports_car.n.01",
        "harvester.n.02",
        "migration.n.04",
        "barrier.n.01",
        "idea.n.01",
        "bottle.n.01"
    ]

    # Conparisons --------------------------------
    print("\n ----- Confronto Semantico ----- \n")
    synset_vectors, valid_synsets = setup_wordnet()
    for entry in mistery:
        syn, value = synsets2(entry, synset_vectors, valid_synsets)
        cosine_values.append(value)
        sem_results.append(syn)
        print(f"Definizione: {entry} \n Synset: {syn.name()} \n Definizione Synset: {syn.definition()} \n Similarità: {value:.2f}\n")

    print("\n\n ----- Confronto Lessicale ----- \n")
    cleaned_mistery = clean_up(mistery)
    for entry in cleaned_mistery:
        syn, value = synsets(entry)
        jaccard_values.append(value)
        lex_results.append(syn)
        print(f"Definizione: {entry} \n Synset: {syn.name()} \n Definizione Synset: {syn.definition()} \n Similarità: {value:.2f}\n")

    # Jaccard x Cosine Scatter Plot -------------------
    plt.figure(figsize=(12, 8))
    plt.xlabel('Cosine Similarity')
    color_map = plt.cm.get_cmap('tab20', len(cosine_values))

    for i in range(len(cosine_values)):
        plt.scatter(
            cosine_values[i],
            jaccard_values[i],
            color=color_map(i),
            edgecolors='black',
            s=100,
            label=f'{i + 1}'
        )

    plt.xlim(0.80, 1.00)
    plt.ylim(0.10, 0.40)
    plt.title('Confronto Indici Oggetti', fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Jaccard Similarity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()

    # Accuracy Comparison with Gold Standard -------------------
    print("\n\n ----- Confronto con Gold Standard ----- \n")
    distance_sem = []
    distance_lex = []
    for i in range(len(gold_standard)):
        if gold_standard[i] != "/":
            syn = wn.synset(gold_standard[i])

            # Calculate path similarity for both methods, handling None cases
            s_sim = sem_results[i].path_similarity(syn) if sem_results[i] is not None else 0
            l_sim = lex_results[i].path_similarity(syn) if lex_results[i] is not None else 0

            distance_sem.append(s_sim)
            distance_lex.append(l_sim)
        else:
            # Append 0 if the gold standard is not valid
            distance_sem.append(0)
            distance_lex.append(0)

    # Print average distances for both methods
    for d in distance_sem:
        print(f"{d:.2f}")
    print(f"Media Semantico: {np.mean(distance_sem)} \n\n")
    for dl in distance_lex:
        print(f"{dl:.2f}")
    print(f"Media Lessicale: {np.mean(distance_lex)} \n\n")

    # Distances Bar Plot -------------------
    indices = np.arange(len(gold_standard))
    width = 0.35  # Larghezza delle barre
    plt.figure(figsize=(12, 7))
    plt.bar(indices - width / 2, distance_sem, width, label='Distanza Semantica (Spacy)', color='skyblue', edgecolor='black')
    plt.bar(indices + width / 2, distance_lex, width, label='Distanza Lessicale (Jaccard)', color='salmon', edgecolor='black')

    plt.xlabel('Indice Definizione (1-25)', fontsize=12)
    plt.ylabel('Path Similarity (0-1)', fontsize=12)
    plt.title('Accuratezza dei Metodi: Somiglianza con il Gold Standard', fontsize=15)
    plt.xticks(indices, range(1, 26))
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()