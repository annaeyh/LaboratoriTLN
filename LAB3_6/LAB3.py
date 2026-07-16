import math
from nltk.corpus import wordnet as wn
import pandas as pd
import ollama

# Caricamento del csv
df = pd.read_csv('../CSV/Dataset_medico.csv')

# Calcola l'entropia per ogni parola
def calculate_entropy(word):
    word = word.strip().lower()

    lemmas = wn.lemmas(word)
    if not lemmas:
        return 0.0

    counts = [lemma.count() for lemma in lemmas]
    total = sum(counts)

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

# Applicazione funzione entropia e creazione della colonna
df['Entropy'] = df['English'].apply(calculate_entropy)

# Arrotondamento dei valori a 2 cifre decimali e esclusione dei termini a entropia 0
df['Entropy'] = df['Entropy'].round(2)
df_active = df[df['Entropy'] > 0]

# Creazione colonna quintili e applicazione funzione pandas dedicata
df_active['Quintile'] = pd.qcut(df_active['Entropy'], 5, labels=[1,2,3,4,5])

# Ordinamento in base ai quintili
df_ordered = df_active.sort_values(by='Entropy', ascending=False)

output_file = '../CSV/Dataset_medico_con_Entropia.csv'
df_ordered.to_csv(output_file, index=False)

# Genera il benchmark prompt per i 5 gruppi per entropia
df_prompt = pd.DataFrame(columns=['Term', 'Synset Medical', 'Synset Common', 'Sentence Medical', 'Sentence Common', 'Quintile'])
for i in range(df_ordered.shape[0]):
    # Prendo termini e synset necessari come input e contesto
    word = df_ordered.iloc[i]['English']
    synset_medical = wn.synset(df_ordered.iloc[i]['Medical'])
    synset_common = wn.synset(df_ordered.iloc[i]['Common'])

    # Prompt utilizzato per la generazione puntuale in ambito medico
    prompt = f"Write a natural English sentence using the word '{word}'. It MUST be used in this exact sense: '{synset_medical.definition()}'. Output ONLY the sentence, nothing else."
    # Generazione frase in ambito medico
    response = ollama.chat(model='phi3', messages=[
        {'role': 'user', 'content': prompt}
    ])
    sentence = response['message']['content'].strip('"')

    # Stessa cosa ma nel contesto comune
    prompt_common = f"Write a natural English sentence using the word '{word}'. It MUST be used in this exact sense: '{synset_common.definition()}'. Output ONLY the sentence, nothing else."
    response_common = ollama.chat(model='phi3', messages=[
        {'role': 'user', 'content': prompt_common}
    ])
    sentence_common = response_common['message']['content'].strip('"')

    # Aggiunta al dataframe di destinazione
    df_prompt.loc[len(df_prompt)] = [word, synset_medical.name(), synset_common.name(), sentence, sentence_common, df_ordered.iloc[i]['Quintile']]

output_file = '../CSV/Benchmark.csv'
df_prompt.to_csv(output_file, index=False)