from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import re
from Bio import Entrez
import pickle
import nltk

nltk.data.path.append('nltk_data_dir')
nltk.download('stopwords', download_dir='nltk_data_dir')
nltk.download('punkt', download_dir='nltk_data_dir')
nltk.download('punkt_tab', download_dir='nltk_data_dir')
nltk.download('wordnet', download_dir='nltk_data_dir')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

Entrez.email = "annacarelli02@gmail.com"

# Recupero degli abstract dai record
def extract_abstracts(records):
    abstracts = []
    for record in records:
        article = record.get("MedlineCitation", {}).get("Article", {})
        abstract_obj = article.get("Abstract", {})
        abstract_parts = abstract_obj.get("AbstractText", [])

        abstract_text = " ".join(str(part) for part in abstract_parts).strip()
        if abstract_text:
            abstracts.append(abstract_text)

    return abstracts

# Preprocess dei dati
def preprocess(data):
    results = []
    stopword_list = stopwords.words('english')
    domain_stopwords = ['paper', 'method', 'propose', 'proposed', 'approach', 'model']
    stopword_list.extend(domain_stopwords)
    lemmatizer = WordNetLemmatizer()

    for record in data:
        record = re.sub(r'http\S+|www\S+|https\S+', '', record, flags=re.MULTILINE)
        sentence = sent_tokenize(record)
        table = str.maketrans('', '', string.punctuation)
        abstract_tokens = []
        for s in sentence:
            lower_tokens = [w.lower() for w in word_tokenize(s)]
            filtered_tokens = [t.translate(table) for t in lower_tokens if t.isalpha()]
            tokens = [word for word in filtered_tokens if word not in stopword_list]
            tokens_fin = [lemmatizer.lemmatize(word) for word in tokens]
            abstract_tokens.extend(tokens_fin)
        clean_abstract_string = " ".join(abstract_tokens)
        results.append(clean_abstract_string)
    return results

if __name__ == "__main__":
    target_k = 45000
    valid_abstracts = []
    # Usiamo set() per salvare dati unici e controllare i duplicati
    seen_abstracts = set()
    all_pmids = set()

    print("Raccolta di PMID unici... ")
    # Facciamo ricerche separate per anno per aggirare il limite di 9999
    years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012]
    for year in years:
        print(f"  -> Cerco articoli pubblicati nel {year}...")
        search_handle = Entrez.esearch(db="pubmed", term=f"{year}[dp]", retmax=4000)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        ids = search_results.get("IdList", [])
        all_pmids.update(ids)
        # Recuperiamo più id di quelli richiesti per sicurezza
        if len(all_pmids) >= 55000:
            break

    # Convertiamo il set in una lista
    all_pmids = list(all_pmids)
    print(f"Trovati {len(all_pmids)} ID unici. Inizio il download degli abstract...")

    # Recupero dei record dalla lista di id in batch di 500 elementi
    batch_size = 500
    for i in range(0, len(all_pmids), batch_size):
        batch_ids = all_pmids[i:i + batch_size]
        try:
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(batch_ids), retmode="xml")
            fetch_data = Entrez.read(fetch_handle)
            fetch_handle.close()
            records = fetch_data.get("PubmedArticle", [])
            batch_abstracts = extract_abstracts(records)

            # Controllo duplicati e inserimento
            for abstract in batch_abstracts:
                if abstract not in seen_abstracts:
                    seen_abstracts.add(abstract)
                    valid_abstracts.append(abstract)
                    if len(valid_abstracts) == target_k:
                        break

            print(f"\t--> Ottenuti {len(valid_abstracts)} abstract VALIDI e UNICI")
            if len(valid_abstracts) == target_k:
                break
        except Exception as e:
            print(f"Errore durante il download del blocco (ID saltati): {e}")

    print(f"\nRaccolti con successo esattamente {len(valid_abstracts)} abstract unici e validi.")

    # Preprocess dei dati
    print("Pre-processing dei dati in corso...")
    preprocessed_data = preprocess(valid_abstracts)

    # Salvataggio in formato pickle
    print("Salvataggio in processed_data45.pkl...")
    with open('processed_data45.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    #with open('raw_data.pkl', 'wb') as f:
        #pickle.dump(valid_abstracts, f)

    print("Pipeline completata.")