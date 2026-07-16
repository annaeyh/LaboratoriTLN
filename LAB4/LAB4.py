import pickle
import time
import umap
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import os

load_dotenv()
token = os.environ.get("HF_TOKEN")
login(token=token)

def print_statistics(topic_model, topics, execution_time):
    print("Statistics =============================================")
    # 1. Numero di Topic
    topic_info = topic_model.get_topic_info()
    print(f"\tNumber of topics: {len(topic_info) - 1}")
    # 2. Percentuale di Outliers
    outlier_percentage = (topics.count(-1) / len(topics)) * 100
    print(f"\tOutliers percentage: {outlier_percentage:.2f}%")
    # 3. Top 5 keyword usate nei primi 10 topic più usati
    print(f"\tTop 5 words from 10 most used topics:")
    print_top_topics(topic_model, topic_info)
    # 4. Tempo di esecuzione
    print(f"\tExecution time: {execution_time:.2f} seconds")
    # 5. Documenti rappresentativi per 5 topic
    print(f"\tRepresentative documents for 5 topics:")
    print_representative_docs(topic_model, topic_info)

# Stampa dei top 10 topic più usati con le prime 5 parole chiave
def print_top_topics(topic_model, topic_info):
    top_10_topics = topic_info[topic_info['Topic'] != -1].head(10)
    for index, row in top_10_topics.iterrows():
        topic_id = row['Topic']
        doc_count = row['Count']
        top_words = [word_tuple[0] for word_tuple in topic_model.get_topic(topic_id)[:5]]
        words_string = ", ".join(top_words)
        print(f"\t\t• Topic {topic_id} ({doc_count} documents): {words_string}")

# Stampa dei documenti rappresentativi
def print_representative_docs(topic_model, topic_info):
    random_topics = topic_info[topic_info['Topic'] != -1].sample(5 if len(topic_info) -1 > 5 else len(topic_info) -1, random_state=42)['Topic'].tolist()
    for topic_id in random_topics:
        print(f"\t\tTopic {topic_id}:")
        representative_docs = topic_model.get_representative_docs(topic_id)
        for i in range(len(representative_docs)):
            print("\t\t\t" + representative_docs[i])

# Creaz
def visualize(topic_model, data, reduced_embeddings):
    fig_docs = topic_model.visualize_documents(data, reduced_embeddings=reduced_embeddings, hide_document_hover=False)
    fig_bar = topic_model.visualize_barchart(top_n_topics=10)
    fig_hier = topic_model.visualize_hierarchy()
    fig_heat = topic_model.visualize_heatmap()

    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Saving visualizations to the '{output_dir}' directory...")
    fig_docs.write_html(os.path.join(output_dir, "documents_visualization.html"))
    fig_bar.write_html(os.path.join(output_dir, "barchart_visualization.html"))
    fig_hier.write_html(os.path.join(output_dir, "hierarchy_visualization.html"))
    fig_heat.write_html(os.path.join(output_dir, "heatmap_visualization.html"))

if __name__ == "__main__":
    # Input dell'utente ===========================================================================
    in_model = input("Insert one of these: \n\t1. all-MiniLM-L6-v2 \n\t2. all-mpnet-base-v2 \n\t3. pritamdeka/S-PubMedBert-MS-MARCO \n\t4. thenlper/gte-small \nEmbedding model: ")
    in_compon = int(input("Insert one of these: \n\t 1. 5 \n\t 2. 10 \n\t 3. 15 \nUMAP n_components = "))
    in_neigh = int(input("Insert one of these: \n\t 1. 15 \n\t 2. 30 \n\t 3. 50 \nUMAP n_neighbors = "))
    in_mincl = int(input("Insert one of these: \n\t 1. 50 \n\t 2. 100 \n\t 3. 200 \nHDBSCAN min_cluster_size = "))

    # Stampa di info ===========================================================================
    print("You chose: ")
    print(f"\t- Model {in_model}")
    print(f"\t- UMAP with n_components={in_compon}, n_neighbors={in_neigh}")
    print(f"\t- HDBSCAN with min_cluster_size={in_mincl}")

    # Apertura dei dati ================================================================================
    with open('processed_data12.pkl', 'rb') as f:
        data = pickle.load(f)

    # Esecuzione pipeline =========================================================================
    print("Loading the model...")
    model = SentenceTransformer(in_model)
    # Inizia il timer
    print("Running the pipeline...")
    start_time = time.time()
    # Embedding con SentenceTransformer -------------------------------------------------------
    print("\t--> Computing embeddings...")
    embeddings = model.encode(data, show_progress_bar=True)
    # UMAP reduction -----------------------------------------------------------------------
    print("\t--> Reducing dimensionality with UMAP...")
    umap_model = umap.UMAP(n_components=in_compon,
                           min_dist=0.0,
                           metric="cosine",
                           n_neighbors=in_neigh,
                           random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    # HDBSCAN clustering ----------------------------------------------------------------
    print("\t--> Clustering with HDBSCAN...")
    hdbscan_model = HDBSCAN(min_cluster_size=in_mincl,
                            metric="euclidean",
                            cluster_selection_method="eom",
                            prediction_data=True)
    clusters = hdbscan_model.fit_predict(reduced_embeddings)
    # BERTopic modeling ----------------------------------------------------------------
    print("\t--> Executing with BERTopic...")
    topic_model = BERTopic(embedding_model=model,
                           umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           verbose=True)
    topics, probs = topic_model.fit_transform(data, embeddings)
    # Stop timer ---------------------------------------------------------------------
    print("Ending the pipeline...")
    end_time = time.time()
    execution_time = end_time - start_time

    # Esplorazione dei risultati ====================================================================
    print_statistics(topic_model, topics, execution_time)

    # Visualizzazioni ============================================================================
    visualize(topic_model, data, reduced_embeddings)