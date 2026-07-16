import nltk
import pandas as pd
from nltk.wsd import lesk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from ewiser.spacy.disambiguate import Disambiguator

# Caricamento strumenti necessari
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
weights = "C:/Users/Anna/Desktop/UNI/TLN/Di Caro/LAB3_6/ewiser/ewiser.semcor_base.pt"
wsd = Disambiguator(weights, lang="en")
nlp.add_pipe(wsd, last=True)
df_bench = pd.read_csv('../CSV/Benchmark.csv')

# Algoritmo di Lesk
def lesk_algorithm(df):
    synsets_medical = []
    synsets_common = []
    for index, row in df.iterrows():
        word = row['Term']
        sentence_medical = row['Sentence Medical']
        sentence_common = row['Sentence Common']
        synset_medical = lesk(sentence_medical.split(), word)
        synset_common = lesk(sentence_common.split(), word)
        synsets_medical.append(synset_medical)
        synsets_common.append(synset_common)
    return synsets_medical, synsets_common

# Algoritmo Most Frequent Sense
def MFS(df):
    synsets = []
    for index, row in df.iterrows():
        word = row['Term']
        synset = wn.synsets(word)[0] if wn.synsets(word) else None
        synsets.append(synset)
    return synsets

# Ottenimento del synset a partire dal codice ofset
def offset_to_synset(offset):
    if not isinstance(offset, str) or not offset.startswith('wn:'):
        return None

    codice = offset.replace('wn:', '')
    pos = codice[-1]
    offset_num = int(codice[:-1])
    try:
        return wn.synset_from_pos_and_offset(pos, offset_num)
    except:
        return None

# Algoritmo di disambiguazione tramite embedding
def embedding(df):
    synsets_medical = []
    synsets_common = []
    for index, row in df.iterrows():
        word = row['Term']
        sentence_medical = row['Sentence Medical']
        sentence_common = row['Sentence Common']
        # Calcolo embedding delle frasi di esempio
        docmed = nlp(sentence_medical)
        doccom = nlp(sentence_common)
        # Salvataggio synset del termine nella frase di contesto
        for token in docmed:
            if token.text.lower().strip() == word.lower().strip():
                synset_medical = token._.offset
                break

        for token in doccom:
            if token.text.lower().strip() == word.lower().strip():
                synset_common = token._.offset
                break

        synsets_medical.append(offset_to_synset(synset_medical))
        synsets_common.append(offset_to_synset(synset_common))
    return synsets_medical, synsets_common

# Risultati
synsets_lesk_medical, synsets_lesk_common = lesk_algorithm(df_bench)
synsets_mfs = MFS(df_bench)
sunsets_ewiser_medical, synsets_ewiser_common = embedding(df_bench)

# Gold Standard
synset_true_medical = df_bench['Synset Medical'].tolist()
synset_true_common = df_bench['Synset Common'].tolist()

# Composizione di un dataset completo
df_synsets = pd.DataFrame({'Term': df_bench['Term'],
                           'Synset Medical': df_bench['Synset Medical'],
                           'Synset Common': df_bench['Synset Common'],
                           'Synset Lesk Medical': [s.name() for s in synsets_lesk_medical],
                           'Synset Lesk Common': [s.name() for s in synsets_lesk_common],
                            'Synset MFS': [s.name() for s in synsets_mfs],
                           'Synset EWISER Medical': [s.name() if s is not None else "NON TROVATO" for s in sunsets_ewiser_medical],
                            'Synset EWISER Common': [s.name() if s is not None else "NON TROVATO" for s in synsets_ewiser_common]})
df_synsets.to_csv('../CSV/Synsets_Predicted.csv', index=False)

# Valutazione dei risultati (globale e diviso per entropia)
def evaluate(synsets_predicted, synsets_true):
    corrects = 0
    corrects_quintile = [0,0,0,0,0]
    len_quintile = [0,0,0,0,0]
    for i in range(len(synsets_predicted)):
        len_quintile[df_bench['Quintile'][i]-1] += 1
        if hasattr(synsets_predicted[i], 'name'):
            if synsets_predicted[i].name() == synsets_true[i]:
                corrects += 1
                corrects_quintile[df_bench['Quintile'][i] - 1] += 1
    return corrects / len(synsets_predicted), [corrects_quintile[i] / len_quintile[i] for i in range(5)]

accuracy_lesk_medical, accuracies_lesk_medical_q = evaluate(synsets_lesk_medical, synset_true_medical)
accuracy_lesk_common, accuracies_lesk_common_q = evaluate(synsets_lesk_common, synset_true_common)
print(f"Lesk Medical Accuracy Total: {accuracy_lesk_medical:.2f}")
print(f"Lesk Common Accuracy Total: {accuracy_lesk_common:.2f}")

accuracy_ewiser_medical, accuracies_ewiser_medical_q = evaluate(sunsets_ewiser_medical, synset_true_medical)
accuracy_ewiser_common, accuracies_ewiser_common_q = evaluate(synsets_ewiser_common, synset_true_common)
print(f"Ewiser Medical Accuracy Total: {accuracy_ewiser_medical:.2f}")
print(f"Ewiser Common Accuracy Total: {accuracy_ewiser_common:.2f}")

accuracy_mfs_medical, accuracies_mfs_medical_q = evaluate(synsets_mfs, synset_true_medical)
accuracy_mfs_common, accuracies_mfs_common_q = evaluate(synsets_mfs, synset_true_common)
print(f"MFS Medical Accuracy Total: {accuracy_mfs_medical:.2f}")
print(f"MFS Common Accuracy Total: {accuracy_mfs_common:.2f}")

df_entropy = pd.read_csv('../CSV/Dataset_medico_con_Entropia.csv')

# Costruzione del dataframe da utilizzare per i grafici
# Calcolo entropia media per quantile
df_medie = df_entropy.groupby('Quintile')['Entropy'].mean().reset_index()
data = []
for i in range(len(df_medie)):
    # Estraggo il quintile della parola corrente
    q = df_medie['Quintile'][i]
    m = df_medie['Entropy'][i]
    # Dati Lesk
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_lesk_medical_q[i], 'Algorithm': 'Lesk', 'Domain': 'Medical'})
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_lesk_common_q[i], 'Algorithm': 'Lesk', 'Domain': 'Common'})
    # Dati MFS
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_mfs_medical_q[i], 'Algorithm': 'MFS', 'Domain': 'Medical'})
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_mfs_common_q[i], 'Algorithm': 'MFS', 'Domain': 'Common'})
    # Dati Ewiser
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_ewiser_medical_q[i], 'Algorithm': 'EWISER', 'Domain': 'Medical'})
    data.append({'Quintile': q, 'Entropy': m, 'Accuracy': accuracies_ewiser_common_q[i], 'Algorithm': 'EWISER', 'Domain': 'Common'})

df_sumup = pd.DataFrame(data)

# 1. Box Plot per confronto accuracies globale Lesk vs MFS ------------------------------------------------------
data_global = [
    {'Algorithm': 'MFS', 'Domain': 'Common', 'Accuracy': accuracy_mfs_common},
    {'Algorithm': 'Lesk', 'Domain': 'Common', 'Accuracy': accuracy_lesk_common},
    {'Algorithm': 'MFS', 'Domain': 'Medical', 'Accuracy': accuracy_mfs_medical},
    {'Algorithm': 'Lesk', 'Domain': 'Medical', 'Accuracy': accuracy_lesk_medical},
    {'Algorithm': 'EWISER', 'Domain': 'Common', 'Accuracy': accuracy_ewiser_common},
    {'Algorithm': 'EWISER', 'Domain': 'Medical', 'Accuracy': accuracy_ewiser_medical}
]
df_global = pd.DataFrame(data_global)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_global,
    x="Algorithm",
    y="Accuracy",
    hue="Domain",
    palette={"Medical": "mediumseagreen", "Common": "steelblue"},
    edgecolor="black",
    linewidth=1.5
)
plt.title("Accuratezza Globale dei 3 algoritmi", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Algoritmo", fontsize=12)
plt.ylabel("Accuratezza Globale", fontsize=12)
plt.ylim(0, 1.1)
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=12)
plt.legend(title="Dominio", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../images/Grafico_Globale_Barplot.png", dpi=300)
plt.show()

# 2. Heat Map per il confronto tra entropia media del quintile e accuratezza ---------------------------------------------------
df_sumup['Alg_Dom'] = df_sumup['Algorithm'] + ' - ' + df_sumup['Domain']
heatmap_data = df_sumup.pivot(index='Alg_Dom', columns='Quintile', values='Accuracy')

plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=.5,
    cbar_kws={'label': 'Accuratezza Relativa (Auto-scalata)'}
)
plt.title("Heatmap delle Performance: Algoritmi e Domini", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Quintile di Entropia")
plt.ylabel("Modello - Dominio")
plt.yticks(rotation=0)
plt.savefig("../images/4_Heatmap_Migliorata.png", dpi=300, bbox_inches='tight')
plt.show()


