import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
import csv

def remove_urls(text):
    if not text: return ""
    url_pattern = re.compile(r'http\S+|www\S+|https\S+', re.IGNORECASE)
    return url_pattern.sub('', text)

def load_labels(path):
    """Carica le label in un dizionario {ID: Label}."""
    labels = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0].strip()] = parts[1].strip()
    return labels

# Depressione
path_dep = "/home/luca/Scrivania/UNI/Magistrale/Big Data/progetto/eRisk2018/2018/task_1_depression/chunk_estratti"
label_file_dep = "/home/luca/Scrivania/UNI/Magistrale/Big Data/progetto/eRisk2018/2018/task_1_depression/risk-golden-truth-test.txt"

# Anoressia
path_ano = "/home/luca/Scrivania/UNI/Magistrale/Big Data/progetto/eRisk2018/2018/task 2 - anorexia/train"
label_file_ano = "/home/luca/Scrivania/UNI/Magistrale/Big Data/progetto/eRisk2018/2018/task 2 - anorexia/train/risk_golden_truth.txt"

# Output
output_file = "/home/luca/Scrivania/UNI/Magistrale/Big Data/progetto/all_user_with_anorexia.csv"

labels_dep = load_labels(label_file_dep)
labels_ano = load_labels(label_file_ano)

data = []

# Funzione helper per processare una cartella
def process_folder(base_path, label_dict, task_type):
    """
    task_type: 'depression' o 'anorexia'
    Serve per decidere come mappare la label 1.
    """
    files_found = 0
    for root_dir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".xml"):
                files_found += 1
                full_path = os.path.join(root_dir, file)
                
                try:
                    tree = ET.parse(full_path)
                    root = tree.getroot()
                    id_ = root.findtext("ID", default="").strip()
                    
                    # Recupera la label dal dizionario
                    raw_label = label_dict.get(id_)
                    
                    # Logica assegnazione label numerica
                    final_label = None
                    if raw_label is not None:
                        val = int(raw_label)
                        if task_type == 'depression':
                            final_label = val # 0 o 1
                        elif task_type == 'anorexia':
                            # Se è 1 (malato) diventa 2, se 0 resta 0
                            final_label = 2 if val == 1 else 0
                    
                    # Se non abbiamo la label (es. file non nel ground truth), saltiamo
                    if final_label is None:
                        continue

                    # Estrae i writing
                    for writings in root.findall("WRITING"):
                        title = remove_urls(writings.findtext("TITLE", default="")).strip().lower()
                        date = writings.findtext("DATE", default="").strip()
                        text = remove_urls(writings.findtext("TEXT", default="")).strip().lower()
                        info = writings.findtext("INFO", default="").strip()
                        
                        data.append({
                            "id": id_,
                            "title": title,
                            "date": date,
                            "text": text,
                            "info": info,
                            "label": final_label
                        })
                except Exception as e:
                    print(f"Errore nel file {file}: {e}")
    print(f"Processati {files_found} file per {task_type}")


print("Inizio elaborazione Depressione...")
process_folder(path_dep, labels_dep, 'depression')

print("Inizio elaborazione Anoressia...")
process_folder(path_ano, labels_ano, 'anorexia')

# Creazione DataFrame
df = pd.DataFrame(data, columns=["id", "title", "date", "text", "info", "label"])

print(f"Totale righe estratte: {len(df)}")
print(df.head(5))

# Salvataggio
df_sorted = df.sort_values(by="id").reset_index(drop=True)
df_sorted.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
print(f"File salvato in: {output_file}")
