from Bio import SeqIO
import pandas as pd
import random
from collections import defaultdict
import os
import shutil

# === Paso 1: Cargar datos ===
def read_fasta_to_dict(path):
    return {record.id: record for record in SeqIO.parse(path, "fasta")}

positive_train = read_fasta_to_dict("positive_epitopes_train.fasta")
negative_train = read_fasta_to_dict("negative_epitopes_train.fasta")
negative_extra = read_fasta_to_dict("negatives_matched10.fasta")
positive_test = read_fasta_to_dict("positive_epitopes_test.fasta")
negative_test = read_fasta_to_dict("negative_epitopes_test.fasta")
positives_final = read_fasta_to_dict("positivos_final.fasta")

# === Paso 2: Construir clústeres desde combined.tsv ===
def build_clusters(pairs):
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for a, b in pairs:
        union(a, b)

    clusters = defaultdict(set)
    for seq in parent:
        root = find(seq)
        clusters[root].add(seq)

    return list(clusters.values())

combined = pd.read_csv("combined.tsv", sep="\t", header=None)
cluster_pairs = combined[[0, 1]].values.tolist()
clusters = build_clusters(cluster_pairs)

# Crear mapa: secuencia → clúster_id
id_to_cluster = {}
for i, cluster in enumerate(clusters):
    for seq_id in cluster:
        id_to_cluster[seq_id] = i

# === Paso 3: Selección de clústeres para test (10% positivos, 10% negativos) ===
random.seed(42)

# Obtener clústeres con positivos y negativos
pos_clusters = {id_to_cluster[k] for k in positives_final if k in id_to_cluster}
neg_clusters = {id_to_cluster[k] for k in negative_extra if k in id_to_cluster}

# Seleccionar 10% de cada tipo
pos_clusters = list(pos_clusters)
neg_clusters = list(neg_clusters)
random.shuffle(pos_clusters)
random.shuffle(neg_clusters)

n_pos_test = max(1, int(0.10 * len(pos_clusters)))
n_neg_test = max(1, int(0.10 * len(neg_clusters)))

test_clusters = set(pos_clusters[:n_pos_test] + neg_clusters[:n_neg_test])

# === Paso 4: Identificar secuencias de test ===
test_ids = set(positive_test) | set(negative_test)
test_ids |= {seq_id for cluster_id in test_clusters for seq_id in clusters[cluster_id]}

# === Paso 5: Construir train pool con todo lo demás ===
all_data = {}
all_data.update(positives_final)
all_data.update(positive_train)
all_data.update(negative_train)
all_data.update(negative_extra)

train_ids = [k for k in all_data if k not in test_ids]

# === Paso 6: Generar folds ===
# Etiquetas por ID
def get_label(seq_id):
    if seq_id in positives_final or seq_id in positive_train:
        return 1
    return 0

train_data = [(k, get_label(k)) for k in train_ids]

# Agrupar por clúster
cluster_map = defaultdict(list)
for seq_id, label in train_data:
    cluster = id_to_cluster.get(seq_id, f"no_cluster_{seq_id}")
    cluster_map[cluster].append((seq_id, label))

# Distribuir clústeres
folds = [[] for _ in range(5)]
fold_labels = [[] for _ in range(5)]
cluster_items = list(cluster_map.items())
random.shuffle(cluster_items)

for cluster_id, items in cluster_items:
    scores = [abs(sum(flab) - (len(flab) - sum(flab))) for flab in fold_labels]
    idx = scores.index(min(scores))
    folds[idx].extend([seq for seq, _ in items])
    fold_labels[idx].extend([lab for _, lab in items])

# === Paso 7: Guardar test y folds ===
os.makedirs("test", exist_ok=True)
positive_test_records = [positive_test.get(k) for k in positive_test if k in test_ids]
positive_test_records += [all_data[k] for k in test_ids if k in all_data and get_label(k) == 1]

negative_test_records = [negative_test.get(k) for k in negative_test if k in test_ids]
negative_test_records += [all_data[k] for k in test_ids if k in all_data and get_label(k) == 0]

SeqIO.write(positive_test_records, "test/positive_test.fasta", "fasta")
SeqIO.write(negative_test_records, "test/negative_test.fasta", "fasta")

# Guardar folds
shutil.rmtree("folds", ignore_errors=True)
os.makedirs("folds", exist_ok=True)

summary = []

for i, fold_ids in enumerate(folds):
    fold_dir = f"folds/fold_{i+1}"
    os.makedirs(fold_dir, exist_ok=True)
    pos_records = [all_data[k] for k in fold_ids if k in all_data and get_label(k) == 1]
    neg_records = [all_data[k] for k in fold_ids if k in all_data and get_label(k) == 0]
    SeqIO.write(pos_records, f"{fold_dir}/train_pos_fold{i+1}.fasta", "fasta")
    SeqIO.write(neg_records, f"{fold_dir}/train_neg_fold{i+1}.fasta", "fasta")
    summary.append((f"Fold {i+1}", len(pos_records), len(neg_records)))

# === Resumen final ===
print("Resumen de secuencias por conjunto:\n")
print(f"Test positivo: {len(positive_test_records)}")
print(f"Test negativo: {len(negative_test_records)}\n")

for fold_name, n_pos, n_neg in summary:
    print(f"{fold_name}: {n_pos} positivos, {n_neg} negativos")
