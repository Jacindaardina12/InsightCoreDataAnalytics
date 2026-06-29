# ============================================================
# clustering_sku.ipynb
# Tujuan: Segmentasi SKU IL VASTO → Fast Mover, Slow Mover, Dead Stock
# Mata Kuliah: Pengenalan Pola
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- KONEKSI DATABASE ---
conn = sqlite3.connect('data/database/replan_report.db')

# --- AMBIL FITUR PER SKU ---
df = pd.read_sql("""
    SELECT
        p.product_name,
        COALESCE(SUM(f.sales_amount), 0)       AS total_sales,
        COALESCE(AVG(ip.days_of_inventory), 0)  AS avg_doi,
        COALESCE(AVG(ip.average_sales_flow), 0) AS avg_flow,
        COALESCE(SUM(fi.stock_quantity), 0)     AS total_stock
    FROM dim_product p
    LEFT JOIN fact_inventory_month f
        ON f.product_id = p.product_id
    LEFT JOIN inventory_parameter ip
        ON ip.product_id = p.product_id
    LEFT JOIN fact_inventory fi
        ON fi.product_id = p.product_id
    GROUP BY p.product_name
    HAVING total_stock > 0
""", conn)

print(f"Total SKU: {len(df)}")
df.head()

# --- NORMALISASI FITUR ---
features = ['total_sales', 'avg_doi', 'avg_flow', 'total_stock']
X = df[features].fillna(0)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ELBOW METHOD (tentukan k optimal) ---
inertia = []
K_range = range(1, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-', linewidth=2)
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method — Penentuan Jumlah Cluster Optimal')
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/elbow_chart.png', dpi=150)
plt.show()
print("→ Pilih k di titik 'siku' grafik di atas")

# --- K-MEANS CLUSTERING (k=3) ---
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(X_scaled)

# --- EVALUASI CLUSTERING ---
sil_score = silhouette_score(X_scaled, df['cluster'])

print("\n=== EVALUASI CLUSTERING ===")
print(f"Jumlah Cluster      : 3")
print(f"Silhouette Score    : {sil_score:.4f}")

if sil_score >= 0.50:
    print("Interpretasi        : Cluster terpisah dengan baik")
elif sil_score >= 0.25:
    print("Interpretasi        : Cluster cukup representatif")
else:
    print("Interpretasi        : Cluster masih saling tumpang tindih")

# Beri label berdasarkan rata-rata total_sales per cluster
cluster_means = (
    df.groupby('cluster')['total_sales']
      .mean()
      .sort_values(ascending=False)
)

label_map = {}

labels = [
    'Fast Mover',
    'Slow Mover',
    'Dead Stock'
]

for i, cluster_id in enumerate(cluster_means.index):
    label_map[cluster_id] = labels[i]

df['segment'] = df['cluster'].map(label_map)

# Beri label berdasarkan rata-rata total_sales per cluster
cluster_means = df.groupby('cluster')['total_sales'].mean().sort_values(ascending=False)
label_map = {}
labels    = ['Fast Mover', 'Slow Mover', 'Dead Stock']
for i, cluster_id in enumerate(cluster_means.index):
    label_map[cluster_id] = labels[i]

df['segment'] = df['cluster'].map(label_map)

# Ringkasan per segment
summary = df.groupby('segment')[features].mean().round(2)
print("\n=== RINGKASAN PER SEGMENT ===")
print(summary)
print(f"\nJumlah SKU per segment:")
print(df['segment'].value_counts())

# --- PCA UNTUK VISUALISASI 2 DIMENSI ---
pca = PCA(n_components=2)

coords = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print("\n=== PCA ===")
print(
    f"Explained Variance (PC1 + PC2): "
    f"{explained_variance.sum()*100:.2f}%"
)

df['pca_x'] = coords[:, 0]
df['pca_y'] = coords[:, 1]

colors = {'Fast Mover': '#2563EB', 'Slow Mover': '#F59E0B', 'Dead Stock': '#EF4444'}

plt.figure(figsize=(10, 6))
for segment, color in colors.items():
    mask = df['segment'] == segment
    plt.scatter(
        df.loc[mask, 'pca_x'],
        df.loc[mask, 'pca_y'],
        c=color, label=segment, alpha=0.7, s=60
    )

plt.title('Segmentasi SKU IL VASTO — K-Means Clustering (PCA 2D)', fontsize=13)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('output/clustering_sku.png', dpi=150)
plt.show()

# --- TAMPILKAN HASIL PER SEGMENT ---
for seg in ['Fast Mover', 'Slow Mover', 'Dead Stock']:
    print(f"\n{'='*50}")
    print(f"SEGMENT: {seg} ({len(df[df['segment']==seg])} SKU)")
    print('='*50)
    print(df[df['segment']==seg][['product_name','total_sales','avg_doi','total_stock']]
          .sort_values('total_sales', ascending=False)
          .head(10)
          .to_string(index=False))

# --- EXPORT HASIL CLUSTERING ---
df.to_csv(
    'output/clustering_sku_result.csv',
    index=False
)

# Export metrik evaluasi
evaluation = pd.DataFrame({
    'metric': [
        'silhouette_score',
        'explained_variance_pca'
    ],
    'value': [
        sil_score,
        explained_variance.sum()
    ]
})

evaluation.to_csv(
    'output/clustering_evaluation.csv',
    index=False
)

print(
    "\n✅ Hasil clustering disimpan ke "
    "output/clustering_sku_result.csv"
)

print(
    "✅ Evaluasi clustering disimpan ke "
    "output/clustering_evaluation.csv"
)