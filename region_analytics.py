# ============================================================
# region_analytics.ipynb
# Tujuan: Segmentasi Region → High, Medium, Low Performer
# Mata Kuliah: People Analytics
# ============================================================

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

conn = sqlite3.connect('data/database/replan_report.db')

# --- AMBIL DATA PER REGION ---
df = pd.read_sql("""
    SELECT
        r.region_name,
        COALESCE(SUM(f.sales_amount), 0)        AS total_sales,
        COALESCE(AVG(ip.days_of_inventory), 0)   AS avg_doi,
        COALESCE(SUM(fi.stock_quantity), 0)      AS total_stock,
        COALESCE(AVG(ip.average_sales_flow), 0)  AS avg_flow,
        COUNT(DISTINCT f.product_id)             AS jumlah_sku_aktif
    FROM dim_region r
    LEFT JOIN fact_inventory_month f
        ON f.region_id = r.region_id
    LEFT JOIN inventory_parameter ip
        ON ip.region_id = r.region_id
    LEFT JOIN fact_inventory fi
        ON fi.region_id = r.region_id
    WHERE r.region_name NOT IN ('NASIONAL','Old Code','item','product')
    GROUP BY r.region_name
    ORDER BY total_sales DESC
""", conn)

print(f"Total Region: {len(df)}")
df.round(2)

# --- SEGMENTASI BERDASARKAN QUANTILE ---
q33 = df['total_sales'].quantile(0.33)
q66 = df['total_sales'].quantile(0.66)

def segmentasi(val):
    if val >= q66:   return 'High Performer'
    elif val >= q33: return 'Medium Performer'
    else:            return 'Low Performer'

df['segment'] = df['total_sales'].apply(segmentasi)

# --- INFORMASI BATAS SEGMENTASI ---
print("\n=== BATAS SEGMENTASI QUANTILE ===")
print(f"Q33 (Low → Medium)  : {q33:,.0f}")
print(f"Q66 (Medium → High) : {q66:,.0f}")

print("=== SEGMENTASI REGION ===\n")
print(df[['region_name','total_sales','avg_doi','segment']]
      .to_string(index=False))

print(f"\nJumlah region per segment:")
print(df['segment'].value_counts())

# Persentase distribusi segment
segment_pct = (
    df['segment']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPersentase Region per Segment:")
print(segment_pct)

# --- VISUALISASI BAR CHART SALES PER REGION ---
colors_map = {
    'High Performer':   '#22c55e',
    'Medium Performer': '#f59e0b',
    'Low Performer':    '#ef4444'
}
bar_colors = df['segment'].map(colors_map)

plt.figure(figsize=(12, 5))
bars = plt.barh(df['region_name'], df['total_sales'],
                color=bar_colors, edgecolor='white')
plt.xlabel('Total Sales')
plt.title('Performa Penjualan per Region — Segmentasi Distribusi IL VASTO')
plt.gca().invert_yaxis()

# Legend
patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
plt.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.savefig('output/region_segmentasi.png', dpi=150)
plt.show()

# --- VISUALISASI SCATTER: Sales vs DOI per Region ---
plt.figure(figsize=(10, 6))
for seg, color in colors_map.items():
    mask = df['segment'] == seg
    plt.scatter(
        df.loc[mask, 'avg_doi'],
        df.loc[mask, 'total_sales'],
        c=color, label=seg, s=100, alpha=0.8
    )
    for _, row in df[mask].iterrows():
        plt.annotate(
            row['region_name'].split('(')[0].strip(),
            (row['avg_doi'], row['total_sales']),
            fontsize=8, ha='left',
            xytext=(5, 3), textcoords='offset points'
        )

plt.xlabel('Rata-rata DOI (hari)')
plt.ylabel('Total Sales')
plt.title('Hubungan DOI vs Penjualan per Region')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('output/region_doi_vs_sales.png', dpi=150)
plt.show()

# --- RINGKASAN PER SEGMENT ---
summary = df.groupby('segment').agg(
    jumlah_region   = ('region_name', 'count'),
    rata_sales      = ('total_sales', 'mean'),
    rata_doi        = ('avg_doi', 'mean'),
    rata_stock      = ('total_stock', 'mean'),
    rata_sku_aktif  = ('jumlah_sku_aktif', 'mean')
).round(2)

print("\n=== RINGKASAN KARAKTERISTIK PER SEGMENT ===")
print(summary.to_string())

# --- INTERPRETASI SEGMENTASI ---
print("\n=== INTERPRETASI BISNIS ===")

for seg in ['High Performer', 'Medium Performer', 'Low Performer']:

    subset = df[df['segment'] == seg]

    if len(subset) == 0:
        continue

    avg_sales = subset['total_sales'].mean()
    avg_doi   = subset['avg_doi'].mean()

    print(f"\n{seg}")
    print(f"- Jumlah Region : {len(subset)}")
    print(f"- Rata-rata Sales : {avg_sales:,.0f}")
    print(f"- Rata-rata DOI   : {avg_doi:.2f}")

# --- EXPORT HASIL ---
df.to_csv(
    'output/region_segmentasi_result.csv',
    index=False
)

summary.to_csv(
    'output/region_segmentasi_summary.csv'
)

print(
    "\n✅ Hasil segmentasi disimpan ke "
    "output/region_segmentasi_result.csv"
)

print(
    "✅ Ringkasan segmentasi disimpan ke "
    "output/region_segmentasi_summary.csv"
)