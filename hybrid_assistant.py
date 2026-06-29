from __future__ import annotations

import re
import sqlite3
from typing import Any
from difflib import SequenceMatcher
import logging

from src.model_client import DistilLabsLLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# REFERENSI STRUKTUR DATABASE
# =========================================================
# fact_inventory_month : product_id, region_id, month_date, year, month, sales_amount, bbk
# inventory_parameter  : product_id, region_id, lead_time, average_sales_flow, days_of_inventory, replenishment
# fact_inventory       : product_id, region_id, stock_quantity, intransit_quantity, snapshot_date, year, month
# dim_product          : product_id, item_code, old_code, product_name
# dim_region           : region_name, region_code, region_id
# product_category     : product_id, product_name, category_name  <-- PERBAIKAN: Didaftarkan di skema

# =========================================================
# CITY ALIAS
# =========================================================
CITY_ALIAS = {
    "surabaya": "JATIM (Surabaya)", "jatim": "JATIM (Surabaya)",
    "jakarta": "JABAR-JKT (Jakarta)", "jabar": "JABAR-JKT (Jakarta)",
    "bandung": "JABAR-JKT (Jakarta)", "jkt": "JABAR-JKT (Jakarta)",
    "yogyakarta": "YGY (Yogyakarta)", "jogja": "YGY (Yogyakarta)", "ygy": "YGY (Yogyakarta)",
    "medan": "MDN (Medan)", "mdn": "MDN (Medan)",
    "padang": "PDG (Padang)", "pdg": "PDG (Padang)",
    "pekanbaru": "PKU (Pekanbaru)", "pku": "PKU (Pekanbaru)",
    "palembang": "PLG (Palembang)", "plg": "PLG (Palembang)",
    "banjarmasin": "BJM (Banjarmasin)", "bjm": "BJM (Banjarmasin)",
    "balikpapan": "BKP (Balikpapan)", "bkp": "BKP (Balikpapan)",
    "pontianak": "PTK (Pontianak)", "ptk": "PTK (Pontianak)",
    "makassar": "MKS (Makassar)", "mks": "MKS (Makassar)",
    "manado": "MND (Manado)", "mnd": "MND (Manado)",
    "ambon": "AMB (Ambon)", "amb": "AMB (Ambon)",
    "kupang": "KPG (Kupang)", "kpg": "KPG (Kupang)",
    "bali": "BALI (Denpasar)", "denpasar": "BALI (Denpasar)",
    "marketplace": "MKP (Marketplace)", "online": "MKP (Marketplace)", "mkp": "MKP (Marketplace)",
}

# Region yang dikecualikan dari semua perhitungan
EXCLUDE_REGIONS = "('NASIONAL','Old Code','item','product')"
EXCLUDE_LIST    = ['NASIONAL', 'Old Code', 'item', 'product']

# =========================================================
# MONTH MAP
# =========================================================
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "januari": "01", "februari": "02", "maret": "03",
    "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
    "oktober": "10", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
    "agt": "08", "okt": "10", "des": "12", "nopember": "11",
}

# =========================================================
# SEMANTIC KEYWORDS
# =========================================================
SEMANTIC_KEYWORDS = {
    "region": [
        "region", "wilayah", "daerah", "area", "zona", "cabang",
        "kota", "provinsi", "lokasi", "kota mana"
    ],
    "sales": [
        "sales", "penjualan", "revenue", "omset", "pendapatan",
        "terjual", "laris", "jual", "dijual", "terjual"
    ],
    "highest": [
        "tertinggi", "paling tinggi", "terbesar", "top", "highest",
        "max", "terbanyak", "terlaris", "paling laris", "best seller",
        "paling bagus", "paling banyak", "paling baik", "terbaik"
    ],
    "lowest": [
        "terendah", "paling rendah", "terkecil", "lowest", "min",
        "tersedikit", "paling sedikit", "paling tidak laku",
        "tidak laku", "jarang terjual", "paling jarang",
        "paling jelek", "jelek", "paling buruk", "buruk",
        "paling parah", "terburuk", "paling lemah"
    ],
    "product": [
        "product", "produk", "barang", "item", "sku",
        "nama produk", "produk apa", "barang apa", "item apa"
    ],
    "stock": [
        "stock", "stok", "persediaan", "jumlah stok",
        "barang tersedia", "sisa barang", "stok barang"
    ],
    "doi": [
        "doi", "days of inventory", "hari stok", "ketahanan stok",
        "tahan berapa hari", "berapa hari lagi", "bisa tahan"
    ],
    "replenishment": [
        "replenishment", "replan", "perlu ditambah",
        "perlu restock", "perlu diisi", "tambah stok"
    ],
    "dead_stock": [
        "dead stock", "tidak terjual", "stok mati", "menumpuk", "numpuk",
        "tidak laku-laku", "tidak pernah terjual", "tidak bergerak",
        "barang tidak bergerak", "stok numpuk"
    ],
    "intransit": [
        "intransit", "in transit", "dalam perjalanan",
        "masih dikirim", "sedang dikirim"
    ],
    "bbk": [
        "bbk", "barang belum kirim", "belum dikirim",
        "pengiriman belum selesai"
    ],
    "trend": [
        "tren", "trend", "per bulan", "setiap bulan", "tiap bulan",
        "dari bulan", "perkembangan", "monthly", "penjualan per bulan",
        "4 bulan", "bulanan"
    ],
    "low_stock": [
        "hampir habis", "hampir kosong", "mau habis", "kritis",
        "perlu segera", "segera habis"
    ],
    "aggregate": [
        "berapa total", "total keseluruhan", "jumlah total",
        "total semua", "berapa jumlah", "total stok",
        "total sales", "total penjualan"
    ],
    "summary": [
        "keseluruhan", "ringkasan", "kondisi bisnis", "performa",
        "overview", "secara umum", "bagaimana bisnis", "bagaimana kondisi"
    ],
    "product_location": [
        "dijual di", "dijual dimana", "dijual di kota", "dijual di mana",
        "lokasi penjualan", "kota mana saja dijual", "dimana dijual",
        "ada di kota mana", "tersedia di"
    ],
    "range": [
        "hingga", "sampai", "s/d", "sd", "dari bulan", "dari tahun"
    ],
}


def contains_semantic(q_lower: str, key: str) -> bool:
    return any(k in q_lower for k in SEMANTIC_KEYWORDS.get(key, []))


def detect_order(q_lower: str) -> str:
    if any(w in q_lower for w in SEMANTIC_KEYWORDS["lowest"]):
        return "ASC"
    return "DESC"


# =========================================================
# PERBAIKAN: DETECT MONTH (Prioritas Format Angka over Teks)
# =========================================================
def detect_month(question: str):
    """
    Deteksi bulan dan tahun dari pertanyaan.
    PERBAIKAN: Urutan regex diubah agar format angka seperti 11-25 / 11-2025 dievaluasi
    terlebih dahulu sebelum teks nama bulan agar tidak bocor ke deteksi tahun global.
    """
    q_lower = question.lower()

    # 1. Format MM-YYYY (misal 11-2025)
    match = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", q_lower)
    if match:
        return match.group(2), f"{int(match.group(1)):02d}"

    # 2. Format YYYY-MM (misal 2025-11)
    match = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", q_lower)
    if match:
        return match.group(1), f"{int(match.group(2)):02d}"

    # 3. Format MM-YY (misal 11-25 → November 2025)
    match = re.search(r"\b(0?[1-9]|1[0-2])[-/](\d{2})\b", q_lower)
    if match:
        return f"20{match.group(2)}", f"{int(match.group(1)):02d}"

    # Fallback ke pencarian tahun dan teks bulan standar
    year_match = re.search(r"(20\d{2})", q_lower)
    year = year_match.group(1) if year_match else None

    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", q_lower):
            return year, num

    # Format: bulan N atau month N
    match = re.search(r"(bulan|month)\s*(\d{1,2})", q_lower)
    if match:
        m = int(match.group(2))
        if 1 <= m <= 12:
            return year, f"{m:02d}"

    return year, None


# =========================================================
# HELPER GENERATE INSIGHT OTOMATIS VIA LLM
# =========================================================
def generate_llm_insight(question: str, data: list, client: DistilLabsLLM) -> str:
    """
    PERBAIKAN: Menjawab tanda silang (X) pada evaluasi LLM.
    Berfungsi untuk otomatisasi pembuatan kalimat insight 2-3 kalimat yang kontekstual
    berdasarkan data riil yang ditarik dari SQLite database.
    """
    if not data:
        return ""
    try:
        prompt = (
            f"Pertanyaan User: {question}\n"
            f"Data Hasil Query: {str(data[:15])}\n\n"
            f"Tugas: Buatlah kalimat insight/analisis singkat sebanyak 2 sampai 3 kalimat saja "
            f"dalam Bahasa Indonesia yang cerdas dan relevan dengan bisnis retail berdasarkan data diatas. "
            f"Jangan memberikan asumsi angka yang di luar dari data riil tersebut. Format langsung dalam teks bersih."
        )
        insight_text = client.generate_text(prompt)
        return f"\n\nInsight : \n💡 {insight_text.strip()}"
    except Exception:
        return ""


def detect_date_range(question: str):
    q_lower = question.lower()
    if not contains_semantic(q_lower, "range"):
        return None

    months_found = []
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", q_lower):
            months_found.append((q_lower.index(name), num))
    months_found.sort()

    years_found = re.findall(r"20\d{2}", q_lower)

    if len(months_found) >= 2:
        m_start = months_found[0][1]
        m_end   = months_found[-1][1]
        yr      = years_found[0] if years_found else None
        return {"type": "month_range", "start_month": m_start, "end_month": m_end, "year": yr}

    if len(years_found) >= 2:
        return {"type": "year_range", "start_year": years_found[0], "end_year": years_found[-1]}

    return None


def detect_top_n(question: str) -> int | None:
    match = re.search(r"top\s+(\d+)", question.lower())
    return int(match.group(1)) if match else None


def detect_numeric_condition(question: str):
    q = question.lower()
    if any(w in q for w in ["bukan 0", "tidak 0", "selain 0", "non zero"]):
        return "> 0"
    if any(w in q for w in ["= 0", "nol", "habis", "kosong", "zero", "tidak memiliki"]):
        return "= 0"
    patterns = [
        (r"(>=|lebih dari sama dengan)\s*(\d+)", ">="),
        (r"(<=|kurang dari sama dengan)\s*(\d+)", "<="),
        (r"(>|lebih dari|di atas)\s*(\d+)", ">"),
        (r"(<|kurang dari|di bawah)\s*(\d+)", "<"),
        (r"(=|sama dengan)\s*(\d+)", "="),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, q)
        if match:
            return f"{operator} {match.group(2)}"
    return None


def detect_all(question: str) -> bool:
    return any(w in question.lower() for w in ["semua", "seluruh", "all", "semuanya"])


def detect_count(question: str) -> bool:
    return any(w in question.lower() for w in ["berapa banyak", "berapa jumlah", "count"])


def detect_trend_query(question: str) -> bool:
    return contains_semantic(question.lower(), "trend")


def detect_doi_query(question: str) -> bool:
    return contains_semantic(question.lower(), "doi")


def detect_dead_stock_query(question: str) -> bool:
    return contains_semantic(question.lower(), "dead_stock")


def detect_replenishment_query(question: str) -> bool:
    return contains_semantic(question.lower(), "replenishment")


def detect_low_stock_query(question: str) -> bool:
    return contains_semantic(question.lower(), "low_stock")


def detect_intransit_query(question: str) -> bool:
    return contains_semantic(question.lower(), "intransit")


def detect_bbk_query(question: str) -> bool:
    return contains_semantic(question.lower(), "bbk")


def detect_product_location_query(question: str) -> bool:
    return contains_semantic(question.lower(), "product_location")


def detect_top_region_by_month(question: str):
    q_lower = question.lower()
    if not contains_semantic(q_lower, "region"):
        return None
    if not contains_semantic(q_lower, "sales"):
        return None
    if not (contains_semantic(q_lower, "highest") or contains_semantic(q_lower, "lowest")):
        return None
    year, month = detect_month(question)
    return (year, month) if month else None


def detect_product_query(question: str):
    q_lower = question.lower()
    if not contains_semantic(q_lower, "product"):
        return None
    if any(w in q_lower for w in ["saja", "semua", "seluruh", "semuanya", "apa saja", "apa aja"]):
        if not (contains_semantic(q_lower, "highest") or contains_semantic(q_lower, "lowest")):
            return None
    if contains_semantic(q_lower, "aggregate"):
        return None
    metric = "stock" if contains_semantic(q_lower, "stock") else "sales"
    year, month = detect_month(question)
    return {
        "metric": metric,
        "year": year,
        "month": month,
        "has_region": contains_semantic(q_lower, "region"),
        "top_n": detect_top_n(question) or 5,
        "order": detect_order(q_lower),
    }


def detect_region_query(question: str):
    q_lower = question.lower()
    if not contains_semantic(q_lower, "region"):
        return None
    return {
        "is_count": detect_count(question),
        "is_all": detect_all(question),
        "metric": "stock" if contains_semantic(q_lower, "stock") else "sales",
        "order": detect_order(q_lower),
        "condition_numeric": detect_numeric_condition(question),
    }


def is_region_focus(question: str) -> bool:
    return contains_semantic(question.lower(), "region") and not contains_semantic(question.lower(), "product")


def detect_master_data_query(question: str):
    q_lower = question.lower()
    trigger_words = [
        "tampilkan", "list", "daftar", "show", "semua", "seluruh",
        "apa saja", "mana saja", "sebutkan", "semuanya", "saja"
    ]
    has_aggregation = any(w in q_lower for w in [
        "tertinggi", "terendah", "top", "rata", "average",
        "terlaris", "paling", "terbesar", "terkecil",
        "berapa total", "total"
    ])
    if any(w in q_lower for w in trigger_words) and not has_aggregation:
        return {
            "product": contains_semantic(q_lower, "product"),
            "region": contains_semantic(q_lower, "region"),
        }
    return None


def detect_specific_entity(question: str, conn: sqlite3.Connection):
    q_lower = question.lower()
    for alias, region_name in CITY_ALIAS.items():
        if re.search(rf"\b{re.escape(alias)}\b", q_lower):
            return {"type": "region", "value": region_name}
    regions = conn.execute("SELECT region_name FROM dim_region").fetchall()
    for r in regions:
        tokens = re.findall(r"[a-zA-Z]+", r[0].lower())
        for t in tokens:
            if len(t) > 2 and re.search(rf"\b{re.escape(t)}\b", q_lower):
                return {"type": "region", "value": r[0]}
    products = conn.execute("SELECT product_name FROM dim_product").fetchall()
    for p in products:
        if p[0].lower() in q_lower:
            return {"type": "product", "value": p[0]}
    return None


# =========================================================
# SAFETY & SCHEMA HELPERS
# =========================================================
def ensure_select_only(sql: str):
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")


def extract_schema_columns(schema: str) -> dict:
    tables = {}
    current_table = None
    for line in schema.splitlines():
        line = line.strip()
        if line.lower().startswith("create table"):
            current_table = line.split()[2]
            tables[current_table] = []
        elif current_table and line and not line.startswith(");"):
            col = line.split()[0]
            tables[current_table].append(col)
    return tables


def get_top_k_tables(question: str, schema_dict: dict, k: int = 3) -> list:
    question_lower = question.lower()
    scores = []
    for table, cols in schema_dict.items():
        combined = table + " " + " ".join(cols)
        score = SequenceMatcher(None, question_lower, combined.lower()).ratio()
        if "sales" in question_lower and "sales_amount" in cols: score += 1.0
        if "doi" in question_lower and "days_of_inventory" in cols: score += 1.0
        if "replen" in question_lower and "replenishment" in cols: score += 1.0
        if "region" in question_lower and any(c in cols for c in ["region_id", "region_name"]): score += 0.5
        if "product" in question_lower and any(c in cols for c in ["product_id", "product_name"]): score += 0.5
        if any(w in question_lower for w in ["bulan", "month"]) and "month_date" in cols: score += 0.5
        if any(w in question_lower for w in ["stok", "stock"]) and "stock_quantity" in cols: score += 0.8
        if any(w in question_lower for w in ["kategori", "category"]) and "category_name" in cols: score += 1.2
        scores.append((score, table))
    scores.sort(reverse=True)
    return [t for s, t in scores[:k] if s > 0.3]


def build_filtered_schema(selected_tables: set, schema_dict: dict) -> str:
    return "\n".join(f"Table {t}: columns {', '.join(schema_dict.get(t, []))}" for t in selected_tables)


def validate_columns(sql: str, schema_dict: dict) -> bool:
    alias_map = {
        "f": "fact_inventory_month", "fi": "fact_inventory",
        "ip": "inventory_parameter", "r": "dim_region", "p": "dim_product", "pc": "product_category",
    }
    for col in re.findall(r"\b[a-zA-Z_]+\.[a-zA-Z_]+\b", sql):
        alias, column_name = col.split(".")
        table_name = alias_map.get(alias)
        if table_name and column_name not in schema_dict.get(table_name, []):
            return False
    return True


def format_natural_answer(data: list[dict]) -> str:
    if not data:
        return "Maaf, data tidak ditemukan."

    keys = list(data[0].keys())

    if len(keys) == 1:
        col = keys[0]
        val = data[0][col]
        if len(data) == 1 and isinstance(val, (int, float)):
            return f"Total **{col.replace('_', ' ')}**: **{int(val):,}**"
        lines = [f"Berikut daftar {col.replace('_', ' ')}:\n"]
        for i, row in enumerate(data, 1):
            lines.append(f"{i}. {row[col]}")
        lines.append(f"\nTotal: {len(data)} data")
        return "\n".join(lines)

    elif len(keys) == 2:
        label_key, value_key = keys
        try:
            data_sorted = sorted(data, key=lambda x: x[value_key] or 0, reverse=True)
        except TypeError:
            data_sorted = data
        lines = [f"**{value_key.replace('_', ' ').title()}** per **{label_key.replace('_', ' ').title()}**:\n"]
        for i, row in enumerate(data_sorted, 1):
            val = row[value_key]
            val_fmt = f"{int(val):,}" if isinstance(val, (int, float)) else str(val)
            lines.append(f"{i}. {row[label_key]} → {val_fmt}")
        return "\n".join(lines)

    lines = [f"Ditemukan **{len(data)} data**:\n"]
    for i, row in enumerate(data[:20], 1):
        parts = " | ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(f"{i}. {parts}")
    if len(data) > 20:
        lines.append(f"... dan {len(data) - 20} data lainnya")
    return "\n".join(lines)


# =========================================================
# MAIN ENGINE
# =========================================================
def answer_question(
    question: str,
    mode: str | None,
    conn: sqlite3.Connection,
    schema: str,
    client: DistilLabsLLM,
    index=None,
    top_k: int = 3,
    hybrid: bool = False,
) -> dict[str, Any]:
    
    GREETINGS = ["halo", "hai", "hello", "hi", "hey", "selamat pagi",
                 "selamat siang", "selamat malam", "pagi", "siang", "malam"]
    if question.strip().lower() in GREETINGS or any(
        question.strip().lower().startswith(g) for g in GREETINGS
    ):
        return {
            "mode": "greeting",
            "answer": (
                "Halo! Ada yang bisa saya bantu hari ini?\n\n"
            ),
            "data": [],
            "sql": "",
        }

    q_lower = question.lower()
    order   = detect_order(q_lower)
    top_n   = detect_top_n(question) or 5
    year, month = detect_month(question)

    def run(sql: str) -> list[dict]:
        cursor = conn.execute(sql)
        cols   = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in cursor.fetchall()]

    # =====================================================
    # DATE RANGE QUERY
    # =====================================================
    date_range = detect_date_range(question)
    if date_range and contains_semantic(q_lower, "aggregate"):

        if date_range["type"] == "month_range":
            m_start = date_range["start_month"]
            m_end   = date_range["end_month"]
            yr      = date_range["year"]

            yr_filter = f"AND strftime('%Y', f.month_date) = '{yr}'" if yr else ""
            sql = f"""
                SELECT f.month_date AS bulan,
                       SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE strftime('%m', f.month_date) BETWEEN '{m_start}' AND '{m_end}'
                  AND r.region_name NOT IN {EXCLUDE_REGIONS}
                  {yr_filter}
                GROUP BY f.month_date
                ORDER BY f.month_date;
            """
            data     = run(sql)
            grand    = sum(r["total_penjualan"] for r in data)
            insight  = generate_llm_insight(question, data, client)
            answer   = (
                f"**Total penjualan periode bulan {m_start} hingga {m_end}"
                f"{' tahun '+yr if yr else ''}:** **{grand:,}**\n\n"
                + format_natural_answer(data) + insight
            )
            return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql.strip(), "chart_type": "line"}

        elif date_range["type"] == "year_range":
            y_start = date_range["start_year"]
            y_end   = date_range["end_year"]
            sql = f"""
                SELECT strftime('%Y', f.month_date) AS tahun,
                       f.month_date AS bulan,
                       SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE strftime('%Y', f.month_date) BETWEEN '{y_start}' AND '{y_end}'
                  AND r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY f.month_date
                ORDER BY f.month_date;
            """
            data = run(sql)
            per_year: dict[str, int] = {}
            for row in data:
                yr = row["tahun"]
                per_year[yr] = per_year.get(yr, 0) + (row["total_penjualan"] or 0)
            grand = sum(per_year.values())
            year_lines = "\n".join(f"  - Tahun {yr}: {int(total):,}" for yr, total in sorted(per_year.items()))
            insight  = generate_llm_insight(question, data, client)
            answer = (
                f"**Total penjualan {y_start} hingga {y_end}: {grand:,}**\n\n"
                f"Rincian per tahun:\n{year_lines}\n\n"
                + format_natural_answer(data) + insight
            )
            return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql.strip(), "chart_type": "line"}

    # =====================================================
    # AGGREGATE QUERY
    # =====================================================
    if contains_semantic(q_lower, "aggregate"):

        if contains_semantic(q_lower, "stock") and not contains_semantic(q_lower, "sales"):
            sql = f"""
                SELECT SUM(fi.stock_quantity) AS total_stok_keseluruhan
                FROM fact_inventory fi
                JOIN dim_region r ON fi.region_id = r.region_id
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
            """
            data = run(sql)
            insight = generate_llm_insight(question, data, client)
            return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

        if contains_semantic(q_lower, "sales") or "penjualan" in q_lower:
            yr, mo = detect_month(question)
            where_parts = [f"r.region_name NOT IN {EXCLUDE_REGIONS}"]
            if yr:
                where_parts.append(f"strftime('%Y', f.month_date) = '{yr}'")
            if mo:
                where_parts.append(f"strftime('%m', f.month_date) = '{mo}'")
            where = " AND ".join(where_parts)
            sql = f"""
                SELECT SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE {where}
            """
            data = run(sql)
            insight = generate_llm_insight(question, data, client)
            return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # SUMMARY BISNIS
    # =====================================================
    if contains_semantic(q_lower, "summary"):
        yr, mo = detect_month(question)
        time_filter = ""
        if yr and mo:
            time_filter = f"AND strftime('%Y', f.month_date) = '{yr}' AND strftime('%m', f.month_date) = '{mo}'"
        elif yr:
            time_filter = f"AND strftime('%Y', f.month_date) = '{yr}'"

        total_sales = run(f"""
            SELECT SUM(f.sales_amount) AS total_penjualan
            FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
            WHERE r.region_name NOT IN {EXCLUDE_REGIONS} {time_filter}
        """)[0].get("total_penjualan", 0) or 0

        top_region = run(f"""
            SELECT r.region_name, SUM(f.sales_amount) AS total_sales
            FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
            WHERE r.region_name NOT IN {EXCLUDE_REGIONS} {time_filter}
            GROUP BY r.region_name ORDER BY total_sales DESC LIMIT 3
        """)
        top_product = run(f"""
            SELECT p.product_name, SUM(f.sales_amount) AS total_sales
            FROM fact_inventory_month f JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_region r ON f.region_id = r.region_id
            WHERE r.region_name NOT IN {EXCLUDE_REGIONS} {time_filter}
            GROUP BY p.product_name ORDER BY total_sales DESC LIMIT 3
        """)
        stok_kritis = run(f"""
            SELECT COUNT(*) AS jumlah_sku_kritis FROM inventory_parameter ip
            JOIN dim_region r ON ip.region_id = r.region_id
            WHERE ip.days_of_inventory < 7 AND ip.days_of_inventory > 0
              AND r.region_name NOT IN {EXCLUDE_REGIONS}
        """)[0].get("jumlah_sku_kritis", 0) or 0

        bulan_map = {"01":"Januari","02":"Februari","03":"Maret","04":"April",
                     "05":"Mei","06":"Juni","07":"Juli","08":"Agustus",
                     "09":"September","10":"Oktober","11":"November","12":"Desember"}
        period_label = f"bulan {bulan_map.get(mo,mo)} {yr}" if mo and yr else (f"tahun {yr}" if yr else "keseluruhan")

        answer = (
            f"📊 **Ringkasan Bisnis IL VASTO — {period_label.title()}**\n\n"
            f"💰 **Total Penjualan:** {int(total_sales):,}\n\n"
            f"🏆 **Top 3 Wilayah Terlaris:**\n" +
            "\n".join(f"  {i+1}. {r['region_name']} → {int(r['total_sales']):,}" for i, r in enumerate(top_region)) +
            f"\n\n🥇 **Top 3 Produk Terlaris:**\n" +
            "\n".join(f"  {i+1}. {r['product_name']} → {int(r['total_sales']):,}" for i, r in enumerate(top_product)) +
            f"\n\n⚠️ **SKU stok kritis (< 7 hari):** {stok_kritis} SKU"
        )
        return {"mode": "sql_deterministic", "answer": answer, "data": top_region,
                "sql": f"-- Summary untuk {period_label}"}

    # =====================================================
    # TREND QUERY
    # =====================================================
    if detect_trend_query(question):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""

        if entity and entity["type"] == "region":
            sql = f"""
                SELECT f.month_date AS bulan, r.region_name,
                       SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
                GROUP BY f.month_date, r.region_name ORDER BY f.month_date;
            """
        else:
            sql = f"""
                SELECT f.month_date AS bulan, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY f.month_date ORDER BY f.month_date;
            """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight,
                "data": data, "sql": sql.strip(), "chart_type": "line"}

    # =====================================================
    # TOP N REGION BY MONTH
    # =====================================================
    top_result = detect_top_region_by_month(question)
    if top_result:
        yr, mo = top_result
        cond = f"strftime('%m', f.month_date) = '{mo}'"
        if yr:
            cond += f" AND strftime('%Y', f.month_date) = '{yr}'"
        sql = f"""
            SELECT r.region_name, SUM(f.sales_amount) AS total_sales
            FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
            WHERE {cond} AND r.region_name NOT IN {EXCLUDE_REGIONS}
            GROUP BY r.region_name ORDER BY total_sales {order} LIMIT {top_n};
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # TOP N REGION BY YEAR
    # =====================================================
    year_match = re.search(r"(20\d{2})", question)
    if (
        contains_semantic(q_lower, "region") and contains_semantic(q_lower, "sales")
        and (contains_semantic(q_lower, "highest") or contains_semantic(q_lower, "lowest"))
        and year_match and month is None and "rata" not in q_lower
    ):
        yr  = year_match.group(1)
        sql = f"""
            SELECT r.region_name, SUM(f.sales_amount) AS total_sales
            FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
            WHERE strftime('%Y', f.month_date) = '{yr}'
              AND r.region_name NOT IN {EXCLUDE_REGIONS}
            GROUP BY r.region_name ORDER BY total_sales {order} LIMIT {top_n};
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # TOP REGION GLOBAL
    # =====================================================
    if (
        contains_semantic(q_lower, "region") and contains_semantic(q_lower, "sales")
        and not re.search(r"(20\d{2})", question) and month is None
        and (contains_semantic(q_lower, "highest") or contains_semantic(q_lower, "lowest"))
    ):
        sql = f"""
            SELECT r.region_name, SUM(f.sales_amount) AS total_sales
            FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
            WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
            GROUP BY r.region_name ORDER BY total_sales {order} LIMIT {top_n};
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # DOI QUERY
    # =====================================================
    if detect_doi_query(question):
        num_match = re.search(r"(\d+)", question)
        threshold = num_match.group(1) if num_match else "30"
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""

        if any(w in q_lower for w in ["di atas", "lebih dari", ">"]):
            where = f"ip.days_of_inventory > {threshold}"
        elif any(w in q_lower for w in ["di bawah", "kurang dari", "<"]):
            where = f"ip.days_of_inventory < {threshold}"
        elif any(w in q_lower for w in ["kritis", "hampir", "mau habis"]):
            where = "ip.days_of_inventory < 7"
        else:
            where = "ip.days_of_inventory IS NOT NULL"

        limit = 9999 if detect_all(question) else max(top_n * 3, 15)
        sql = f"""
            SELECT p.product_name, r.region_name,
                   ROUND(ip.days_of_inventory, 1) AS doi_hari,
                   ROUND(ip.average_sales_flow, 1) AS rata_penjualan_harian
            FROM inventory_parameter ip
            JOIN dim_product p ON ip.product_id = p.product_id
            JOIN dim_region  r ON ip.region_id  = r.region_id
            WHERE {where} AND r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
            ORDER BY ip.days_of_inventory {order} LIMIT {limit};
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # STOCK HABIS
    # =====================================================
    if contains_semantic(q_lower, "stock") and any(w in q_lower for w in ["habis", "kosong", "= 0", "nol"]):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""

        sql = f"""
            SELECT p.product_name, r.region_name,
                   fi.stock_quantity AS stok
            FROM fact_inventory fi
            JOIN dim_product p ON fi.product_id = p.product_id
            JOIN dim_region  r ON fi.region_id  = r.region_id
            WHERE fi.stock_quantity = 0
              AND r.region_name NOT IN {EXCLUDE_REGIONS}
              {region_filter}
            ORDER BY p.product_name;
        """
        data = run(sql)
        loc_label = f" di {entity['value']}" if entity and entity["type"] == "region" else ""
        insight = generate_llm_insight(question, data, client)
        answer = (
            f"📦 **Produk dengan stok habis (stock = 0){loc_label}:**\n"
            f"Ditemukan **{len(data)} produk** dengan stok kosong.\n\n"
            + format_natural_answer(data[:20])
            + (f"\n\n... dan {len(data)-20} produk lainnya." if len(data) > 20 else "")
            + insight
        )
        return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql.strip()}

    # =====================================================
    # DEAD STOCK + filter region
    # =====================================================
    if detect_dead_stock_query(question):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""

        sql = f"""
            SELECT p.product_name, r.region_name,
                   fi.stock_quantity AS stok_tersedia,
                   COALESCE(ts.total, 0) AS total_penjualan
            FROM fact_inventory fi
            JOIN dim_product p ON fi.product_id = p.product_id
            JOIN dim_region  r ON fi.region_id  = r.region_id
            LEFT JOIN (
                SELECT product_id, region_id, SUM(sales_amount) AS total
                FROM fact_inventory_month
                GROUP BY product_id, region_id
            ) ts ON ts.product_id = fi.product_id AND ts.region_id = fi.region_id
            WHERE fi.stock_quantity > 0
              AND COALESCE(ts.total, 0) = 0
              AND r.region_name NOT IN {EXCLUDE_REGIONS}
              {region_filter}
            ORDER BY fi.stock_quantity DESC
            LIMIT 20;
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # REPLENISHMENT / LOW STOCK
    # =====================================================
    if detect_replenishment_query(question) or detect_low_stock_query(question):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""
        sql = f"""
            SELECT p.product_name, r.region_name,
                   fi.stock_quantity AS stok_sekarang,
                   ROUND(ip.days_of_inventory, 1) AS perkiraan_habis_hari,
                   SUM(f.sales_amount) AS total_penjualan
            FROM fact_inventory fi
            JOIN dim_product p ON fi.product_id = p.product_id
            JOIN dim_region  r ON fi.region_id  = r.region_id
            JOIN inventory_parameter ip ON ip.product_id = fi.product_id AND ip.region_id = fi.region_id
            JOIN fact_inventory_month f ON f.product_id = fi.product_id AND f.region_id = fi.region_id
            WHERE ip.days_of_inventory > 0 AND ip.days_of_inventory < 15
              AND r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
            GROUP BY p.product_name, r.region_name, fi.stock_quantity, ip.days_of_inventory
            HAVING total_penjualan > 0
            ORDER BY ip.days_of_inventory ASC, total_penjualan DESC
            LIMIT 20;
        """
        data = run(sql)
        if not data:
            sql = f"""
                SELECT p.product_name, r.region_name,
                       fi.stock_quantity AS stok_sekarang,
                       ROUND(ip.days_of_inventory, 1) AS perkiraan_habis_hari
                FROM inventory_parameter ip
                JOIN dim_product p ON ip.product_id = p.product_id
                JOIN dim_region  r ON ip.region_id  = r.region_id
                JOIN fact_inventory fi ON fi.product_id = ip.product_id AND fi.region_id = ip.region_id
                WHERE ip.days_of_inventory > 0 AND r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
                ORDER BY ip.days_of_inventory ASC LIMIT 15;
            """
            data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # INTRANSIT QUERY
    # =====================================================
    if detect_intransit_query(question):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""
        limit = 9999 if detect_all(question) else max(top_n * 3, 15)
        sql = f"""
            SELECT p.product_name, r.region_name,
                   fi.intransit_quantity AS jumlah_dalam_perjalanan,
                   fi.stock_quantity AS stok_tersedia
            FROM fact_inventory fi
            JOIN dim_product p ON fi.product_id = p.product_id
            JOIN dim_region  r ON fi.region_id  = r.region_id
            WHERE fi.intransit_quantity > 0
              AND r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
            ORDER BY fi.intransit_quantity {order} LIMIT {limit};
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # BBK QUERY
    # =====================================================
    if detect_bbk_query(question):
        entity = detect_specific_entity(question, conn)
        region_filter = f"AND r.region_name = '{entity['value']}'" if entity and entity["type"] == "region" else ""
        sql = f"""
            SELECT p.product_name, r.region_name,
                   SUM(f.bbk) AS total_bbk, f.month_date AS bulan
            FROM fact_inventory_month f
            JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_region  r ON f.region_id  = r.region_id
            WHERE f.bbk > 0
              AND r.region_name NOT IN {EXCLUDE_REGIONS} {region_filter}
            GROUP BY p.product_name, r.region_name, f.month_date
            ORDER BY total_bbk {order} LIMIT 20;
        """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # PRODUCT LOCATION QUERY
    # =====================================================
    if detect_product_location_query(question):
        entity = detect_specific_entity(question, conn)
        if entity and entity["type"] == "product":
            pn  = entity["value"]
            sql = f"""
                SELECT DISTINCT r.region_name,
                       SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f
                JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region  r ON f.region_id  = r.region_id
                WHERE p.product_name = '{pn}'
                  AND r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY r.region_name
                HAVING total_penjualan > 0
                ORDER BY total_penjualan DESC;
            """
            data = run(sql)
            insight = generate_llm_insight(question, data, client)
            answer = (
                f"📦 **{pn}**\n\n"
                f"Dijual di **{len(data)} wilayah**:\n" +
                "\n".join(f"  {i+1}. {r['region_name']} → {int(r['total_penjualan']):,}" for i, r in enumerate(data))
                + insight
            )
            return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql.strip()}

    # =====================================================
    # ENTITY INSIGHT REGION (PERBAIKAN: Dashboard Intel Lengkap)
    # =====================================================
    entity = detect_specific_entity(question, conn)

    if entity and entity["type"] == "region" and any(w in q_lower for w in ["saja", "semua", "seluruh", "apa"]):
        if not (contains_semantic(q_lower, "highest") or contains_semantic(q_lower, "lowest") or detect_top_n(question)):
            rn  = entity["value"]
            sql = f"""
                SELECT DISTINCT p.product_name
                FROM fact_inventory_month f
                JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region  r ON f.region_id  = r.region_id
                WHERE r.region_name = '{rn}'
                ORDER BY p.product_name;
            """
            data = run(sql)
            insight = generate_llm_insight(question, data, client)
            return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    if entity and not detect_master_data_query(question):
        def to_dict(rows, keys): return [dict(zip(keys, r)) for r in rows]

        if entity["type"] == "region":
            rn = entity["value"]

            top_sales = to_dict(conn.execute(f"""
                SELECT p.product_name, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f
                JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region  r ON f.region_id  = r.region_id
                WHERE r.region_name = '{rn}'
                GROUP BY p.product_name ORDER BY total_sales DESC LIMIT 5
            """).fetchall(), ["product_name", "total_sales"])

            # PERBAIKAN: Penggunaan HAVING SUM(f.sales_amount) > 0 yang konsisten
            lowest_sales = to_dict(conn.execute(f"""
                SELECT p.product_name, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f
                JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region  r ON f.region_id  = r.region_id
                WHERE r.region_name = '{rn}'
                GROUP BY p.product_name
                HAVING SUM(f.sales_amount) > 0
                ORDER BY total_sales ASC LIMIT 5
            """).fetchall(), ["product_name", "total_sales"])

            stok_kritis = to_dict(conn.execute(f"""
                SELECT p.product_name, fi.stock_quantity,
                       ROUND(ip.days_of_inventory,1) AS doi
                FROM inventory_parameter ip
                JOIN dim_product p ON ip.product_id = p.product_id
                JOIN dim_region  r ON ip.region_id  = r.region_id
                JOIN fact_inventory fi ON fi.product_id = ip.product_id AND fi.region_id = ip.region_id
                WHERE r.region_name = '{rn}' AND ip.days_of_inventory > 0
                ORDER BY ip.days_of_inventory ASC LIMIT 5
            """).fetchall(), ["product_name", "stock_quantity", "doi"])

            stok_habis = to_dict(conn.execute(f"""
                SELECT p.product_name, fi.stock_quantity AS stok
                FROM fact_inventory fi
                JOIN dim_product p ON fi.product_id = p.product_id
                JOIN dim_region  r ON fi.region_id  = r.region_id
                WHERE r.region_name = '{rn}' AND fi.stock_quantity = 0
                LIMIT 5
            """).fetchall(), ["product_name", "stok"])

            tren_bulanan = to_dict(conn.execute(f"""
                SELECT f.month_date AS bulan, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name = '{rn}'
                GROUP BY f.month_date ORDER BY f.month_date
            """).fetchall(), ["bulan", "total_sales"])

            # PERBAIKAN: Mengambil performa penjualan berdasarkan tabel baru product_category khusus region terkait
            category_sales = to_dict(conn.execute(f"""
                SELECT pc.category_name, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f
                JOIN product_category pc ON f.product_id = pc.product_id
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name = '{rn}'
                GROUP BY pc.category_name ORDER BY total_sales DESC
            """).fetchall(), ["category_name", "total_sales"])

            ranking = conn.execute(f"""
                SELECT region_name, total_sales, rank FROM (
                    SELECT r.region_name, SUM(f.sales_amount) AS total_sales,
                           RANK() OVER (ORDER BY SUM(f.sales_amount) DESC) AS rank
                    FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
                    WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                    GROUP BY r.region_name
                ) t WHERE region_name = '{rn}'
            """).fetchone()

            total_region = sum(r["total_sales"] or 0 for r in tren_bulanan)
            insight = generate_llm_insight(question, top_sales, client)

            answer = (
                f"📍 **{rn}**\n\n"
                f"💰 **Total penjualan: {int(total_region):,}**\n"
                + (f"📊 **Ranking wilayah:** #{ranking[2]} dari semua wilayah\n" if ranking else "") +
                f"\n**🏆 Top 5 Produk Terlaris:**\n" +
                "\n".join(f"  {i+1}. {r['product_name']} → {int(r['total_sales']):,}" for i, r in enumerate(top_sales)) +
                f"\n\n**📉 Produk Terjual Paling Sedikit (bukan 0):**\n" +
                "\n".join(f"  {i+1}. {r['product_name']} → {int(r['total_sales'])}" for i, r in enumerate(lowest_sales)) +
                f"\n\n**🏷️ Penjualan Berdasarkan Kategori Produk:**\n" +
                "\n".join(f"  - {r['category_name']}: {int(r['total_sales']):,}" for r in category_sales) +
                f"\n\n**⚠️ Stok Kritis (DOI terendah):**\n" +
                "\n".join(f"  {i+1}. {r['product_name']} — stok: {r['stock_quantity']}, cukup: {r['doi']} hari" for i, r in enumerate(stok_kritis)) +
                (f"\n\n**📦 Produk Stok Habis ({len(stok_habis)} produk):**\n" +
                 "\n".join(f"  {i+1}. {r['product_name']}" for i, r in enumerate(stok_habis)) if stok_habis else "") +
                f"\n\n**📅 Tren Penjualan Bulanan:**\n" +
                "\n".join(f"  - {r['bulan']}: {int(r['total_sales']):,}" for r in tren_bulanan)
                + insight
            )

            return {
                "mode": "entity_insight",
                "entity_type": "region",
                "entity": rn,
                "data": tren_bulanan,
                "answer": answer,
                "sql": f"-- Entity insight untuk region: {rn}",
                "chart_type": "line",
            }

        if entity["type"] == "product":
            pn = entity["value"]
            if contains_semantic(q_lower, "stock"):
                sql = f"""
                    SELECT r.region_name, SUM(fi.stock_quantity) AS total_stok
                    FROM fact_inventory fi
                    JOIN dim_region  r ON fi.region_id  = r.region_id
                    JOIN dim_product p ON fi.product_id = p.product_id
                    WHERE p.product_name = '{pn}' AND r.region_name NOT IN {EXCLUDE_REGIONS}
                    GROUP BY r.region_name ORDER BY total_stok DESC
                """
                data  = run(sql)
                total = sum(r.get("total_stok", 0) or 0 for r in data)
                insight = generate_llm_insight(question, data, client)
                answer = (
                    f"📦 **{pn}**\n\n**Total stok: {int(total):,}**\n\n"
                    f"**Stok per wilayah:**\n" +
                    "\n".join(f"  {i+1}. {r['region_name']} → {int(r['total_stok']):,}" for i, r in enumerate(data[:10]))
                    + insight
                )
                return {"mode": "entity_insight", "entity_type": "product", "entity": pn,
                        "data": data, "answer": answer, "sql": sql.strip()}
            else:
                sql = f"""
                    SELECT r.region_name, SUM(f.sales_amount) AS total_penjualan
                    FROM fact_inventory_month f
                    JOIN dim_region  r ON f.region_id  = r.region_id
                    JOIN dim_product p ON f.product_id = p.product_id
                    WHERE p.product_name = '{pn}' AND r.region_name NOT IN {EXCLUDE_REGIONS}
                    GROUP BY r.region_name ORDER BY total_penjualan DESC
                """
                data  = run(sql)
                total = sum(r.get("total_penjualan", 0) or 0 for r in data)
                total_stock = conn.execute(f"""
                    SELECT COALESCE(SUM(fi.stock_quantity), 0)
                    FROM fact_inventory fi JOIN dim_product p ON fi.product_id = p.product_id
                    WHERE p.product_name = '{pn}'
                """).fetchone()[0]
                insight = generate_llm_insight(question, data, client)
                answer = (
                    f"📦 **{pn}**\n\n"
                    f"**Total penjualan: {int(total):,}** | **Total stok: {int(total_stock):,}**\n\n"
                    f"**Penjualan per wilayah:**\n" +
                    "\n".join(f"  {i+1}. {r['region_name']} → {int(r['total_penjualan']):,}" for i, r in enumerate(data[:10]))
                    + insight
                )
                return {"mode": "entity_insight", "entity_type": "product", "entity": pn,
                        "data": data, "answer": answer, "sql": sql.strip()}

    # =====================================================
    # MASTER DATA QUERY
    # =====================================================
    master_intent = detect_master_data_query(question)
    if master_intent:
        has_product = master_intent["product"]
        has_region  = master_intent["region"]

        if has_product and not has_region:
            sql  = "SELECT DISTINCT product_name FROM dim_product ORDER BY product_name;"
            data = run(sql)
            return {"mode": "sql_deterministic", "answer": format_natural_answer(data), "data": data, "sql": sql}

        elif has_region and not has_product:
            sql  = "SELECT DISTINCT region_name FROM dim_region WHERE region_name NOT IN ('Old Code','item','product') ORDER BY region_name;"
            data = run(sql)
            answer = (
                format_natural_answer(data) +
                "\n\n📌 *Catatan: Region **NASIONAL** mencakup keseluruhan total penjualan dan stok dari semua wilayah di atas.*"
            )
            return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql}

        elif has_product and has_region:
            sql  = f"SELECT p.product_name, r.region_name FROM dim_product p CROSS JOIN dim_region r WHERE r.region_name NOT IN {EXCLUDE_REGIONS} ORDER BY p.product_name, r.region_name;"
            data = run(sql)
            return {"mode": "sql_deterministic", "answer": format_natural_answer(data), "data": data, "sql": sql}

        return {"mode": "error", "answer": "Query tidak dikenali.", "data": []}

    # =====================================================
    # REGION QUERY
    # =====================================================
    region_intent = detect_region_query(question)
    if region_intent and is_region_focus(question):
        is_count          = region_intent["is_count"]
        is_all            = region_intent["is_all"]
        metric            = region_intent["metric"]
        condition_numeric = region_intent["condition_numeric"]

        if metric == "sales":
            table    = "fact_inventory_month f"
            sum_expr = "SUM(f.sales_amount)"
            joins    = "JOIN dim_region r ON f.region_id = r.region_id"
        else:
            table    = "fact_inventory fi"
            sum_expr = "SUM(fi.stock_quantity)"
            joins    = "JOIN dim_region r ON fi.region_id = r.region_id"

        if is_count and not condition_numeric:
            sql = f"SELECT COUNT(DISTINCT region_id) AS total_region FROM dim_region WHERE region_name NOT IN {EXCLUDE_REGIONS};"
        elif is_count and condition_numeric:
            sql = f"""SELECT COUNT(*) AS total_region FROM (
                SELECT r.region_id FROM {table} {joins}
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY r.region_id HAVING {sum_expr} {condition_numeric}) sub;"""
        else:
            having = f"HAVING {sum_expr} {condition_numeric}" if condition_numeric else ""
            limit  = "" if is_all else f"LIMIT {top_n}"
            sql = f"""
                SELECT r.region_name, {sum_expr} AS total_value
                FROM {table} {joins}
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY r.region_name {having}
                ORDER BY total_value {order} {limit};
            """
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_deterministic", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql.strip()}

    # =====================================================
    # PRODUCT QUERY
    # =====================================================
    if "dan" in q_lower and contains_semantic(q_lower, "product"):
        has_top    = bool(re.search(r"(top\s*\d+|terlaris|tertinggi)", q_lower))
        has_bottom = bool(re.search(r"(tidak laris|terendah|jarang|terkecil|sedikit|jelek)", q_lower))
        n_match    = re.findall(r"top\s*(\d+)", q_lower)

        if has_top or has_bottom:
            n_top    = int(n_match[0]) if n_match else 5
            n_bottom = int(n_match[1]) if len(n_match) > 1 else n_top

            top_data = run(f"""
                SELECT p.product_name, SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY p.product_name ORDER BY total_penjualan DESC LIMIT {n_top};
            """)
            # PERBAIKAN: Gunakan fungsi agregat aslinya di klausa HAVING agar kompatibel
            bottom_data = run(f"""
                SELECT p.product_name, SUM(f.sales_amount) AS total_penjualan
                FROM fact_inventory_month f JOIN dim_product p ON f.product_id = p.product_id
                JOIN dim_region r ON f.region_id = r.region_id
                WHERE r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY p.product_name
                HAVING SUM(f.sales_amount) > 0
                ORDER BY total_penjualan ASC LIMIT {n_bottom};
            """)
            insight = generate_llm_insight(question, top_data, client)
            answer = (
                f"🏆 **Top {n_top} Produk Terlaris:**\n" +
                "\n".join(f"  {i+1}. {r['product_name']} → {int(r['total_penjualan']):,}" for i, r in enumerate(top_data)) +
                f"\n\n📉 **Top {n_bottom} Produk Paling Tidak Laris (bukan 0):**\n" +
                "\n".join(f"  {i+1}. {r['product_name']} → {int(r['total_penjualan'])}" for i, r in enumerate(bottom_data))
                + insight
            )
            return {"mode": "sql_deterministic", "answer": answer,
                    "data": top_data + bottom_data, "sql": "-- Dual top/bottom query"}

    product_intent = detect_product_query(question)
    if product_intent:
        is_all            = detect_all(question)
        is_count          = detect_count(question)
        metric            = product_intent["metric"]
        yr                = product_intent["year"]
        mo                = product_intent["month"]
        has_region_filter = product_intent["has_region"]
        top_n_p           = None if is_all else product_intent["top_n"]
        condition_numeric = detect_numeric_condition(question)
        p_order           = product_intent["order"]

        if p_order == "ASC" and not detect_top_n(question):
            top_n_p = 10

        if metric == "sales":
            table    = "fact_inventory_month f"
            sum_expr = "SUM(f.sales_amount)"
            joins    = "JOIN dim_product p ON f.product_id = p.product_id"
            if has_region_filter:
                joins += " JOIN dim_region r ON f.region_id = r.region_id"
            condition = f"r.region_name NOT IN {EXCLUDE_REGIONS}" if has_region_filter else "1=1"
            if yr: condition += f" AND strftime('%Y', f.month_date) = '{yr}'"
            if mo: condition += f" AND strftime('%m', f.month_date) = '{mo}'"
        else:
            table    = "fact_inventory fi"
            sum_expr = "SUM(fi.stock_quantity)"
            joins    = "JOIN dim_product p ON fi.product_id = p.product_id"
            if has_region_filter:
                joins += " JOIN dim_region r ON fi.region_id = r.region_id"
            condition = f"r.region_name NOT IN {EXCLUDE_REGIONS}" if has_region_filter else "1=1"

        select_cols = "p.product_name" + (", r.region_name" if has_region_filter else "")
        group_by    = "p.product_name" + (", r.region_name" if has_region_filter else "")

        # PERBAIKAN: Mengganti total_value menjadi sum_expr asli di dalam HAVING agar kompatibel penuh dengan SQLite
        if p_order == "ASC" and not is_count:
            having = f"HAVING {sum_expr} > 0"
        elif condition_numeric and not is_count:
            having = f"HAVING {sum_expr} {condition_numeric}"
        else:
            having = ""

        limit = f"LIMIT {top_n_p}" if top_n_p else ""

        if is_count:
            sql = f"SELECT COUNT(*) AS total_count FROM (SELECT p.product_id FROM {table} {joins} WHERE {condition} GROUP BY p.product_id) sub;"
        else:
            sql = f"""
                SELECT {select_cols}, {sum_expr} AS total_value
                FROM {table} {joins}
                WHERE {condition}
                GROUP BY {group_by} {having}
                ORDER BY total_value {p_order} {limit};
            """

        data = run(sql)

        if p_order == "DESC" and metric == "sales" and data and not has_region_filter:
            top_pn = data[0].get("product_name", "")
            top_regions = run(f"""
                SELECT r.region_name, SUM(f.sales_amount) AS total_sales
                FROM fact_inventory_month f JOIN dim_region r ON f.region_id = r.region_id
                JOIN dim_product p ON f.product_id = p.product_id
                WHERE p.product_name = '{top_pn}' AND r.region_name NOT IN {EXCLUDE_REGIONS}
                GROUP BY r.region_name ORDER BY total_sales DESC LIMIT 3
            """)
            loc_str = ", ".join(r["region_name"] for r in top_regions)
            extra   = f"\n\n📍 **Lokasi penjualan terbesar {top_pn}:** {loc_str}"
        else:
            extra = ""

        insight = generate_llm_insight(question, data, client)
        answer = format_natural_answer(data) + extra + insight
        return {"mode": "sql_deterministic", "answer": answer, "data": data, "sql": sql.strip()}

    # =====================================================
    # FALLBACK — LLM
    # =====================================================
    try:
        schema_dict     = extract_schema_columns(schema)
        selected_tables = set(get_top_k_tables(question, schema_dict, k=top_k))
        if "product" in q_lower or "produk" in q_lower or "item" in q_lower:
            selected_tables.add("dim_product")
        if "region" in q_lower or "kota" in q_lower:
            selected_tables.add("dim_region")
        if contains_semantic(q_lower, "doi") or contains_semantic(q_lower, "replenishment"):
            selected_tables.add("inventory_parameter")
        if contains_semantic(q_lower, "sales"):
            selected_tables.add("fact_inventory_month")
        if contains_semantic(q_lower, "stock"):
            selected_tables.add("fact_inventory")
        if "kategori" in q_lower or "category" in q_lower:
            selected_tables.add("product_category")

        filtered_schema = build_filtered_schema(selected_tables, schema_dict)
        sql, _          = client.invoke_sql(filtered_schema, question)
        ensure_select_only(sql)
        if not validate_columns(sql, schema_dict):
            sql, _ = client.invoke_sql(filtered_schema, question)
            ensure_select_only(sql)
        data = run(sql)
        insight = generate_llm_insight(question, data, client)
        return {"mode": "sql_llm", "answer": format_natural_answer(data) + insight, "data": data, "sql": sql}

    except Exception as e:
        return {"mode": "error", "error": str(e),
                "answer": f"Maaf, terjadi kesalahan: {str(e)}", "data": []}