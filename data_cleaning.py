import pandas as pd
import sqlite3
import re

# ==============================
# CONFIG
# ==============================
EXCEL_FILE = "data/replan_report.xlsx"
DB_FILE = "data/database/replan_report.db"
SNAPSHOT_DATE = "2026-02-27"

# ==============================
# LOAD EXCEL + FIX HEADER
# ==============================
df_raw = pd.read_excel(EXCEL_FILE, header=None)

header1 = df_raw.iloc[0]
header2 = df_raw.iloc[1]

columns = []
current_region = ""

for h1, h2 in zip(header1, header2):

    if pd.notna(h1):
        current_region = str(h1).strip()

    if pd.notna(h2):
        columns.append(f"{current_region}_{h2}".strip())
    else:
        columns.append(current_region)

df = df_raw.iloc[2:].copy()
df.columns = columns
df = df.reset_index(drop=True)

# ==============================
# AUTO DETECT PRODUCT COLUMNS
# ==============================
def find_col(keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    raise Exception(f"Kolom {keyword} tidak ditemukan")

# ==============================
# DETECT KOLOM PRODUK OTOMATIS
# ==============================
item_code_col = find_col("Item Code")
old_code_col = find_col("Old Code")
item_name_col = find_col("Item Name")

# ==============================
# PRODUCT CATEGORY FUNCTION
# ==============================
def categorize_product(product_name):

    if pd.isna(product_name):
        return "Other"

    name = str(product_name).upper()


    # hapus simbol agar matching lebih akurat
    name = re.sub(r'[^A-Z0-9 ]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # =================================
    # HAIR COLOR
    # =================================
    if any(k in name for k in [
        "COLOR",
        "COLOUR",
        "COLORANT",
        "COLOURANT",
        "TONING",
        "PEARL",
        "SILVER",
        "BLONDE",
        "CORRECTOR",
        "ROSE PINK",
        "RED",
        "GREEN"
    ]):
        return "Hair Color"

    # =================================
    # HAIR CHEMICAL
    # =================================
    elif any(k in name for k in [
        "BLEACH",
        "BLEACHING",
        "PEROXIDE",
        "DEVELOPER",
        "LIGHTENING",
        "DECOLORANTE",
        "PERMANENT WAVE",
        "PERMING",
        "NEUTRALIZER",
        "STRAIGHTENING"
    ]):
        return "Hair Chemical"

    # =================================
    # HAIR TREATMENT
    # =================================
    elif any(k in name for k in [
        "TREATMENT",
        "KERATIN",
        "MASK",
        "SPA",
        "ESSENCE",
        "SERUM",
        "CONDITIONER",
        "FILLER",
        "INFUSION",
        "LOTION",
        "REVITAE"

    ]):
        return "Hair Treatment"

    # =================================
    # SHAMPOO
    # =================================
    elif any(k in name for k in [
        "SHAMPOO",
        "CLEANSER"
    ]):
        return "Shampoo & Cleanser"

    # =================================
    # HAIR STYLING
    # =================================
    elif any(k in name for k in [
        "GEL",
        "SPRAY",
        "WAX",
        "POMADE",
        "MOUSSE",
        "MOUSSY",
        "CURL",
        "CURLY",
        "TEXTURE",
        "SEA SALT",
        "HOLD",
        "HAIR CREAM",
        "TEXTURIZING",
        "FIBER CREAM"
    ]):
        return "Hair Styling"

    # =================================
    # SALON EQUIPMENT
    # =================================
    elif any(k in name for k in [
        "IRON",
        "CATOK",
        "DRYER",
        "MACHINE",
        "BOWL",
        "BRUSH",
        "PADDLE",
        "TINTING",
        "TROLLY",
        "APRON",
        "CAPE",
        "CHART",
        "WASHBAK",
        "TOWEL",
        "MAGIC LITE"
    ]):
        return "Salon Equipment"

    # =================================
    # SKINCARE
    # =================================
    elif any(k in name for k in [
        "TONER",
        "FOUNDATION",
        "CUSHION",
        "FACE",
        "DAY CREAM",
        "NIGHT CREAM"
    ]):
        return "Skincare"

    # =================================
    # PROMO / BUNDLE
    # =================================
    elif any(k in name for k in [
        "PAKET",
        "SET",
        "POIN",
        "BUNDLE"
    ]):
        return "Promo / Bundle"

    return "Other"

# ==============================
# DIM PRODUCT
# ==============================

dim_product = df[[
    item_code_col,
    old_code_col,
    item_name_col
]].drop_duplicates()

dim_product.columns = [
    "item_code",
    "old_code",
    "product_name"
]

dim_product.insert(0, "product_id", range(1, len(dim_product)+1))

# ==============================
# PRODUCT CATEGORY TABLE
# ==============================
product_category = dim_product[[
    "product_id",
    "product_name"
]].copy()

product_category["category_name"] = product_category[
    "product_name"
].apply(categorize_product)

# mapping product_id ke df
df = df.merge(
    dim_product[["product_id","item_code"]],
    left_on=item_code_col,
    right_on="item_code",
    how="left"
)

# ==============================
# DIM REGION
# ==============================
regions = sorted({
    col.split("_")[0]
    for col in df.columns if "_" in col
    and not col.startswith("Item")
})

dim_region = pd.DataFrame({
    "region_name": regions
})

dim_region["region_code"] = dim_region["region_name"].str.extract(r"(^[A-Z]+)")
dim_region["region_id"] = range(1, len(dim_region)+1)

# ==============================
# FACT INVENTORY
# ==============================
inventory_rows = []

for _, reg in dim_region.iterrows():

    region = reg["region_name"]
    region_id = reg["region_id"]

    stock_col = f"{region}_Stock"
    intransit_col = f"{region}_Intransit"

    if stock_col not in df.columns:
        continue

    temp = df[["product_id"]].copy()
    temp["region_id"] = region_id

    temp["stock_quantity"] = pd.to_numeric(
        df[stock_col], errors="coerce"
    ).fillna(0).astype(int)

    if intransit_col in df.columns:
        temp["intransit_quantity"] = pd.to_numeric(
            df[intransit_col], errors="coerce"
        ).fillna(0).astype(int)
    else:
        temp["intransit_quantity"] = 0

    temp["snapshot_date"] = SNAPSHOT_DATE

    snapshot_dt = pd.to_datetime(SNAPSHOT_DATE)
    temp["year"] = snapshot_dt.year
    temp["month"] = snapshot_dt.month
    
    inventory_rows.append(temp)

fact_inventory = pd.concat(inventory_rows, ignore_index=True)

# ==============================
# INVENTORY (DENORMALIZED)
# ==============================
inventory = fact_inventory.merge(
    dim_region[["region_id","region_name"]],
    on="region_id"
)[["product_id","region_name","stock_quantity"]]

inventory.columns = ["product_id","region_name","stock_quantity"]
inventory.insert(0, "id", range(1,len(inventory)+1))

# ==============================
# FACT INVENTORY MONTH
# ==============================
month_map = {
    "Jan":"01","Feb":"02","Mar":"03","Apr":"04",
    "May":"05","Jun":"06","Jul":"07","Aug":"08",
    "Sep":"09","Oct":"10","Nov":"11","Dec":"12"
}

fact_month_list = []

for col in df.columns:

    if "Sales" not in col and "BBK" not in col:
        continue

    region, metric = col.split("_",1)

    match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d+)", metric)

    if not match:
        continue

    month = match.group(1)
    year = "20" + match.group(2)
    month_date = f"{year}-{month_map[month]}-01"

    metric_type = "sales_amount" if "Sales" in metric else "bbk"

    temp = df[["product_id"]].copy()

    temp["region_id"] = dim_region.loc[
        dim_region.region_name==region,
        "region_id"
    ].values[0]

    temp["month_date"] = month_date
    temp["year"] = int(year)
    temp["month"] = int(month_map[month])
    temp[metric_type] = pd.to_numeric(
        df[col], errors="coerce"
    ).fillna(0)

    fact_month_list.append(temp)

fact_inventory_month = (
    pd.concat(fact_month_list)
    .groupby(["product_id","region_id","month_date","year","month"])
    .sum()
    .reset_index()
)

# ==============================
# INVENTORY PARAMETER
# ==============================
inventory_parameter = df[[
    "product_id",
    "NASIONAL_Avg Flow",
    "NASIONAL_DOI",
    "NASIONAL_REPLENISHMENT"
]].copy()

inventory_parameter.columns = [
    "product_id",
    "average_sales_flow",
    "days_of_inventory",
    "replenishment"
]

inventory_parameter["region_id"] = 1
inventory_parameter["lead_time"] = 0

# ✅ urutan sesuai database awal
inventory_parameter = inventory_parameter[[
    "product_id",
    "region_id",
    "lead_time",
    "average_sales_flow",
    "days_of_inventory",
    "replenishment"
]]

# ✅ dtype FIX
inventory_parameter["lead_time"] = inventory_parameter["lead_time"].astype(float)
inventory_parameter["average_sales_flow"] = pd.to_numeric(inventory_parameter["average_sales_flow"], errors="coerce")
inventory_parameter["days_of_inventory"] = pd.to_numeric(inventory_parameter["days_of_inventory"], errors="coerce")
inventory_parameter["replenishment"] = pd.to_numeric(inventory_parameter["replenishment"], errors="coerce")

# ==============================
# INVENTORY TEXT (EMPTY INIT)
# ==============================
inventory_text = pd.DataFrame({
    "id": pd.Series(dtype="int"),
    "product_id": pd.Series(dtype="int"),
    "region_name": pd.Series(dtype="str"),
    "description_text": pd.Series(dtype="str")
})

fact_inventory_month["month_date"] = pd.to_datetime(
    fact_inventory_month["month_date"]
).astype(str)

fact_inventory_month["sales_amount"] = fact_inventory_month["sales_amount"].astype(int)
fact_inventory_month["bbk"] = fact_inventory_month["bbk"].astype(int)

# ==============================
# CHECK UNCATEGORIZED PRODUCTS
# ==============================
other_products = product_category[
    product_category["category_name"] == "Other"
]

print("\n===== OTHER CATEGORY PRODUCTS =====")

for p in other_products["product_name"].tolist():
    print("-", p)

# ==============================
# SAVE TO SQLITE
# ==============================
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS fact_inventory_month;
DROP TABLE IF EXISTS inventory_parameter;
DROP TABLE IF EXISTS inventory_text;
DROP TABLE IF EXISTS product_category;

CREATE TABLE fact_inventory_month (
    product_id INTEGER,
    region_id INTEGER,
    month_date TIMESTAMP,
    year INTEGER,
    month INTEGER,
    sales_amount INTEGER,
    bbk INTEGER
);

CREATE TABLE inventory_parameter (
    product_id INTEGER,
    region_id INTEGER,
    lead_time REAL,
    average_sales_flow REAL,
    days_of_inventory REAL,
    replenishment REAL
);

CREATE TABLE inventory_text (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    region_name TEXT,
    description_text TEXT
);

CREATE TABLE product_category (
    product_id INTEGER,
    product_name TEXT,
    category_name TEXT
);
                  
""")

conn.commit()

dim_product.to_sql("dim_product", conn, if_exists="replace", index=False)
dim_region.to_sql("dim_region", conn, if_exists="replace", index=False)
fact_inventory.to_sql("fact_inventory", conn, if_exists="replace", index=False)
inventory.to_sql("inventory", conn, if_exists="replace", index=False)
fact_inventory_month.to_sql("fact_inventory_month", conn, if_exists="append", index=False)
inventory_parameter.to_sql("inventory_parameter", conn, if_exists="append", index=False)
inventory_text.to_sql("inventory_text", conn, if_exists="append", index=False)
product_category.to_sql(
    "product_category",
    conn,
    if_exists="append",
    index=False
)

conn.close()

print("✅ DATA CLEANING SELESAI — 8 TABLE BERHASIL DIBUAT")