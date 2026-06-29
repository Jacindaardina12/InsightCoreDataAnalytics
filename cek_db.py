import sqlite3

# Sesuaikan dengan path database-mu
db_path = 'data/database/replan_report.db'

try:
    conn = sqlite3.connect(db_path)
    # Ambil semua nama tabel
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    if not tables:
        print("Database berhasil dibaca, tapi tidak ada tabel di dalamnya.")
        
    for t in tables:
        table_name = t[0]
        print(f'\n=== {table_name} ===')
        # Ambil info kolom dari masing-masing tabel
        columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        for c in columns:
            print(f'  {c[1]} ({c[2]})')

except sqlite3.DatabaseError as e:
    print(f"Error: {e}. Pastikan file ini benar-benar database SQLite, bukan file kosong!")
finally:
    if 'conn' in locals():
        conn.close()