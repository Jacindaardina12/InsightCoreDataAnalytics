"""
app.py — IL VASTO InsightBot
==============================
UI: Clean editorial light theme
- Warm white background (#FAFAF8)
- Amber accent (#C8A96E)
- DM Serif Display + DM Sans typography
- WhatsApp-style chat bubbles
- Chart tabs (Grafik / Tabel)
- SQL expander per jawaban

Jalankan: streamlit run app.py
"""

import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from src.hybrid_assistant import answer_question
from src.model_client import DistilLabsLLM
from src.chart_builder import (
    auto_select_chart,
    build_line_chart,
    generate_interpretation,
)

load_dotenv()
#===========================================================
import base64

def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="InsightCore Data Analytics",
    page_icon="assets/InsightCorePutih.png",
    layout="centered",
)

# ── Load CSS ──
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# =========================================================
# CONNECTIONS (cached)
# =========================================================
DB_PATH = os.getenv("DB_PATH", "data/database/replan_report.db")


@st.cache_resource
def load_engine():
    if not os.path.exists(DB_PATH):
        st.error(f"Database tidak ditemukan: {DB_PATH}")
        st.stop()
    conn   = sqlite3.connect(DB_PATH, check_same_thread=False)
    client = DistilLabsLLM()
    return conn, client


conn, client = load_engine()


@st.cache_data
def _get_schema(_conn):
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            return f.read()
    cursor = _conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    return "\n".join(r[0] for r in cursor.fetchall() if r[0])


SCHEMA = _get_schema(conn)

# =========================================================
# SESSION STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# =========================================================
# CHART HELPER — light theme Plotly
# =========================================================
AMBER = "#C8A96E"
INK   = "#1A1A18"
PAPER = "#FAFAF8"

COLOR_SEQ = [
    "#1A1A18", "#C8A96E", "#8B7355", "#D4B896",
    "#5A5A55", "#E8E6E0", "#3D3D3A", "#A89070",
]


def _light_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        font=dict(family="DM Sans", size=12, color="#5A5A55"),
        title=dict(text=title, font=dict(size=13, color=INK, family="DM Sans")) if title else None,
        xaxis=dict(gridcolor="#E8E6E0", zerolinecolor="#E8E6E0", color="#9A9890", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#E8E6E0", zerolinecolor="#E8E6E0", color="#9A9890", tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5A5A55", size=11)),
        margin=dict(l=8, r=8, t=32 if title else 8, b=8),
        colorway=COLOR_SEQ,
    )
    return fig


def make_chart(data: list[dict], chart_type: str = "") -> go.Figure | None:
    """Build an appropriate Plotly chart from query result data."""
    if not data or len(data) < 2:
        return None

    df   = pd.DataFrame(data)
    keys = list(df.columns)

    # Detect date column for line chart
    has_date = any("date" in k.lower() or "bulan" in k.lower() or "month" in k.lower() for k in keys)

    if chart_type == "line" or has_date:
        date_col  = next((c for c in keys if "date" in c.lower() or "bulan" in c.lower() or "month" in c.lower()), keys[0])
        value_col = next((c for c in keys if "sales" in c.lower() or "total" in c.lower() or "value" in c.lower()), keys[-1])
        group_col = next((c for c in keys if "region" in c.lower() or "produk" in c.lower() or "product" in c.lower()), None)

        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        except Exception:
            pass

        if group_col and group_col != date_col and group_col != value_col:
            fig = px.line(df, x=date_col, y=value_col, color=group_col,
                          markers=True, line_shape="spline", color_discrete_sequence=COLOR_SEQ)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[value_col],
                mode="lines+markers",
                line=dict(color=AMBER, width=2.5),
                marker=dict(size=7, color=AMBER, line=dict(color="white", width=1.5)),
                fill="tozeroy",
                fillcolor=f"rgba(200,169,110,0.07)",
            ))
        return _light_theme(fig)

    # Label + value cols
    label_key = keys[0]
    value_key = keys[1] if len(keys) > 1 else keys[0]
    for c in ["region_name", "product_name", "region", "bulan"]:
        if c in keys:
            label_key = c
            break
    for c in ["total_sales", "total_value", "total_penjualan", "total_stok", "stok_tersedia"]:
        if c in keys:
            value_key = c
            break

    if label_key == value_key:
        return None

    # Pie for small region sets
    n_cats = df[label_key].nunique()
    if "region" in label_key and n_cats <= 10:
        fig = px.pie(df, names=label_key, values=value_key,
                     color_discrete_sequence=COLOR_SEQ, hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont=dict(size=11, family="DM Sans"))
        return _light_theme(fig)

    # Horizontal bar for products (long names)
    if "product" in label_key and n_cats > 5:
        df_s = df.sort_values(value_key, ascending=True).tail(15)
        fig  = px.bar(df_s, y=label_key, x=value_key, orientation="h",
                      color_discrete_sequence=[AMBER])
        fig.update_traces(marker_line_width=0, texttemplate="%{x:,}", textposition="outside")
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)))
        return _light_theme(fig)

    # Vertical bar default
    fig = px.bar(df, x=label_key, y=value_key,
                 color_discrete_sequence=[AMBER], text_auto=True)
    fig.update_traces(marker_line_width=0, textposition="outside",
                      textfont=dict(size=10))
    return _light_theme(fig)


# =========================================================
# HTML HELPERS
# =========================================================

def _user_bubble(text: str) -> str:
    return f"""
    <div style="display:flex;justify-content:flex-end;margin:16px 0 4px;">
      <div style="
        background:#1A1A18;
        color:#FAFAF8;
        padding:12px 16px;
        border-radius:16px 16px 2px 16px;
        max-width:78%;
        font-size:14px;
        line-height:1.6;
        font-family:'DM Sans',sans-serif;
        box-shadow:0 2px 8px rgba(0,0,0,0.12);
      ">{text}</div>
    </div>"""


def _bot_bubble(text: str) -> str:
    # Convert markdown-style **bold** to <strong>
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = text.replace("\n", "<br>")
    return f"""
    <div style="display:flex;justify-content:flex-start;margin:4px 0 8px;">
      <div style="
        background:#FFFFFF;
        color:#1A1A18;
        padding:14px 18px;
        border-radius:2px 16px 16px 16px;
        max-width:88%;
        font-size:14px;
        line-height:1.7;
        font-family:'DM Sans',sans-serif;
        border:1px solid #E8E6E0;
        box-shadow:0 1px 4px rgba(0,0,0,0.04);
      ">{text}</div>
    </div>"""


def _insight_box(text: str) -> str:
    return f"""
    <div style="
      background:#FFFBF0;
      border-left:3px solid {AMBER};
      padding:14px 16px;
      border-radius:0 8px 8px 0;
      margin:6px 0 12px;
      font-size:13.5px;
      color:#5A4A2A;
      font-family:'DM Sans',sans-serif;
      line-height:1.65;
    ">
      <span style="font-weight:600;color:{AMBER};">💡 Insight</span>
      <br><br>{text}
    </div>"""


def _divider() -> str:
    return "<div style='height:1px;background:#F0EDE6;margin:8px 0 16px;'></div>"


# =========================================================
# HEADER
# =========================================================
logo_b64 = _img_to_base64("assets/InsightCore.png")

# Gunakan fallback emoji jika logo.png belum ada di folder assets
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 72px; height: 72px; border-radius: 16px; object-fit: cover;" />' if logo_b64 else '<div style="width:72px;height:72px;background:#1A1A18;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:28px;flex-shrink:0;">📊</div>'

st.markdown(f"""
<div style="padding: 24px 0 16px 0; border-bottom: 1px solid var(--text-color-light, #E8E6E0); margin-bottom: 24px;">
  <div style="display: flex; align-items: center; gap: 16px;">
    {logo_html}
    <div style="display: flex; flex-direction: column; justify-content: center;">
      <div style="font-family: 'DM Serif Display', Georgia, serif; font-size: 22px; font-weight: 600; color: var(--text-color, #1A1A18); line-height: 1.2; letter-spacing: -0.3px;">
        InsightCore Data Analytics
      </div>
      <div style="font-size: 13px; color: var(--secondary-text-color, #9A9890); margin-top: 2px; letter-spacing: 0.3px; font-family: 'DM Sans', sans-serif;">
      AI-powered assistant untuk eksplorasi data, analisis, dan penyajian insight secara interaktif.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SUGGESTION SCREEN (tampil jika chat kosong)
# =========================================================
SUGGESTIONS = [
    ("📈", "Bagaimana tren penjualan tiap bulannya?"),
    ("🏆", "Produk apa dengan penjualan tertinggi?"),
    ("📍", "Bagaimana performa wilayah Surabaya?"),
    ("📦", "Produk apa yang stoknya hampir habis?"),
    ("🌍", "Kota mana dengan penjualan tertinggi?"),
    ("🔍", "Produk mana yang tidak pernah terjual tapi stoknya masih ada?"),
]

if st.session_state.show_suggestions and not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 32px;">
      <div style="
        font-family:'DM Serif Display',serif;
        font-size:26px;
        color:#1A1A18;
        margin-bottom:10px;
        letter-spacing:-0.4px;
      ">Halo, mau lihat insight apa hari ini?</div>
      <div style="font-size:14px;color:#9A9890;font-family:'DM Sans',sans-serif;">
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Grid 2 kolom
    for i in range(0, len(SUGGESTIONS), 2):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            icon, label = SUGGESTIONS[i]
            if st.button(f"{icon}  {label}", key=f"sug_{i}"):
                st.session_state.show_suggestions = False
                st.session_state._pending_query   = label
                st.rerun()
        if i + 1 < len(SUGGESTIONS):
            with c2:
                icon, label = SUGGESTIONS[i + 1]
                if st.button(f"{icon}  {label}", key=f"sug_{i+1}"):
                    st.session_state.show_suggestions = False
                    st.session_state._pending_query   = label
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# RENDER CHAT HISTORY
# =========================================================
for idx, msg in enumerate(st.session_state.chat_history):

    if msg["role"] == "user":
        st.markdown(_user_bubble(msg["text"]), unsafe_allow_html=True)

    else:
        # Bot bubble — teks jawaban
        st.markdown(_bot_bubble(msg["text"]), unsafe_allow_html=True)

        # Insight box (amber)
        if msg.get("interpretation"):
            st.markdown(_insight_box(msg["interpretation"]), unsafe_allow_html=True)

        # Chart + Tabel dalam tabs
        if msg.get("df") is not None and len(msg["df"]) > 0:
            df_msg = msg["df"]

            tab_chart, tab_data = st.tabs(["📊 Grafik", "🗂️ Tabel Data"])

            with tab_chart:
                fig = make_chart(df_msg.to_dict("records"), msg.get("chart_type", ""))
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.caption("Chart tidak tersedia untuk data ini.")

            with tab_data:
                st.dataframe(df_msg, use_container_width=True, hide_index=True)

        # SQL expander
        if msg.get("sql"):
            with st.expander("💻 lihat query sql yang dijalankan"):
                st.code(msg["sql"], language="sql")

        st.markdown(_divider(), unsafe_allow_html=True)

# =========================================================
# PROCESS QUERY (dari suggestion atau input)
# =========================================================
def _process(query: str):
    """Run answer_question, save result to chat history, rerun."""
    # Tambah user bubble dulu
    st.session_state.chat_history.append({"role": "user", "text": query})

    with st.spinner("Mengekstrak data dari database..."):
        res = answer_question(
            question=query,
            mode=None,
            conn=conn,
            schema=SCHEMA,
            client=client,
        )

    data        = res.get("data", [])
    sql_shown   = res.get("sql", "")
    chart_type  = res.get("chart_type", "")
    answer_text = res.get("answer", "Maaf, data tidak dapat ditarik.")

    df_payload     = pd.DataFrame(data) if data else None
    interpretation = None

    # Auto-detect line chart jika kata "tren/bulanan" ada
    if not chart_type and any(w in query.lower() for w in ["tren", "per bulan", "bulanan"]):
        chart_type = "line"

    # Generate interpretasi LLM
    if data and client:
        try:
            from src.chart_builder import generate_interpretation
            interpretation = generate_interpretation(data, query, client)
        except Exception:
            pass

    st.session_state.chat_history.append({
        "role":           "bot",
        "text":           answer_text,
        "df":             df_payload,
        "chart_type":     chart_type,
        "sql":            sql_shown,
        "interpretation": interpretation,
    })
    st.session_state.show_suggestions = False
    st.rerun()


# Pending query dari suggestion button
if hasattr(st.session_state, "_pending_query") and st.session_state._pending_query:
    q = st.session_state._pending_query
    del st.session_state._pending_query
    _process(q)

# =========================================================
# CHAT INPUT
# =========================================================
user_input = st.chat_input("Tanyakan tentang penjualan, produk, stok, wilayah...")

if user_input:
    _process(user_input)

# =========================================================
# CLEAR BUTTON (muncul jika ada history)
# =========================================================
if st.session_state.chat_history:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([5, 1])
    with col_r:
        if st.button("↺ Reset", key="clear_btn"):
            st.session_state.chat_history   = []
            st.session_state.show_suggestions = True
            st.rerun()