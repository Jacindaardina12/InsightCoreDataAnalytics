"""
chart_builder.py — Project A: IL VASTO Replan Intelligence
============================================================
FITUR BARU:
- Chart variatif: bar, pie, line (untuk tren bulanan)
- auto_select_chart: pilih chart otomatis berdasarkan tipe data
- generate_interpretation: LLM generate 2-3 kalimat narasi otomatis
"""
from __future__ import annotations

from typing import Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os


# =========================================================
# FITUR BARU #3 — INTERPRETASI OTOMATIS VIA LLM
# =========================================================

def generate_interpretation(data: list[dict], question: str, client=None) -> str | None:
    """
    Generate 2-3 kalimat interpretasi dari data hasil query.
    Dikirim ke Groq, dikembalikan sebagai string narasi Bahasa Indonesia.
    Jika client tidak tersedia atau error, return None.
    """
    if not client or not data:
        return None

    try:
        # Buat ringkasan data untuk dikirim ke LLM
        keys = list(data[0].keys()) if data else []
        sample = data[:5]  # ambil 5 baris pertama sebagai konteks

        summary_lines = []
        for row in sample:
            parts = ", ".join(f"{k}: {v}" for k, v in row.items())
            summary_lines.append(parts)

        summary_text = "\n".join(summary_lines)
        total_rows = len(data)

        prompt = f"""Kamu adalah analis bisnis yang menjelaskan data kepada klien non-teknis.

Pertanyaan yang diajukan: "{question}"

Data hasil analisis ({total_rows} baris, berikut 5 contoh teratas):
{summary_text}

Tulis 2-3 kalimat interpretasi singkat dalam Bahasa Indonesia yang:
- Mudah dipahami orang awam (tanpa jargon teknis)
- Menyebutkan angka/fakta penting dari data
- Memberikan kesimpulan atau rekomendasi sederhana
- JANGAN sebut nama kolom teknis seperti "sales_amount", "stock_quantity" dll

Langsung tulis kalimatnya saja, tanpa pembuka seperti "Berikut interpretasinya:"."""

        narasi, _ = client.invoke_rag(question="interpretasi", retrieved_context=prompt)
        return narasi.strip() if narasi and narasi != "insufficient information" else None

    except Exception:
        return None


# =========================================================
# CHART: BAR — untuk ranking / perbandingan
# =========================================================

def build_bar_chart(data: list[dict], title: str = "") -> go.Figure | None:
    """Bar chart standar untuk data ranking/perbandingan."""
    if not data or len(data) < 2:
        return None

    df = pd.DataFrame(data)
    keys = list(df.columns)

    label_key = keys[0]
    value_key = keys[1] if len(keys) > 1 else keys[0]

    for candidate in ["region_name", "product_name", "region", "bulan"]:
        if candidate in keys:
            label_key = candidate
            break

    for candidate in ["total_sales", "total_value", "total_stock", "stok_tersedia",
                      "total_penjualan", "doi_hari", "replenishment"]:
        if candidate in keys:
            value_key = candidate
            break

    if label_key == value_key:
        return None

    try:
        fig = px.bar(
            df,
            x=label_key,
            y=value_key,
            title=title or f"{value_key.replace('_', ' ').title()} per {label_key.replace('_', ' ').title()}",
            color=value_key,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title=label_key.replace("_", " ").title(),
            yaxis_title=value_key.replace("_", " ").title(),
        )
        fig.update_traces(textposition="outside")
        return fig
    except Exception:
        return None


# =========================================================
# CHART: PIE — untuk distribusi/proporsi
# =========================================================

def build_pie_chart(data: list[dict], title: str = "") -> go.Figure | None:
    """
    Pie chart untuk distribusi/proporsi.
    Cocok untuk: sales per region, distribusi stok per wilayah.
    Dipakai otomatis jika jumlah kategori <= 8.
    """
    if not data or len(data) < 2:
        return None

    df = pd.DataFrame(data)
    keys = list(df.columns)

    label_key = keys[0]
    value_key = keys[1] if len(keys) > 1 else keys[0]

    for candidate in ["region_name", "product_name", "region", "bulan"]:
        if candidate in keys:
            label_key = candidate
            break

    for candidate in ["total_sales", "total_value", "total_stock",
                      "total_penjualan", "stok_tersedia"]:
        if candidate in keys:
            value_key = candidate
            break

    if label_key == value_key:
        return None

    try:
        fig = px.pie(
            df,
            names=label_key,
            values=value_key,
            title=title or f"Distribusi {value_key.replace('_', ' ').title()}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        return fig
    except Exception:
        return None


# =========================================================
# FITUR BARU #2 — LINE CHART untuk tren bulanan
# =========================================================

def build_line_chart(data: list[dict], title: str = "") -> go.Figure | None:
    """
    Line chart untuk tren waktu (bulanan).
    Otomatis dipakai jika data punya kolom tanggal/bulan.
    """
    if not data or len(data) < 2:
        return None

    df = pd.DataFrame(data)
    keys = list(df.columns)

    date_col  = next((c for c in keys if "date" in c.lower() or "bulan" in c.lower() or "month" in c.lower()), None)
    value_col = next((c for c in keys if "sales" in c.lower() or "value" in c.lower() or "total" in c.lower()), None)
    group_col = next((c for c in keys if "region" in c.lower() or "product" in c.lower()), None)

    if not date_col or not value_col:
        return None

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            color=group_col if group_col and group_col != date_col else None,
            title=title or "Tren Penjualan per Bulan",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            xaxis_title="Bulan",
            yaxis_title=value_col.replace("_", " ").title(),
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        return fig
    except Exception:
        return None


# =========================================================
# SUPPLY CHAIN CHARTS
# =========================================================

def build_doi_chart(data: list[dict]) -> go.Figure | None:
    """Bar chart DOI dengan garis threshold 30 hari."""
    if not data:
        return None

    df = pd.DataFrame(data)
    doi_col = next((c for c in df.columns if "doi" in c.lower() or "days" in c.lower()), None)
    if not doi_col:
        return None

    label = "product_name" if "product_name" in df.columns else df.columns[0]

    try:
        fig = px.bar(
            df,
            x=label,
            y=doi_col,
            color="region_name" if "region_name" in df.columns else None,
            title="Days of Inventory (DOI) per SKU",
            barmode="group",
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="red",
            annotation_text="Batas aman 30 hari",
            annotation_position="top right",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            xaxis_title="SKU",
            yaxis_title="DOI (hari)",
        )
        return fig
    except Exception:
        return None


def build_replenishment_chart(data: list[dict]) -> go.Figure | None:
    """Bar chart kebutuhan replenishment per SKU."""
    if not data:
        return None

    df = pd.DataFrame(data)
    rep_col = next((c for c in df.columns if "replan" in c.lower() or "replen" in c.lower()), None)
    if not rep_col:
        return None

    label = "product_name" if "product_name" in df.columns else df.columns[0]

    try:
        fig = px.bar(
            df,
            x=label,
            y=rep_col,
            color="region_name" if "region_name" in df.columns else None,
            title="Kebutuhan Replenishment per SKU",
            barmode="group",
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        return fig
    except Exception:
        return None


def build_region_heatmap(data: list[dict]) -> go.Figure | None:
    """Heatmap SKU vs Region."""
    if not data:
        return None

    df = pd.DataFrame(data)
    if "product_name" not in df.columns or "region_name" not in df.columns:
        return None

    val_col = next((c for c in df.columns if "sales" in c.lower() or "value" in c.lower()), None)
    if not val_col:
        return None

    try:
        pivot = df.pivot_table(index="product_name", columns="region_name", values=val_col, aggfunc="sum")
        fig = px.imshow(
            pivot,
            title="Heatmap Sales — SKU vs Region",
            color_continuous_scale="Blues",
            text_auto=True,
            aspect="auto",
        )
        fig.update_layout(font=dict(size=11))
        return fig
    except Exception:
        return None


# =========================================================
# FITUR BARU #2 — AUTO SELECT CHART
# Pilih chart yang paling tepat berdasarkan tipe data
# =========================================================

def auto_select_chart(data: list[dict], force_type: str = "") -> go.Figure | None:
    """
    Pilih chart otomatis berdasarkan tipe data:
    - Ada kolom tanggal/bulan → LINE chart
    - Ada region_name + value, kategori <= 8 → PIE chart
    - Ada doi/days → DOI bar chart dengan threshold
    - Default → BAR chart
    
    force_type: 'bar', 'pie', 'line' untuk paksa tipe tertentu
    """
    if not data:
        return None

    keys = list(data[0].keys())

    if force_type == "line":
        return build_line_chart(data)
    if force_type == "pie":
        return build_pie_chart(data)
    if force_type == "bar":
        return build_bar_chart(data)

    # Auto detect
    has_date  = any("date" in k.lower() or "bulan" in k.lower() or "month" in k.lower() for k in keys)
    has_doi   = any("doi" in k.lower() or "days" in k.lower() for k in keys)
    has_region = "region_name" in keys
    n_rows     = len(data)

    if has_date:
        return build_line_chart(data)

    if has_doi:
        return build_doi_chart(data)

    if has_region and n_rows <= 8:
        return build_pie_chart(data)

    return build_bar_chart(data)