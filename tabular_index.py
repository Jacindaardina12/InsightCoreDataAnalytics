from __future__ import annotations

"""
tabular_index.py — Project A: IL VASTO Replan Intelligence
============================================================
File ini TIDAK ADA PERUBAHAN dari versi lama.
Dipindah langsung ke src/ tanpa modifikasi.

Fungsi:
- SimpleEmbedder    : hash-based text embedding (tanpa model ML)
- TabularChunk      : dataclass untuk chunk data tabular
- TabularIndex      : vector store + hybrid retrieval
- build_tabular_chunks : buat chunks dari DataFrame
- format_retrieved_context : format chunks untuk LLM prompt
"""

import json
import math
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd


# ==============================
# TEXT UTILITIES
# ==============================

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_to_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# ==============================
# DATE + KEY HELPERS
# ==============================

def detect_date_range(values: Iterable[Any]) -> tuple[str, str] | None:
    parsed: list[datetime] = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            parsed.append(pd.to_datetime(v))
        except Exception:
            continue
    if not parsed:
        return None
    parsed.sort()
    return parsed[0].date().isoformat(), parsed[-1].date().isoformat()


def infer_key_columns(columns: list[str]) -> list[str]:
    keys = [col for col in columns if col.lower() == "id" or col.lower().endswith("_id")]
    if not keys and columns:
        keys.append(columns[0])
    return keys


# ==============================
# CHUNK OBJECT
# ==============================

@dataclass
class TabularChunk:
    chunk_id: str
    table: str
    text: str
    tokens: set[str]
    row_ids: list[int]
    key_columns: list[str]
    date_range: tuple[str, str] | None
    categories: list[str]

    def metadata(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "row_ids": self.row_ids,
            "date_range": list(self.date_range) if self.date_range else None,
            "categories": self.categories,
        }


# ==============================
# EMBEDDING (STABLE HASH)
# ==============================

class SimpleEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            stable_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = stable_hash % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ==============================
# TABULAR INDEX
# ==============================

class TabularIndex:
    def __init__(self, embedder: SimpleEmbedder | None = None):
        self.embedder = embedder or SimpleEmbedder()
        self.chunks: list[TabularChunk] = []
        self.embeddings: list[list[float]] = []

    def add_chunk(self, chunk: TabularChunk) -> None:
        self.chunks.append(chunk)
        self.embeddings.append(self.embedder.embed(chunk.text))

    def add_chunks(self, chunks: Iterable[TabularChunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        hybrid: bool = True,
        alpha: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:

        if not self.chunks:
            return []

        query_emb = self.embedder.embed(question)
        token_set = set(tokenize(question))
        candidates = list(range(len(self.chunks)))

        if filters:
            table_filter = filters.get("table")
            if table_filter:
                candidates = [i for i in candidates if self.chunks[i].table == table_filter]

        scored = []
        for idx in candidates:
            chunk = self.chunks[idx]
            vec_score = cosine_similarity(query_emb, self.embeddings[idx])
            keyword_score = 0.0
            if token_set:
                overlap = token_set.intersection(chunk.tokens)
                keyword_score = len(overlap) / max(1, len(token_set))
            score = alpha * vec_score + (1 - alpha) * keyword_score if hybrid else vec_score
            scored.append({
                "chunk": chunk,
                "score": score,
                "vector_score": vec_score,
                "keyword_score": keyword_score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ==============================
# BUILD CHUNKS
# ==============================

def compact_row_text(row: pd.Series, key_columns: list[str]) -> str:
    parts: list[str] = []
    for col, value in row.items():
        if pd.isna(value):
            continue
        text = safe_to_string(value)
        if text:
            parts.append(f"{col}={text}")
    key_values = [
        safe_to_string(row[k])
        for k in key_columns
        if k in row and not pd.isna(row[k])
    ]
    return f"key={','.join(key_values)} | " + ", ".join(parts)


def build_tabular_chunks(
    df: pd.DataFrame,
    table_name: str,
    rows_per_chunk: int = 50,
) -> list[TabularChunk]:

    columns = list(df.columns)
    key_columns = infer_key_columns(columns)
    chunks: list[TabularChunk] = []
    rows_per_chunk = max(1, rows_per_chunk)

    for start in range(0, len(df), rows_per_chunk):
        end = min(start + rows_per_chunk, len(df))
        sub = df.iloc[start:end]
        row_ids = list(range(start, end))
        row_texts = [compact_row_text(row, key_columns) for _, row in sub.iterrows()]
        text = normalize_whitespace(f"table={table_name} | " + " || ".join(row_texts))
        tokens = set(tokenize(text))

        date_range = None
        for col in columns:
            if "date" in col.lower():
                date_range = detect_date_range(sub[col].values)
                if date_range:
                    break

        categories: list[str] = []
        for col in columns:
            if df[col].dtype == object:
                uniques = df[col].dropna().unique()
                if 1 <= len(uniques) <= 10:
                    categories.extend([safe_to_string(u) for u in uniques[:10]])

        chunks.append(TabularChunk(
            chunk_id=f"{table_name}:rows:{start}-{end - 1}",
            table=table_name,
            text=text,
            tokens=tokens,
            row_ids=row_ids,
            key_columns=key_columns,
            date_range=date_range,
            categories=categories,
        ))

    return chunks


# ==============================
# CONTEXT FORMATTERS
# ==============================

def format_retrieved_context(retrieved: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in retrieved:
        chunk: TabularChunk = item["chunk"]
        meta_text = json.dumps(chunk.metadata(), ensure_ascii=False)
        lines.append(f"[{chunk.chunk_id}] {chunk.text} | meta={meta_text}")
    return "\n".join(lines)


def make_sources(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "table": item["chunk"].table,
            "row_ids": item["chunk"].row_ids,
            "chunk_id": item["chunk"].chunk_id,
            "snippet": item["chunk"].text[:240],
        }
        for item in retrieved
    ]


def make_retrieved_chunks_payload(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk"].chunk_id,
            "text": item["chunk"].text,
            "score": round(item["score"], 6),
            "metadata": item["chunk"].metadata(),
        }
        for item in retrieved
    ]
