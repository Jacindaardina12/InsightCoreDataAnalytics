from __future__ import annotations

import re
import time
from typing import Any

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


class DistilLabsLLM:
    """
    LLM client untuk Text-to-SQL pipeline.
    Sebelumnya: OpenAI client → Ollama lokal (http://127.0.0.1:11434/v1)
    Sekarang  : Groq API (cloud, gratis, model LLaMA3 identik)

    Perubahan dari file lama:
    - Hapus dependency openai
    - Ganti ke groq.Groq client
    - API key dibaca dari .env (GROQ_API_KEY)
    - Semua method (invoke_sql, invoke_rag, invoke_sql_grounded_explanation) TIDAK BERUBAH
    """

    def __init__(
        self,
        model_name: str = None,
        rag_max_output_chars: int = 900,
        request_timeout_seconds: float = 30.0,
    ):
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.rag_max_output_chars = max(120, int(rag_max_output_chars))
        self.request_timeout_seconds = max(5.0, float(request_timeout_seconds))
        self.available = True
        self._stub_reason = ""

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY tidak ditemukan di .env\n"
                "Daftar gratis di https://console.groq.com → buat API key → isi .env"
            )

        self.client = Groq(api_key=api_key)

        # Test koneksi saat startup
        try:
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(f"Gagal konek ke Groq API: {e}")

    # ==============================
    # PROMPTS (tidak berubah dari file lama)
    # ==============================

    def get_sql_prompt(self, schema: str, question: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """
You are a data assistant. Produce exactly ONE SQLite-compatible SQL query.
- Return SQL only.
- No commentary.
- No markdown.
- No backticks.
- Use only tables and columns listed in SCHEMA.
If the question is slightly ambiguous, make a reasonable assumption and still return one query.
""",
            },
            {
                "role": "user",
                "content": f"""
SCHEMA:
{schema}

QUESTION:
{question}
""",
            },
        ]

    def get_rag_prompt(self, question: str, retrieved_context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """
You are a retrieval-augmented assistant.
Use ONLY the RETRIEVED INFORMATION to answer.
If the answer is not explicitly supported, reply exactly:
insufficient information

Rules:
- English only
- Plain text only
- No emojis
- No markdown
- Keep answer concise
""",
            },
            {
                "role": "user",
                "content": f"""
QUESTION:
{question}

RETRIEVED INFORMATION:
{retrieved_context}
""",
            },
        ]

    def get_sql_explanation_prompt(
        self,
        question: str,
        sql: str,
        result_columns: list[str],
        result_rows: list[dict[str, Any]],
    ) -> list[dict[str, str]]:

        evidence = {
            "columns": result_columns,
            "rows": result_rows,
        }

        return [
            {
                "role": "system",
                "content": """
You are a SQL-results analyst.

Write a concise explanation using ONLY RESULT EVIDENCE.

Rules:
- Do not claim anything not present in RESULT EVIDENCE.
- If no rows exist, reply exactly: insufficient information
- Use plain business language in Bahasa Indonesia.
- Max 120 words.
""",
            },
            {
                "role": "user",
                "content": f"""
QUESTION:
{question}

SQL:
{sql}

RESULT EVIDENCE:
{evidence}
""",
            },
        ]

    # ==============================
    # INVOCATION METHODS (tidak berubah dari file lama)
    # ==============================

    def invoke_sql(self, schema: str, question: str) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError(self._stub_reason or "LLM unavailable")

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_sql_prompt(schema, question),
            temperature=0,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        content = self._clean_sql(content)

        return content, time.perf_counter() - start

    def invoke_rag(self, question: str, retrieved_context: str) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError(self._stub_reason or "LLM unavailable")

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_rag_prompt(question, retrieved_context),
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        content = self._trim_output(content)

        return content, time.perf_counter() - start

    def invoke_sql_grounded_explanation(
        self,
        question: str,
        sql: str,
        result_columns: list[str],
        result_rows: list[dict[str, Any]],
    ) -> tuple[str, float]:
        if not self.available:
            raise RuntimeError(self._stub_reason or "LLM unavailable")

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_sql_explanation_prompt(
                question, sql, result_columns, result_rows
            ),
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        content = self._clean_sql(content)

        return content, time.perf_counter() - start

    # ==============================
    # OUTPUT CLEANING (tidak berubah dari file lama)
    # ==============================

    def _trim_output(self, text: str) -> str:
        cleaned = text.strip()

        if len(cleaned) <= self.rag_max_output_chars:
            return cleaned

        head = cleaned[: self.rag_max_output_chars]
        split_point = max(head.rfind("."), head.rfind("!"), head.rfind("?"))

        if split_point < int(self.rag_max_output_chars * 0.5):
            split_point = head.rfind(" ")

        if split_point <= 0:
            split_point = self.rag_max_output_chars

        trimmed = head[:split_point].rstrip()

        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."

        return trimmed

    def _clean_sql(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = re.sub(r"^sql\s*:\s*", "", text, flags=re.IGNORECASE)

        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lower().startswith("select"):
                return "\n".join(lines[i:]).strip()

        return text.strip()
