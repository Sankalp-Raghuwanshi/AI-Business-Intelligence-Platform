"""
SQL agent.

Turns a natural-language instruction into a SQLite query against
master_orders, runs it, and returns the resulting dataframe. This is a
direct refactor of the generate_sql / extract_sql / run_query functions
from the original app.py, just wrapped in a class so the orchestrator can
call it like every other agent.
"""

import re
import sqlite3

import pandas as pd
from groq import Groq


class SQLAgent:
    name = "sql"

    def __init__(self, client: Groq, db_path: str, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.db_path = db_path
        self.model = model

    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_schema(self) -> str:
        conn = self._connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(master_orders)")
        columns = cursor.fetchall()
        conn.close()
        schema = "Table: master_orders, Columns: "
        for col in columns:
            schema += col[1] + " (" + col[2] + "), "
        return schema

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = (
            "You are a SQL expert. Database schema: " + schema +
            " Write a SQLite SQL query to answer: " + question +
            " Rules: table name is master_orders, use payment_value for revenue, "
            "use customer_state for state. Return ONLY the SQL query inside "
            "```sql ``` blocks. For monthly trend questions, always group by "
            "both order_year and order_month together and order by "
            "order_year, order_month."
        )
        return self._call_llm(prompt)

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        match = re.search(r"```sql\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"SELECT.*?;", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return None

    def run(self, instruction: str) -> dict:
        schema = self.get_schema()
        raw = self._generate_sql(instruction, schema)
        sql_query = self._extract_sql(raw)

        if not sql_query:
            return {
                "agent": self.name,
                "success": False,
                "error": "Could not generate a valid SQL query for this question.",
            }

        try:
            conn = self._connection()
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
        except Exception as e:
            return {"agent": self.name, "success": False, "sql": sql_query, "error": str(e)}

        if df is None or len(df) == 0:
            return {
                "agent": self.name,
                "success": False,
                "sql": sql_query,
                "error": "Query ran successfully but returned no rows.",
            }

        return {"agent": self.name, "success": True, "sql": sql_query, "dataframe": df}
