"""
Planner agent.

Given a user's natural-language question, the planner decides which of the
specialist agents (sql, ml, rag, viz) actually need to run, and what
instruction to give each one. It does not execute anything itself -
the orchestrator dispatches the plan it returns.
"""

import json
import re

from groq import Groq

PLANNER_SYSTEM_PROMPT = """You are the planning agent for a business intelligence system.
Given a user's question, decide which specialist agents are needed to answer it.

Available agents:
- sql: answers questions answerable by querying structured order/sales data
  (revenue, counts, averages, trends, group-bys). Use for anything with
  numbers that live in the database.
- ml: predicts delivery delay risk or review score for a hypothetical
  order, based on attributes like customer/seller state, category, price,
  freight, delivery days, and month. Use ONLY when the question asks to
  predict/estimate delay risk or review score for an order - not for
  aggregate historical stats (those go through sql).
- rag: answers questions that require company policy, definitions, or other
  unstructured knowledge-base documents (e.g. "what is our return policy",
  "how are review scores defined"). Use ONLY when the answer is not a number
  from the database.
- viz: produces a chart. Include this whenever the sql or ml result would be
  clearer as a chart (comparisons, trends, distributions).

Rules:
- Only include agents that are actually needed. Most questions need just
  "sql", optionally plus "viz".
- Only use "ml" if the question is explicitly about a future/predicted value.
- Only use "rag" if the question is about policy/definitions/documents, not
  raw numbers from the database.
- Always respond with ONLY valid JSON, no explanation, no markdown fences,
  in this exact shape:
{"tasks": [{"agent": "sql", "instruction": "..."}, {"agent": "viz", "instruction": "..."}]}
"""

VALID_AGENTS = {"sql", "ml", "rag", "viz"}


class PlannerAgent:
    name = "planner"

    def __init__(self, client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.model = model

    def plan(self, user_query: str, schema: str):
        prompt = (
            f"Database schema: {schema}\n\n"
            f"User question: {user_query}\n\n"
            f"Return the JSON plan now."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse(raw, user_query)

    def _parse(self, raw: str, user_query: str):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        json_str = match.group(0) if match else raw
        try:
            data = json.loads(json_str)
            tasks = data.get("tasks", [])
            cleaned = [
                t for t in tasks
                if isinstance(t, dict) and t.get("agent") in VALID_AGENTS and t.get("instruction")
            ]
            if cleaned:
                return cleaned
        except json.JSONDecodeError:
            pass
        # Fallback: if the planner didn't return usable JSON, default to a
        # plain SQL task so the system still attempts an answer.
        return [{"agent": "sql", "instruction": user_query}]
