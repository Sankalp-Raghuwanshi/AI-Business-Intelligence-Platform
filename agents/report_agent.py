"""
Report agent.

Takes whatever the sql / ml / rag agents produced (some tasks may have
failed or been skipped) and synthesizes one coherent, management-facing
answer. Also generates the fixed-format executive summary from raw KPIs,
same as the original generate_executive_summary function.
"""

from groq import Groq


class ReportAgent:
    name = "report"

    def __init__(self, client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.model = model

    def _call_llm(self, prompt: str, max_tokens: int = 400) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def synthesize(self, state: dict) -> str:
        parts = []
        results = state.get("results", {})

        if "sql" in results:
            r = results["sql"]
            if r.get("success"):
                parts.append("SQL result:\n" + r["dataframe"].head(10).to_string())
            else:
                parts.append("SQL agent failed: " + r.get("error", "unknown error"))

        if "ml" in results:
            r = results["ml"]
            if r.get("success"):
                parts.append(
                    "ML prediction: " + str(r.get("prediction")) +
                    " - " + str(r.get("explanation", ""))
                )
            else:
                parts.append("ML agent failed: " + r.get("error", "unknown error"))

        if "rag" in results:
            r = results["rag"]
            if r.get("success"):
                parts.append(
                    "Knowledge base answer: " + r["answer"] +
                    " (sources: " + ", ".join(r.get("sources", [])) + ")"
                )
            else:
                parts.append("RAG agent failed: " + r.get("error", "unknown error"))

        if not parts:
            return "I couldn't retrieve any data to answer this question. Try rephrasing it."

        context = "\n\n".join(parts)
        chart_note = (
            "A chart has been generated alongside this answer."
            if state.get("chart") is not None else ""
        )

        prompt = (
            "You are a Senior Business Analyst presenting to management. "
            "The user asked: " + state["user_query"] + "\n\n"
            "Here is what each specialist agent found:\n" + context + "\n\n" + chart_note +
            "\n\nWrite a structured response with exactly these three parts: "
            "1. WHAT: What does this data show? (1-2 sentences with specific numbers) "
            "2. WHY: Why might this have happened, or what does it mean? (1 sentence) "
            "3. ACTION: What should management do? (1 concrete recommendation) "
            "If any agent failed, acknowledge it briefly instead of ignoring it. "
            "Plain text only, no markdown symbols. Keep the whole thing under 180 words."
        )
        return self._call_llm(prompt)

    def executive_summary(self, kpis: dict) -> str:
        kpi_text = ", ".join(f"{k}: {v}" for k, v in kpis.items())
        prompt = (
            "You are the Head of Business Intelligence presenting to the CEO. "
            "Here are the company KPIs: " + kpi_text + ". "
            "Write a professional executive summary with exactly these 4 sections: "
            "PERFORMANCE: Overall business performance in 2 sentences. "
            "STRENGTHS: Top 2 strengths with specific numbers. "
            "RISKS: Top 2 risks or concerns with specific numbers. "
            "RECOMMENDATIONS: 2 concrete actionable recommendations for management. "
            "Plain text only, no markdown symbols. Maximum 250 words."
        )
        return self._call_llm(prompt, max_tokens=500)
