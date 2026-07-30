# Multi-agent refactor

This adds the planner -> {sql, ml, rag} -> viz -> report architecture on top
of your existing app. Drop these files into your repo, keeping the same
relative paths:

```
agents/
  __init__.py
  planner_agent.py
  sql_agent.py
  ml_agent.py
  rag_agent.py
  viz_agent.py
  report_agent.py
orchestrator.py
app.py                       (replaces your current app.py)
data/knowledge_base/*.md     (sample docs for the RAG agent)
```

Your `predict.py` and `data/olist.db` stay exactly where they are.

## Install the one new dependency

```
pip install sentence-transformers
```

This powers the RAG agent's embeddings locally - no extra API key needed.
First run will download the `all-MiniLM-L6-v2` model (~90MB) and cache
embeddings to `data/kb_embeddings.npz`; subsequent runs are fast.

## One thing you need to do: expose `run_prediction` from your model

Your current `predict.py` is a Streamlit page (loaded via `exec()`), so it
likely mixes model logic with `st.*` widget calls. To wrap it as the ML
agent, pull the model-loading + inference into a new `predict_core.py`:

```python
# predict_core.py
def run_prediction(question: str) -> dict:
    # your existing model loading / inference code
    return {
        "prediction": ...,     # the predicted value
        "explanation": "...",  # one sentence on what it means
    }
```

Keep `predict.py`'s Streamlit rendering as-is; just have it call the same
`run_prediction()` so there's one source of truth. Until you add this file,
the ML agent will report a clear "not configured" message instead of
crashing - the rest of the pipeline (sql, rag, viz, report) works
regardless.

## What changed vs. the original app.py

- Planner now decides which agents run per question (most questions still
  just hit `sql`, but "what's our return policy" now correctly routes to
  `rag` instead of failing as a SQL query).
- `sql`, `ml`, `rag` run concurrently via a thread pool when more than one
  is needed - not one after another.
- The old "Deep Analysis" question-decomposition feature is superseded by
  the planner + RAG agent combination; remove that block from your old
  app.py (already dropped in the new one).
- Executive Summary button and KPI cards behave identically to before.
- Add more `.md`/`.txt` files to `data/knowledge_base/` any time - the RAG
  agent rebuilds its embedding cache automatically when it sees a newer
  file.
