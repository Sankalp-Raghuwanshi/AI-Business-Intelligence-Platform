"""
Orchestrator.

Wires the whole pipeline together:

    user query -> planner agent -> {sql, ml, rag} in parallel -> viz agent -> report agent

The planner decides which of sql/ml/rag are actually needed (most
questions only need sql). Those run concurrently since they're
independent, LLM/IO-bound calls. The viz agent runs afterward because it
depends on the sql agent's dataframe. The report agent always runs last
and synthesizes whatever came back, including partial failures.
"""

import concurrent.futures as cf


class BIOrchestrator:
    def __init__(self, planner, sql_agent, ml_agent, rag_agent, viz_agent, report_agent):
        self.planner = planner
        self.agents = {
            "sql": sql_agent,
            "ml": ml_agent,
            "rag": rag_agent,
        }
        self.viz_agent = viz_agent
        self.report_agent = report_agent

    def handle(self, user_query: str, schema: str) -> dict:
        state = {
            "user_query": user_query,
            "plan": [],
            "results": {},
            "errors": [],
            "chart": None,
            "final_report": "",
        }

        plan = self.planner.plan(user_query, schema)
        state["plan"] = plan

        wants_viz = any(t["agent"] == "viz" for t in plan)
        core_tasks = [t for t in plan if t["agent"] in self.agents]

        if core_tasks:
            with cf.ThreadPoolExecutor(max_workers=len(core_tasks)) as executor:
                futures = {}
                for task in core_tasks:
                    agent = self.agents.get(task["agent"])
                    if agent is None:
                        state["errors"].append(
                            f"{task['agent']} agent is not configured, skipping."
                        )
                        continue
                    futures[executor.submit(agent.run, task["instruction"])] = task["agent"]

                for future in cf.as_completed(futures):
                    agent_name = futures[future]
                    try:
                        state["results"][agent_name] = future.result()
                    except Exception as e:
                        state["errors"].append(f"{agent_name} agent raised an exception: {e}")

        if wants_viz:
            sql_result = state["results"].get("sql")
            if sql_result and sql_result.get("success"):
                state["chart"] = self.viz_agent.run(sql_result["dataframe"], user_query)

        state["final_report"] = self.report_agent.synthesize(state)
        return state
