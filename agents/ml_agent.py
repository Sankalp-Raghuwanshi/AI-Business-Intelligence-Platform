"""
ML agent.

Wraps the project's existing predict.py so it can be called like every
other agent, instead of only being usable as a Streamlit page.

IMPORTANT - one-time setup needed on your side:
Your current predict.py is invoked with `exec(open("predict.py").read())`,
which means it's written as a Streamlit *page* that likely calls st.*
widgets directly rather than returning a value. To plug it into this agent,
pull the model-loading + inference code out into a plain function in a new
predict_core.py (keep predict.py's st.* rendering as-is, just have it call
this same function too, so there's one source of truth):

    # predict_core.py
    def run_prediction(question: str) -> dict:
        # ... your existing model loading / inference logic ...
        return {
            "prediction": <the predicted value>,
            "explanation": "<one sentence on what this means>",
        }

Once predict_core.py exists with that function, this agent will pick it up
automatically - no changes needed here.
"""

import importlib


class MLAgent:
    name = "ml"

    def __init__(self, module_name: str = "predict_core"):
        self.module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)

    def run(self, instruction: str) -> dict:
        try:
            self._load()
        except ImportError:
            return {
                "agent": self.name,
                "success": False,
                "error": (
                    f"No {self.module_name}.py found. Create it with a "
                    f"run_prediction(question) function that wraps your "
                    f"existing predict.py model (see module docstring)."
                ),
            }

        if not hasattr(self._module, "run_prediction"):
            return {
                "agent": self.name,
                "success": False,
                "error": f"{self.module_name}.py has no run_prediction(question) function.",
            }

        try:
            result = self._module.run_prediction(instruction)
            return {"agent": self.name, "success": True, **result}
        except Exception as e:
            return {"agent": self.name, "success": False, "error": str(e)}
