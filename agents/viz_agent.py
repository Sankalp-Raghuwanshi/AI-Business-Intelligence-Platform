"""
Visualization agent.

Picks a chart type based on the shape of a dataframe and the wording of
the question, and returns a Plotly figure. Direct refactor of the original
generate_chart function.
"""

import plotly.express as px


class VizAgent:
    name = "viz"

    def run(self, df, question: str):
        if df is None or len(df) <= 1:
            return None

        question_lower = question.lower()
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if not numeric_cols:
            return None

        numeric_col = numeric_cols[0]
        time_keywords = ["month", "year", "trend", "over time", "monthly", "yearly", "daily"]

        if any(word in question_lower for word in time_keywords) and len(cols) >= 2:
            fig = px.line(
                df,
                x=cols[0],
                y=numeric_col,
                title="\U0001F4C8 " + question.title(),
                markers=True,
                color_discrete_sequence=["#00C9A7"],
            )
        elif text_cols:
            fig = px.bar(
                df.sort_values(by=numeric_col, ascending=False).head(15),
                x=numeric_col,
                y=text_cols[0],
                orientation="h",
                title="\U0001F4CA " + question.title(),
                color=numeric_col,
                color_continuous_scale="Teal",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        else:
            fig = px.bar(
                df.sort_values(by=numeric_col, ascending=False).head(15),
                x=cols[0],
                y=numeric_col,
                title="\U0001F4CA " + question.title(),
                color=numeric_col,
                color_continuous_scale="Teal",
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        return fig
