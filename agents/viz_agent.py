"""
Visualization agent.

Picks a chart type based on the shape of a dataframe and the wording of
the question, and returns a Plotly figure. Direct refactor of the original
generate_chart function.
"""

import plotly.express as px


class VizAgent:
    name = "viz"

    def generate_chart(df, question):
    # Detect chart type based on data shape and question keywords
    question_lower = question.lower()
    
    # If only 1 row returned, no chart needed
    if len(df) <= 1:
        return None
    
    cols = df.columns.tolist()
    
    # Find numeric and text columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    numeric_col = numeric_cols[0]
    
    # Time series — line chart
    time_keywords = ['month', 'year', 'trend', 'over time', 'monthly', 'yearly', 'daily']
    if any(word in question_lower for word in time_keywords):
        if len(cols) >= 2:
            fig = px.line(
                df,
                x=cols[0],
                y=numeric_col,
                title="📈 " + question.title(),
                markers=True,
                color_discrete_sequence=["#00C9A7"]
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            return fig
    
    # Categorical comparison — horizontal bar chart
    if text_cols and numeric_cols:
        fig = px.bar(
            df.sort_values(by=numeric_col, ascending=False).head(15),
            x=numeric_col,
            y=text_cols[0],
            orientation='h',
            title="📊 " + question.title(),
            color=numeric_col,
            color_continuous_scale="Teal"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig
    
    # Pure numeric — vertical bar
    if len(numeric_cols) >= 1:
        fig = px.bar(
            df.sort_values(by=numeric_col, ascending=False).head(15),
            x=cols[0],
            y=numeric_col,
            title="📊 " + question.title(),
            color=numeric_col,
            color_continuous_scale="Teal"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        return fig
    
    return None