import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Load the correlation data from the provided JSON
correlation_data = {
    "Survived": [1.0, -0.338, -0.077, -0.035, -0.082, 0.257],
    "Pclass": [-0.338, 1.0, -0.369, 0.083, 0.018, -0.554],
    "Age": [-0.077, -0.369, 1.0, -0.308, -0.189, 0.096],
    "SibSp": [-0.035, 0.083, -0.308, 1.0, 0.414, 0.159],
    "Parch": [-0.082, 0.018, -0.189, 0.414, 1.0, 0.216],
    "Fare": [0.257, -0.554, 0.096, 0.159, 0.216, 1.0]
}

# Create correlation matrix
columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
correlation_matrix = np.array([correlation_data[col] for col in columns])

# Create heatmap with annotations
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix,
    x=columns,
    y=columns,
    colorscale='RdBu',
    zmid=0,
    text=[[f"{val:.3f}" for val in row] for row in correlation_matrix],
    texttemplate="%{text}",
    textfont={"size": 12},
    hoverongaps=False,
    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
))

# Update layout
fig.update_layout(
    title="Titanic Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features"
)

# Save as both PNG and SVG
fig.write_image("correlation_heatmap.png")
fig.write_image("correlation_heatmap.svg", format="svg")

print("Heatmap saved successfully!")