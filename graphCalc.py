import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Scientific Graphing Calculator")

st.write("Enter your experimental data")

# Data input
x_values = st.text_input("Enter X values (comma separated)", "1,2,3,4,5")
y_values = st.text_input("Enter Y values (comma separated)", "2,4,5,4,5")

x = np.array([float(i) for i in x_values.split(",")])
y = np.array([float(i) for i in y_values.split(",")])

# Variable names
x_label = st.text_input("X axis variable", "X")
y_label = st.text_input("Y axis variable", "Y")

if st.button("Generate Graph"):

    # reshape for sklearn
    X = x.reshape(-1,1)

    # Linear regression
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_

    r2 = r2_score(y, y_pred)

    # Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(symbol="x", size=10),
        name="Data Points"
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode="lines",
        name="Best Fit Line"
    ))

    fig.update_layout(
        title="Scientific Graph",
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    st.plotly_chart(fig)

    st.subheader("Graph Analysis")

    st.write(f"**Slope:** {slope:.4f}")
    st.write(f"**Intercept:** {intercept:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    st.write("### Equation")

    st.latex(f"{y_label} = {slope:.3f}{x_label} + {intercept:.3f}")