import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def nice_number(value):
    exponent = np.floor(np.log10(value))
    fraction = value / 10**exponent

    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * 10**exponent

def calculate_axis_limits(data):

    minimum = np.min(data)
    maximum = np.max(data)

    range_val = maximum - minimum

    tick_spacing = nice_number(range_val / 8)

    axis_min = np.floor(minimum / tick_spacing) * tick_spacing
    axis_max = np.ceil(maximum / tick_spacing) * tick_spacing

    return axis_min, axis_max, tick_spacing



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

    x_min, x_max, x_tick = calculate_axis_limits(x)
    y_min, y_max, y_tick = calculate_axis_limits(y)

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
    yaxis_title=y_label,

    xaxis=dict(
        range=[x_min, x_max],
        dtick=x_tick
    ),

    yaxis=dict(
        range=[y_min, y_max],
        dtick=y_tick
    )
)

    st.plotly_chart(fig)

    st.subheader("Axis Information")

    st.write(f"X-axis range: {x_min} to {x_max}")
    st.write(f"X-axis scale: {x_tick}")

    st.write(f"Y-axis range: {y_min} to {y_max}")
    st.write(f"Y-axis scale: {y_tick}")

    st.subheader("Graph Analysis")

    st.write(f"**Slope:** {slope:.4f}")
    st.write(f"**Intercept:** {intercept:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    st.write("### Equation")

    st.latex(f"{y_label} = {slope:.3f}{x_label} + {intercept:.3f}")