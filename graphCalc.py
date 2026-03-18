import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==============================
# Utility Functions
# ==============================

def detect_outliers(data, threshold=2.5):
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return np.zeros_like(data, dtype=bool)

    z_scores = (data - mean) / std
    return np.abs(z_scores) > threshold


def nice_number(value):
    if value == 0:
        return 1

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

    if range_val == 0:
        range_val = abs(minimum) if minimum != 0 else 1

    tick_spacing = nice_number(range_val / 8)

    axis_min = np.floor(minimum / tick_spacing) * tick_spacing
    axis_max = np.ceil(maximum / tick_spacing) * tick_spacing

    return axis_min, axis_max, tick_spacing


# ==============================
# Streamlit UI
# ==============================

st.title("Scientific Graphing Calculator")

input_mode = st.radio(
    "Select Input Method",
    ["Manual Input", "Upload CSV"]
)

st.write("Enter your experimental data")


if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        import pandas as pd

        df = pd.read_csv(uploaded_file)

        st.write("Preview of data:")
        st.dataframe(df)

        columns = df.columns.tolist()

        x_column = st.selectbox("Select X variable", columns)
        y_column = st.selectbox("Select Y variable", columns)

        x = df[x_column].values
        y = df[y_column].values

        x_label = x_column
        y_label = y_column
elif input_mode == "Manual Input":
    x_values = st.text_input("Enter X values (comma-separated)", "1, 2, 3, 4, 5")
    y_values = st.text_input("Enter Y values (comma-separated)", "2, 3, 5, 7, 11")

    
    # Parse input safely
    try:
        x = np.array([float(i.strip()) for i in x_values.split(",")])
        y = np.array([float(i.strip()) for i in y_values.split(",")])
    except:
        st.error("Invalid input. Please enter numeric values separated by commas.")
        st.stop()

    # Validate length
    if len(x) != len(y):
        st.error("X and Y must have the same number of values.")
        st.stop()

    # Axis labels
    x_label = st.text_input("X axis variable", "X")
    y_label = st.text_input("Y axis variable", "Y")

if 'x' not in locals() or 'y' not in locals():
    st.warning("Please provide valid input data.")
    st.stop()
    
# ==============================
# Main Logic
# ==============================

if st.button("Generate Graph"):

    # -------- Initial Regression (ALL data) --------
    X_all = x.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_all, y)

    y_pred_all = model.predict(X_all)

    # -------- Residual-based Outlier Detection --------
    residuals = y - y_pred_all
    outliers = detect_outliers(residuals)

    # Separate data
    x_outliers = x[outliers]
    y_outliers = y[outliers]

    x_normal = x[~outliers]
    y_normal = y[~outliers]

    # -------- Refit model WITHOUT outliers --------
    if len(x_normal) >= 2:
        X_normal = x_normal.reshape(-1, 1)
        model.fit(X_normal, y_normal)
    else:
        st.warning("Not enough non-outlier points for regression. Using all data.")
        model.fit(X_all, y)

    # Final prediction for full graph
    y_pred = model.predict(X_all)

    # -------- Final metrics --------
    slope = model.coef_[0]
    intercept = model.intercept_

    if len(x_normal) >= 2:
        y_pred_normal = model.predict(x_normal.reshape(-1, 1))
        r2 = r2_score(y_normal, y_pred_normal)
    else:
        r2 = r2_score(y, y_pred)

    # -------- Axis scaling --------
    x_min, x_max, x_tick = calculate_axis_limits(x)
    y_min, y_max, y_tick = calculate_axis_limits(y)

    # ==============================
    # Plotting
    # ==============================

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=x_normal,
        y=y_normal,
        mode="markers",
        marker=dict(symbol="x", size=10, color="blue"),
        name="Data Points"
    ))

    # Outliers
    if len(x_outliers) > 0:
        fig.add_trace(go.Scatter(
            x=x_outliers,
            y=y_outliers,
            mode="markers",
            marker=dict(symbol="x", size=12, color="red"),
            name="Outliers"
        ))

    # Best fit line
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode="lines",
        line=dict(color="green", width=3),
        name="Best Fit Line"
    ))

    # Layout with scientific grid
    fig.update_layout(
        title="Scientific Graph",

        xaxis=dict(
            title=x_label,
            range=[x_min, x_max],
            dtick=x_tick,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True
        ),

        yaxis=dict(
            title=y_label,
            range=[y_min, y_max],
            dtick=y_tick,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Output Info
    # ==============================

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

    st.subheader("Outlier Detection")

    st.write(f"Number of outliers detected: {len(x_outliers)}")

    if len(x_outliers) > 0:
        st.write("Outlier points:")
        st.write(list(zip(x_outliers, y_outliers)))