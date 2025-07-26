import streamlit as st
import pandas as pd


st.title(" dashboards Explaining Black-Box Classifiers with SHAP and LIME")
st.write(
    "This app demonstrates how to use SHAP and LIME to explain predictions "
    "from a black-box model."
)
st.sidebar.header("Select an Instance to Explain")
st.sidebar.write("### Selected Instance Details")
#st.sidebar.dataframe(instance)
st.sidebar.write("True Label")
st.sidebar.write("Model prediction")
st.sidebar.write("Predicted Probability")

st.markdown("---")

st.header("1. SHAP (SHapley Additive exPlanations)")
st.write(
    "SHAP uses game theory to explain the output of any machine learning model. "
    "The plot below shows features pushing the prediction higher (in red) and "
    "features pushing it lower (in blue)."
)
st.subheader("SHAP Force Plot")
st.markdown("---")
st.header("2. LIME (Local Interpretable Model-agnostic Explanations)")
st.write(
    "LIME explains a prediction by creating a simple, interpretable local model "
    "(like a linear model) around the specific instance. The plot below shows "
    "the features that were most influential for this single prediction."
)
st.subheader("LIME Explanation Plot")