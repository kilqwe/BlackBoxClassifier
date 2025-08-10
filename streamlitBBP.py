import streamlit as st
import pandas as pd
from model import load_and_prepare_data, train_model
from SHAPpredictor import shap_explainer
from streamlit.components.v1 import html
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
#shap.initjs()

st.title(" Dashboard Explaining Black-Box Classifiers with SHAP and LIME")
st.write(
    "This app demonstrates how to use SHAP and LIME to explain predictions "
    "from a black-box model."
)
X, y, data = load_and_prepare_data()
model = train_model(X, y)
explainer, shap_values = shap_explainer(model, X)

# Sidebar
st.sidebar.header("Select an Instance to Explain")
index = st.sidebar.slider("Instance Index", 0, len(X) - 1, 0)
st.sidebar.write("### Selected Instance Details")
st.sidebar.dataframe(data.iloc[[index]])
st.sidebar.write("True Label:", y.iloc[index])
st.sidebar.write("Model Prediction:", model.predict(X.iloc[[index]])[0])

st.markdown("---")

st.header("1. SHAP (SHapley Additive exPlanations)")
st.write(
    "SHAP uses game theory to explain the output of any machine learning model. "
    "The plot below shows features pushing the prediction higher (in red) and "
    "features pushing it lower (in blue)."
)
st.subheader("SHAP Force Plot")

# darkMode = '''
# <style>
#     body {
#         background-color: #0E1117;
#         color: #FAFAFA;
#     }
# </style>
# '''



shapForce = shap.plots.force(
    explainer.expected_value[1], # First argument is the base value
    shap_values=shap_values[index,:,1].values,
    features=X.iloc[index],
    matplotlib=False,
    show=False
)

# Use st.pyplot to display the figure
# st.pyplot(shapforce, bbox_inches='tight')
# plt.clf()
components.html(
    shapForce.html(),
    height=200,
    scrolling=True,
)
#darkPlot = darkMode + shapForce


st.subheader("SHAP Summary Plot (Bar)")

fig2 = plt.figure()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig2)
st.markdown("---")



st.header("2. LIME (Local Interpretable Model-agnostic Explanations)")
st.write(
    "LIME explains a prediction by creating a simple, interpretable local model "
    "(like a linear model) around the specific instance. The plot below shows "
    "the features that were most influential for this single prediction."
)
st.subheader("LIME Explanation Plot")