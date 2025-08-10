import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

# Import the necessary functions from your other files
from model import load_and_prepare_data, train_model
from LIMEpredictor import create_lime_explainer, explain_instance
from SHAPpredictor import create_shap_explainer_and_values

# Set page configuration for a cleaner look
st.set_page_config(layout="wide")


# --- Cached Functions for Performance ---
@st.cache_data
def load_data():
    """Cached function to load and prepare data."""
    X, y_class, original_data = load_and_prepare_data()
    return X, y_class, original_data


@st.cache_resource
def get_explainers_and_model(_X, _y_class):
    """Cached function to train model and create all explainers."""
    model = train_model(_X, _y_class)
    lime_explainer = create_lime_explainer(
        training_data=_X.values,
        feature_names=_X.columns.tolist(),
        class_names=model.classes_
    )
    shap_explainer, shap_values = create_shap_explainer_and_values(model, _X)
    return model, lime_explainer, shap_explainer, shap_values


def create_lime_plot(explanation_list: list, predicted_class: str) -> plt.figure:
    """Creates a bar plot from the LIME explanation list."""
    exp_df = pd.DataFrame(explanation_list, columns=["feature", "weight"])
    exp_df["color"] = exp_df["weight"].apply(lambda x: '#90EE90' if x > 0 else '#F08080')
    exp_df = exp_df.sort_values(by="weight", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(exp_df["feature"], exp_df["weight"], color=exp_df["color"])
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.set_title(f"LIME explanation (Prediction: '{predicted_class.upper()}')", fontsize=14)
    ax.set_xlabel("Feature Contribution (Weight)", fontsize=10)
    plt.tight_layout()
    return fig


# --- Main App Logic ---

st.title("SHAP and LIME Explanations for Black-Box Classifiers")
st.write("Comparing LIME and SHAP explanations for a Random Forest model's predictions.")

# Load data and get models/explainers using cached functions
X, y_class, original_data = load_data()
model, lime_explainer, shap_explainer, shap_values = get_explainers_and_model(X, y_class)

# --- Interactive Sidebar ---
st.sidebar.header("Select an instance to explain")
sample_index = st.sidebar.slider(
    "Instance Index:",
    min_value=0,
    max_value=len(X) - 1,
    value=25,
    step=1
)

# --- Display Predictions & Data in Sidebar ---
sample_to_explain = X.iloc[sample_index]
true_class = y_class.iloc[sample_index]
predicted_class = model.predict(sample_to_explain.values.reshape(1, -1))[0]

st.sidebar.subheader("Model Prediction")
st.sidebar.metric(label="Model Predicts", value=predicted_class)
st.sidebar.metric(label="True Class", value=true_class)
st.sidebar.subheader("Selected Instance Data")
st.sidebar.write("Features:")
st.sidebar.dataframe(sample_to_explain)


# --- Main Page for Explanations ---


# Create two tabs for LIME and SHAP
tab1, tab2 = st.tabs(["LIME Explanation", "SHAP Explanation"])

with tab1:
    st.subheader("Local Interpretable Model-agnostic Explanations (LIME)")
    st.write("LIME explains a single prediction by creating a simpler, interpretable local model around it.")
    lime_explanation_list = explain_instance(lime_explainer, model, sample_to_explain.values)
    lime_fig = create_lime_plot(lime_explanation_list, predicted_class)
    st.pyplot(lime_fig)
    
    with st.expander("Plot Interpretation for example instance"):
        st.info("""
            This plot acts as a **pro and con list** for the selected abalone.
            - **What to Look For**: The longest bars on the right (positive) tell us the features of the abalone that pushed the model towards the prediction it made and left (negative) that tells us about the features that pushed the model against the prediction it made.
            - **Example Summary for Instance Index = 0**: "For this abalone instance, the plot shows that its **Shucked Weight and Sex(here Infant/Immature)** were the strongest reasons the model predicted **OLD**. Conversely, its **Shell Height and Height** were the biggest factors arguing against it."
        """)


with tab2:
    st.subheader("SHapley Additive exPlanations (SHAP)")
    st.write("SHAP uses game theory to explain the contribution of each feature to the prediction.")

    class_names = model.classes_.tolist()
    predicted_class_index = class_names.index(predicted_class)
    
    st.write(f"Showing SHAP explanation for the **'{predicted_class.upper()}'** class prediction.")
    
    # --- Section for Decision Plot ---
    st.divider() 
    st.subheader("Decision Plot")
    st.write("Shows how the model prediction moves from the base value to the final score as each feature is added.")
    plt.figure()
    shap.decision_plot(
        base_value=shap_explainer.expected_value[predicted_class_index],
        shap_values=shap_values.values[sample_index, :, predicted_class_index],
        features=X.iloc[sample_index],
        show=False,
        auto_size_plot=True
    )
    st.pyplot(plt.gcf())
    plt.clf()
    with st.expander("Plot Interpretation for example instance"):
        st.info("""
            This plot visualizes the **step-by-step journey** of the prediction for this abalone.
            
            **What to look for:**
            * **X-axis (Model output value)**: Shows the cumulative contribution to the prediction as each feature is added.
            * **Y-axis**: Lists the features in the order they were used in this decision path.
            * **Color bar (top)**: Shows low-to-high model output values (blue → red).
            * **Numbers in parentheses**: The SHAP value for that feature (how much it changed the prediction).
            
            **Example Summary for Abalone Instance = 0:**
            * "The plot for this abalone reveals its prediction journey. Starting from the **average (Base Value) of ~0.21**, the score sharply increased when the model considered its **Sex (M) and Shell Weight**. The final position of the line at the top is the model's exact score and prediction."
        """)


    # --- Section for Waterfall Plot ---
    st.divider()
    st.subheader("Waterfall Plot")
    st.write("Displays the positive and negative contributions of each feature for a single prediction.")
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[sample_index, :, predicted_class_index], show=False)
    st.pyplot(plt.gcf())
    plt.clf()
    with st.expander("Plot Interpretation for example instance"):
        st.info("""
            This plot shows the **building blocks** of the prediction for the selected abalone.
            
            **What to Look For:**
            * **E[f(X)] = 0.23**: The model’s baseline prediction (the average over all samples).
            * **f(x) = 0.63**: The final predicted output for this sample after adding each feature’s effect.
            * **Red bars**: Features that increased the prediction.
            * **Blue bars**: Features that decreased the prediction.
            * **Numbers on bars**: The SHAP value (change in prediction) caused by that feature.
            
            **Example Summary:**
            * "The most significant factor pushing this abalone's prediction higher was its **Shucked Weight**. On the other hand, its **Shell Weight** had the largest negative impact (in this case, almost no impact)."
        """)


    # --- Section for Force Plot ---
    st.divider()
    st.subheader("Force Plot")
    st.write("Illustrates the balance of features pushing the prediction higher (red) versus lower (blue).")
    shap.force_plot(
        base_value=shap_explainer.expected_value[predicted_class_index],
        shap_values=shap_values.values[sample_index, :, predicted_class_index],
        features=X.iloc[sample_index],
        matplotlib=True,
        show=False,
        figsize=(16, 4)
    )
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()
    with st.expander("Plot Interpretation for example instance"):
        st.info("""
            This plot illustrates the **tug-of-war** between features for this specific abalone.

            **What to Look For:**
            * **Base value (~0.55)**: The model’s average prediction over the training data.
            * **Final prediction (0.88)**: The final score, strongly above the baseline after feature contributions are applied.
            * **Red arrows**: Features pushing the prediction higher (toward 1).
            * **Blue arrows**: Features pushing the prediction lower (toward 0).
            * **Arrow length**: Represents the strength of the contribution.

            **Example Summary:**
            * "Features in red, like **Shucked Weight and Length**, are pushing the prediction higher, while features in blue, like **Height**, are pulling it lower. The final prediction is the result of which side wins."
        """)