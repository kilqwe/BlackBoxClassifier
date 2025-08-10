import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

# Import functions from the model.py file
from model import load_and_prepare_data, train_model

def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: list,
    class_names: list,
    mode: str = 'classification'
) -> LimeTabularExplainer:
    """
    Initializes and returns a LimeTabularExplainer.
    
    Args:
        training_data: The numpy array of the data used to train the model.
        feature_names: A list of feature names.
        class_names: A list of the target class names.
        mode: The explanation mode ('classification' or 'regression').
        
    Returns:
        An configured instance of LimeTabularExplainer.
    """
    print("\n--- Creating LIME Explainer ---")
    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode
    )
    print("Explainer created successfully.")
    return explainer

def explain_instance(
    explainer: LimeTabularExplainer,
    model: RandomForestClassifier,
    data_row: np.ndarray
) -> list:
    """
    Explains a single prediction using the LIME explainer.
    
    Args:
        explainer: The configured LIME explainer instance.
        model: The trained classifier model.
        data_row: The specific data instance (as a numpy array) to explain.
        
    Returns:
        A list of tuples representing the feature contributions.
    """
    print("\n--- Explaining a Single Instance ---")
    explanation = explainer.explain_instance(
        data_row=data_row,
        predict_fn=model.predict_proba
    )
    return explanation.as_list()

if __name__ == "__main__":
    # 1. Load data and train the model using functions from model.py
    X, y_class, _ = load_and_prepare_data()
    model = train_model(X, y_class)

    # 2. Create the LIME explainer
    explainer = create_lime_explainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=model.classes_
    )

    # 3. Select a sample and get its explanation
    sample_index = 25 # You can change this index to explain a different sample
    sample_to_explain = X.iloc[sample_index]
    
    explanation_list = explain_instance(
        explainer=explainer,
        model=model,
        data_row=sample_to_explain.values
    )
    
    # 4. Print the results
    predicted_class = model.predict(sample_to_explain.values.reshape(1, -1))[0]
    print(f"\n--- LIME Explanation for Sample {sample_index} ---")
    print(f"Model Prediction: '{predicted_class}'\n")
    print("Feature Contributions:")
    for feature, weight in explanation_list:
        print(f"- {feature:<25} | Weight: {weight:.4f}")