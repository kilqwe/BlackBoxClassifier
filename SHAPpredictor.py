import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import functions from the model.py file
from model import load_and_prepare_data, train_model

def create_shap_explainer_and_values(
    model: RandomForestClassifier, 
    data: pd.DataFrame
) -> tuple[shap.TreeExplainer, shap.Explanation]:
    """
    Creates a SHAP TreeExplainer and calculates SHAP values for the given data.

    Args:
        model: A trained tree-based model (e.g., RandomForestClassifier).
        data: The input feature data as a pandas DataFrame.

    Returns:
        A tuple containing:
        - The SHAP explainer object.
        - The calculated SHAP values object.
    """
    print("\n--- Creating SHAP Explainer and calculating values ---")
    # SHAP's TreeExplainer is optimized for tree-based models like Random Forest
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for all instances in the provided data
    shap_values = explainer(data)
    
    print("SHAP values calculated successfully.")
    return explainer, shap_values

if __name__ == "__main__":
    # 1. Load data and train the model using functions from model.py
    X, y_class, _ = load_and_prepare_data()
    model = train_model(X, y_class)

    # 2. Create the SHAP explainer and calculate values for the entire dataset
    explainer, shap_values = create_shap_explainer_and_values(model, X)

    # 3. Print a confirmation and some details to show it worked
    print("\n--- SHAP Refactoring Complete ---")
    print(f"SHAP Explainer Type: {type(explainer)}")
    
    # The shap_values object contains the values, base values, and original data
    print(f"Shape of SHAP values array: {shap_values.values.shape}")
    print(f"Shape of base values array: {shap_values.base_values.shape}")
    print(f"An example base value (average prediction): {shap_values.base_values[0]}")