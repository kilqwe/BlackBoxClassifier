import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
@st.cache_data
def load_and_prepare_data():
    """Loads and prepares the abalone data for a classification task."""
    print("\n--- Loading and Preparing Data ---")
    data = pd.read_csv("abalone.data",
                       names=["sex", "length", "diameter", "height", "whole_weight",
                              "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    
    X = data[["sex", "length", "height", "shucked_weight", "viscera_weight", "shell_weight"]].copy()

    X["sex.M"] = [1 if s == "M" else 0 for s in X["sex"]]
    X["sex.F"] = [1 if s == "F" else 0 for s in X["sex"]]
    X["sex.I"] = [1 if s == "I" else 0 for s in X["sex"]]
    X = X.drop("sex", axis=1)

    def ring_to_class(r):
        if r < 8: return "young"
        elif r <= 11: return "adult"
        else: return "old"

    y_class = data["rings"].apply(ring_to_class)
    
    print(f"Data loaded. Features shape: {X.shape}")
    print("Target class distribution:")
    print(y_class.value_counts())
    
    return X, y_class, data
@st.cache_resource
def train_model(X, y_class):
    """Trains a RandomForestClassifier model."""
    print("\n--- Training RandomForestClassifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_class)
    print(f"Model trained successfully. Model classes: {model.classes_}")
    return model