import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
import os
import importlib


@st.cache_data
def load_data():
    """
    Loads the Dry Bean dataset from the UCI Machine Learning Repository.
    """
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/Dry_Bean_Dataset.xlsx"
    #ssl._create_default_https_context = ssl._create_unverified_context
    #df = pd.read_excel(url, engine='openpyxl')
    ##url = "https://archive.ics.uci.edu/static/public/602/data/DryBeanDataset/Dry_Bean_Dataset.xlsx"
    ##ssl._create_default_https_context = ssl._create_unverified_context
    ##df = pd.read_excel(url)

    # Set the path to the file you'd like to load
    file_path = "Dry_Bean_Dataset.xlsx"

    # Load the latest version
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

@st.cache_data
def preprocess_data(df):
    """
    Preprocesses the Dry Bean dataset by creating a hold-out test set,
    encoding the target variable, splitting the remaining data, and
    scaling the features.
    """
    # Separate the hold-out test set
    main_df, holdout_df = train_test_split(df, test_size=500, random_state=42, stratify=df['Class'])

    X_main = main_df.drop("Class", axis=1)
    y_main = main_df["Class"]
    X_holdout = holdout_df.drop("Class", axis=1)
    y_holdout = holdout_df["Class"]

    # Encode target variable on the entire main dataset before splitting
    le = LabelEncoder()
    y_main_encoded = le.fit_transform(y_main)
    class_names = le.classes_  # Store class names for later

    # Transform holdout y using the same encoder
    y_holdout_encoded = le.transform(y_holdout)

    # Split the main data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_main, y_main_encoded, test_size=0.3, random_state=42, stratify=y_main_encoded)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_holdout = scaler.transform(X_holdout)

    return X_train, X_val, y_train, y_val, X_holdout, y_holdout_encoded, class_names

@st.cache_resource
def load_models():
    """
    Dynamically loads all models from the 'models' directory.
    """
    models = {}
    model_files = [f for f in os.listdir('models') if f.endswith('.py') and not f.startswith('__')]

    for model_file in model_files:
        module_name = model_file[:-3]
        module = importlib.import_module(f'models.{module_name}')
        # Format the name for display, e.g., 'logistic_regression' -> 'Logistic Regression'
        model_name_display = module_name.replace('_', ' ').title()
        models[model_name_display] = module.get_model()

    return models

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a single classification model.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Confusion Matrix": cm,
    }

def evaluate_holdout_set(trained_models, X_holdout, y_holdout):
    """
    Evaluates the trained models on the hold-out test set.
    """
    holdout_results = {}

    for model_name, model in trained_models.items():
        # Make predictions
        y_pred = model.predict(X_holdout)
        y_pred_proba = model.predict_proba(X_holdout)

        # Calculate metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        auc = roc_auc_score(y_holdout, y_pred_proba, multi_class='ovr')
        precision = precision_score(y_holdout, y_pred, average='weighted')
        recall = recall_score(y_holdout, y_pred, average='weighted')
        f1 = f1_score(y_holdout, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_holdout, y_pred)

        holdout_results[model_name] = {
            "Holdout Accuracy": accuracy,
            "Holdout AUC": auc,
            "Holdout Precision": precision,
            "Holdout Recall": recall,
            "Holdout F1 Score": f1,
            "Holdout MCC": mcc,
        }

    return holdout_results

# Main app
st.title("Machine Learning Classification Dashboard")

# --- Data Loading Section ---
st.sidebar.title("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

df = None
dataset_name = ""
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        dataset_name = uploaded_file.name
        st.sidebar.success("Custom dataset loaded.")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        st.stop()
else:
    # Use the default dataset if no file is uploaded
    df = load_data()
    dataset_name = "Dry Bean Dataset"

st.sidebar.markdown(f"**Current Dataset:** {dataset_name}")

# --- The rest of the app assumes 'df' is loaded ---
# Load and preprocess data
X_train, X_val, y_train, y_val, X_holdout, y_holdout, class_names = preprocess_data(df)

# Load models dynamically
available_models = load_models()

# Sidebar for view and model selection
st.sidebar.title("View and Model Selection")
view = st.sidebar.radio("Choose a view", ["Single Model View", "Model Comparison View"])

model_names = sorted(list(available_models.keys()))
selected_model_names = []

if view == "Single Model View":
    # Model selection for single view
    model_name = st.sidebar.selectbox("Choose a model", model_names)
    if model_name:
        selected_model_names = [model_name]
else: # Model Comparison View
    # Model selection for comparison view
    selected_model_names = st.sidebar.multiselect(
        "Choose models to compare",
        model_names,
        default=model_names
    )

# Main panel view
if not selected_model_names:
    st.warning("Please select at least one model from the sidebar to continue.")
    st.stop()

# --- Model Training and Evaluation ---
st.header("Model Training Progress")
progress_bar = st.progress(0)
status_text = st.empty()

results = {}
trained_models = {}
total_models = len(selected_model_names)

for i, model_name in enumerate(selected_model_names):
    status_text.text(f"Training {model_name}... ({i+1}/{total_models})")
    model = available_models[model_name]

    # Train and evaluate a single model
    results[model_name] = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
    trained_models[model_name] = model # Store the trained model instance

    progress_bar.progress((i + 1) / total_models)

status_text.text(f"All {total_models} models trained successfully!")


# Evaluate models on the hold-out test set
holdout_results = evaluate_holdout_set(trained_models, X_holdout, y_holdout)


if view == "Single Model View":
    st.header(f"Single Model View: {selected_model_names[0]}")
    model_name = selected_model_names[0] # There will only be one
    metrics = results[model_name]
    holdout_metrics = holdout_results[model_name]

    # Use columns for a side-by-side layout for metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Validation Set Metrics")
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("AUC", f"{metrics['AUC']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        st.metric("MCC", f"{metrics['MCC']:.4f}")

    with col2:
        st.subheader("Hold-out Set Metrics")
        st.metric("Accuracy", f"{holdout_metrics['Holdout Accuracy']:.4f}")
        st.metric("AUC", f"{holdout_metrics['Holdout AUC']:.4f}")
        st.metric("Precision", f"{holdout_metrics['Holdout Precision']:.4f}")
        st.metric("Recall", f"{holdout_metrics['Holdout Recall']:.4f}")
        st.metric("F1 Score", f"{holdout_metrics['Holdout F1 Score']:.4f}")
        st.metric("MCC", f"{holdout_metrics['Holdout MCC']:.4f}")

    # Display confusion matrix below the metrics, centered
    st.subheader("Confusion Matrix (Validation Set)")
    cm = metrics["Confusion Matrix"]
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=class_names,
                    y=class_names
                   )
    fig.update_layout(title_text=f'Confusion Matrix for {model_name}', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)


else: # Model Comparison View
    st.header("Model Comparison View")

    # Combine validation and holdout results for comparison
    summary_df = pd.DataFrame(results).T.drop(columns="Confusion Matrix")
    holdout_summary_df = pd.DataFrame(holdout_results).T

    comparison_df = summary_df.join(holdout_summary_df)
    comparison_df.index.name = "Model"

    # Use columns for a side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        # Display summary table
        st.subheader("Model Performance")
        st.dataframe(comparison_df)

    with col2:
        # Interactive bar chart
        st.subheader("Compare Models by Metric")
        # Let user select from combined metrics
        metric_to_compare = st.selectbox("Select a metric", comparison_df.columns)

        fig = px.bar(comparison_df, x=comparison_df.index, y=metric_to_compare,
                     title=f"Model Comparison for {metric_to_compare}",
                     labels={'x': 'Model', 'y': metric_to_compare},
                     text_auto=True)
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)