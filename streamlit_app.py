import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


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
    Preprocesses the Dry Bean dataset by encoding the target variable,
    splitting the data, and scaling the features.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

@st.cache_data
def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "Confusion Matrix": cm,
        }
    
    return results

# Main app
st.title("Machine Learning Classification Dashboard")

# Load and preprocess data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train and evaluate models
results = train_and_evaluate(X_train, y_train, X_test, y_test)

# Sidebar for navigation
st.sidebar.title("Navigation")
view = st.sidebar.radio("Choose a view", ["Single Model View", "Model Comparison View"])

if view == "Single Model View":
    st.header("Single Model View")
    
    # Model selection
    model_name = st.sidebar.selectbox("Choose a model", list(results.keys()))
    
    # Display metrics
    st.subheader(f"Metrics for {model_name}")
    metrics = results[model_name]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    cm = metrics["Confusion Matrix"]
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=df['Class'].unique(),
                    y=df['Class'].unique()
                   )
    fig.update_layout(title_text=f'Confusion Matrix for {model_name}', title_x=0.5)
    st.plotly_chart(fig)


else: # Model Comparison View
    st.header("Model Comparison View")
    
    # Display summary table
    st.subheader("Model Performance Summary")
    summary_df = pd.DataFrame(results).T.drop(columns="Confusion Matrix")
    summary_df.index.name = "Model"
    st.dataframe(summary_df)
    
    # Interactive bar chart
    st.subheader("Compare Models by Metric")
    metric_to_compare = st.selectbox("Select a metric", summary_df.columns)
    
    fig = px.bar(summary_df, x=summary_df.index, y=metric_to_compare,
                 title=f"Model Comparison for {metric_to_compare}",
                 labels={'x': 'Model', 'y': metric_to_compare},
                 text_auto=True)
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig)