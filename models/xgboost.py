from xgboost import XGBClassifier

def get_model():
    """
    Returns an instance of the XGBoost Classifier model.
    """
    return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
