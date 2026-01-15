from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Returns an instance of the Random Forest Classifier model.
    """
    return RandomForestClassifier(random_state=42, n_jobs=-1)
