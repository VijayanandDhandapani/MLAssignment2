from sklearn.linear_model import LogisticRegression

def get_model():
    """
    Returns an instance of the Logistic Regression model.
    """
    return LogisticRegression(random_state=42, n_jobs=-1)
