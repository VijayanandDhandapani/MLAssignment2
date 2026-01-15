from sklearn.tree import DecisionTreeClassifier

def get_model():
    """
    Returns an instance of the Decision Tree Classifier model.
    """
    return DecisionTreeClassifier(random_state=42)
