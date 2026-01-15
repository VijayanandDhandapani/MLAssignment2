from sklearn.neighbors import KNeighborsClassifier

def get_model():
    """
    Returns an instance of the K-Nearest Neighbors Classifier model.
    """
    return KNeighborsClassifier(n_jobs=-1)
