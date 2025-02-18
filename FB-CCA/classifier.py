import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

class EEGClassifier:
    """Handles classification using KNN with leave-subject-out paradigm."""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def classify(self, X, y):
        """Perform leave-subject-out classification."""
        loo = LeaveOneOut()
        y_pred = cross_val_predict(self.classifier, X, y, cv=loo)
        accuracy = accuracy_score(y, y_pred)
        return accuracy
