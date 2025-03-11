import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class EEGClassifier:
    """Performs EEG classification using Max-CCA or K-Nearest Neighbors (KNN) methods."""

    def __init__(self, n_neighbors=5):
        """
        Initialize the EEG classifier.

        Parameters:
        - n_neighbors (int): Number of neighbors used in KNN classification (default: 5).
        """
        self.n_neighbors = n_neighbors

    def classify_max_cca(self, features_by_subject, labels_by_subject, stim_frequencies, num_channels=1, num_harmonics=1):
        """
        Classify EEG trials by selecting the stimulus frequency with the highest CCA correlation.

        Parameters:
        - features_by_subject (dict): Dictionary of subject-wise features arrays (num_trials, num_freqs*num_channels*num_harmonics).
        - labels_by_subject (dict): Dictionary of subject-wise labels (integers representing stimulus frequencies).
        - stim_frequencies (list): List of stimulus frequencies.
        - num_channels (int): Number of EEG channels used.
        - num_harmonics (int): Number of harmonics per frequency considered.

        Returns:
        - accuracy (float): Classification accuracy across all subjects.
        - conf_matrix (np.ndarray): Confusion matrix summarizing classification results.
        """
        subjects = list(features_by_subject.keys())
        all_preds, all_labels = [], []

        num_freqs = len(stim_frequencies)

        for subject in subjects:
            features = features_by_subject[subject]
            labels = labels_by_subject[subject]

            if labels is None or len(labels) == 0 or features.shape[0] == 0:
                print(f"Skipping subject {subject}: no valid data or labels.")
                continue

            print(f"Classifying subject {subject} with {features.shape[0]} trials.")

            preds = []
            for trial_idx in range(features.shape[0]):
                trial_features = features[trial_idx, :].reshape(num_channels, num_freqs, num_harmonics)

                # Compute maximum correlation across channels and harmonics for each frequency
                freq_corrs = trial_features.max(axis=(0, 2))

                # Predict the frequency with the highest correlation
                pred_label = np.argmax(freq_corrs) + 1  # Frequency labels start from 1
                preds.append(pred_label)

            all_preds.extend(preds)
            all_labels.extend(labels.astype(int))

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(1, num_freqs + 1)))

        print("\nConfusion Matrix (Max-CCA):")
        print(conf_matrix)
        print(f"Overall Max-CCA Classification Accuracy: {accuracy:.4f}")

        return accuracy, conf_matrix

    def classify_knn(self, features_by_subject, labels_by_subject):
        """
        Perform leave-subject-out cross-validation using K-Nearest Neighbors (KNN).

        Parameters:
        - features_by_subject (dict): Dictionary containing subject-wise feature arrays.
        - labels_by_subject (dict): Dictionary containing subject-wise labels.

        Returns:
        - avg_accuracy (float): Average classification accuracy across subjects.
        - overall_conf_matrix (np.ndarray): Aggregated confusion matrix across subjects.
        """
        subjects = list(features_by_subject.keys())
        overall_conf_matrix = None
        accuracies = []

        for test_subject in subjects:
            print(f"Testing on subject {test_subject}, training on remaining subjects.")

            # Construct training data by excluding the current test subject
            X_train = np.vstack([features_by_subject[subj] for subj in subjects if subj != test_subject])
            y_train = np.concatenate([labels_by_subject[subj] for subj in subjects if subj != test_subject])

            # Test data from the excluded subject
            X_test = features_by_subject[test_subject]
            y_test = labels_by_subject[test_subject]

            # Train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(X_train, y_train)

            # Predict and evaluate the model on test subject
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"Subject {test_subject} Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix for Subject {test_subject}:\n{conf_matrix}")

            accuracies.append(accuracy)

            # Aggregate confusion matrices
            if overall_conf_matrix is None:
                overall_conf_matrix = conf_matrix
            else:
                overall_conf_matrix += conf_matrix

        avg_accuracy = np.mean(accuracies)

        print("\nOverall Aggregated Confusion Matrix (KNN):")
        print(overall_conf_matrix)
        print(f"Average Leave-Subject-Out KNN Accuracy: {avg_accuracy:.4f}")

        return avg_accuracy, overall_conf_matrix
