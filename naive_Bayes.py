import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("nb.csv")

# Separate the features and the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.cond_prob = {}
    
    def fit(self, X_train, y_train):
        # Calculate the prior probability of each class
        self.prior["benign"] = np.sum(y_train == "B") / len(y_train)
        self.prior["malignant"] = np.sum(y_train == "M") / len(y_train)
        
        # Calculate the conditional probability of each feature given to each class
        for label in ["benign", "malignant"]:
            label_indices = np.where(y_train == label)[0]
            label_features = X_train.iloc[label_indices, :]
            self.cond_prob[label] = {}
            for feature in label_features.columns:
                unique_vals, counts = np.unique(label_features[feature], return_counts=True)
                prob_dict = dict(zip(unique_vals, counts/np.sum(counts)))
                self.cond_prob[label][feature] = prob_dict
    
    def predict(self, X_test):
        # Predict the class of a given instance using the Naive Bayes algorithm
        predictions = []
        for i in range(len(X_test)):
            x = X_test.iloc[i, :]
            benign_prob = self.prior["benign"]
            malignant_prob = self.prior["malignant"]
            for feature in X_test.columns:
                if x[feature] in self.cond_prob["benign"][feature]:
                    benign_prob *= self.cond_prob["benign"][feature][x[feature]]
                else:
                    benign_prob *= 0
                if x[feature] in self.cond_prob["malignant"][feature]:
                    malignant_prob *= self.cond_prob["malignant"][feature][x[feature]]
                else:
                    malignant_prob *= 0
            if benign_prob > malignant_prob:
                predictions.append("B")
            else:
                predictions.append("M")
        return predictions
    
    def score(self, X_test, y_test):
        # Calculate the accuracy of your Naive Bayes classifier on the testing set
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='M')
        recall = recall_score(y_test, y_pred, pos_label='M')
        f1 = f1_score(y_test, y_pred, pos_label='M')
        return accuracy, cm, precision, recall, f1

# Train the model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Test the model
accuracy, cm, precision, recall, f1 = nb.score(X_test, y_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
