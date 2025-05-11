from Implementations.som import SOM
from Implementations.Hopfield import HopfieldNetwork, binary_to_bipolar, bipolar_to_binary
from Implementations.mlp import MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HeartDiseaseProcessor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            self.data = pd.read_csv(
                url, header=None, names=column_names, na_values='?')
            self.data = self.data.fillna(self.data.median())
            self.data['target'] = self.data['target'].apply(
                lambda x: 1 if x > 0 else 0)
            self.X = self.data.drop('target', axis=1).values
            self.y = self.data['target'].values
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data for training"""
        if self.X is None or self.y is None:
            print("Data not loaded. Call load_data() first.")
            return False

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        # Scale the data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return True

    def train_mlp(self, hidden_sizes=[10, 5], learning_rate=0.01, epochs=1000):
        if self.X_train is None:
            print("Data not preprocessed. Call preprocess_data() first.")
            return None
        input_size = self.X_train.shape[1]
        output_size = 1
        mlp = MLP(input_size, hidden_sizes, output_size)
        losses = mlp.train(self.X_train, self.y_train.reshape(-1, 1), epochs, learning_rate)
        y_pred = mlp.predict(self.X_test)
        accuracy = np.mean(y_pred.flatten() == self.y_test)

        print(f"MLP Accuracy: {accuracy * 100:.2f}%")

        return mlp, losses

    def train_hopfield(self, max_patterns=10):
        """Train Hopfield network on heart disease data"""
        if self.X_train is None:
            print("Data not preprocessed. Call preprocess_data() first.")
            return None
        # We'll take a subset of the features and binarize them
        # Hopfield has limited capacity, so we'll reduce dimensionality
        X_subset = self.X_train[:, :5]

        # Binarize the data (threshold at median)
        X_binary = np.zeros_like(X_subset)
        for i in range(X_subset.shape[1]):
            median = np.median(X_subset[:, i])
            X_binary[:, i] = (X_subset[:, i] > median).astype(int)

        # Select only patterns from class 1 (has heart disease)
        class1_indices = np.where(self.y_train == 1)[0]
        patterns = X_binary[class1_indices]

        # Limit the number of patterns to store
        if len(patterns) > max_patterns:
            patterns = patterns[:max_patterns]

        # Convert to bipolar representation
        bipolar_patterns = binary_to_bipolar(patterns)
        hopfield = HopfieldNetwork(patterns.shape[1])
        hopfield.train(bipolar_patterns)

        print(f"Hopfield Network trained with {len(patterns)} patterns")

        return hopfield, patterns

    def train_som(self, map_size=(5, 5), learning_rate=0.5, sigma=None, epochs=100):
        if self.X_train is None:
            print("Data not preprocessed. Call preprocess_data() first.")
            return None

        # Initialize SOM
        input_dim = self.X_train.shape[1]
        som = SOM(input_dim=input_dim, map_size=map_size,learning_rate=learning_rate, sigma=sigma)
        errors = som.train(self.X_train, epochs=epochs)

        # Get the BMUs for training data
        bmus_train = som.predict(self.X_train)
        bmu_to_label = {}
        for i, bmu in enumerate(map(tuple, bmus_train)):
            if bmu not in bmu_to_label:
                bmu_to_label[bmu] = []
            bmu_to_label[bmu].append(self.y_train[i])

        # For each BMU, assign the majority class
        for bmu, labels in bmu_to_label.items():
            bmu_to_label[bmu] = np.bincount(labels).argmax()

        # Evaluate on test data
        bmus_test = som.predict(self.X_test)
        predictions = np.array([bmu_to_label.get(tuple(bmu), 0)
                               for bmu in bmus_test])
        accuracy = np.mean(predictions == self.y_test)

        print(f"SOM Accuracy: {accuracy * 100:.2f}%")

        return som, errors, bmu_to_label

    def visualize_data(self):
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].hist([
            self.data[self.data['target'] == 0]['age'],
            self.data[self.data['target'] == 1]['age']
        ], bins=10, label=['No Disease', 'Disease'])
        axes[0, 0].set_title('Age Distribution by Heart Disease')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()

        # Plot cholesterol by target
        axes[0, 1].scatter(
            self.data[self.data['target'] == 0]['age'],
            self.data[self.data['target'] == 0]['chol'],
            alpha=0.5, label='No Disease'
        )
        axes[0, 1].scatter(
            self.data[self.data['target'] == 1]['age'],
            self.data[self.data['target'] == 1]['chol'],
            alpha=0.5, label='Disease'
        )
        axes[0, 1].set_title('Age vs. Cholesterol')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Cholesterol')
        axes[0, 1].legend()
        target_counts = self.data['target'].value_counts()
        axes[1, 0].bar(['No Disease', 'Disease'], [
                       target_counts[0], target_counts[1]])
        axes[1, 0].set_title('Heart Disease Distribution')
        axes[1, 0].set_ylabel('Count')
        pd.crosstab(self.data['sex'], self.data['target']).plot(
            kind='bar', ax=axes[1, 1], color=['skyblue', 'salmon'])
        axes[1, 1].set_title('Heart Disease by Sex')
        axes[1, 1].set_xlabel('Sex (0=Female, 1=Male)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(['No Disease', 'Disease'])

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    processor = HeartDiseaseProcessor()
    processor.load_data()
    processor.preprocess_data()
    processor.visualize_data()
    plt.show()
    mlp, losses = processor.train_mlp(epochs=500)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('MLP Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()
    hopfield, patterns = processor.train_hopfield()
    som, errors, bmu_to_label = processor.train_som(epochs=50)
    som.plot_convergence()
    plt.show()
    fig = som.visualize_map(
        processor.X_train, processor.y_train, cmap='coolwarm')
    plt.title('SOM Map with Training Data')
    plt.show()
