import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time


class SOM:
    def __init__(self, input_dim, map_size=(10, 10), learning_rate=0.1, sigma=None, random_seed=None):
        """
        Initialize a Self-Organizing Map

        Parameters:
        input_dim -- dimension of the input data
        map_size -- tuple (width, height) for the SOM grid
        learning_rate -- initial learning rate
        sigma -- initial neighborhood radius (if None, it will be set to max(map_size)/2)
        random_seed -- random seed for reproducibility
        """
        self.input_dim = input_dim
        self.map_size = map_size
        self.initial_learning_rate = learning_rate

        if sigma is None:
            self.initial_sigma = max(map_size) / 2
        else:
            self.initial_sigma = sigma

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize weights randomly between 0 and 1
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

        # For tracking convergence
        self.quantization_errors = []
        self.topographic_errors = []

    def _decay_function(self, initial_value, iteration, max_iterations, decay_type='exponential'):
        """
        Calculate the decayed value (for learning rate and sigma)

        Parameters:
        initial_value -- initial value to decay
        iteration -- current iteration
        max_iterations -- total number of iterations
        decay_type -- type of decay ('exponential' or 'linear')

        Returns:
        decayed value
        """
        if decay_type == 'exponential':
            return initial_value * np.exp(-iteration / max_iterations)
        else:  # linear
            return initial_value * (1 - iteration / max_iterations)

    def _calculate_influence(self, distance, sigma):
        """
        Calculate the neighborhood influence based on distance

        Parameters:
        distance -- distance from BMU
        sigma -- current neighborhood radius

        Returns:
        influence factor
        """
        return np.exp(-(distance**2) / (2 * sigma**2))

    def _find_bmu(self, x):
        # Calculate Euclidean distance between input and all neurons
        distances = np.sum((self.weights - x)**2, axis=2)
        # Find the index of the minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def train(self, data, epochs, verbose=True):
        max_iterations = epochs * len(data)
        iteration = 0

        # Normalize data if not already normalized
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        normalized_data = (data - data_min) / data_range

        for epoch in range(epochs):
            epoch_start_time = time.time()
            quantization_error = 0

            # Shuffle the data for each epoch
            indices = np.random.permutation(len(normalized_data))

            for idx in indices:
                x = normalized_data[idx]

                # Find the Best Matching Unit (BMU)
                bmu = self._find_bmu(x)
                learning_rate = self._decay_function(
                    self.initial_learning_rate, iteration, max_iterations)
                sigma = self._decay_function(
                    self.initial_sigma, iteration, max_iterations)

                # Update weights for all neurons based on distance from BMU
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        # Calculate Manhattan distance to BMU
                        distance = np.abs(i - bmu[0]) + np.abs(j - bmu[1])

                        # Calculate influence based on distance
                        influence = self._calculate_influence(distance, sigma)

                        # Update weights
                        self.weights[i, j] += learning_rate * \
                            influence * (x - self.weights[i, j])

                # Calculate quantization error (distance between input and BMU)
                quantization_error += np.sqrt(
                    np.sum((x - self.weights[bmu[0], bmu[1]])**2))
                iteration += 1

            # Calculate average quantization error for this epoch
            avg_quantization_error = quantization_error / len(data)
            self.quantization_errors.append(avg_quantization_error)

            # Calculate topographic error for this epoch
            topographic_error = self._calculate_topographic_error(
                normalized_data)
            self.topographic_errors.append(topographic_error)

            if verbose and epoch % max(1, epochs // 10) == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch}/{epochs} - QE: {avg_quantization_error:.6f} - "
                        f"TE: {topographic_error:.6f} - Time: {epoch_time:.2f}s")

        return self.quantization_errors

    def _calculate_topographic_error(self, data):
        """
        Calculate topographic error (fraction of data for which first and second BMUs are not adjacent)
        """
        errors = 0
        for x in data:
            bmu1 = self._find_bmu(x)
            temp_weights = self.weights.copy()
            temp_weights[bmu1[0], bmu1[1]] = np.inf

            # Find the second BMU
            distances = np.sum((temp_weights - x)**2, axis=2)
            bmu2 = np.unravel_index(np.argmin(distances), distances.shape)

            # Check if BMUs are adjacent (Manhattan distance = 1)
            manhattan_dist = np.abs(
                bmu1[0] - bmu2[0]) + np.abs(bmu1[1] - bmu2[1])
            if manhattan_dist > 1:
                errors += 1

        return errors / len(data)

    def predict(self, data):
        """
        Find the BMU for each input data point

        Parameters:
        data -- input data of shape (n_samples, input_dim)

        Returns:
        array of BMU coordinates for each input
        """
        # Normalize data using the same normalization as in training
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        normalized_data = (data - data_min) / data_range

        bmus = np.zeros((len(normalized_data), 2), dtype=int)
        for i, x in enumerate(normalized_data):
            bmus[i] = self._find_bmu(x)
        return bmus

    def get_weights(self):
        """Return the weights of the SOM"""
        return self.weights

    def get_convergence_data(self):
        """Return the convergence data (quantization and topographic errors)"""
        return {
            'quantization_errors': self.quantization_errors,
            'topographic_errors': self.topographic_errors
        }

    def plot_convergence(self, figsize=(12, 5)):
        """Plot the convergence of the SOM training"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot quantization error
        ax1.plot(self.quantization_errors)
        ax1.set_title('Quantization Error')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Quantization Error')
        ax1.grid(True)

        # Plot topographic error
        ax2.plot(self.topographic_errors)
        ax2.set_title('Topographic Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Topographic Error')
        ax2.grid(True)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_convergence_from_data(convergence_data, figsize=(12, 5)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if 'quantization_errors' in convergence_data and len(convergence_data['quantization_errors']) > 0:
            ax1.plot(convergence_data['quantization_errors'])
            ax1.set_title('Quantization Error')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Average Quantization Error')
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, 'No quantization error data available',ha='center', va='center', transform=ax1.transAxes)

        if 'topographic_errors' in convergence_data and len(convergence_data['topographic_errors']) > 0:
            ax2.plot(convergence_data['topographic_errors'])
            ax2.set_title('Topographic Error')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Topographic Error')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No topographic error data available',ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        return fig

    def visualize_map(self, data, labels=None, figsize=(10, 8), cmap=None, title=None):
        """
        Visualize the SOM map with data points colored by their labels
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create a grid showing neuron positions
        xx, yy = np.meshgrid(
            np.arange(self.map_size[1]), np.arange(self.map_size[0]))
        ax.scatter(xx.flatten(), yy.flatten(), marker='s',
                   s=100, c='lightgray', alpha=0.3)

        if labels is not None:
            bmu_counts = {}
            for i, sample in enumerate(data):
                bmu = tuple(self.predict(sample.reshape(1, -1))[0])
                if bmu not in bmu_counts:
                    bmu_counts[bmu] = {}
                label = labels[i]
                if label not in bmu_counts[bmu]:
                    bmu_counts[bmu][label] = 0
                bmu_counts[bmu][label] += 1

            # Assign color to each cell based on majority class
            cell_colors = np.zeros(self.map_size)
            for (y, x), counts in bmu_counts.items():
                if len(counts) > 0:
                    majority_class = max(counts.items(), key=lambda x: x[1])[0]
                    cell_colors[y, x] = majority_class

            # Plot heatmap of class distribution
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                # If the task is binary pattern recognition, use black/white
                if cmap is None or (hasattr(cmap, 'colors') and cmap.colors == ['black', 'white']):
                    cmap = ListedColormap(['black', 'white'])
                    im = ax.imshow(cell_colors, cmap=cmap,interpolation='nearest', vmin=0, vmax=1)
                    cbar = plt.colorbar(
                        im, ticks=[0, 1], orientation='vertical')
                    cbar.ax.set_yticklabels(
                        ['<2 white cells', '≥2 white cells'])
                else:
                    # For heart disease, use blue/red
                    cmap = ListedColormap(['blue', 'red'])
                    im = ax.imshow(cell_colors, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
                    cbar = plt.colorbar(
                        im, ticks=[0, 1], orientation='vertical')
                    cbar.ax.set_yticklabels(['Healthy', 'Heart Disease'])
            else:
                # Use default colormap for multi-class
                im = ax.imshow(cell_colors, cmap=cmap or 'viridis', interpolation='nearest')
                cbar = plt.colorbar(im, orientation='vertical')

        ax.set_title(title or "Self-Organizing Map")
        ax.set_xlabel("SOM X coordinate")
        ax.set_ylabel("SOM Y coordinate")

        return fig


def test_som_binary():
    binary_patterns = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
    ])

    # Create labels based on the number of white pixels
    labels = np.array([1 if np.sum(x) >= 2 else 0 for x in binary_patterns])

    som = SOM(input_dim=4, map_size=(4, 4),learning_rate=0.5, sigma=1.0, random_seed=42)
    som.train(binary_patterns, epochs=100, verbose=True)
    som.visualize_map(binary_patterns, labels, cmap='coolwarm')
    plt.show()
    som.plot_convergence()
    plt.show()

    # Test prediction
    test_pattern = np.array([1, 0, 1, 0])
    bmu = som.predict(test_pattern.reshape(1, -1))[0]
    print(f"Test pattern: {test_pattern}")
    print(f"BMU coordinates: {bmu}")

    # Find which patterns are mapped to the same BMU
    bmus = som.predict(binary_patterns)
    similar_patterns = []
    for i, pattern_bmu in enumerate(bmus):
        if np.array_equal(pattern_bmu, bmu):
            similar_patterns.append((i, binary_patterns[i]))

    print("Patterns mapped to the same BMU:")
    for idx, pattern in similar_patterns:
        print(f"Pattern {idx}: {pattern} (Label: {labels[idx]})")


if __name__ == "__main__":
    test_som_binary()
