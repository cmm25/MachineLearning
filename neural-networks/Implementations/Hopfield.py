import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        np.fill_diagonal(self.weights, 0)

    def train(self, patterns):
        """
        Train the Hopfield network using Hebbian learning.

        Parameters:
        patterns -- array of shape (num_patterns, num_neurons) where each row is a pattern  Values should be 1 or -1 (not 0 and 1)
        """
        num_patterns = patterns.shape[0]
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=20, threshold=0, return_iterations=False):
        """
        Recall a pattern from the network.

        Parameters:
        pattern -- initial pattern to start reconstruction
        max_iterations -- maximum number of iterations for convergence
        threshold -- activation threshold (typically 0)
        return_iterations -- if True, also return the number of iterations needed for convergence

        Returns:
        reconstructed pattern, or tuple (reconstructed pattern, iterations) if return_iterations=True
        """
        current_pattern = pattern.copy()
        iterations = 0

        for iterations in range(max_iterations):
            prev_pattern = current_pattern.copy()
            for i in range(self.num_neurons):
                activation = np.dot(self.weights[i], current_pattern)
                # Apply threshold function
                current_pattern[i] = 1 if activation > threshold else -1

            # Check for convergence
            if np.array_equal(prev_pattern, current_pattern):
                break

        # Return the pattern, and optionally the number of iterations
        if return_iterations:
            return current_pattern, iterations + 1
        else:
            return current_pattern

    def energy(self, pattern):
        """
        Calculate energy of the network for a given pattern.
        Lower energy indicates pattern is closer to a stored memory.

        Parameters:
        pattern -- pattern to calculate energy for

        Returns:
        energy value
        """
        return -0.5 * np.dot(np.dot(pattern, self.weights), pattern)
def binary_to_bipolar(binary_patterns):
    """Convert binary (0,1) patterns to bipolar (-1,1) patterns"""
    return 2 * binary_patterns - 1


def bipolar_to_binary(bipolar_patterns):
    """Convert bipolar (-1,1) patterns to binary (0,1) patterns"""
    return (bipolar_patterns + 1) // 2


def flat_to_grid(flat_array):
    """Convert flat array to 2x2 grid"""
    return flat_array.reshape(2, 2)


def visualize_pattern(pattern, title=None, color=None):
    """Visualize a pattern as a 2x2 grid"""
    plt.figure(figsize=(4, 4))
    # Convert to binary (0,1) for visualization
    binary_pattern = bipolar_to_binary(pattern) if -1 in pattern else pattern
    plt.imshow(flat_to_grid(binary_pattern), cmap="gray", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])

    if title:
        plt.title(title, color=color if color else 'black')
    plt.show()


def visualize_patterns(patterns, titles=None):
    """Visualize multiple patterns in a grid"""
    n = len(patterns)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, pattern in enumerate(patterns):
        if i < len(axes):
            binary_pattern = bipolar_to_binary(
                pattern) if -1 in pattern else pattern
            axes[i].imshow(flat_to_grid(binary_pattern),cmap="gray", vmin=0, vmax=1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

            if titles and i < len(titles):
                axes[i].set_title(titles[i])
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def test_hopfield_network():
    binary_patterns = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]

    ])
    patterns = binary_to_bipolar(binary_patterns)
    hopfield = HopfieldNetwork(4)
    hopfield.train(patterns)

    # Visualize the stored patterns
    visualize_patterns(binary_patterns, [ 'Top', 'Bottom', 'Diagonal', 'Opposite Diagonal', "Top Left", "Top Right", "Bottom Left", "Bottom Right"])

    # Test recall with noisy patterns
    print("Testing recall with noisy patterns:")
    noisy_pattern1 = patterns[0].copy()
    noisy_pattern1[np.random.choice(4, 1)] *= -1  # Flip a random bit

    noisy_pattern2 = patterns[1].copy()
    noisy_pattern2[np.random.choice(4, 1)] *= -1  # Flip a random bit

    print("Noisy patterns:")
    visualize_patterns([bipolar_to_binary(noisy_pattern1), bipolar_to_binary(
        noisy_pattern2)], ['Noisy Top', 'Noisy Bottom'])

    # Recall
    recalled_pattern1 = hopfield.recall(noisy_pattern1)
    recalled_pattern2 = hopfield.recall(noisy_pattern2)

    # Visualize the recalled patterns
    print("Recalled patterns:")
    visualize_patterns([bipolar_to_binary(recalled_pattern1), bipolar_to_binary(
        recalled_pattern2)], ['Recalled Top', 'Recalled Bottom'])

    # Interactive testing
    print("\nTesting the Hopfield network:")
    print("Enter 4 binary values (e.g. 1 0 0 1) or 'q' to quit:")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break

        try:
            test_pattern = np.array(list(map(int, user_input.split())))
            if len(test_pattern) != 4 or not all(x in [0, 1] for x in test_pattern):
                print("Please enter exactly 4 binary values (0s and 1s)")
                continue
            bipolar_pattern = binary_to_bipolar(test_pattern)
            visualize_pattern(test_pattern, "Input Pattern")
            print("Energy values:")
            for i, pattern in enumerate(patterns):
                energy = hopfield.energy(pattern)
                print(f"Pattern {i+1}: {energy:.4f}")
            input_energy = hopfield.energy(bipolar_pattern)
            print(f"Input pattern energy: {input_energy:.4f}")

            recalled = hopfield.recall(bipolar_pattern)
            recalled_binary = bipolar_to_binary(recalled)
            matches = False
            match_idx = -1
            for i, pattern in enumerate(patterns):
                if np.array_equal(recalled, pattern):
                    matches = True
                    match_idx = i
                    break
            final_energy = hopfield.energy(recalled)
            print(f"Final energy: {final_energy:.4f}")
            title = f"Recalled: Matches Pattern {match_idx+1}" if matches else "Recalled (No Match)"
            color = 'green' if matches else 'red'
            visualize_pattern(recalled_binary, title, color)

        except ValueError:
            print("Invalid input. Please enter 4 binary values separated by spaces.")


if __name__ == "__main__":
    test_hopfield_network()
