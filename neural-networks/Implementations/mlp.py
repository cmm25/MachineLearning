import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        np.random.seed(None)
        
        # Weights and biases for input to first hidden layer
        self.weights = [np.random.randn(input_size, hidden_sizes[0]) * 0.1]
        self.biases = [np.zeros((1, hidden_sizes[0]))]
        
        # Weights and biases for hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
        
        # Weights and biases for last hidden layer to output
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.layer_inputs = []
        self.activations = [X]
        
        for i in range(len(self.weights)):
            self.layer_inputs.append(np.dot(self.activations[-1], self.weights[i]) + self.biases[i])
            self.activations.append(self.sigmoid(self.layer_inputs[-1]))
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        output_error = y - self.activations[-1]
        output_delta = output_error * self.sigmoid_derivative(self.layer_inputs[-1])
        deltas = [output_delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(self.layer_inputs[i-1])
            deltas.insert(0, delta)
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate, batch_size=None):
        losses = []
        
        if batch_size is None:
            batch_size = len(X)
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            output = self.forward(X)
            loss = np.mean(np.square(y - output))
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)

input_size = 4
hidden_sizes = [10, 4]  
output_size = 1
mlp = MLP(input_size, hidden_sizes, output_size)

X = np.array([
    [0, 0, 0, 0], 
    [0, 0, 0, 1], 
    [0, 0, 1, 0],  
    [0, 0, 1, 1],  
    [0, 1, 0, 0],  
    [0, 1, 0, 1],  
    [0, 1, 1, 0],  
    [0, 1, 1, 1],  
    [1, 0, 0, 0],  
    [1, 0, 0, 1],  
    [1, 0, 1, 0],  
    [1, 0, 1, 1],  
    [1, 1, 0, 0],  
    [1, 1, 0, 1],  
    [1, 1, 1, 0],  
    [1, 1, 1, 1]   
])

y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)

epochs = 1500
learning_rate = 0.1
losses = mlp.train(X, y, epochs, learning_rate)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
fig = plt.gcf()
plt.close(fig)

def count_white_pixels(grid):
    return np.sum(grid)
def flat_to_grid(flat_array):
    return flat_array.reshape(2, 2)
def visualize_all_grids():
    predictions = mlp.predict(X)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        grid = flat_to_grid(X[i])
        pred = predictions[i][0]
        actual = y[i][0]
        white_count = count_white_pixels(grid)
        if white_count <= 1:
            expected = "Black"
        else:
            expected = "White"
        
        predicted = "White" if pred == 1 else "Black"
        
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        color = 'green' if pred == actual else 'red'
        ax.set_title(f"{white_count} white: {predicted}", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
def test_and_visualize(test_flat):
    test_flat = test_flat.reshape(1, -1)
    output = mlp.forward(test_flat)
    pred = (output >= 0.5).astype(int)[0][0]
    white_count = np.sum(test_flat)
    
    predicted = "White" if pred == 1 else "Black"
    
    if white_count <= 1:
        expected = "Black"
    else:
        expected = "White"
    
    print(f"White pixels: {white_count}")
    print(f"Network output: {output[0][0]:.4f}")
    print(f"Classification: {predicted}")
    print(f"Expected: {expected}")
    
    plt.figure(figsize=(4, 4))
    plt.imshow(flat_to_grid(test_flat[0]), cmap="gray", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    color = 'green' if predicted == expected else 'red'
    plt.title(f"{white_count} white pixels: {predicted}", color=color)
    plt.show()

def verify_rule():
    correct = 0
    for i in range(len(X)):
        white_count = count_white_pixels(X[i])
        expected = 1 if white_count >= 2 else 0
        if y[i][0] == expected:
            correct += 1
    
    print(f"Rule verification: {correct}/{len(X)} correct classifications")

if __name__ == "__main__":
    verify_rule()
    visualize_all_grids()
    predictions = mlp.predict(X)
    accuracy = np.mean(predictions.flatten() == y.flatten())
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print("\nTesting the model:")
    print("Enter 4 binary values (e.g. 1 0 0 1) or 'q' to quit:")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break    
        try:
            test_flat = np.array(list(map(int, user_input.split())))
            if len(test_flat) != 4 or not all(x in [0, 1] for x in test_flat):
                print("Please enter exactly 4 binary values (0s and 1s)")
                continue
            test_and_visualize(test_flat)
        except ValueError:
            print("Invalid input. Please enter 4 binary values separated by spaces.")