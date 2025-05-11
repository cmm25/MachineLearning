# Neural Network Studio

An interactive Streamlit application for exploring three different neural network implementations:

1. **Multi-Layer Perceptron (MLP)**: A feedforward neural network for classification.
2. **Hopfield Network**: A recurrent neural network for pattern recall and association.
3. **Self-Organizing Map (SOM)**: An unsupervised neural network for dimensionality reduction and clustering.

## Projects

This application showcases neural networks on two distinct tasks:

### Binary Pattern Recognition (2x2 Grid)

- Interact with a 2x2 grid to create binary patterns
- Train models to recognize patterns with 2 or more white cells
- Visualize model performance and predictions

### Heart Disease Classification

- Analyze the UCI Heart Disease dataset
- Train and evaluate neural networks on the classification task
- Visualize model performance and convergence

## Features

- **Interactive UI**: Select projects and neural network types from the sidebar
- **Neural Network Visualizations**: See convergence graphs for all networks
- **Parameter Customization**: Adjust network parameters through the UI
- **Logging System**: Track operations in a detailed log view
- **Real-time Predictions**: Test trained models on custom inputs

## Neural Network Implementations

### Multi-Layer Perceptron (MLP)

#### Binary Pattern Recognition

- **Learning**: Uses supervised learning with backpropagation to classify 2x2 binary patterns.
- **Training Data**: 16 possible binary patterns with labels (1 if pattern contains ≥2 white cells, 0 otherwise).
- **Prediction**: For a given binary pattern, outputs a value between 0 and 1, which is then thresholded (≥0.5) to determine class.
- **Visualization**: Convergence plot showing mean squared error over training epochs.

#### Heart Disease Classification

- **Learning**: Uses supervised learning with backpropagation on normalized heart disease features.
- **Training Data**: Preprocessed features from the UCI Heart Disease dataset.
- **Prediction**: Outputs a probability of heart disease which is thresholded for binary classification.
- **Performance**: Evaluated using accuracy on test dataset.

### Hopfield Network

#### Binary Pattern Recognition

- **Learning**: Uses Hebbian learning to store binary patterns as stable states in the network.
- **Training Data**: Selected binary patterns from the 16 possible 2x2 grid configurations.
- **Recall Process**: Converges to the closest stored pattern through iterative updates.
- **Energy Function**: Uses an energy function to measure stability (lower energy indicates pattern is closer to stored memory).

#### Heart Disease Classification

- **Learning**: Stores binarized subsets of features from positive heart disease samples.
- **Limitations**: As noted in the application, Hopfield networks are not ideal for complex classification tasks, and this implementation is primarily for demonstration.
- **Recall Process**: Limited application for clinical prediction, but demonstrates associative memory with binarized features.

### Self-Organizing Map (SOM)

#### Binary Pattern Recognition

- **Learning**: Unsupervised competitive learning that maps similar patterns to nearby neurons on a 2D grid.
- **Training**: Uses all 16 binary patterns with no explicit labels during training.
- **Visualization**: Color-coded map showing patterns with similar numbers of white cells clustering together.
- **Prediction**: Identifies the Best Matching Unit (BMU) for a given input pattern.

#### Heart Disease Classification

- **Learning**: Maps high-dimensional heart disease features to a 2D grid while preserving topological relationships.
- **Prediction**: Uses a majority voting scheme where a new sample is classified based on the most common class label in its Best Matching Unit.
- **Visualization**: Color-coded map showing regions associated with different heart disease classifications.
- **Performance**: Evaluated using accuracy on test dataset.

#### Understanding SOM Coordinates and Output

The SOM grid coordinates (such as "2,1") represent the position of the Best Matching Unit (BMU) on the two-dimensional map:

- First number: Row position in the map (vertical coordinate)
- Second number: Column position in the map (horizontal coordinate)

When the application displays "BMU coordinates: (2,1)", it means the input pattern most closely matches the neuron located at row 2, column 1 in the SOM grid. Patterns that map to the same or nearby coordinates are considered similar by the network.

**Color Scheme:**

- For **binary pattern recognition** (e.g., 2x2 grid), the SOM map uses:
  - **Black**: class 0 (patterns with <2 white cells)
  - **White**: class 1 (patterns with ≥2 white cells)
- For **heart disease classification**, the SOM map uses:
  - **Blue**: Healthy (class 0)
  - **Red**: Heart Disease (class 1)
- The color of each cell shows the majority class of the training samples mapped to that neuron.
- The colorbar is labeled with these class names for clarity.

For multi-class problems, a default colormap and numeric class labels are used.

## Heart Disease Model Outputs

After training a model on the Heart Disease dataset, the application provides several important outputs that help you interpret model performance and make predictions:

### Multi-Layer Perceptron (MLP) Outputs

- **Convergence Plot**: Shows the mean squared error (MSE) loss decreasing over epochs during training. A steadily declining curve indicates the model is learning properly, while fluctuations or plateaus may suggest issues with learning rate or model capacity.
- **Test Accuracy**: The percentage of correctly classified samples from the test set. Higher values (closer to 100%) indicate better performance, but be cautious of overfitting if training accuracy is much higher than test accuracy.
- **Raw Output**: For individual predictions, the MLP outputs a value between 0 and 1, representing the probability of heart disease. Values closer to 1 indicate higher confidence in a positive diagnosis.

### Hopfield Network Outputs

- **Stored Patterns**: Displays the binarized feature vectors stored in the network. These represent prototypical patterns of heart disease features the network has memorized.
- **Recall Limitations**: For heart disease data, the application shows the binarized feature patterns but notes the limitations in using Hopfield networks for complex classification tasks. The network works best with binary patterns and limited pattern count due to capacity constraints.

### Self-Organizing Map (SOM) Outputs

- **Quantization Error**: Measures how well the SOM represents the input data. Lower values indicate better representation of the input space by the map neurons.
- **Topographic Error**: Indicates how well the SOM preserves the topology of the input data. Lower values suggest better preservation of neighborhood relationships.
- **SOM Map Visualization**: A color-coded grid showing how different heart disease samples are mapped across the SOM. Similar samples cluster together, with colors typically indicating different classes (presence/absence of heart disease).
- **Test Accuracy**: The classification accuracy achieved by assigning labels to SOM neurons based on majority voting from training data, then using these labels to classify test data.
- **BMU Distribution**: For individual predictions, the application identifies the Best Matching Unit (neuron) for a given input and the majority class associated with that neuron.
