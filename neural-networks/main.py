import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP  
from Hopfield import HopfieldNetwork, binary_to_bipolar, bipolar_to_binary  
import time
import io
from contextlib import redirect_stdout
import pandas as pd

st.set_page_config(
    page_title="Neural Network Pattern Recognition",
    page_icon="🧠",
    layout="wide"
)
def flat_to_grid(flat_array):
    """Convert flat array to 2x2 grid"""
    return flat_array.reshape(2, 2)

def get_logger():
    log_capture = io.StringIO()
    return log_capture

if 'logs' not in st.session_state:
    st.session_state.logs = []

if 'mlp_trained' not in st.session_state:
    st.session_state.mlp_trained = False

if 'hopfield_trained' not in st.session_state:
    st.session_state.hopfield_trained = False

if 'mlp_model' not in st.session_state:
    st.session_state.mlp_model = None

if 'hopfield_model' not in st.session_state:
    st.session_state.hopfield_model = None

# Title and description
st.title("2x2 Pattern Recognition with Neural Networks")
st.markdown("""
This app demonstrates two different neural network approaches for pattern recognition:
1. **Multi-Layer Perceptron (MLP)**: A feedforward neural network for classification
2. **Hopfield Network**: A recurrent neural network for pattern recall and association
""")

st.sidebar.header("Network Configuration")
network_type = st.sidebar.radio("Select Neural Network", ["MLP", "Hopfield Network"])

# Create columns for grid patterns
col1, col2 = st.columns(2)
with col1:
    st.header("2x2 Grid Pattern Input")
    st.markdown("Click on the cells to toggle between black (0) and white (1)")
    if 'grid_values' not in st.session_state:
        st.session_state.grid_values = [[0, 0], [0, 0]]
    grid_cols = st.columns(2)
    updated_grid = [[0, 0], [0, 0]]
    
    for i in range(2):
        for j in range(2):
            with grid_cols[j]:
                cell_value = st.session_state.grid_values[i][j]
                cell_color = "⬜" if cell_value == 1 else "⬛"
                
                if st.button(cell_color, key=f"cell_{i}_{j}"):
                    st.session_state.grid_values[i][j] = 1 - cell_value
                    st.rerun()
                
                updated_grid[i][j] = st.session_state.grid_values[i][j]
    st.markdown("### Current Grid:")
    grid_display = np.array(st.session_state.grid_values)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid_display, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig) 
    flat_input = np.array(st.session_state.grid_values).flatten()
    st.markdown(f"Binary representation: {flat_input}")
with col2:
    if network_type == "MLP":
        st.header("Multi-Layer Perceptron (MLP)")
        
        # MLP parameters
        with st.expander("MLP Parameters"):
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (comma-separated)", "10,4")
            hidden_sizes = [int(size.strip()) for size in hidden_layer_sizes.split(",")]
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            epochs = st.slider("Epochs", 100, 5000, 1500, 100)
        
        # Training data
        st.subheader("Training Data")
        st.markdown("""
        The MLP will be trained on 16 different 2x2 grid patterns with the following rule:
        - Output is 1 (White) if the grid has 2 or more white cells
        - Output is 0 (Black) if the grid has less than 2 white cells
        """)
        
        # Training button
        if st.button("Train MLP Model"):
            with st.spinner("Training MLP..."):
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    X = np.array([
                        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
                    ])
                    y = np.array([1 if np.sum(x) >= 2 else 0 for x in X]).reshape(-1, 1)
                    input_size = 4
                    output_size = 1
                    mlp = MLP(input_size, hidden_sizes, output_size)
                    
                    print(f"Training MLP with hidden layers: {hidden_sizes}")
                    print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
                    
                    start_time = time.time()
                    losses = mlp.train(X, y, epochs, learning_rate)
                    training_time = time.time() - start_time
                    
                    predictions = mlp.predict(X)
                    accuracy = np.mean(predictions.flatten() == y.flatten())
                    
                    print(f"\nTraining completed in {training_time:.2f} seconds")
                    print(f"Final loss: {losses[-1]:.6f}")
                    print(f"Model accuracy: {accuracy * 100:.2f}%")
                    st.session_state.mlp_model = mlp
                    st.session_state.mlp_trained = True
                st.session_state.logs.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "network": "MLP",
                    "action": "Training",
                    "details": log_capture.getvalue()
                })
                
                # Plot the loss curve
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(losses)
                ax.set_title('MLP Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Mean Squared Error')
                ax.grid(True)
                st.pyplot(fig)  
        
        # Prediction section
        st.subheader("Prediction")
        if st.session_state.mlp_trained and st.button("Predict with MLP"):
            with st.spinner("Making prediction..."):
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    mlp = st.session_state.mlp_model
                    test_input = flat_input.reshape(1, -1)
                    output = mlp.forward(test_input)
                    pred = (output >= 0.5).astype(int)[0][0]
                    
                    white_count = np.sum(test_input)
                    predicted = "White" if pred == 1 else "Black"
                    expected = "White" if white_count >= 2 else "Black"
                    
                    print(f"Input pattern: {test_input.flatten()}")
                    print(f"White pixels: {white_count}")
                    print(f"Network output: {output[0][0]:.4f}")
                    print(f"Classification: {predicted}")
                    print(f"Expected: {expected}")
                    print(f"Correct: {predicted == expected}")
                st.session_state.logs.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "network": "MLP",
                    "action": "Prediction",
                    "details": log_capture.getvalue()
                })
                st.markdown(f"**Network Output**: {output[0][0]:.4f}")
                st.markdown(f"**Classification**: {predicted}")
                st.markdown(f"**Expected**: {expected}")
                
                if predicted == expected:
                    st.success("Prediction is correct! ✅")
                else:
                    st.error("Prediction is incorrect! ❌")
        elif not st.session_state.mlp_trained:
            st.warning("Please train the MLP model first.")

    else: 
        st.header("Hopfield Network")
        
        # Hopfield parameters
        with st.expander("Hopfield Network Parameters"):
            max_iterations = st.slider("Max Iterations for Recall", 1, 50, 20)
            patterns_to_store = st.multiselect( "Select Patterns to Store",   ["Top Row", "Bottom Row", "Left Column", "Right Column", "Diagonal", "Anti-diagonal", 
                            "Top Left", "Top Right", "Bottom Left", "Bottom Right"],default=["Top Left", "Top Right", "Bottom Left", "Bottom Right"])
        
        pattern_map = {
            "Top Row": [1, 1, 0, 0],
            "Bottom Row": [0, 0, 1, 1],
            "Left Column": [1, 0, 1, 0],
            "Right Column": [0, 1, 0, 1],
            "Diagonal": [1, 0, 0, 1],
            "Anti-diagonal": [0, 1, 1, 0],
            "Top Left": [1, 0, 0, 0],
            "Top Right": [0, 1, 0, 0],
            "Bottom Left": [0, 0, 1, 0],
            "Bottom Right": [0, 0, 0, 1]
        }
        
        if st.button("Train Hopfield Network"):
            with st.spinner("Training Hopfield Network..."):
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    selected_patterns = np.array([pattern_map[p] for p in patterns_to_store])
                    hopfield = HopfieldNetwork(4)
                    bipolar_patterns = binary_to_bipolar(selected_patterns)
                    
                    # Train the network
                    hopfield.train(bipolar_patterns)
                    
                    print(f"Training Hopfield Network with {len(patterns_to_store)} patterns:")
                    for i, pattern_name in enumerate(patterns_to_store):
                        print(f"  Pattern {i+1}: {pattern_name} - {selected_patterns[i]}")
                    
                    # Store the trained model in session state
                    st.session_state.hopfield_model = hopfield
                    st.session_state.hopfield_trained = True
                    st.session_state.hopfield_patterns = selected_patterns
                    st.session_state.hopfield_pattern_names = patterns_to_store
                st.session_state.logs.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "network": "Hopfield",
                    "action": "Training",
                    "details": log_capture.getvalue()
                })
            
            # Display the stored patterns
            st.subheader("Stored Patterns")
            cols = st.columns(len(patterns_to_store))
            for i, (col, pattern_name) in enumerate(zip(cols, patterns_to_store)):
                with col:
                    pattern = pattern_map[pattern_name]
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(np.array(pattern).reshape(2, 2), cmap='gray', vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(pattern_name)
                    st.pyplot(fig)  
        st.subheader("Pattern Recall")
        if st.session_state.hopfield_trained and st.button("Recall with Hopfield Network"):
            with st.spinner("Recalling pattern..."):
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    hopfield = st.session_state.hopfield_model
                    
                    bipolar_input = binary_to_bipolar(flat_input)
                    
                    print(f"Input pattern: {flat_input}")
                    print(f"Bipolar input: {bipolar_input}")
                    input_energy = hopfield.energy(bipolar_input)
                    print(f"Input energy: {input_energy:.4f}")
                    stored_patterns = binary_to_bipolar(st.session_state.hopfield_patterns)
                    pattern_names = st.session_state.hopfield_pattern_names
                    
                    print("\nEnergy values for stored patterns:")
                    for i, (pattern, name) in enumerate(zip(stored_patterns, pattern_names)):
                        energy = hopfield.energy(pattern)
                        print(f"  {name}: {energy:.4f}")
                    
                    # Recall
                    print(f"\nRecalling pattern (max iterations: {max_iterations})...")
                    recalled = hopfield.recall(bipolar_input, max_iterations)
                    recalled_binary = bipolar_to_binary(recalled)
                    matches = False
                    match_idx = -1
                    match_name = ""
                    
                    for i, pattern in enumerate(stored_patterns):
                        if np.array_equal(recalled, pattern):
                            matches = True
                            match_idx = i
                            match_name = pattern_names[i]
                            break
                    
                    # Calculate final energy
                    final_energy = hopfield.energy(recalled)
                    
                    print(f"Recalled pattern: {recalled_binary}")
                    print(f"Final energy: {final_energy:.4f}")
                    
                    if matches:
                        print(f"Matches stored pattern: {match_name}")
                    else:
                        print("Does not match any stored pattern")
                st.session_state.logs.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "network": "Hopfield",
                    "action": "Recall",
                    "details": log_capture.getvalue()
                })
                
                st.markdown("### Recall Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Input Pattern:**")
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(np.array(st.session_state.grid_values), cmap='gray', vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig)  
                    st.markdown(f"Energy: {input_energy:.4f}")
                
                with col2:
                    st.markdown("**Recalled Pattern:**")
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(recalled_binary.reshape(2, 2), cmap='gray', vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig) 
                    st.markdown(f"Energy: {final_energy:.4f}")
                
                if matches:
                    st.success(f"Successfully recalled pattern: {match_name} ✅")
                else:
                    st.warning("Did not match any stored pattern ⚠️")
                
                # Display energy values for all patterns
                st.subheader("Energy Values")
                energy_data = []
                
                for i, (pattern, name) in enumerate(zip(stored_patterns, pattern_names)):
                    energy = hopfield.energy(pattern)
                    energy_data.append({"Pattern": name, "Energy": energy})
                
                energy_df = pd.DataFrame(energy_data)
                st.dataframe(energy_df)
                
        elif not st.session_state.hopfield_trained:
            st.warning("Please train the Hopfield network first.")
st.header("Execution Logs")
with st.expander("View Logs", expanded=False):
    if st.session_state.logs:
        for i, log in enumerate(reversed(st.session_state.logs)):
            st.markdown(f"### Log Entry {len(st.session_state.logs) - i}")
            st.markdown(f"**Timestamp:** {log['timestamp']}")
            st.markdown(f"**Network:** {log['network']}")
            st.markdown(f"**Action:** {log['action']}")
            st.markdown("**Details:**")
            st.code(log['details'])
            st.divider()
    else:
        st.info("No logs available yet. Train or use a model to generate logs.")

# Clear logs button
if st.session_state.logs and st.button("Clear Logs"):
    st.session_state.logs = []
    st.rerun()

st.header("Network Comparison")
st.markdown("""
| Feature | MLP | Hopfield Network |
|---------|-----|-----------------|
| **Type** | Feedforward | Recurrent |
| **Learning** | Supervised (with backpropagation) | Unsupervised (Hebbian learning) |
| **Purpose** | Classification/Regression | Pattern Association/Memory |
| **Strengths** | Can learn complex decision boundaries | Can recover complete patterns from partial input |
| **Limitations** | Requires labeled training data | Limited storage capacity, can fall into spurious minima |
""")

st.markdown("---")
st.markdown("### About this app")
st.markdown("""
This app demonstrates two different neural networks for pattern recognition:

1. **Multi-Layer Perceptron (MLP)**: A supervised learning algorithm that learns to classify 2x2 grid patterns based on whether they have 2 or more white cells.

2. **Hopfield Network**: An associative memory that can recall stored patterns even when given noisy or partial input.
""")