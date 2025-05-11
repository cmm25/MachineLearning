import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
from contextlib import redirect_stdout
import os
from Implementations.mlp import MLP
from Implementations.Hopfield import (
    HopfieldNetwork, binary_to_bipolar, bipolar_to_binary)
from Implementations.som import SOM
from HeartDisease.processor import HeartDiseaseProcessor

st.set_page_config(
    page_title="Neural Network Studio 🧠",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Neural Network Studio\n"
        "A comprehensive app for exploring neural network implementations "
        "including MLP, Hopfield Network, and Self-Organizing Maps."
    },
)


def get_logger():
    return io.StringIO()


def add_log(project, network, action, details):
    if "logs" not in st.session_state:
        st.session_state.logs = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "project": project,
        "network": network,
        "action": action,
        "details": details,
    }
    # Insert at beginning for recent logs first
    st.session_state.logs.insert(0, log_entry)


def init_session_state():
    if "logs" not in st.session_state:
        st.session_state.logs = []

    # Binary Pattern Project
    if "bp_grid_values" not in st.session_state:
        st.session_state.bp_grid_values = [[0, 0], [0, 0]]
    if "bp_mlp_model" not in st.session_state:
        st.session_state.bp_mlp_model = None
    if "bp_mlp_trained" not in st.session_state:
        st.session_state.bp_mlp_trained = False
    if "bp_mlp_convergence" not in st.session_state:
        st.session_state.bp_mlp_convergence = None
    if "bp_hopfield_model" not in st.session_state:
        st.session_state.bp_hopfield_model = None
    if "bp_hopfield_trained" not in st.session_state:
        st.session_state.bp_hopfield_trained = False
    if "bp_hopfield_patterns" not in st.session_state:
        st.session_state.bp_hopfield_patterns = None
    if "bp_som_model" not in st.session_state:
        st.session_state.bp_som_model = None
    if "bp_som_trained" not in st.session_state:
        st.session_state.bp_som_trained = False
    if "bp_som_convergence" not in st.session_state:
        st.session_state.bp_som_convergence = None

    # Heart Disease Project
    if "hd_processor" not in st.session_state:
        st.session_state.hd_processor = HeartDiseaseProcessor()
    if "hd_data_loaded" not in st.session_state:
        st.session_state.hd_data_loaded = False
    if "hd_mlp_model" not in st.session_state:
        st.session_state.hd_mlp_model = None
    if "hd_mlp_trained" not in st.session_state:
        st.session_state.hd_mlp_trained = False
    if "hd_mlp_convergence" not in st.session_state:
        st.session_state.hd_mlp_convergence = None
    if "hd_hopfield_model" not in st.session_state:
        st.session_state.hd_hopfield_model = None
    if "hd_hopfield_trained" not in st.session_state:
        st.session_state.hd_hopfield_trained = False
    if "hd_hopfield_patterns" not in st.session_state:
        st.session_state.hd_hopfield_patterns = None
    if "hd_som_model" not in st.session_state:
        st.session_state.hd_som_model = None
    if "hd_som_trained" not in st.session_state:
        st.session_state.hd_som_trained = False
    if "hd_som_convergence" not in st.session_state:
        st.session_state.hd_som_convergence = None
    if "hd_som_bmu_to_label" not in st.session_state:
        st.session_state.hd_som_bmu_to_label = None


init_session_state()
st.sidebar.title("⚙️ Configuration Studio")
st.sidebar.markdown("---")

project_type = st.sidebar.selectbox(
    "🚀 Choose Project",
    ["Binary Pattern Recognition (2x2)", "Heart Disease Classification"],
    key="project_selection",
)
st.sidebar.markdown("---")

# Network Selection
network_options = [
    "Multi-Layer Perceptron (MLP)",
    "Hopfield Network",
    "Self-Organizing Map (SOM)",
]
selected_network = st.sidebar.radio(
    "🧠 Select Neural Network", network_options, key="network_selection")
st.sidebar.markdown("---")


def display_binary_patterns_project():
    st.header("Binary Pattern Recognition (2x2 Grid)")
    st.markdown("Interact with the 2x2 grid and use the selected neural network for pattern recognition tasks.")

    col_grid, col_network = st.columns([1, 2])

    with col_grid:
        st.subheader("🎨 Grid Input")
        st.markdown("Click cells to toggle black (0) / white (1).")
        grid_cols = st.columns(2)
        current_grid = st.session_state.bp_grid_values
        for r in range(2):
            for c in range(2):
                with grid_cols[c]:
                    cell_key = f"bp_cell_{r}_{c}"
                    cell_char = "⬜" if current_grid[r][c] == 1 else "⬛"
                    if st.button(cell_char, key=cell_key, use_container_width=True):
                        current_grid[r][c] = 1 - current_grid[r][c]
                        st.rerun()

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(current_grid, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        flat_input = np.array(current_grid).flatten()
        st.caption(f"Binary: {flat_input}")

    with col_network:
        if selected_network == "Multi-Layer Perceptron (MLP)":
            display_bp_mlp(flat_input)
        elif selected_network == "Hopfield Network":
            display_bp_hopfield(flat_input)
        elif selected_network == "Self-Organizing Map (SOM)":
            display_bp_som(flat_input)


def display_bp_mlp(flat_input):
    st.subheader("🤖 Multi-Layer Perceptron (MLP)")
    with st.expander("MLP Parameters & Training", expanded=not st.session_state.bp_mlp_trained):
        hidden_layer_sizes = st.text_input(
            "Hidden Layer Sizes (comma-separated)", "10,4", key="bp_mlp_hidden")
        learning_rate = st.slider(
            "Learning Rate", 0.001, 0.5, 0.05, 0.001, key="bp_mlp_lr", format="%.3f")
        epochs = st.slider("Epochs", 100, 5000, 1000, 100, key="bp_mlp_epochs")

        if st.button("Train MLP Model", key="bp_mlp_train_button"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training MLP..."):
                    hidden_sizes = [
                        int(s.strip())
                        for s in hidden_layer_sizes.split(",")
                        if s.strip()
                    ]
                    X = np.array(
                        [
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
                            [1, 1, 1, 1],
                        ]
                    )
                    y = np.array(
                        [1 if np.sum(p) >= 2 else 0 for p in X]).reshape(-1, 1)
                    mlp = MLP(input_size=4,hidden_sizes=hidden_sizes, output_size=1)
                    print(f"Training MLP: hidden_layers={hidden_sizes}, lr={learning_rate}, epochs={epochs}")
                    losses = mlp.train(X, y, epochs, learning_rate)
                    st.session_state.bp_mlp_model = mlp
                    st.session_state.bp_mlp_trained = True
                    st.session_state.bp_mlp_convergence = losses
                    print(f"MLP training complete. Final loss: {losses[-1]:.4f}")
            add_log("Binary Patterns", "MLP","Training", log_capture.getvalue())
            st.success("MLP Model Trained!")

    if st.session_state.bp_mlp_convergence:
        with st.expander("MLP Convergence Plot", expanded=True):
            fig, ax = plt.subplots()
            ax.plot(st.session_state.bp_mlp_convergence)
            ax.set_title("MLP Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Squared Error")
            ax.grid(True)
            st.pyplot(fig)

    if st.session_state.bp_mlp_trained:
        st.markdown("---")
        st.subheader("🔍 Prediction")
        if st.button("Predict with MLP", key="bp_mlp_predict"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                mlp = st.session_state.bp_mlp_model
                output = mlp.forward(flat_input.reshape(1, -1))
                pred_value = (output >= 0.5).astype(int)[0][0]

                white_count = np.sum(flat_input)
                actual_class = "Pattern ≥2 Whites" if white_count >= 2 else "Pattern <2 Whites"
                predicted_class = "Pattern ≥2 Whites" if pred_value == 1 else "Pattern <2 Whites"

                print(f"Input: {flat_input}, White cells: {white_count}, Raw Output: {output[0][0]:.4f}, Predicted: {predicted_class}, Actual: {actual_class}")
                st.metric("Network Output", f"{output[0][0]:.4f}")
                st.metric("White Cells in Pattern", f"{white_count}")
                st.metric("Predicted Class", predicted_class)
                st.metric("Actual Class", actual_class, delta=("Correct" if predicted_class == actual_class else "Incorrect"))
            add_log("Binary Patterns", "MLP","Prediction", log_capture.getvalue())
def display_bp_hopfield(flat_input):
    st.subheader("🔗 Hopfield Network")
    patterns_map = {
        "Top Row": [1, 1, 0, 0],
        "Bottom Row": [0, 0, 1, 1],
        "Left Col": [1, 0, 1, 0],
        "Right Col": [0, 1, 0, 1],
        "Diagonal \\": [1, 0, 0, 1],
        "Anti-Diag /": [0, 1, 1, 0],
        "Cross": [0, 1, 1, 0],
        "Top Left": [1, 0, 0, 0],
        "Top Right": [0, 1, 0, 0],
        "Bottom Left": [0, 0, 1, 0],
        "Bottom Right": [0, 0, 0, 1],
        "All Black": [0, 0, 0, 0],
        "All White": [1, 1, 1, 1],
    }
    with st.expander("Hopfield Parameters & Training", expanded=not st.session_state.bp_hopfield_trained):
        selected_pattern_names = st.multiselect(
            "Select patterns to store in Hopfield Network:",
            list(patterns_map.keys()),
            default=["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
            key="bp_hop_patterns_select",
        )
        if st.button("Train Hopfield Network", key="bp_hop_train"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training Hopfield Network..."):
                    patterns_to_store = np.array([patterns_map[name] for name in selected_pattern_names])
                    st.session_state.bp_hopfield_patterns = ( patterns_to_store )

                    bipolar_patterns = binary_to_bipolar(patterns_to_store)
                    hopfield = HopfieldNetwork(num_neurons=4)
                    hopfield.train(bipolar_patterns)
                    st.session_state.bp_hopfield_model = hopfield
                    st.session_state.bp_hopfield_trained = True
                    print(f"Hopfield Network trained with {len(selected_pattern_names)} patterns.")
            add_log("Binary Patterns", "Hopfield", "Training", log_capture.getvalue())
            st.success("Hopfield Network Trained!")

    if st.session_state.bp_hopfield_trained:
        st.markdown("---")
        st.subheader("Stored Patterns")
        if (st.session_state.bp_hopfield_patterns is not None and len(st.session_state.bp_hopfield_patterns) > 0):
            num_patterns = len(st.session_state.bp_hopfield_patterns)
            # Display up to 4 patterns per row
            cols = st.columns(min(num_patterns, 4))
            for i, pattern in enumerate(st.session_state.bp_hopfield_patterns):
                with cols[i % min(num_patterns, 4)]:
                    fig, ax = plt.subplots(figsize=(1.5, 1.5))
                    ax.imshow(pattern.reshape(2, 2),cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig)
        else:
            st.info("No patterns stored or training was incomplete.")

        st.markdown("---")
        st.subheader("🔄 Pattern Recall")
        max_iters_recall = st.slider("Max Recall Iterations", 1, 50, 10, key="bp_hop_recall_iters")
        if st.button("Recall with Hopfield", key="bp_hop_recall"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                hopfield = st.session_state.bp_hopfield_model
                bipolar_input = binary_to_bipolar(flat_input)
                print(f"Input (binary): {flat_input}, Input (bipolar): {bipolar_input}")
                input_energy = hopfield.energy(bipolar_input)
                print(f"Input energy: {input_energy:.4f}")

                recalled_bipolar, iterations = hopfield.recall(
                    bipolar_input,
                    max_iterations=max_iters_recall,
                    return_iterations=True,
                )
                recalled_binary = bipolar_to_binary(recalled_bipolar)
                recalled_energy = hopfield.energy(recalled_bipolar)
                print(f"Recalled (bipolar): {recalled_bipolar}, Recalled (binary): {recalled_binary}")
                print(f"Recalled after {iterations} iterations. Recalled energy: {recalled_energy:.4f}")
                st.metric("Input Energy", f"{input_energy:.4f}")
                st.metric("Recalled Energy", f"{recalled_energy:.4f}")
                st.metric("Iterations to Converge", iterations)

                fig_recalled, ax_recalled = plt.subplots(figsize=(2, 2))
                ax_recalled.imshow(recalled_binary.reshape(
                    2, 2), cmap="gray", vmin=0, vmax=1)
                ax_recalled.set_title("Recalled Pattern")
                ax_recalled.set_xticks([])
                ax_recalled.set_yticks([])
                st.pyplot(fig_recalled)
            add_log("Binary Patterns", "Hopfield", "Recall", log_capture.getvalue())
                 
def display_bp_som(flat_input):
    st.subheader("🗺️ Self-Organizing Map (SOM)")
    X_binary_patterns = np.array(
        [
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
            [1, 1, 1, 1],
        ]
    )
    # Using binary labels for color-coding: 0 = <2 white cells, 1 = ≥2 white cells
    y_binary_labels = np.array([1 if np.sum(p) >= 2 else 0 for p in X_binary_patterns])

    with st.expander("SOM Parameters & Training", expanded=not st.session_state.bp_som_trained):
        map_rows = st.slider("Map Rows", 2, 10, 4, key="bp_som_rows")
        map_cols = st.slider("Map Columns", 2, 10, 4, key="bp_som_cols")
        learning_rate_som = st.slider("Learning Rate (Initial)", 0.01, 1.0, 0.5, 0.01, key="bp_som_lr")
        sigma_som = st.slider(
            "Sigma (Initial Neighborhood Radius)",
            0.1,
            float(max(map_rows, map_cols) / 2),
            float(max(map_rows, map_cols) / 4),
            0.1,
            key="bp_som_sigma",
        )
        epochs_som = st.slider("Epochs", 50, 1000, 100,50, key="bp_som_epochs")

        if st.button("Train SOM Model", key="bp_som_train"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training SOM..."):
                    som = SOM(
                        input_dim=4,
                        map_size=(map_rows, map_cols),
                        learning_rate=learning_rate_som,
                        sigma=sigma_som,
                    )
                    print(f"Training SOM: map_size=({map_rows},{map_cols}), lr={learning_rate_som}, sigma={sigma_som}, epochs={epochs_som}")
                    som.train(X_binary_patterns, epochs=epochs_som, verbose=False)
                    st.session_state.bp_som_model = som
                    st.session_state.bp_som_trained = True
                    st.session_state.bp_som_convergence = som.get_convergence_data()
                    print("SOM training complete.")
            add_log("Binary Patterns", "SOM", "Training", log_capture.getvalue())
            st.success("SOM Model Trained!")

    if st.session_state.bp_som_convergence:
        with st.expander("SOM Convergence Plots", expanded=True):
            convergence_data = st.session_state.bp_som_convergence
            # Assumes SOM class has this static/class method
            fig = SOM.plot_convergence_from_data(convergence_data)
            st.pyplot(fig)

    if st.session_state.bp_som_trained:
        st.markdown("---")
        st.subheader("🔍 Prediction & Map Visualization")
        som = st.session_state.bp_som_model
        # Visualize the map with all 16 patterns
        st.write("SOM Map with all 16 binary patterns:")
        from matplotlib.colors import ListedColormap
        som_cmap = ListedColormap(['black', 'white'])
        fig_map = som.visualize_map(X_binary_patterns, labels=y_binary_labels, figsize=(6, 5), cmap=som_cmap)
        st.pyplot(fig_map)
        st.caption("Black = patterns with <2 white cells, White = patterns with ≥2 white cells.")

        if st.button("Find BMU for Current Grid Input", key="bp_som_predict"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                bmu = som.predict(flat_input.reshape(1, -1))[0]
                # Determine the class at the BMU
                cell_class = None
                white_count = np.sum(flat_input)
                actual_class = "White (≥2 white cells)" if white_count >= 2 else "Black (<2 white cells)"

                # Recompute the cell_colors logic for this input
                bmu_counts = {}
                for i, sample in enumerate(X_binary_patterns):
                    bmu_pattern = tuple(som.predict(sample.reshape(1, -1))[0])
                    if bmu_pattern not in bmu_counts:
                        bmu_counts[bmu_pattern] = {0: 0, 1: 0}
                    label = y_binary_labels[i]
                    bmu_counts[bmu_pattern][label] += 1
                if tuple(bmu) in bmu_counts:
                    count_0 = bmu_counts[tuple(bmu)][0]
                    count_1 = bmu_counts[tuple(bmu)][1]
                    if count_0 > count_1:
                        class_val = 0
                        cell_class = "Black (<2 white cells)"
                    elif count_1 > count_0:
                        class_val = 1
                        cell_class = "White (≥2 white cells)"
                    else:
                        class_val = None
                        cell_class = "Ambiguous (equal counts for both classes)"
                else:
                    cell_class = "Unknown"
                print(
                    f"Input: {flat_input}, White cells: {white_count}, BMU: {bmu}, Predicted class: {cell_class}, Actual class: {actual_class}")
                st.metric("Best Matching Unit (BMU) for current input", f"({bmu[0]}, {bmu[1]})")
                st.metric("White Cells in Pattern", f"{white_count}")
                st.metric("Predicted Class", cell_class)
                st.metric("Actual Class", actual_class, delta=("Correct" if (cell_class == actual_class) else "Incorrect"))
                st.info(f"The current input pattern maps to neuron {bmu} on the SOM grid above. This neuron represents patterns with similar features.")
            add_log("Binary Patterns", "SOM",
                    "Prediction", log_capture.getvalue())

# --- Heart Disease Classification Project ---


def display_heart_disease_project():
    st.header("❤️ Heart Disease Classification")
    processor = st.session_state.hd_processor

    if not st.session_state.hd_data_loaded:
        if st.button("Load and Preprocess Heart Disease Data", key="hd_load_data"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Loading and preprocessing data..."):
                    print("Attempting to load Heart Disease dataset...")
                    if processor.load_data():
                        print("Data loaded. Preprocessing...")
                        processor.preprocess_data()
                        st.session_state.hd_data_loaded = True
                        print("Data preprocessing complete.")
                        st.success("Heart Disease data loaded and preprocessed!")
                    else:
                        print("Failed to load data.")
                        st.error("Failed to load or preprocess Heart Disease data.")
            add_log("Heart Disease", "Data",
                    "Load & Preprocess", log_capture.getvalue())
            st.rerun()

    if not st.session_state.hd_data_loaded:
        st.warning("Please load the Heart Disease dataset to proceed.")
        st.stop()

    st.success("Heart Disease data is loaded and preprocessed.")
    if st.checkbox("Show Data Overview", key="hd_show_data_overview"):
        st.subheader("Data Glimpse (First 5 rows of X_train)")
        st.dataframe(pd.DataFrame(processor.X_train).head())
        st.subheader("Target Distribution (y_train)")
        st.write(pd.Series(processor.y_train).value_counts())

        st.subheader("Data Visualizations")
        with st.spinner("Generating data visualizations..."):
            fig_viz = processor.visualize_data()
            st.pyplot(fig_viz)

    st.markdown("---")
    if selected_network == "Multi-Layer Perceptron (MLP)":
        display_hd_mlp(processor)
    elif selected_network == "Hopfield Network":
        display_hd_hopfield(processor)
    elif selected_network == "Self-Organizing Map (SOM)":
        display_hd_som(processor)


def display_hd_mlp(processor):
    st.subheader("🤖 MLP for Heart Disease")
    with st.expander("MLP Parameters & Training", expanded=not st.session_state.hd_mlp_trained):
        hidden_layer_sizes = st.text_input(
            "Hidden Layer Sizes (comma-separated)", "20,10", key="hd_mlp_hidden")
        learning_rate = st.slider(
            "Learning Rate", 0.001, 0.1, 0.01, 0.001, key="hd_mlp_lr", format="%.3f")
        epochs = st.slider("Epochs", 100, 5000, 500, 100, key="hd_mlp_epochs")

        if st.button("Train MLP Model", key="hd_mlp_train_button"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training MLP..."):
                    hidden_sizes = [
                        int(s.strip())
                        for s in hidden_layer_sizes.split(",")
                        if s.strip()
                    ]
                    print(f"Training MLP (Heart Disease): hidden_layers={hidden_sizes}, lr={learning_rate}, epochs={epochs}")
                    mlp_model, losses = processor.train_mlp(
                        hidden_sizes=hidden_sizes,
                        learning_rate=learning_rate,
                        epochs=epochs,
                    )
                    st.session_state.hd_mlp_model = mlp_model
                    st.session_state.hd_mlp_trained = True
                    st.session_state.hd_mlp_convergence = losses

                    y_pred_test = mlp_model.predict(processor.X_test)
                    accuracy_test = np.mean(
                        y_pred_test.flatten() == processor.y_test.flatten())
                    print(f"MLP training complete. Final loss: {losses[-1]:.4f}")
                    print(f"Test Set Accuracy: {accuracy_test*100:.2f}%")
                    st.session_state.hd_mlp_accuracy = accuracy_test

            add_log("Heart Disease", "MLP", "Training", log_capture.getvalue())
            st.success( f"MLP Model Trained! Test Accuracy: {st.session_state.hd_mlp_accuracy*100:.2f}%")

    if st.session_state.hd_mlp_convergence:
        with st.expander("MLP Convergence Plot", expanded=True):
            fig, ax = plt.subplots()
            ax.plot(st.session_state.hd_mlp_convergence)
            ax.set_title("MLP Training Loss (Heart Disease)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean Squared Error")
            ax.grid(True)
            st.pyplot(fig)

    if st.session_state.hd_mlp_trained:
        st.metric("MLP Test Accuracy", f"{st.session_state.get('hd_mlp_accuracy', 0)*100:.2f}%")
        st.markdown("---")
        st.subheader("🔍 Sample Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.info(
                "Each time you click the button below, a different random sample will be selected from the test dataset.")
            if st.button("Predict on Random Test Sample", key="hd_mlp_predict_sample"):
                if len(processor.X_test) > 0:
                    log_capture = get_logger()
                    with redirect_stdout(log_capture):
                        # Select a random sample from test set
                        random_idx = np.random.randint(
                            0, len(processor.X_test))
                        test_sample = processor.X_test[random_idx: random_idx + 1]
                        actual_label = processor.y_test[random_idx]

                        # Make prediction
                        mlp = st.session_state.hd_mlp_model
                        prediction = mlp.predict(test_sample)[0, 0]
                        predicted_class = 1 if prediction >= 0.5 else 0

                        print(f"Selected random test sample index: {random_idx}")
                        print(f"MLP Raw Output: {prediction:.4f}")
                        print(f"Predicted Class: {predicted_class} (Heart Disease=1, Healthy=0)")
                        print(f"Actual Class: {actual_label}")

                        # Display important features for this sample
                        if hasattr(processor, "feature_names"):
                            feature_values = []
                            for i, name in enumerate(processor.feature_names):
                                feature_values.append(
                                    f"{name}: {test_sample[0, i]:.3f}")
                            print("Sample Features:")
                            # Show first 5 features
                            for fv in feature_values[:5]:
                                print(f"- {fv}")

                    add_log("Heart Disease", "MLP", "Sample Prediction", log_capture.getvalue())
                    # Display results to user
                    st.metric("MLP Raw Output", f"{prediction:.4f}")
                    st.metric(
                        "Predicted Class", "Heart Disease" if predicted_class == 1 else "Healthy")
                    st.metric("Actual Class", "Heart Disease" if actual_label == 1 else "Healthy", delta=(
                        "Correct" if predicted_class == actual_label else "Incorrect"))
                    # Add confidence indicator
                    confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
                    st.progress(confidence)
                    st.text(f"Prediction Confidence: {confidence:.2%}")
                else:
                    st.error( "No test samples available. Please ensure data is loaded correctly." )

        with col2:
            st.subheader("Manual Input Prediction")
            st.info("Enter values for key features to predict heart disease:")

            # Create simplified form with a few important features
            age = st.slider("Age", 20, 100, 55, 1, key="hd_mlp_manual_age")
            sex = st.selectbox("Sex", ["Male", "Female"], key="hd_mlp_manual_sex")
            chest_pain = st.selectbox(
                "Chest Pain Type",
                [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic",
                ],
                key="hd_mlp_manual_cp",
            )
            resting_bp = st.slider("Resting BP (mm Hg)", 90, 200, 130, 5, key="hd_mlp_manual_bp")
            cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250, 10, key="hd_mlp_manual_chol")

            if st.button("Predict", key="hd_mlp_manual_predict"):
                if st.session_state.hd_mlp_trained:
                    log_capture = get_logger()
                    with redirect_stdout(log_capture):
                        # Convert inputs to normalized feature vector
                        try:
                            # Create a feature vector with default values
                            manual_features = np.zeros(
                                (1, processor.X_train.shape[1]))
                            print("Creating manual prediction input...")
                            if hasattr(processor, "feature_index_map"):
                                feature_map = processor.feature_index_map
                                print(f"Available features: {feature_map}")
                                # Map form inputs to feature indices based on your data
                                sex_val = 1 if sex == "Male" else 0
                                cp_val = [
                                    "Typical Angina",
                                    "Atypical Angina",
                                    "Non-anginal Pain",
                                    "Asymptomatic",
                                ].index(chest_pain)
                                manual_features[0, 0] = (age - 50) / 20
                                manual_features[0, 1] = sex_val
                                manual_features[0, 2] = cp_val / 3
                                manual_features[0, 3] = (resting_bp - 120) / 40
                                manual_features[0, 4] = (
                                    cholesterol - 200) / 200
                            else:
                                print("Feature mapping not available, using default positions")
                                # Use default positions if feature map isn't available
                                manual_features[0, 0] = (age - 50) / 20
                                manual_features[0,1] = 1 if sex == "Male" else 0
                                manual_features[0, 2] = [
                                    "Typical Angina",
                                    "Atypical Angina",
                                    "Non-anginal Pain",
                                    "Asymptomatic",
                                ].index(chest_pain) / 3
                                manual_features[0, 3] = (resting_bp - 120) / 40
                                manual_features[0, 4] = (
                                    cholesterol - 200) / 200

                            # Make prediction
                            mlp = st.session_state.hd_mlp_model
                            prediction = mlp.predict(manual_features)[0, 0]
                            predicted_class = 1 if prediction >= 0.5 else 0

                            print(f"Manual input normalized features: {manual_features}")
                            print(f"MLP Raw Output: {prediction:.4f}")
                            print(f"Predicted Class: {predicted_class} (Heart Disease=1, Healthy=0)")
                        except Exception as e:
                            print(f"Error during manual prediction: {str(e)}")
                            st.error(f"Error processing input: {str(e)}")

                    add_log("Heart Disease", "MLP","Manual Prediction", log_capture.getvalue())
                    # Display results
                    st.metric("Prediction Score", f"{prediction:.4f}")
                    st.metric("Result", "Heart Disease Risk" if predicted_class == 1 else "Healthy")
                    confidence = abs(prediction - 0.5) * 2
                    st.progress(confidence)
                    st.text(f"Confidence: {confidence:.2%}")
                else:
                    st.error("Please train the MLP model first")


def display_hd_hopfield(processor):
    st.subheader("🔗 Hopfield Network for Heart Disease")
    st.warning("Hopfield Networks are generally not ideal for complex classification tasks like Heart Disease due to limited capacity and requirement for binary/bipolar patterns. This is for demonstration.")

    with st.expander("Hopfield Parameters & Training", expanded=not st.session_state.hd_hopfield_trained):
        max_patterns_hd = st.slider(
            "Max Patterns to Store (from Class 1)", 5, 50, 10, key="hd_hop_max_patterns")

        if st.button("Train Hopfield Network", key="hd_hop_train"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training Hopfield (Heart Disease)..."):
                    print(f"Training Hopfield (Heart Disease) with max_patterns={max_patterns_hd}")
                    hopfield_model, stored_binary_patterns = processor.train_hopfield(
                        max_patterns=max_patterns_hd)
                    st.session_state.hd_hopfield_model = hopfield_model
                    st.session_state.hd_hopfield_trained = True
                    st.session_state.hd_hopfield_patterns = stored_binary_patterns
                    print(f"Hopfield (Heart Disease) trained with {len(stored_binary_patterns)} patterns.")
            add_log("Heart Disease", "Hopfield", "Training", log_capture.getvalue())
            st.success(f"Hopfield Model Trained with {len(st.session_state.hd_hopfield_patterns)} patterns.")

    if st.session_state.hd_hopfield_trained:
        st.markdown("---")
        st.subheader("Stored Patterns (Binarized Subset of Features)")
        stored_patterns_hd = st.session_state.hd_hopfield_patterns
        if stored_patterns_hd is not None and len(stored_patterns_hd) > 0:
            st.write(f"Displaying {len(stored_patterns_hd)} stored patterns (each is a {stored_patterns_hd.shape[1]}-dim binarized feature vector).")
            st.dataframe(pd.DataFrame(stored_patterns_hd))
        else:
            st.info("No patterns stored or training incomplete.")
        st.markdown("---")
        st.subheader("🔍 Pattern Recall Demonstration")

        st.info("Each time you click the button below, a different random sample will be selected from the test dataset.")
        if st.button("Recall with Random Test Sample", key="hd_hop_recall"):
            if len(processor.X_test) > 0:
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    random_idx = np.random.randint(0, len(processor.X_test))
                    test_sample = processor.X_test[random_idx]
                    actual_label = processor.y_test[random_idx]
                    try:
                        print(
                            f"Selected random test sample index: {random_idx} (Class {actual_label})")
                        test_features = processor.binarize_features(
                            test_sample.reshape(1, -1))[0]
                        print(f"Binarized test features: {test_features}")
                        bipolar_features = binary_to_bipolar(test_features)
                        noise_level = 0.2  # Flip 20% of bits
                        noise_count = int(len(bipolar_features) * noise_level)
                        noise_indices = np.random.choice(
                            range(len(bipolar_features)), size=noise_count, replace=False)
                        noisy_features = bipolar_features.copy()
                        for idx in noise_indices:
                            noisy_features[idx] *= -1  # Flip the bit

                        print(f"Added noise at indices: {noise_indices}")
                        print(f"Noisy features: {noisy_features}")

                        # Recall pattern
                        hopfield = st.session_state.hd_hopfield_model
                        # Convert back to binary for display
                        recalled_features, iterations = hopfield.recall(
                            noisy_features, max_iterations=20, return_iterations=True)
                        recalled_binary = bipolar_to_binary(recalled_features)

                        print(f"Recalled after {iterations} iterations")
                        print(f"Recalled features: {recalled_binary}")

                    except Exception as e:
                        print(f"Error during Hopfield recall: {str(e)}")
                        st.error(f"Error during pattern recall: {str(e)}")

                add_log("Heart Disease", "Hopfield", "Recall", log_capture.getvalue())
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sample Index", f"{random_idx} of {len(processor.X_test)-1}")
                    st.subheader("Original Binary Features")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(test_features.reshape(1, -1), cmap='binary')
                    ax.set_title("Original")
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.subheader("Noisy Binary Features")
                    # Convert bipolar noisy features back to binary for visualization
                    noisy_binary = bipolar_to_binary(noisy_features)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(noisy_binary.reshape(1, -1), cmap='binary')
                    ax.set_title(f"With {noise_level*100:.0f}% Noise")
                    ax.axis('off')
                    st.pyplot(fig)

                with col3:
                    st.subheader("Recalled Binary Features")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(recalled_binary.reshape(1, -1), cmap='binary')
                    ax.set_title(f"After {iterations} Iterations")
                    ax.axis('off')
                    st.pyplot(fig)

                # Calculate and display recovery accuracy
                recovery_accuracy = np.mean(recalled_binary == test_features)
                st.metric("Recovery Accuracy", f"{recovery_accuracy*100:.1f}%")
            else:
                st.error("No test samples available. Please ensure data is loaded correctly.")


def display_hd_som(processor):
    st.subheader("🗺️ SOM for Heart Disease")
    with st.expander("SOM Parameters & Training", expanded=not st.session_state.hd_som_trained):
        map_rows_hd = st.slider("Map Rows", 5, 20, 10, key="hd_som_rows")
        map_cols_hd = st.slider("Map Columns", 5, 20, 10, key="hd_som_cols")
        learning_rate_som_hd = st.slider("Learning Rate (Initial)", 0.01, 1.0, 0.5, 0.01, key="hd_som_lr")
        sigma_som_hd = st.slider(
            "Sigma (Initial)",
            0.1,
            float(max(map_rows_hd, map_cols_hd) / 2),
            float(max(map_rows_hd, map_cols_hd) / 4),
            0.1,
            key="hd_som_sigma",
        )
        epochs_som_hd = st.slider("Epochs", 50, 1000, 100, 50, key="hd_som_epochs")

        if st.button("Train SOM Model", key="hd_som_train"):
            log_capture = get_logger()
            with redirect_stdout(log_capture):
                with st.spinner("Training SOM (Heart Disease)..."):
                    print(f"Training SOM (Heart Disease): map_size=({map_rows_hd},{map_cols_hd}), lr={learning_rate_som_hd}, sigma={sigma_som_hd}, epochs={epochs_som_hd}")
                    som_model, convergence_data_som, bmu_to_label = processor.train_som(
                        map_size=(map_rows_hd, map_cols_hd),
                        learning_rate=learning_rate_som_hd,
                        sigma=sigma_som_hd,
                        epochs=epochs_som_hd,
                    )
                    st.session_state.hd_som_model = som_model
                    st.session_state.hd_som_trained = True
                    st.session_state.hd_som_convergence = (
                        som_model.get_convergence_data())
                    st.session_state.hd_som_bmu_to_label = bmu_to_label

                    # Evaluate on test data using the bmu_to_label map
                    bmus_test = som_model.predict(processor.X_test)
                    # Default to class 0 if BMU not in map
                    predictions_som = np.array(
                        [bmu_to_label.get(tuple(bmu), 0) for bmu in bmus_test])
                    accuracy_som = np.mean(predictions_som == processor.y_test)
                    st.session_state.hd_som_accuracy = accuracy_som
                    print(f"SOM training complete. Test Accuracy: {accuracy_som*100:.2f}%")
            add_log("Heart Disease", "SOM", "Training", log_capture.getvalue())
            st.success(f"SOM Model Trained! Test Accuracy: {st.session_state.hd_som_accuracy*100:.2f}%")

    if st.session_state.hd_som_convergence:
        with st.expander("SOM Convergence Plots", expanded=True):
            convergence_data = st.session_state.hd_som_convergence
            fig = SOM.plot_convergence_from_data(convergence_data)
            st.pyplot(fig)

    if st.session_state.hd_som_trained:
        st.metric("SOM Test Accuracy",f"{st.session_state.get('hd_som_accuracy', 0)*100:.2f}%")
        st.markdown("---")
        st.subheader("🗺️ SOM Map Visualization (Color-coded by Class)")
        som = st.session_state.hd_som_model
        fig_map_hd = som.visualize_map(
            processor.X_train, labels=processor.y_train, figsize=(8, 7), cmap="coolwarm",
            title="Heart Disease Classification Map"
        )
        st.pyplot(fig_map_hd)
        st.caption("Map neurons are colored based on the majority class of training samples mapped to them (blue = Healthy, red = Heart Disease)." )

        # Add prediction functionality
        st.markdown("---")
        st.subheader("🔍 Sample Prediction")

        st.info("Each time you click the button below, a different random sample will be selected from the test dataset.")

        if st.button("Predict on Random Test Sample", key="hd_som_predict_sample"):
            if st.session_state.hd_som_trained and len(processor.X_test) > 0:
                log_capture = get_logger()
                with redirect_stdout(log_capture):
                    # Select a random sample from test set
                    random_idx = np.random.randint(0, len(processor.X_test))
                    test_sample = processor.X_test[random_idx: random_idx + 1]
                    actual_label = processor.y_test[random_idx]

                    # Make prediction
                    som = st.session_state.hd_som_model
                    bmu_to_label = st.session_state.hd_som_bmu_to_label

                    # Find the Best Matching Unit (BMU)
                    bmu = som.predict(test_sample)[0]
                    bmu_tuple = tuple(bmu)

                    # Get the predicted class from the BMU mapping
                    predicted_class = bmu_to_label.get(bmu_tuple, 0)
                    print(f"Selected random test sample index: {random_idx}")
                    print(f"BMU coordinates: ({bmu[0]}, {bmu[1]})")
                    print(f"Predicted Class: {predicted_class} (Heart Disease=1, Healthy=0)")
                    print(f"Actual Class: {actual_label}")
                    neighborhood_counts = {0: 0, 1: 0}
                    neighborhood_size = 1  # Look at immediate neighbors

                    # Get training samples that map to this BMU and its neighbors
                    bmus_train = som.predict(processor.X_train)
                    for i, bmu_train in enumerate(bmus_train):
                        # Check if this BMU is within the neighborhood
                        dist = np.abs(bmu_train[0] - bmu[0]) + \
                            np.abs(bmu_train[1] - bmu[1])
                        if dist <= neighborhood_size:
                            label = processor.y_train[i]
                            neighborhood_counts[label] = (
                                neighborhood_counts.get(label, 0) + 1)

                    total_neighbors = sum(neighborhood_counts.values())
                    if total_neighbors > 0:
                        print(f"Class distribution in BMU neighborhood:")
                        for label, count in neighborhood_counts.items():
                            class_name = "Healthy" if label == 0 else "Heart Disease"
                            percentage = (count / total_neighbors) * 100
                            print(f"- {class_name}: {count} samples ({percentage:.1f}%)")

                    # Display feature values for the sample
                    if hasattr(processor, "feature_names"):
                        feature_values = []
                        for i, name in enumerate(processor.feature_names):
                            feature_values.append(
                                f"{name}: {test_sample[0, i]:.3f}")
                        print("Sample Features:")
                        for fv in feature_values[:5]:  # Show first 5 features
                            print(f"- {fv}")

                add_log("Heart Disease", "SOM", "Sample Prediction", log_capture.getvalue())
                # Display results to user
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("BMU Coordinates", f"({bmu[0]}, {bmu[1]})")
                    st.metric("Predicted Class", "Heart Disease" if predicted_class == 1 else "Healthy")
                    st.metric(
                        "Actual Class",
                        "Heart Disease" if actual_label == 1 else "Healthy",
                        delta=(
                            "Correct"
                            if predicted_class == actual_label
                            else "Incorrect"
                        ),
                    )
                    st.info(
                        f"The BMU ({bmu[0]}, {bmu[1]}) represents the neuron on the SOM map that best matches this patient's features.")

                with col2:
                    # Show class distribution in neighborhood
                    if total_neighbors > 0:
                        st.subheader("BMU Neighborhood Classes")
                        labels = ["Healthy", "Heart Disease"]
                        values = [
                            neighborhood_counts.get(0, 0),
                            neighborhood_counts.get(1, 0),
                        ]

                        fig, ax = plt.subplots()
                        ax.bar(labels, values, color=["blue", "red"])
                        ax.set_ylabel("Count")
                        ax.set_title("Class Distribution in BMU Neighborhood")
                        st.pyplot(fig)

                        confidence = max(values) / \
                            sum(values) if sum(values) > 0 else 0
                        st.text(f"Prediction Confidence: {confidence:.2%}")
            else:
                st.error("No test samples available or SOM model not trained.")


# --- Main Application Flow ---
st.title("🚀 Neural Network Exploration Studio")
st.markdown("Welcome! Choose a project and a neural network from the sidebar to begin.")
st.markdown("---")

if project_type == "Binary Pattern Recognition (2x2)":
    display_binary_patterns_project()
elif project_type == "Heart Disease Classification":
    display_heart_disease_project()

# --- Execution Logs Display ---
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Execution Logs")
if st.sidebar.button("Clear All Logs", key="clear_logs_button"):
    st.session_state.logs = []
    st.rerun()

with st.sidebar.expander("View Logs", expanded=False):
    if st.session_state.logs:
        for log in st.session_state.logs:  # Already reversed on insert
            st.markdown(
                f"""
            **Timestamp:** {log['timestamp']}
            **Project:** {log['project']}
            **Network:** {log['network']}
            **Action:** {log['action']}
            """
            )
            with st.popover("Details"):
                st.code(log["details"])
            st.markdown("---")
    else:
        st.info("No logs yet.")
