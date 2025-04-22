import streamlit as st
import numpy as np
from mlp import MLP  # Import your neural network class

st.title("Neural Network Binary Classifier")

# Initialize your model (only once)
if 'model' not in st.session_state:
    st.session_state.model = MLP(input_size=4, hidden_size=8, output_size=1)
    st.session_state.trained = False

# Training section
with st.expander("Training", expanded=not st.session_state.trained):
    if st.button("Train Model") or not st.session_state.trained:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Training code here
        epochs = 1500
        for epoch in range(epochs):
            loss = st.session_state.model.train_epoch()  # Your training function

            # Update progress and status every 100 epochs
            if epoch % 100 == 0:
                progress_bar.progress(epoch / epochs)
                status_text.text(f"Epoch {epoch}, Loss: {loss:.4f}")

        st.session_state.trained = True
        st.success("Training complete!")

        # Display accuracy metrics and plots
        st.write("Rule verification results and accuracy metrics will appear here")
        # Call your accuracy evaluation functions
        # Use st.pyplot() instead of plt.show()

# Prediction section
st.header("Test the Model")
st.write("Enter 4 binary values")

# Create 4 number inputs for the binary values
col1, col2, col3, col4 = st.columns(4)
with col1:
    b1 = st.number_input("Value 1", min_value=0, max_value=1, step=1)
with col2:
    b2 = st.number_input("Value 2", min_value=0, max_value=1, step=1)
with col3:
    b3 = st.number_input("Value 3", min_value=0, max_value=1, step=1)
with col4:
    b4 = st.number_input("Value 4", min_value=0, max_value=1, step=1)

if st.button("Predict"):
    if st.session_state.trained:
        input_data = np.array([b1, b2, b3, b4])
        prediction = st.session_state.model.predict(
            input_data)  # Your prediction function
        st.write(f"Prediction: {prediction}")
    else:
        st.warning("Please train the model first!")
