# Replace the plt.show() calls in your mlp.py file with this pattern:

# Import streamlit if not already imported
import streamlit as st

# Instead of:
# plt.show()

# Use:
fig = plt.gcf()  # Get current figure
st.pyplot(fig)   # Display in Streamlit
plt.close(fig)   # Close the figure to free memory
