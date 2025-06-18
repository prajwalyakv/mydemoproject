import streamlit as st
import pandas as pd
import numpy as np

st.title("🎉 Welcome to Your First Streamlit App")

st.write("This is running inside a uv-managed virtual environment!")

# Create a simple DataFrame
data = pd.DataFrame(
    np.random.randn(5, 3),
    columns=["Column A", "Column B", "Column C"]
)

st.write("📊 Here's a random DataFrame:")
st.dataframe(data)

# Add a simple button
if st.button("Say Hello"):
    st.success("👋 Hello from Streamlit!")
