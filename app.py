import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd

st.title("Handwritten Number Collector")

# Initialize a temporary "database" in the app's memory
if 'db' not in st.secrets and 'db' not in st.session_state:
    st.session_state.db = []

# 1. Drawing Area
canvas_result = st_canvas(
    stroke_width=15, stroke_color="#FFF", background_color="#000",
    height=280, width=280, drawing_mode="freedraw", key="canvas"
)

# 2. Number Label
label = st.selectbox("Which number did you draw?", list(range(1, 10)))

if st.button("Add to List"):
    if canvas_result.image_data is not None:
        # Convert to 28x28 grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = img.resize((28, 28))
        pixels = np.array(img).flatten().tolist()
        
        # Save to temporary session list
        st.session_state.db.append([label] + pixels)
        st.success(f"Saved! Total entries: {len(st.session_state.db)}")

# 3. Download the "Database"
if len(st.session_state.db) > 0:
    df = pd.DataFrame(st.session_state.db)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Collected Data (CSV)", data=csv, file_name="digits_data.csv")
