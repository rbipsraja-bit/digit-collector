import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Digit Collector (0-9)")

# 1. Initialize variables (Counter starts at 0)
if 'current_digit' not in st.session_state:
    st.session_state.current_digit = 0
if 'db' not in st.session_state:
    st.session_state.db = []

st.title(f"Please draw the number: {st.session_state.current_digit}")
st.write(f"Progress: {len(st.session_state.db)} total entries saved.")

# 2. Drawing Area
canvas_result = st_canvas(
    stroke_width=18, 
    stroke_color="#FFFFFF", 
    background_color="#000000",
    height=280, 
    width=280, 
    drawing_mode="freedraw", 
    key=f"canvas_{st.session_state.current_digit}" # Key changes to clear canvas
)

# 3. "Save & Next" Logic
if st.button("Save & Next Number"):
    if canvas_result.image_data is not None:
        # Convert to 28x28 grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = img.resize((28, 28))
        pixels = np.array(img).flatten().tolist()
        
        # Save Label + 784 pixels
        st.session_state.db.append([st.session_state.current_digit] + pixels)
        
        # Increment the digit (Go from 0 -> 9, then loop back to 0)
        if st.session_state.current_digit < 9:
            st.session_state.current_digit += 1
        else:
            st.session_state.current_digit = 0
            st.balloons()
            st.success("Great job! You finished a set of 0-9!")
        
        # Force the app to refresh to the next number
        st.rerun()
    else:
        st.warning("Please draw the number before clicking next!")

# 4. Download Section (Always visible at the bottom)
if len(st.session_state.db) > 0:
    st.divider()
    df = pd.DataFrame(st.session_state.db)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download Dataset ({len(st.session_state.db)} images)",
        data=csv,
        file_name="handwritten_data.csv",
        mime="text/csv",
    )
    
    if st.button("Clear All Data & Restart"):
        st.session_state.db = []
        st.session_state.current_digit = 0
        st.rerun()
