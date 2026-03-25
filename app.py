import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Smart Digit Learner")

# 1. Initialize States
if 'db' not in st.session_state:
    st.session_state.db = []
if 'mode' not in st.session_state:
    st.session_state.mode = "collect"
if 'current_digit' not in st.session_state:
    st.session_state.current_digit = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_pixels' not in st.session_state:
    st.session_state.last_pixels = None
if 'canvas_step' not in st.session_state:
    st.session_state.canvas_step = 0 # This will be used to refresh the canvas

# Helper function to refresh canvas
def refresh_canvas():
    st.session_state.canvas_step += 1
    st.session_state.last_prediction = None

# --- MODE 1: INITIAL COLLECTION (0-9) ---
if st.session_state.mode == "collect":
    st.title(f"Step 1: Teach me the number {st.session_state.current_digit}")
    
    # The key includes canvas_step so it clears when we increment it
    canvas = st_canvas(
        stroke_width=18, stroke_color="#FFF", background_color="#000", 
        height=280, width=280, key=f"canvas_collect_{st.session_state.canvas_step}"
    )
    
    if st.button("Save & Next"):
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L').resize((28, 28))
            st.session_state.db.append([st.session_state.current_digit] + np.array(img).flatten().tolist())
            
            if st.session_state.current_digit < 9:
                st.session_state.current_digit += 1
            else:
                st.session_state.mode = "predict"
            
            refresh_canvas()
            st.rerun()

# --- MODE 2: PREDICTION & LEARNING ---
else:
    st.title("Step 2: Draw any random number!")
    st.write(f"My Brain Size: {len(st.session_state.db)} examples")
    
    canvas_p = st_canvas(
        stroke_width=18, stroke_color="#FFF", background_color="#000", 
        height=280, width=280, key=f"canvas_predict_{st.session_state.canvas_step}"
    )

    if st.button("Detect Number"):
        if canvas_p.image_data is not None:
            # Train AI
            data = np.array(st.session_state.db)
            model = KNeighborsClassifier(n_neighbors=1).fit(data[:, 1:], data[:, 0])
            
            # Process drawing
            img = Image.fromarray(canvas_p.image_data.astype('uint8')).convert('L').resize((28, 28))
            st.session_state.last_pixels = np.array(img).flatten().tolist()
            
            # Predict
            pred = model.predict([st.session_state.last_pixels])
            st.session_state.last_prediction = int(pred[0])
        else:
            st.warning("Please draw something first!")

    # Feedback Logic
    if st.session_state.last_prediction is not None:
        st.subheader(f"I think that's a: {st.session_state.last_prediction}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                st.session_state.db.append([st.session_state.last_prediction] + st.session_state.last_pixels)
                st.success("Learned!")
                refresh_canvas()
                st.rerun()
        
        with col2:
            correct_label = st.number_input("❌ No, it's actually:", 0, 9)
            if st.button("Teach Me"):
                st.session_state.db.append([correct_label] + st.session_state.last_pixels)
                st.info(f"Fixed! I now know this is a {correct_label}.")
                refresh_canvas()
                st.rerun()

# Download Button
if st.session_state.db:
    st.divider()
    df = pd.DataFrame(st.session_state.db)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download Dataset ({len(df)} entries)", csv, "digits_dataset.csv")
