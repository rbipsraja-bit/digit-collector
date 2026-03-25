import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Digit Trainer & Predictor")

if 'db' not in st.session_state:
    st.session_state.db = []
if 'current_digit' not in st.session_state:
    st.session_state.current_digit = 0
if 'mode' not in st.session_state:
    st.session_state.mode = "collect" # Modes: "collect" or "predict"

# --- MODE 1: DATA COLLECTION ---
if st.session_state.mode == "collect":
    st.title(f"Step 1: Draw the number {st.session_state.current_digit}")
    st.write(f"Goal: Collect 0-9 first. Current samples: {len(st.session_state.db)}")

    canvas_collect = st_canvas(
        stroke_width=18, stroke_color="#FFF", background_color="#000",
        height=280, width=280, drawing_mode="freedraw", key=f"c_{st.session_state.current_digit}"
    )

    if st.button("Save & Next"):
        if canvas_collect.image_data is not None:
            img = Image.fromarray(canvas_collect.image_data.astype('uint8')).convert('L').resize((28, 28))
            pixels = np.array(img).flatten().tolist()
            st.session_state.db.append([st.session_state.current_digit] + pixels)
            
            if st.session_state.current_digit < 9:
                st.session_state.current_digit += 1
            else:
                st.session_state.mode = "predict" # Switch to prediction mode
            st.rerun()

# --- MODE 2: RANDOM DETECTION ---
else:
    st.title("Step 2: Draw a Random Number!")
    st.write("The system will now try to detect what you draw based on your previous inputs.")

    canvas_predict = st_canvas(
        stroke_width=18, stroke_color="#FFF", background_color="#000",
        height=280, width=280, drawing_mode="freedraw", key="predict_canvas"
    )

    if st.button("Detect Number"):
        if canvas_predict.image_data is not None and len(st.session_state.db) > 0:
            # Prepare Training Data
            data = np.array(st.session_state.db)
            X_train = data[:, 1:] # Pixels
            y_train = data[:, 0]  # Labels
            
            # Simple AI Model
            model = KNeighborsClassifier(n_neighbors=1)
            model.fit(X_train, y_train)
            
            # Process Current Drawing
            img = Image.fromarray(canvas_predict.image_data.astype('uint8')).convert('L').resize((28, 28))
            current_pixels = np.array(img).flatten().reshape(1, -1)
            
            # Predict
            prediction = model.predict(current_pixels)
            st.header(f"Detected Number: {int(prediction[0])}")
            st.confetti()
        else:
            st.warning("Draw something first!")

    if st.button("Back to Collecting"):
        st.session_state.mode = "collect"
        st.session_state.current_digit = 0
        st.rerun()
