import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Smart Digit Learner")

# 1. Initialize Database & States
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

# --- MODE 1: INITIAL COLLECTION (0-9) ---
if st.session_state.mode == "collect":
    st.title(f"Step 1: Teach me the number {st.session_state.current_digit}")
    canvas = st_canvas(stroke_width=18, stroke_color="#FFF", background_color="#000", height=280, width=280, key="c1")
    
    if st.button("Save & Next"):
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L').resize((28, 28))
            st.session_state.db.append([st.session_state.current_digit] + np.array(img).flatten().tolist())
            if st.session_state.current_digit < 9:
                st.session_state.current_digit += 1
            else:
                st.session_state.mode = "predict"
            st.rerun()

# --- MODE 2: PREDICTION & LEARNING ---
else:
    st.title("Step 2: Draw any number!")
    st.write(f"My Brain Size: {len(st.session_state.db)} examples")
    
    canvas_p = st_canvas(stroke_width=18, stroke_color="#FFF", background_color="#000", height=280, width=280, key="c2")

    if st.button("Detect Number"):
        # Train AI on current database
        data = np.array(st.session_state.db)
        model = KNeighborsClassifier(n_neighbors=1).fit(data[:, 1:], data[:, 0])
        
        # Process current drawing
        img = Image.fromarray(canvas_p.image_data.astype('uint8')).convert('L').resize((28, 28))
        st.session_state.last_pixels = np.array(img).flatten().tolist()
        st.session_state.last_prediction = int(model.predict([st.session_state.last_pixels])[0])

    # Show prediction and ask for feedback
    if st.session_state.last_prediction is not None:
        st.subheader(f"I think that's a: {st.session_state.last_prediction}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                # If correct, add it to DB to reinforce learning
                st.session_state.db.append([st.session_state.last_prediction] + st.session_state.last_pixels)
                st.success("Awesome! I've learned this style.")
                st.session_state.last_prediction = None
                st.rerun()
        
        with col2:
            correct_label = st.number_input("❌ No, it's actually:", 0, 9)
            if st.button("Teach Me"):
                # If wrong, add it to DB with the CORRECT label
                st.session_state.db.append([correct_label] + st.session_state.last_pixels)
                st.info(f"Fixed! I now know this is a {correct_label}.")
                st.session_state.last_prediction = None
                st.rerun()

# Download Button at the bottom
if st.session_state.db:
    st.divider()
    csv = pd.DataFrame(st.session_state.db).to_csv(index=False).encode('utf-8')
    st.download_button("Download Brain (CSV)", csv, "my_digit_model.csv")
