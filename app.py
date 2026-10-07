import os
import sqlite3
import urllib.request
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION & MODEL SETUP ---
MODEL_PATH = "mobilenet_v3_small.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
DB_FILE = "lost_and_found.db"
IMAGE_DIR = "uploaded_images"

os.makedirs(IMAGE_DIR, exist_ok=True)


@st.cache_resource
def download_model():
    """Download the MediaPipe MobileNetV3 TFLite embedder model if missing."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading MediaPipe Embedder Model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


@st.cache_resource
def get_embedder():
    """Initialize the MediaPipe ImageEmbedder pipeline."""
    model_file = download_model()
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True)
    return vision.ImageEmbedder.create_from_options(options)


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS items
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  item_type
                  TEXT,
                  location_found
                  TEXT,
                  date_found
                  TEXT,
                  notes
                  TEXT,
                  image_path
                  TEXT,
                  vector
                  BLOB,
                  hsv_hist
                  BLOB,
                  is_returned
                  INTEGER
                  DEFAULT
                  0
              )
              ''')
    # Safely migrate existing databases if missing the 'notes' column
    try:
        c.execute("ALTER TABLE items ADD COLUMN notes TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()


init_db()


# --- FEATURE EXTRACTION & SIMILARITY FUNCTIONS ---
def extract_mediapipe_vector(image_bytes):
    """Extract a deep feature vector using MediaPipe."""
    embedder = get_embedder()
    pil_img = Image.open(image_bytes).convert("RGB")
    image_np = np.array(pil_img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    result = embedder.embed(mp_image)
    embedding_values = result.embeddings[0].embedding
    return np.array(embedding_values, dtype=np.float32)


def extract_hsv_histogram(image_bytes):
    """Extract baseline HSV Color Histogram representation."""
    pil_img = Image.open(image_bytes).convert("RGB")
    image_np = np.array(pil_img)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))


def hsv_similarity(hist1, hist2):
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


# --- STREAMLIT UI ---
st.set_page_config(page_title="School Lost & Found System", layout="wide")
st.title("🎒 School Lost & Found Hub")

tabs = st.tabs(["🔎 Search Lost Items", "📸 Report/Register Item", "🔐 Staff Inventory Management"])

# --- TAB 1: SEARCH LOST ITEMS ---
with tabs[0]:
    st.header("Find Your Missing Item")

    search_mode = st.radio(
        "How would you like to search?",
        ["📸 Upload a Photo (AI Visual Search)", "📝 Type Description & Filter"],
        horizontal=True
    )
    st.divider()

    # --- MODE 1: PHOTO SEARCH ---
    if "Upload a Photo" in search_mode:
        uploaded_file = st.file_uploader("Upload a photo of your missing item (or store/past photo)",
                                         type=["jpg", "jpeg", "png"], key="search_upload")

        col1, col2 = st.columns(2)
        with col1:
            search_algorithm = st.radio("Matching Algorithm",
                                        ["MediaPipe Deep Learning Features", "HSV Color Histogram (Baseline)"])
        with col2:
            similarity_threshold = st.slider("Match Confidence Threshold", 0.0, 1.0, 0.60, 0.05)

        if uploaded_file:
            st.image(uploaded_file, caption="Query Photo", width=180)

            if st.button("Search Collection", type="primary"):
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute(
                    "SELECT id, item_type, location_found, date_found, notes, image_path, vector, hsv_hist FROM items WHERE is_returned = 0")
                rows = c.fetchall()
                conn.close()

                if not rows:
                    st.warning("The lost and found database is currently empty.")
                else:
                    matches = []
                    use_mediapipe = "MediaPipe" in search_algorithm

                    if use_mediapipe:
                        query_vec = extract_mediapipe_vector(uploaded_file)
                        for row in rows:
                            db_vec = np.frombuffer(row[6], dtype=np.float32)
                            score = cosine_similarity(query_vec, db_vec)
                            matches.append((score, row))
                    else:
                        query_hist = extract_hsv_histogram(uploaded_file)
                        for row in rows:
                            db_hist = np.frombuffer(row[7], dtype=np.float32)
                            score = hsv_similarity(query_hist, db_hist)
                            matches.append((score, row))

                    matches.sort(key=lambda x: x[0], reverse=True)
                    top_3 = matches[:3]

                    st.subheader("Top Matches")
                    if top_3[0][0] < similarity_threshold:
                        st.error(
                            f"⚠️ No matching item found above threshold `{similarity_threshold:.2%}` (Best match score: `{top_3[0][0]:.2%}`).")
                    else:
                        cols = st.columns(len(top_3))
                        for idx, (score, row) in enumerate(top_3):
                            item_id, item_type, loc, date_found, notes, img_path, _, _ = row
                            with cols[idx]:
                                if os.path.exists(img_path):
                                    st.image(img_path, use_container_width=True)

                                is_match = score >= similarity_threshold
                                badge = "🟢 Top Match" if idx == 0 and is_match else (
                                    "🟡 Candidate" if is_match else "🔴 Low Confidence")

                                st.markdown(f"### {badge}")
                                st.markdown(f"**Similarity Score:** `{score:.2%}`")
                                st.markdown(f"**Category:** {item_type}")
                                st.markdown(f"**Location:** {loc}")
                                st.markdown(f"**Date:** {date_found}")
                                if notes:
                                    st.markdown(f"**Details:** {notes}")
                                st.caption(f"Ref ID: #{item_id}")

    # --- MODE 2: TEXT & FILTER SEARCH ---
    else:
        col1, col2 = st.columns(2)
        with col1:
            category_filter = st.selectbox("Filter by Category",
                                           ["All Categories", "Water Bottle", "Jacket", "Lunch Box", "Backpack",
                                            "Other"])
        with col2:
            search_text = st.text_input("Search Description Keywords",
                                        placeholder="e.g., black, hydroflask, stickers, Nike, room 211")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        query = "SELECT id, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0"
        params = []

        if category_filter != "All Categories":
            query += " AND item_type = ?"
            params.append(category_filter)

        if search_text.strip():
            query += " AND (item_type LIKE ? OR location_found LIKE ? OR notes LIKE ?)"
            term = f"%{search_text.strip()}%"
            params.extend([term, term, term])

        c.execute(query, params)
        results = c.fetchall()
        conn.close()

        st.subheader(f"Matching Items ({len(results)} found)")

        if not results:
            st.info("No active items match your description or filters.")
        else:
            cols = st.columns(3)
            for idx, item in enumerate(results):
                item_id, item_type, loc, date_found, notes, img_path = item
                with cols[idx % 3]:
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    st.markdown(f"### {item_type}")
                    st.markdown(f"**Location:** {loc}")
                    st.markdown(f"**Date Found:** {date_found}")
                    if notes:
                        st.markdown(f"**Details:** {notes}")
                    st.caption(f"Ref ID: #{item_id}")
                    st.divider()

# --- TAB 2: REGISTER / SPOT AN ITEM ---
with tabs[1]:
    st.header("Report or Register a Found Item")
    st.info("Found something on campus? Upload a picture so the owner can locate it!")

    with st.form("register_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            item_type = st.selectbox("Category", ["Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
            location_found = st.text_input("Found Location",
                                           placeholder="e.g., Cafeteria Chair 4, Room 211, Gym Bleachers")
        with col2:
            date_found = st.date_input("Date Discovered")
            notes = st.text_input("Description Details / Color / Brand",
                                  placeholder="e.g., Black Hydroflask with red stickers, size M blue jacket")

        registered_image = st.file_uploader("Take or Upload Item Photo", type=["jpg", "jpeg", "png"], key="reg_upload")

        submitted = st.form_submit_button("Submit Item to System", type="primary")

        if submitted:
            if not registered_image or not location_found:
                st.error("Please provide both a photo and the location where the item was found.")
            else:
                img_filename = f"{date_found}_{item_type.lower().replace(' ', '_')}_{registered_image.name}"
                save_path = os.path.join(IMAGE_DIR, img_filename)

                with open(save_path, "wb") as f:
                    f.write(registered_image.getbuffer())

                vector = extract_mediapipe_vector(registered_image)
                hsv_hist = extract_hsv_histogram(registered_image)

                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute('''
                          INSERT INTO items (item_type, location_found, date_found, notes, image_path, vector, hsv_hist,
                                             is_returned)
                          VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                          ''', (item_type, location_found, str(date_found), notes, save_path, vector.tobytes(),
                                hsv_hist.tobytes()))
                conn.commit()
                conn.close()

                st.success(f"Item #{c.lastrowid} successfully logged! Students can now search for it.")

# --- TAB 3: STAFF INVENTORY & RETURNS ---
with tabs[2]:
    st.header("Staff Admin Console")

    passcode = st.text_input("Enter Staff Passcode to manage returns:", type="password")

    # Simple admin lock (Passcode: 1234)
    if passcode != "1234":
        st.warning("🔒 Enter passcode `1234` to authorize return sign-offs.")
    else:
        st.success("🔓 Authorized as Front Office Staff")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT id, item_type, location_found, date_found, notes, image_path, is_returned FROM items ORDER BY id DESC")
        all_items = c.fetchall()
        conn.close()

        if not all_items:
            st.info("Inventory is empty.")
        else:
            for item in all_items:
                item_id, item_type, loc, date_found, notes, img_path, is_returned = item
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        if os.path.exists(img_path):
                            st.image(img_path, width=120)
                    with col2:
                        status_badge = "🔴 Active" if is_returned == 0 else "🟢 Returned"
                        st.markdown(f"### #{item_id}: {item_type} ({status_badge})")
                        st.text(f"Location: {loc} | Found on: {date_found}")
                        if notes:
                            st.caption(f"Details: {notes}")
                    with col3:
                        if is_returned == 0:
                            if st.button("Mark as Returned", key=f"return_{item_id}"):
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                c.execute("UPDATE items SET is_returned = 1 WHERE id = ?", (item_id,))
                                conn.commit()
                                conn.close()
                                st.rerun()
                    st.divider()