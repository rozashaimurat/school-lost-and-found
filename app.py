import os
import sqlite3
import urllib.request
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# 1. CONFIGURATION & DIRECTORY SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="School Lost & Found Hub",
    page_icon="🎒",
    layout="wide"
)

DB_FILE = "lost_and_found.db"
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectors"
MODEL_PATH = "mobilenet_v3_small.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Automatically download the MediaPipe embedder model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading vision AI model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------------------------------------
# 2. DATABASE INITIALIZATION
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            location_found TEXT NOT NULL,
            date_found TEXT NOT NULL,
            notes TEXT,
            image_path TEXT NOT NULL,
            vector_path TEXT,
            is_returned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 3. AI FEATURE EXTRACTION (MEDIAPIPE)
# ---------------------------------------------------------
@st.cache_resource
def get_embedder():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageEmbedder.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True,
        quantize=False
    )
    return vision.ImageEmbedder.create_from_options(options)

def extract_mediapipe_vector(image_file):
    """Extract 1024-d feature vector using MediaPipe MobileNetV3."""
    try:
        embedder = get_embedder()
        pil_img = Image.open(image_file).convert("RGB")
        image_np = np.array(pil_img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        result = embedder.embed(mp_image)
        embedding_values = result.embeddings[0].embedding
        return np.array(embedding_values, dtype=np.float32)
    except Exception as e:
        st.error(f"Error extracting image features: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two feature vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

# ---------------------------------------------------------
# 4. USER INTERFACE & NAVIGATION
# ---------------------------------------------------------
st.title("🎒 School Lost & Found Hub")

tab1, tab2, tab3 = st.tabs([
    "🔍 Search Lost Items", 
    "📷 Report/Register Item", 
    "🔐 Staff Inventory Management"
])

# =========================================================
# TAB 1: SEARCH LOST ITEMS
# =========================================================
with tab1:
    st.header("Find Your Missing Item")
    
    search_mode = st.radio(
        "How would you like to search?",
        ["📷 Upload a Photo (AI Visual Search)", "✏️ Type Description & Filter"],
        horizontal=True
    )

    # --- MODE 1: VISUAL AI SEARCH ---
    if search_mode == "📷 Upload a Photo (AI Visual Search)":
        uploaded_query = st.file_uploader("Upload a photo of your lost item (or similar image)", type=["jpg", "jpeg", "png"])
        
        if uploaded_query is not None:
            st.image(uploaded_query, caption="Query Image", width=250)
            
            with st.spinner("Extracting AI features & matching against inventory..."):
                query_vec = extract_mediapipe_vector(uploaded_query)
            
            if query_vec is not None:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT id, item_type, location_found, date_found, notes, image_path, vector_path FROM items WHERE is_returned = 0")
                active_items = c.fetchall()
                conn.close()

                matches = []
                for item in active_items:
                    item_id, item_type, loc, date_found, notes, img_path, vec_path = item
                    if vec_path and os.path.exists(vec_path):
                        saved_vec = np.load(vec_path)
                        sim_score = cosine_similarity(query_vec, saved_vec)
                        matches.append((sim_score, item))

                # Sort by similarity score descending
                matches.sort(key=lambda x: x[0], reverse=True)

                st.subheader("Top Visual Matches")
                if not matches:
                    st.info("No active inventory items available to match.")
                else:
                    cols = st.columns(3)
                    for idx, (score, item) in enumerate(matches[:6]):
                        item_id, item_type, loc, date_found, notes, img_path, vec_path = item
                        with cols[idx % 3]:
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            st.markdown(f"### {item_type}")
                            st.progress(float(max(0.0, min(1.0, score))))
                            st.caption(f"Match Confidence: {score*100:.1f}%")
                            st.markdown(f"**Location:** {loc}")
                            st.markdown(f"**Date Found:** {date_found}")
                            if notes:
                                st.markdown(f"**Details:** {notes}")
                            st.caption(f"Ref ID: #{item_id}")
                            st.divider()

    # --- MODE 2: TEXT & FILTER SEARCH ---
    else:
        col1, col2 = st.columns(2)
        with col1:
            category_filter = st.selectbox("Filter by Category", ["All Categories", "Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
        with col2:
            search_text = st.text_input("Search Description Keywords", placeholder="e.g., black, hydroflask, room 211")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0")
        all_active_items = c.fetchall()
        conn.close()

        # Word-by-word keyword matching across all description fields
        keywords = search_text.lower().strip().split()
        filtered_results = []

        for item in all_active_items:
            item_id, item_type, loc, date_found, notes, img_path = item
            
            # Check Category filter
            if category_filter != "All Categories" and item_type != category_filter:
                continue

            # Combine category, location, and notes into one searchable text string
            combined_text = f"{item_type} {loc} {notes or ''}".lower()

            # Ensure all typed keywords exist in the item record
            if all(kw in combined_text for kw in keywords):
                filtered_results.append(item)

        st.subheader(f"Matching Items ({len(filtered_results)} found)")
        
        if not filtered_results:
            st.info("No active items match your description or filters.")
        else:
            cols = st.columns(3)
            for idx, item in enumerate(filtered_results):
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

# =========================================================
# TAB 2: REPORT / REGISTER ITEM
# =========================================================
with tab2:
    st.header("Register Found Item")
    st.write("Found something on campus? Register it here so the owner can claim it!")

    with st.form("register_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            item_type = st.selectbox("Category", ["Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
            location_found = st.text_input("Location Found", placeholder="e.g., Gymnasium bleachers, Room 104")
        with col_b:
            date_found = st.date_input("Date Found").strftime("%Y-%m-%d")
            notes = st.text_area("Additional Details / Description", placeholder="e.g., Navy blue Hydroflask with a sticker")

        item_image = st.file_uploader("Upload Photo of Found Item", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Submit Found Item")

        if submitted:
            if not item_image:
                st.error("Please upload a photo of the item.")
            elif not location_found.strip():
                st.error("Please specify where the item was found.")
            else:
                # 1. Save Image to disk
                file_ext = os.path.splitext(item_image.name)[1]
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO items (item_type, location_found, date_found, notes, image_path) VALUES (?, ?, ?, ?, ?)",
                          (item_type, location_found, date_found, notes, ""))
                new_id = c.lastrowid
                
                img_path = os.path.join(UPLOAD_DIR, f"item_{new_id}{file_ext}")
                with open(img_path, "wb") as f:
                    f.write(item_image.getbuffer())

                # 2. Extract & Save Vector
                vec_path = os.path.join(VECTOR_DIR, f"vector_{new_id}.npy")
                vector = extract_mediapipe_vector(item_image)
                
                if vector is not None:
                    np.save(vec_path, vector)
                else:
                    vec_path = None

                # 3. Update Database with file paths
                c.execute("UPDATE items SET image_path = ?, vector_path = ? WHERE id = ?", (img_path, vec_path, new_id))
                conn.commit()
                conn.close()

                st.success(f"Item successfully registered! (Ref ID: #{new_id})")

# =========================================================
# TAB 3: STAFF INVENTORY MANAGEMENT
# =========================================================
with tab3:
    st.header("Staff Console")
    
    passcode = st.text_input("Enter Passcode to Access Staff Controls", type="password")
    
    if passcode == "1234":
        st.success("Authenticated as Staff")
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, item_type, location_found, date_found, notes, is_returned, image_path FROM items ORDER BY id DESC")
        all_records = c.fetchall()
        conn.close()

        if not all_records:
            st.info("No records in database.")
        else:
            for record in all_records:
                rec_id, rec_type, rec_loc, rec_date, rec_notes, rec_returned, rec_img = record
                
                col_img, col_info, col_action = st.columns([1, 2, 1])
                
                with col_img:
                    if os.path.exists(rec_img):
                        st.image(rec_img, width=120)
                
                with col_info:
                    status_badge = "✅ Returned" if rec_returned == 1 else "🟡 Active"
                    st.markdown(f"**#{rec_id} - {rec_type}** ({status_badge})")
                    st.write(f"📍 **Found at:** {rec_loc} on {rec_date}")
                    if rec_notes:
                        st.caption(f"Notes: {rec_notes}")
                
                with col_action:
                    if rec_returned == 0:
                        if st.button("Mark Returned", key=f"ret_{rec_id}"):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            c.execute("UPDATE items SET is_returned = 1 WHERE id = ?", (rec_id,))
                            conn.commit()
                            conn.close()
                            st.rerun()
                    else:
                        st.write("Item Returned")

                st.divider()
    elif passcode:
        st.error("Incorrect passcode.")
