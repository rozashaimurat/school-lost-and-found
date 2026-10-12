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
# 1. PAGE CONFIGURATION & CUSTOM CSS STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="School Lost & Found Hub",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Injection for Modern UI
st.markdown("""
    <style>
    /* Global Page Styling */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
    }
    .hero-banner p {
        color: #E0E7FF !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }

    /* Tab Button Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }

    /* Card Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
    }

    /* Custom Badges */
    .badge-active {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-returned {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Subtle Divider Styling */
    hr {
        border-color: #E2E8F0 !important;
        margin: 1.5rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "lost_and_found.db"
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectors"
MODEL_PATH = "mobilenet_v3_small.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

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
    options = vision.ImageEmbedderOptions(
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
# 4. HERO HEADER & NAVIGATION
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <h1>🎒 School Lost & Found Hub</h1>
        <p>AI-powered visual search and smart inventory tracking for campus items</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🔍 Search Lost Items", 
    "📷 Report / Register Item", 
    "🔐 Staff Console"
])

# =========================================================
# TAB 1: SEARCH LOST ITEMS
# =========================================================
with tab1:
    st.markdown("### 🔎 Find Your Missing Item")
    st.caption("Choose how you want to locate your lost property below.")
    
    search_mode = st.radio(
        "Search Mode",
        ["📷 Upload a Photo (AI Visual Search)", "✏️ Type Description & Filter"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # --- MODE 1: VISUAL AI SEARCH ---
    if search_mode == "📷 Upload a Photo (AI Visual Search)":
        with st.container(border=True):
            st.markdown("##### 📸 AI Camera Match")
            uploaded_query = st.file_uploader(
                "Upload a photo of your lost item (or a similar picture from the web)", 
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_query is not None:
                st.image(uploaded_query, caption="Your Uploaded Photo", width=220)
                
                with st.spinner("Extracting AI features & calculating vector similarity..."):
                    query_vec = extract_mediapipe_vector(uploaded_query)
                
                if query_vec is not None:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("SELECT id, item_type, location_found, date_found, notes, image_path, vector_path FROM items WHERE is_returned = 0")
                    active_items = c.fetchall()
                    conn.close()

                    # Set 35% minimum confidence threshold
                    SIMILARITY_THRESHOLD = 0.35

                    matches = []
                    for item in active_items:
                        item_id, item_type, loc, date_found, notes, img_path, vec_path = item
                        if vec_path and os.path.exists(vec_path):
                            saved_vec = np.load(vec_path)
                            sim_score = cosine_similarity(query_vec, saved_vec)
                            
                            if sim_score >= SIMILARITY_THRESHOLD:
                                matches.append((sim_score, item))

                    matches.sort(key=lambda x: x[0], reverse=True)

                    st.markdown("#### 🎯 Best Visual Matches")
                    if not matches:
                        st.info("No matching items found with sufficient visual similarity (minimum 35% match required). Try uploading a clearer photo or using Keyword Search.")
                    else:
                        cols = st.columns(3)
                        for idx, (score, item) in enumerate(matches[:6]):
                            item_id, item_type, loc, date_found, notes, img_path, vec_path = item
                            with cols[idx % 3]:
                                with st.container(border=True):
                                    if os.path.exists(img_path):
                                        st.image(img_path, use_container_width=True)
                                    
                                    st.markdown(f"#### {item_type}")
                                    conf_pct = max(0.0, min(1.0, score))
                                    st.progress(float(conf_pct))
                                    st.caption(f"**Match Score:** {score*100:.1f}%")
                                    
                                    st.markdown(f"📍 **Found at:** {loc}")
                                    st.markdown(f"📅 **Date:** {date_found}")
                                    if notes:
                                        st.markdown(f"📝 **Notes:** {notes}")
                                    st.caption(f"Ref ID: #{item_id}")

    # --- MODE 2: TEXT & FILTER SEARCH ---
    else:
        with st.container(border=True):
            st.markdown("##### 🔍 Keyword & Category Search")
            col1, col2 = st.columns(2)
            with col1:
                category_filter = st.selectbox(
                    "Category Filter", 
                    ["All Categories", "Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"]
                )
            with col2:
                search_text = st.text_input(
                    "Keyword Search", 
                    placeholder="e.g., navy blue, hydroflask, gym, room 104"
                )

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0")
        all_active_items = c.fetchall()
        conn.close()

        keywords = search_text.lower().strip().split()
        filtered_results = []

        for item in all_active_items:
            item_id, item_type, loc, date_found, notes, img_path = item
            
            if category_filter != "All Categories" and item_type != category_filter:
                continue

            combined_text = f"{item_type} {loc} {notes or ''}".lower()

            if all(kw in combined_text for kw in keywords):
                filtered_results.append(item)

        st.markdown(f"#### 📦 Available Items ({len(filtered_results)} found)")
        
        if not filtered_results:
            st.info("No active inventory items match your criteria.")
        else:
            cols = st.columns(3)
            for idx, item in enumerate(filtered_results):
                item_id, item_type, loc, date_found, notes, img_path = item
                with cols[idx % 3]:
                    with st.container(border=True):
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.markdown(f"#### {item_type}")
                        st.markdown(f"📍 **Location:** {loc}")
                        st.markdown(f"📅 **Date:** {date_found}")
                        if notes:
                            st.markdown(f"📝 **Details:** {notes}")
                        st.caption(f"Ref ID: #{item_id}")

# =========================================================
# TAB 2: REPORT / REGISTER ITEM
# =========================================================
with tab2:
    st.markdown("### 📥 Register a Found Item")
    st.caption("Found an item on campus? Fill out the details below to log it into the lost and found system.")

    with st.container(border=True):
        with st.form("register_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                item_type = st.selectbox("Item Category", ["Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
                location_found = st.text_input("Location Found", placeholder="e.g., Gymnasium bleachers, Library, Room 204")
            with col_b:
                date_found = st.date_input("Date Found").strftime("%Y-%m-%d")
                notes = st.text_area("Description & Unique Identifiers", placeholder="e.g., Blue stainless steel bottle with stickers")

            item_image = st.file_uploader("Item Photo (Used for AI feature matching)", type=["jpg", "jpeg", "png"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Submit Item to Inventory", use_container_width=True)

            if submitted:
                if not item_image:
                    st.error("Please upload a photo of the item.")
                elif not location_found.strip():
                    st.error("Please specify where the item was found.")
                else:
                    file_ext = os.path.splitext(item_image.name)[1]
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO items (item_type, location_found, date_found, notes, image_path) VALUES (?, ?, ?, ?, ?)",
                        (item_type, location_found, date_found, notes, "")
                    )
                    new_id = c.lastrowid
                    
                    img_path = os.path.join(UPLOAD_DIR, f"item_{new_id}{file_ext}")
                    with open(img_path, "wb") as f:
                        f.write(item_image.getbuffer())

                    vec_path = os.path.join(VECTOR_DIR, f"vector_{new_id}.npy")
                    vector = extract_mediapipe_vector(item_image)
                    
                    if vector is not None:
                        np.save(vec_path, vector)
                    else:
                        vec_path = None

                    c.execute("UPDATE items SET image_path = ?, vector_path = ? WHERE id = ?", (img_path, vec_path, new_id))
                    conn.commit()
                    conn.close()

                    st.balloons()
                    st.success(f"Item successfully registered! Reference ID: #{new_id}")

# =========================================================
# TAB 3: STAFF INVENTORY MANAGEMENT
# =========================================================
with tab3:
    st.markdown("### 🔐 Staff Management Console")
    st.caption("Authorized staff members can update item statuses and confirm returns.")
    
    passcode = st.text_input("Enter Staff Passcode", type="password")
    
    if passcode == "1234":
        st.success("Authenticated as Staff")
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, item_type, location_found, date_found, notes, is_returned, image_path FROM items ORDER BY id DESC")
        all_records = c.fetchall()
        conn.close()

        total_items = len(all_records)
        active_cnt = sum(1 for r in all_records if r[5] == 0)
        returned_cnt = sum(1 for r in all_records if r[5] == 1)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Items Logged", total_items)
        m_col2.metric("Active Unclaimed", active_cnt)
        m_col3.metric("Returned to Owner", returned_cnt)

        st.markdown("---")

        if not all_records:
            st.info("No records present in the database.")
        else:
            for record in all_records:
                rec_id, rec_type, rec_loc, rec_date, rec_notes, rec_returned, rec_img = record
                
                with st.container(border=True):
                    col_img, col_info, col_action = st.columns([1.2, 2.8, 1.5])
                    
                    with col_img:
                        if os.path.exists(rec_img):
                            st.image(rec_img, use_container_width=True)
                    
                    with col_info:
                        if rec_returned == 1:
                            badge_html = '<span class="badge-returned">✅ Returned</span>'
                        else:
                            badge_html = '<span class="badge-active">🟡 Active Unclaimed</span>'
                        
                        st.markdown(f"#### Item #{rec_id}: {rec_type} {badge_html}", unsafe_allow_html=True)
                        st.markdown(f"📍 **Location:** {rec_loc}")
                        st.markdown(f"📅 **Date Found:** {rec_date}")
                        if rec_notes:
                            st.caption(f"Notes: {rec_notes}")
                    
                    with col_action:
                        if rec_returned == 0:
                            if st.button("Mark as Returned", key=f"ret_{rec_id}", use_container_width=True):
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                c.execute("UPDATE items SET is_returned = 1 WHERE id = ?", (rec_id,))
                                conn.commit()
                                conn.close()
                                st.rerun()
                        else:
                            st.markdown("---")
                            st.caption("Status: Complete")
    elif passcode:
        st.error("Incorrect passcode.")
