import os
import sqlite3
import urllib.request
import base64
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
# Automatically detect logo file name
if os.path.exists("1200x630wa.png"):
    LOGO_PATH = "1200x630wa.png"
elif os.path.exists("logo.png"):
    LOGO_PATH = "logo.png"
else:
    LOGO_PATH = None

def get_base64_image(image_path):
    """Convert local image to base64 string for direct HTML embedding inside hero banner."""
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_image(LOGO_PATH)

st.set_page_config(
    page_title="School Lost & Found Hub",
    page_icon=LOGO_PATH if LOGO_PATH else "🎒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Injection
st.markdown("""
    <style>
    /* Global Page Styling */
    .main { background-color: #F8FAFC; }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
        display: flex;
        align-items: center;
        gap: 22px;
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0 !important;
    }
    .hero-banner p {
        color: #E0E7FF !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }

    /* Tab Button Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
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

    /* Custom Status Badges & Slot Tags */
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
    .slot-tag {
        background-color: #4F46E5;
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 800;
        display: inline-block;
    }
    .room-tag {
        background-color: #8B5CF6;
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 800;
        display: inline-block;
    }
    hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }
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
# 2. DATABASE INITIALIZATION & SLOT LOGIC
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_number INTEGER,
            holding_location TEXT DEFAULT 'Front Office',
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
    # Database migration support for existing DB files
    c.execute("PRAGMA table_info(items)")
    columns = [col[1] for col in c.fetchall()]
    if "slot_number" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN slot_number INTEGER")
    if "holding_location" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN holding_location TEXT DEFAULT 'Front Office'")
    conn.commit()
    conn.close()

init_db()

def get_lowest_available_slot():
    """Finds the lowest positive slot number not occupied by an active Front Office item."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT slot_number FROM items WHERE is_returned = 0 AND holding_location = 'Front Office' AND slot_number IS NOT NULL")
    occupied_slots = set(row[0] for row in c.fetchall())
    conn.close()

    slot = 1
    while slot in occupied_slots:
        slot += 1
    return slot

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
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

# ---------------------------------------------------------
# 4. HERO HEADER WITH INTEGRATED LOGO
# ---------------------------------------------------------
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height: 95px; border-radius: 12px; background-color: #FFFFFF; padding: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); flex-shrink: 0;" />'
else:
    logo_html = '<span style="font-size: 3.5rem; line-height: 1;">🎒</span>'

st.markdown(f"""
    <div class="hero-banner">
        {logo_html}
        <div>
            <h1>School Lost & Found Hub</h1>
            <p>AI-powered visual search and smart inventory tracking for campus items</p>
        </div>
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
                    c.execute("SELECT id, slot_number, holding_location, item_type, location_found, date_found, notes, image_path, vector_path FROM items WHERE is_returned = 0")
                    active_items = c.fetchall()
                    conn.close()

                    # Threshold set to 0.15 to account for background noise
                    SIMILARITY_THRESHOLD = 0.15
                    matches = []

                    for item in active_items:
                        item_id, slot_num, holding_loc, item_type, loc, date_found, notes, img_path, vec_path = item
                        if vec_path and os.path.exists(vec_path):
                            saved_vec = np.load(vec_path)
                            sim_score = cosine_similarity(query_vec, saved_vec)
                            
                            if sim_score >= SIMILARITY_THRESHOLD:
                                matches.append((sim_score, item))

                    matches.sort(key=lambda x: x[0], reverse=True)

                    st.markdown("#### 🎯 Best Visual Matches")
                    if not matches:
                        st.info("No matching items found. Try uploading a clearer photo or using Keyword Search.")
                    else:
                        cols = st.columns(3)
                        for idx, (score, item) in enumerate(matches[:6]):
                            item_id, slot_num, holding_loc, item_type, loc, date_found, notes, img_path, vec_path = item
                            with cols[idx % 3]:
                                with st.container(border=True):
                                    if os.path.exists(img_path):
                                        st.image(img_path, use_container_width=True)
                                    
                                    if holding_loc == "Front Office" and slot_num:
                                        tag_html = f"<span class='slot-tag'>Office Slot #{slot_num}</span>"
                                    else:
                                        tag_html = f"<span class='room-tag'>🏫 Classroom Pickup</span>"

                                    st.markdown(f"#### {item_type} {tag_html}", unsafe_allow_html=True)
                                    conf_pct = max(0.0, min(1.0, score))
                                    st.progress(float(conf_pct))
                                    st.caption(f"**Match Score:** {score*100:.1f}%")
                                    
                                    st.markdown(f"📍 **Location / Room:** {loc}")
                                    st.markdown(f"📅 **Date Found:** {date_found}")
                                    if notes:
                                        st.markdown(f"📝 **Notes:** {notes}")

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
        c.execute("SELECT id, slot_number, holding_location, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0")
        all_active_items = c.fetchall()
        conn.close()

        keywords = search_text.lower().strip().split()
        filtered_results = []

        for item in all_active_items:
            item_id, slot_num, holding_loc, item_type, loc, date_found, notes, img_path = item
            if category_filter != "All Categories" and item_type != category_filter:
                continue

            combined_text = f"{item_type} {loc} {notes or ''} {holding_loc}".lower()
            if all(kw in combined_text for kw in keywords):
                filtered_results.append(item)

        st.markdown(f"#### 📦 Available Items ({len(filtered_results)} found)")
        
        if not filtered_results:
            st.info("No active inventory items match your criteria.")
        else:
            cols = st.columns(3)
            for idx, item in enumerate(filtered_results):
                item_id, slot_num, holding_loc, item_type, loc, date_found, notes, img_path = item
                with cols[idx % 3]:
                    with st.container(border=True):
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        
                        if holding_loc == "Front Office" and slot_num:
                            tag_html = f"<span class='slot-tag'>Office Slot #{slot_num}</span>"
                        else:
                            tag_html = f"<span class='room-tag'>🏫 Classroom Pickup</span>"

                        st.markdown(f"#### {item_type} {tag_html}", unsafe_allow_html=True)
                        st.markdown(f"📍 **Location / Room:** {loc}")
                        st.markdown(f"📅 **Date Found:** {date_found}")
                        if notes:
                            st.markdown(f"📝 **Details:** {notes}")

# =========================================================
# TAB 2: REPORT / REGISTER ITEM
# =========================================================
with tab2:
    st.markdown("### 📥 Register a Found Item")
    
    holding_choice = st.radio(
        "Current Item Storage Location",
        ["🏢 Sent to Front Office (Assign Storage Tag)", "🏫 Kept in Classroom (Student picks up from room)"],
        horizontal=True
    )

    if "Front Office" in holding_choice:
        next_slot = get_lowest_available_slot()
        st.info(f"🏷️ **Auto-Assigned Office Tag:** This item will be assigned to **Bin / Hook Slot #{next_slot}** in the front office.")
    else:
        st.success("📍 **Classroom Pickup:** No office slot tag required. Students will see the classroom location directly in search results.")

    with st.container(border=True):
        with st.form("register_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                item_type = st.selectbox("Item Category", ["Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
                location_found = st.text_input("Location / Room Found", placeholder="e.g., Room 204, Gymnasium bleachers, Library")
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
                    st.error("Please specify the location or room where the item was found.")
                else:
                    if "Front Office" in holding_choice:
                        assigned_slot = get_lowest_available_slot()
                        holding_loc_val = "Front Office"
                    else:
                        assigned_slot = None
                        holding_loc_val = "Classroom"

                    file_ext = os.path.splitext(item_image.name)[1]
                    
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO items (slot_number, holding_location, item_type, location_found, date_found, notes, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (assigned_slot, holding_loc_val, item_type, location_found, date_found, notes, "")
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
                    if assigned_slot:
                        st.success(f"Item successfully registered! Attach Tag **Slot #{assigned_slot}** and place in office bin.")
                    else:
                        st.success(f"Item successfully registered! It is logged as available for pickup in **{location_found}**.")

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
        c.execute("SELECT id, slot_number, holding_location, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0 ORDER BY holding_location ASC, slot_number ASC")
        active_records = c.fetchall()

        c.execute("SELECT id, slot_number, holding_location, item_type, location_found, date_found, notes, image_path FROM items WHERE is_returned = 1 ORDER BY id DESC")
        returned_records = c.fetchall()
        conn.close()

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Items Logged", len(active_records) + len(returned_records))
        m_col2.metric("Active Unclaimed", len(active_records))
        m_col3.metric("Returned to Owner", len(returned_records))

        st.markdown("---")
        st.markdown("#### 🏷️ Active Unclaimed Inventory")
        
        if not active_records:
            st.info("No active unclaimed items in inventory.")
        else:
            for record in active_records:
                rec_id, rec_slot, rec_holding, rec_type, rec_loc, rec_date, rec_notes, rec_img = record
                
                with st.container(border=True):
                    col_img, col_info, col_action = st.columns([1.2, 2.8, 1.5])
                    
                    with col_img:
                        if os.path.exists(rec_img):
                            st.image(rec_img, use_container_width=True)
                    
                    with col_info:
                        if rec_holding == "Front Office" and rec_slot:
                            tag_html = f"<span class='slot-tag'>Office Slot #{rec_slot}</span>"
                        else:
                            tag_html = f"<span class='room-tag'>🏫 Classroom Pickup</span>"

                        st.markdown(f"#### {rec_type} {tag_html}", unsafe_allow_html=True)
                        st.markdown(f"📍 **Location / Room:** {rec_loc}")
                        st.markdown(f"📅 **Date Found:** {rec_date}")
                        if rec_notes:
                            st.caption(f"Notes: {rec_notes}")
                    
                    with col_action:
                        btn_label = f"Mark Returned (Frees Slot #{rec_slot})" if rec_slot else "Mark Returned"
                        if st.button(btn_label, key=f"ret_{rec_id}", use_container_width=True):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            c.execute("UPDATE items SET is_returned = 1 WHERE id = ?", (rec_id,))
                            conn.commit()
                            conn.close()
                            st.rerun()

        if returned_records:
            st.markdown("---")
            st.markdown("#### ✅ Archive (Returned Items)")
            for record in returned_records:
                rec_id, rec_slot, rec_holding, rec_type, rec_loc, rec_date, rec_notes, rec_img = record
                with st.container(border=True):
                    loc_info = f"Slot #{rec_slot}" if rec_slot else f"Classroom ({rec_loc})"
                    st.markdown(f"**Item #{rec_id} ({rec_type})** — Formerly {loc_info} — Returned to Owner")
    elif passcode:
        st.error("Incorrect passcode.")
