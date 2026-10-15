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
# 1. PAGE CONFIGURATION & CALM NAVY/WHITE CSS THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="LISA Lost & Found",
    page_icon="🎒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Calm white background, navy typography, subtle borders
st.markdown("""
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Header Bar */
    .header-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #64748B;
        font-size: 0.9rem;
        margin: 2px 0 0 0;
    }

    /* Item Cards */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"]:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }

    /* Status Badges */
    .badge-office {
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-classroom {
        background-color: #F5F3FF;
        color: #6D28D9;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-match {
        background-color: #ECFDF5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Clean Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        border-radius: 8px 8px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1E40AF !important;
        border-bottom: 2px solid #1E40AF !important;
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "lost_and_found.db"
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectors"
MODEL_PATH = "mobilenet_v3_small.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"
APP_URL = "https://school-lost-and-found-exey2emwyf5hcxwktbyskv.streamlit.app/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Preparing image matching system..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------------------------------------
# 2. DATABASE INITIALIZATION & SLOT RECYCLING
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_number INTEGER,
            holding_location TEXT DEFAULT 'Front Office',
            holding_room_details TEXT,
            item_type TEXT NOT NULL,
            title TEXT,
            location_found TEXT NOT NULL,
            date_found TEXT NOT NULL,
            notes TEXT,
            image_path TEXT NOT NULL,
            vector_path TEXT,
            is_returned INTEGER DEFAULT 0,
            claimed_by TEXT,
            returned_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Database migration checks
    c.execute("PRAGMA table_info(items)")
    columns = [col[1] for col in c.fetchall()]
    if "slot_number" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN slot_number INTEGER")
    if "holding_location" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN holding_location TEXT DEFAULT 'Front Office'")
    if "holding_room_details" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN holding_room_details TEXT")
    if "title" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN title TEXT")
    if "claimed_by" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN claimed_by TEXT")
    if "returned_at" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN returned_at TIMESTAMP")
    conn.commit()
    conn.close()

init_db()

def get_lowest_available_slot():
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
    except Exception:
        return None

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "show_photo_search" not in st.session_state:
    st.session_state.show_photo_search = False
if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None
if "staff_auth" not in st.session_state:
    st.session_state.staff_auth = False

# ---------------------------------------------------------
# 5. HEADER BAR & STAFF QUICK LINK
# ---------------------------------------------------------
head_col1, head_col2 = st.columns([4, 1])

with head_col1:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px;'>
            <span style='font-size: 2rem;'>🎒</span>
            <div>
                <h2 style='margin: 0; color: #0F172A; font-weight: 800; font-size: 1.5rem;'>LISA Lost & Found</h2>
                <p style='margin: 0; color: #64748B; font-size: 0.88rem;'>Find it. Pick it up.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with head_col2:
    st.markdown("<div style='text-align: right; padding-top: 6px;'>", unsafe_allow_html=True)
    if st.button("🔐 Staff Sign In" if not st.session_state.staff_auth else "✅ Staff Active", key="header_staff_btn"):
        st.session_state.active_tab_override = "Staff Console"
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 12px 0 20px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

# Main Navigation
tab_gallery, tab_report, tab_staff = st.tabs([
    "📦 Found Items Gallery", 
    "➕ Report a Found Item", 
    "🔐 Staff Console"
])

# =========================================================
# TAB 1: FOUND ITEMS GALLERY (HOMEPAGE & CENTERPIECE)
# =========================================================
with tab_gallery:
    
    # 1. Prominent Single Search Field
    search_query = st.text_input(
        "Search items", 
        placeholder="🔍 Search by item, color, brand, or location (e.g. Blue Stanley, Gym, Jacket)",
        label_visibility="collapsed"
    )

    # 2. Controls & Filter Row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.2, 1.2, 0.8])
    
    with ctrl_col1:
        if st.button("📷 Search with a Photo", use_container_width=True):
            st.session_state.show_photo_search = not st.session_state.show_photo_search

    with ctrl_col2:
        cat_filter = st.selectbox(
            "Category", 
            ["All Categories", "Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"],
            label_visibility="collapsed"
        )

    with ctrl_col3:
        loc_filter = st.selectbox(
            "Pickup Location", 
            ["All Locations", "Front Office", "Classroom Pickup"],
            label_visibility="collapsed"
        )

    with ctrl_col4:
        if st.button("Clear filters", use_container_width=True):
            st.session_state.show_photo_search = False
            st.rerun()

    # 3. Secondary Photo Search Drawer (Expands when clicked)
    photo_query_vec = None
    if st.session_state.show_photo_search:
        with st.container(border=True):
            st.markdown("##### 📸 Photo Matching")
            st.caption("Upload a photo or snap a picture of your lost item to find visually similar matches.")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                uploaded_img = st.file_uploader("Upload reference photo", type=["jpg", "png", "jpeg"])
            with p_col2:
                camera_img = st.camera_input("Or snap live photo")
            
            selected_photo = uploaded_img or camera_img
            if selected_photo:
                with st.spinner("Looking for similar items…"):
                    photo_query_vec = extract_mediapipe_vector(selected_photo)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Detail View Overlay (Finish Student's Journey)
    if st.session_state.selected_item_id is not None:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, slot_number, holding_location, holding_room_details, item_type, title, location_found, date_found, notes, image_path FROM items WHERE id = ?", (st.session_state.selected_item_id,))
        det_item = c.fetchone()
        conn.close()

        if det_item:
            d_id, d_slot, d_holding, d_room_det, d_type, d_title, d_loc, d_date, d_notes, d_img = det_item
            display_title = d_title if d_title else f"{d_type}"

            with st.container(border=True):
                st.markdown("### 📋 Item Details & Pickup Instructions")
                
                det_col1, det_col2 = st.columns([1, 1.5])
                with det_col1:
                    if os.path.exists(d_img):
                        st.image(d_img, use_container_width=True)
                
                with det_col2:
                    st.markdown(f"## {display_title}")
                    
                    if d_holding == "Front Office":
                        pickup_label = f"🏢 Front Office (Slot #{d_slot})" if d_slot else "🏢 Front Office"
                        instruction_text = "Visit the **Front Office** during passing period, lunch, or after dismissal and describe your item to staff."
                    else:
                        room_text = d_room_det or d_loc
                        pickup_label = f"🏫 Classroom Pickup ({room_text})"
                        instruction_text = f"Visit **{room_text}** before school or after dismissal to collect your item from the teacher."

                    st.markdown(f"📍 **Found at:** {d_loc} on {d_date}")
                    st.markdown(f"📦 **Pick up at:** `{pickup_label}`")
                    if d_notes:
                        st.markdown(f"📝 **Identifying Notes:** {d_notes}")
                    
                    st.info(f"💡 **How to collect:**\n\n{instruction_text}")

                    if st.button("Close Details", use_container_width=False):
                        st.session_state.selected_item_id = None
                        st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)

    # 5. Fetch Active Items & Apply Search Filters
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, slot_number, holding_location, holding_room_details, item_type, title, location_found, date_found, notes, image_path, vector_path FROM items WHERE is_returned = 0 ORDER BY id DESC")
    active_items = c.fetchall()
    conn.close()

    display_list = []

    # Filter logic
    for item in active_items:
        i_id, i_slot, i_holding, i_room_det, i_type, i_title, i_loc, i_date, i_notes, i_img, i_vec_path = item
        
        # Category filter
        if cat_filter != "All Categories" and i_type != cat_filter:
            continue

        # Location filter
        if loc_filter == "Front Office" and i_holding != "Front Office":
            continue
        if loc_filter == "Classroom Pickup" and i_holding == "Front Office":
            continue

        # Text search query
        if search_query.strip():
            kw = search_query.lower().strip()
            combined_info = f"{i_type} {i_title or ''} {i_loc} {i_notes or ''} {i_holding}".lower()
            if kw not in combined_info:
                continue

        # Photo similarity matching
        sim_score = 0.0
        if photo_query_vec is not None and i_vec_path and os.path.exists(i_vec_path):
            saved_vec = np.load(i_vec_path)
            sim_score = cosine_similarity(photo_query_vec, saved_vec)
            if sim_score < 0.15:  # Similarity threshold
                continue

        display_list.append((sim_score, item))

    # Sort by photo score if photo vector is active
    if photo_query_vec is not None:
        display_list.sort(key=lambda x: x[0], reverse=True)

    # 6. Render Item Cards Centerpiece
    st.markdown(f"#### Found Items ({len(display_list)})")

    if not display_list:
        st.info("No found items match your search. Try clearing filters or searching for general keywords.")
    else:
        cols = st.columns(3)
        for idx, (score, item) in enumerate(display_list):
            i_id, i_slot, i_holding, i_room_det, i_type, i_title, i_loc, i_date, i_notes, i_img, i_vec_path = item
            card_title = i_title if i_title else f"{i_type}"
            
            with cols[idx % 3]:
                with st.container(border=True):
                    # Consistent image aspect ratio preview
                    if os.path.exists(i_img):
                        st.image(i_img, use_container_width=True)
                    
                    # Pickup location badge
                    if i_holding == "Front Office" and i_slot:
                        badge_html = f"<span class='badge-office'>Pick up: Front Office (Slot #{i_slot})</span>"
                    else:
                        room_disp = i_room_det if i_room_det else i_loc
                        badge_html = f"<span class='badge-classroom'>Pick up: {room_disp}</span>"

                    st.markdown(f"**{card_title}**")
                    st.caption(f"Found {i_date} · {i_loc}")
                    st.markdown(badge_html, unsafe_allow_html=True)

                    if photo_query_vec is not None and score > 0:
                        st.markdown(f"<span class='badge-match'>Visually similar ({int(score*100)}%)</span>", unsafe_allow_html=True)

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("View details", key=f"view_btn_{i_id}", use_container_width=True):
                        st.session_state.selected_item_id = i_id
                        st.rerun()

# =========================================================
# TAB 2: REPORT A FOUND ITEM (SIMPLIFIED & LOGICAL)
# =========================================================
with tab_report:
    st.markdown("### Report a Found Item")
    st.caption("Help a student reunite with their lost property by logging what you found below.")

    with st.container(border=True):
        # Step 1: Add a photo
        st.markdown("##### 1. Add a Photo")
        rep_photo_source = st.radio("Photo Input", ["📁 Upload Image File", "📷 Take Photo with Camera"], horizontal=True, label_visibility="collapsed")
        
        item_photo = None
        if rep_photo_source == "📁 Upload Image File":
            item_photo = st.file_uploader("Upload clear photo of the item", type=["jpg", "jpeg", "png"])
        else:
            item_photo = st.camera_input("Snap picture of found item")

        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

        # Step 2: Describe the item
        st.markdown("##### 2. Describe the Item")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            item_title = st.text_input("Short Item Title", placeholder="e.g. Blue Stanley Water Bottle, Black North Face Jacket")
            item_category = st.selectbox("Category", ["Water Bottle", "Jacket", "Lunch Box", "Backpack", "Other"])
        with col_r2:
            item_notes = st.text_area("Identifying Details / Notes", placeholder="e.g. Has a pink sticker on the bottom, slightly scratched lid")

        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

        # Step 3: Where and when it was found
        st.markdown("##### 3. Enter Where and When It Was Found")
        col_r3, col_r4 = st.columns(2)
        with col_r3:
            location_found = st.text_input("Location Found", placeholder="e.g. Gym, Library, Hallway by Room 104")
        with col_r4:
            date_found = st.date_input("Date Found").strftime("%Y-%m-%d")

        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

        # Step 4: Where it is being held (Separated from Location Found)
        st.markdown("##### 4. Enter Where It Is Being Held")
        holding_choice = st.radio(
            "Holding Location",
            ["🏢 Front Office (Assigns automatic bin/slot tag)", "🏫 Kept in Classroom / Specific Room"],
            horizontal=True
        )

        room_details = ""
        if "Classroom" in holding_choice:
            room_details = st.text_input("Classroom / Room Number for Pickup", placeholder="e.g. Room 211")

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 5: Save found item
        if st.button("Save Found Item", use_container_width=True):
            if not item_photo:
                st.error("Please upload or snap a photo of the item first.")
            elif not location_found.strip():
                st.error("Please enter where the item was found.")
            elif "Classroom" in holding_choice and not room_details.strip():
                st.error("Please enter the room number where the item is being held.")
            else:
                if "Front Office" in holding_choice:
                    assigned_slot = get_lowest_available_slot()
                    holding_loc_val = "Front Office"
                else:
                    assigned_slot = None
                    holding_loc_val = "Classroom Pickup"

                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("""
                    INSERT INTO items (slot_number, holding_location, holding_room_details, item_type, title, location_found, date_found, notes, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (assigned_slot, holding_loc_val, room_details.strip(), item_category, item_title.strip(), location_found.strip(), date_found, item_notes.strip(), ""))
                new_id = c.lastrowid

                # Save image file
                img_path = os.path.join(UPLOAD_DIR, f"item_{new_id}.jpg")
                with open(img_path, "wb") as f:
                    f.write(item_photo.getbuffer())

                # Save vector embedding for AI search
                vec_path = os.path.join(VECTOR_DIR, f"vector_{new_id}.npy")
                vector = extract_mediapipe_vector(item_photo)
                if vector is not None:
                    np.save(vec_path, vector)
                else:
                    vec_path = None

                c.execute("UPDATE items SET image_path = ?, vector_path = ? WHERE id = ?", (img_path, vec_path, new_id))
                conn.commit()
                conn.close()

                st.success("Item saved! Students can now view and search for it in the gallery.")

# =========================================================
# TAB 3: STAFF CONSOLE
# =========================================================
with tab_staff:
    st.markdown("### 🔐 Staff Console")
    st.caption("Authorized staff and teachers can manage unclaimed items, record returns, and view claim history.")

    staff_code = st.text_input("Enter Staff Passcode", type="password")
    if staff_code == "1234":
        st.session_state.staff_auth = True
        st.success("Authenticated as Staff")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, slot_number, holding_location, holding_room_details, item_type, title, location_found, date_found, notes, image_path FROM items WHERE is_returned = 0 ORDER BY id DESC")
        staff_active = c.fetchall()

        c.execute("SELECT id, slot_number, holding_location, holding_room_details, item_type, title, location_found, date_found, notes, image_path, claimed_by, returned_at FROM items WHERE is_returned = 1 ORDER BY returned_at DESC")
        staff_returned = c.fetchall()
        conn.close()

        st.markdown(f"#### Unclaimed Items ({len(staff_active)})")
        
        for item in staff_active:
            s_id, s_slot, s_holding, s_room_det, s_type, s_title, s_loc, s_date, s_notes, s_img = item
            s_display_title = s_title if s_title else s_type

            with st.container(border=True):
                sc1, sc2, sc3 = st.columns([1, 2.5, 1.5])
                with sc1:
                    if os.path.exists(s_img):
                        st.image(s_img, use_container_width=True)
                with sc2:
                    st.markdown(f"**{s_display_title}**")
                    if s_holding == "Front Office" and s_slot:
                        st.markdown(f"<span class='badge-office'>Office Slot #{s_slot}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge-classroom'>Room: {s_room_det}</span>", unsafe_allow_html=True)
                    st.caption(f"Found at {s_loc} on {s_date}")
                    if s_notes:
                        st.caption(f"Notes: {s_notes}")
                with sc3:
                    claimer = st.text_input("Claimed by Student Name/ID", key=f"claimer_{s_id}")
                    if st.button("Mark Returned", key=f"mark_ret_{s_id}", use_container_width=True):
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                        c.execute("UPDATE items SET is_returned = 1, claimed_by = ?, returned_at = CURRENT_TIMESTAMP WHERE id = ?", (claimer.strip() if claimer.strip() else "Verified Student", s_id))
                        conn.commit()
                        conn.close()
                        st.success(f"Item #{s_id} marked as returned!")
                        st.rerun()

        if staff_returned:
            st.markdown("---")
            st.markdown(f"#### Returned History Archive ({len(staff_returned)})")
            for r_item in staff_returned:
                r_id, r_slot, r_holding, r_room_det, r_type, r_title, r_loc, r_date, r_notes, r_img, r_claimed, r_ret_at = r_item
                st.caption(f"✅ **{r_title or r_type}** — Claimed by `{r_claimed}` on {r_ret_at}")

    elif staff_code:
        st.error("Incorrect passcode.")
