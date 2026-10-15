"""
School Lost & Found Hub - LISA Academy North
Redesigned UI (requires Streamlit >= 1.46 for top navigation).
"""
import base64
import html
import io
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageOps

# =========================================================
# 1. SETTINGS
# =========================================================
DB_FILE = "lost_and_found.db"
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectors"
MODEL_PATH = "mobilenet_v3_small.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_embedder/"
    "mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"
)
APP_URL = "https://school-lost-and-found-exey2emwyf5hcxwktbyskv.streamlit.app/"

CATEGORIES = ["Water bottle", "Jacket", "Lunch box", "Backpack", "Other"]

# Photo-match labels. Tune these after testing with real photos.
# Turn on SHOW_SCORES to see each card's score while you test.
STRONG_MATCH = 0.45
POSSIBLE_MATCH = 0.20
# If nothing reaches POSSIBLE_MATCH, still show this many closest items
# (labeled "Worth a look") so a real match is never hidden completely.
FALLBACK_RESULTS = 3
SHOW_SCORES = False

DONATE_AFTER_DAYS = 30

# Logo: the wide 1200x630 logo is trimmed to a square automatically.
# (If you ever add a square "crest.png", it is used first.)
LOGO_PATH = next(
    (p for p in ["crest.png", "1200x630wa.png", "logo.png"] if os.path.exists(p)),
    None,
)

st.set_page_config(
    page_title="Lost & Found · LISA Academy North",
    page_icon=LOGO_PATH or ":material/backpack:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Setting up photo search…"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def get_staff_passcode():
    """Passcode lives in Streamlit secrets, not in the code."""
    try:
        return st.secrets["STAFF_PASSCODE"]
    except Exception:
        return None


# =========================================================
# 2. STYLES
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp select, .stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Public Sans', 'Segoe UI', system-ui, sans-serif;
}
.block-container { max-width: 1200px; padding-top: 4.5rem; padding-bottom: 4rem; }

/* Page headings */
.lf-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .35rem; color: #0F172A; line-height: 1.15; }
.lf-sub { font-size: 1.1rem; color: #475569; margin: 0 0 1.4rem; max-width: 640px; }
.lf-h2 { font-size: 1.35rem; font-weight: 800; margin: 1.6rem 0 .2rem; color: #0F172A; }
.lf-meta { font-size: .9rem; color: #475569; margin: 0 0 1rem; }

/* Brand block */
.lf-brand { display: flex; align-items: center; gap: 16px; margin-bottom: 2rem; }
.lf-brand img { width: 68px; height: 68px; object-fit: contain; border-radius: 14px; background: #fff; border: 1px solid #E2E8F0; }
.lf-brand .crest { width: 68px; height: 68px; border-radius: 14px; background: #B91C1C; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 18px; }
.lf-brand small { display: block; font-size: 15px; font-weight: 600; color: #475569; letter-spacing: .02em; }
.lf-brand strong { display: block; font-size: 28px; font-weight: 800; color: #0F172A; line-height: 1.15; }

/* Item cards */
.lf-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
.lf-card { background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; }
.lf-card.best { border: 2px solid #4338CA; }
.lf-photo { aspect-ratio: 4 / 3; width: 100%; object-fit: cover; display: block; background: #E2E8F0; }
.lf-nophoto { aspect-ratio: 4 / 3; background: #E2E8F0; color: #475569; display: flex; align-items: center; justify-content: center; font-size: 14px; }
.lf-body { padding: 16px 18px 18px; display: flex; flex-direction: column; gap: 6px; flex-grow: 1; }
.lf-row { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: #475569; }
.lf-row .cat { font-weight: 700; }
.lf-name { margin: 0; font-size: 1.1rem; font-weight: 700; color: #0F172A; line-height: 1.3; }
.lf-found { margin: 0 0 6px; font-size: 14px; color: #475569; }
.lf-pickup { margin-top: auto; padding: 10px 12px; border-radius: 10px; font-size: 14px; font-weight: 700; }
.lf-pickup.office { background: #EEF2FF; color: #3730A3; }
.lf-pickup.room { background: #FFF7ED; color: #9A3412; }
.lf-match { align-self: flex-start; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; }
.lf-match.strong { background: #4338CA; color: #fff; }
.lf-match.possible { background: #FFF7ED; color: #9A3412; border: 1px solid #FDBA74; }
.lf-match.low { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }

/* How-to-pick-up steps */
.lf-steps { margin-top: 2rem; background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px; display: flex; flex-wrap: wrap; gap: 24px; }
.lf-step { flex: 1 1 240px; display: flex; gap: 14px; }
.lf-step .n { width: 38px; height: 38px; flex-shrink: 0; border-radius: 999px; background: #EEF2FF; color: #3730A3; display: flex; align-items: center; justify-content: center; font-weight: 800; }
.lf-step b { display: block; font-size: 16px; color: #0F172A; margin-bottom: 2px; }
.lf-step span { font-size: 14px; color: #475569; }

/* Register: step headings and tag preview */
.lf-stephead { display: flex; align-items: center; gap: 12px; margin-bottom: .4rem; }
.lf-stephead .n { width: 28px; height: 28px; border-radius: 999px; background: #4338CA; color: #fff; font-size: 14px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
.lf-stephead b { font-size: 1.15rem; font-weight: 800; color: #0F172A; }
.lf-tag { background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px; text-align: center; }
.lf-tag .label { font-size: 13px; font-weight: 700; color: #475569; }
.lf-tag .slot { width: 140px; height: 140px; margin: 14px auto; border-radius: 24px; background: #4338CA; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.lf-tag .slot small { font-size: 14px; font-weight: 600; color: #E0E7FF; }
.lf-tag .slot span { font-size: 72px; font-weight: 800; line-height: 1; }
.lf-tag .slot.room { background: #FFF7ED; color: #9A3412; }
.lf-tag .slot.room span { font-size: 34px; }
.lf-tag .slot.room small { color: #9A3412; }
.lf-tag p { margin: 6px 0 0; font-size: 15px; color: #0F172A; }
.lf-tag p.muted { font-size: 14px; color: #475569; }
.lf-note { margin-top: 16px; background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 16px; padding: 18px; font-size: 14px; color: #7C2D12; }

/* Staff metrics */
.lf-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin: 1.2rem 0 1.6rem; }
.lf-metric { background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 20px; }
.lf-metric span { display: block; font-size: 14px; font-weight: 600; color: #475569; }
.lf-metric b { display: block; font-size: 2rem; font-weight: 800; color: #0F172A; }
.lf-metric.warn { background: #FFF7ED; border-color: #FED7AA; }
.lf-metric.warn span { color: #9A3412; }
.lf-metric.warn b { color: #7C2D12; }

/* Empty state */
.lf-empty { background: #fff; border: 1px dashed #CBD5E1; border-radius: 16px; padding: 36px 24px; text-align: center; color: #334155; }
.lf-empty { margin-top: 1.4rem; }
.lf-empty b { display: block; font-size: 1.1rem; color: #0F172A; margin-bottom: 4px; }

/* Full-width primary submit buttons */
.st-key-submit_item button { width: 100%; min-height: 52px; font-size: 17px; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATABASE
# =========================================================
def db():
    return sqlite3.connect(DB_FILE, timeout=10)


def init_db():
    with db() as conn:
        c = conn.cursor()
        c.execute(
            """
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
                claimed_by TEXT,
                returned_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute("PRAGMA table_info(items)")
        cols = {row[1] for row in c.fetchall()}
        migrations = {
            "slot_number": "INTEGER",
            "holding_location": "TEXT DEFAULT 'Front Office'",
            "claimed_by": "TEXT",
            "returned_at": "TIMESTAMP",
            "pickup_room": "TEXT",  # new: classroom where the student collects it
        }
        for name, kind in migrations.items():
            if name not in cols:
                c.execute(f"ALTER TABLE items ADD COLUMN {name} {kind}")


init_db()


def lowest_free_slot(cursor):
    cursor.execute(
        "SELECT slot_number FROM items WHERE is_returned = 0 "
        "AND holding_location = 'Front Office' AND slot_number IS NOT NULL"
    )
    taken = {row[0] for row in cursor.fetchall()}
    slot = 1
    while slot in taken:
        slot += 1
    return slot


def next_free_slot():
    with db() as conn:
        return lowest_free_slot(conn.cursor())


ITEM_COLUMNS = (
    "id, slot_number, holding_location, item_type, location_found, date_found, "
    "notes, image_path, vector_path, pickup_room, claimed_by, returned_at"
)


def load_items(returned=False):
    order = "returned_at DESC" if returned else "date_found DESC, id DESC"
    with db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {ITEM_COLUMNS} FROM items WHERE is_returned = ? ORDER BY {order}",
            (1 if returned else 0,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_returned(item_ids, claimed_by):
    with db() as conn:
        conn.executemany(
            "UPDATE items SET is_returned = 1, claimed_by = ?, "
            "returned_at = CURRENT_TIMESTAMP WHERE id = ?",
            [(claimed_by, i) for i in item_ids],
        )


def undo_return(item_ids):
    with db() as conn:
        conn.executemany(
            "UPDATE items SET is_returned = 0, claimed_by = NULL, returned_at = NULL WHERE id = ?",
            [(i,) for i in item_ids],
        )


# =========================================================
# 4. PHOTO AI (MEDIAPIPE)
# =========================================================
@st.cache_resource
def get_embedder():
    options = vision.ImageEmbedderOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        l2_normalize=True,
        quantize=False,
    )
    return vision.ImageEmbedder.create_from_options(options)


def photo_vector(image_bytes):
    """Whole-photo vector saved with each item (kept for older-data compatibility)."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
        result = get_embedder().embed(mp_image)
        return np.array(result.embeddings[0].embedding, dtype=np.float32)
    except Exception as e:
        st.error(f"That photo couldn't be read. Try a JPG or PNG. ({e})")
        return None


def _embed(img):
    data = np.ascontiguousarray(np.array(img.convert("RGB")))
    result = get_embedder().embed(mp.Image(image_format=mp.ImageFormat.SRGB, data=data))
    return np.array(result.embeddings[0].embedding, dtype=np.float32)


def _center_crop(img, keep):
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def image_vectors(img):
    """Vectors for the whole photo plus tighter center crops.
    The crops cut away background (brick walls, desks, laps) so the same
    item photographed in two different places still matches."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return [_embed(img), _embed(_center_crop(img, 0.7)), _embed(_center_crop(img, 0.5))]


def query_vectors(image_bytes):
    try:
        return image_vectors(Image.open(io.BytesIO(image_bytes)))
    except Exception as e:
        st.error(f"That photo couldn't be read. Try a JPG or PNG. ({e})")
        return None


@st.cache_data(show_spinner=False)
def stored_vectors(image_path, mtime):
    """Same multi-crop vectors for a saved item photo (cached, so it runs once per item)."""
    return image_vectors(Image.open(image_path))


def item_vectors(item):
    path = item.get("image_path")
    if path and os.path.exists(path):
        try:
            return stored_vectors(path, os.path.getmtime(path))
        except Exception:
            pass
    vp = item.get("vector_path")  # fallback: older items with only a saved vector
    if vp and os.path.exists(vp):
        return [np.load(vp)]
    return []


def best_score(qvecs, ivecs):
    return max((cosine(q, v) for q in qvecs for v in ivecs), default=0.0)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


# =========================================================
# 5. DISPLAY HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def thumb_data_uri(path, mtime, size=480):
    """Square-ish cropped thumbnail as a data URI (cached; mtime busts the cache)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = ImageOps.fit(img, (size, int(size * 0.75)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb(path, size=480):
    if path and os.path.exists(path):
        try:
            return thumb_data_uri(path, os.path.getmtime(path), size)
        except Exception:
            return None
    return None


def nice_date(value):
    """'2026-10-15' or '2026-10-15 14:02:11' -> 'Oct 15'."""
    if not value:
        return ""
    try:
        d = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return f"{d:%b} {d.day}" if d.year == date.today().year else f"{d:%b} {d.day}, {d.year}"
    except ValueError:
        return str(value)


def days_waiting(item):
    try:
        return (date.today() - datetime.strptime(item["date_found"], "%Y-%m-%d").date()).days
    except (ValueError, TypeError):
        return 0


def pickup_text(item):
    if item["holding_location"] == "Front Office" and item["slot_number"]:
        return f"Front Office · Slot {item['slot_number']}", "office"
    room = item.get("pickup_room") or item["location_found"]
    return f"Pick up from {room}", "room"


def item_title(item):
    notes = (item.get("notes") or "").strip()
    return notes if notes else item["item_type"]


def card_html(item, match=None, score=None):
    e = html.escape
    uri = thumb(item["image_path"])
    photo = (
        f'<img class="lf-photo" src="{uri}" alt="Photo of {e(item_title(item))}">'
        if uri
        else '<div class="lf-nophoto">No photo</div>'
    )
    pickup, kind = pickup_text(item)
    badge = ""
    best = ""
    if match == "strong":
        badge, best = '<span class="lf-match strong">Strong match</span>', " best"
    elif match == "possible":
        badge = '<span class="lf-match possible">Possible match</span>'
    elif match == "low":
        badge = '<span class="lf-match low">Worth a look</span>'
    if score is not None:
        badge += f'<span class="lf-match low">score {score:.2f}</span>'
    return f"""
<article class="lf-card{best}">
  {photo}
  <div class="lf-body">
    {badge}
    <div class="lf-row"><span class="cat">{e(item['item_type'])}</span><span>{e(nice_date(item['date_found']))}</span></div>
    <h3 class="lf-name">{e(item_title(item))}</h3>
    <p class="lf-found">Found in {e(item['location_found'])}</p>
    <div class="lf-pickup {kind}">{e(pickup)}</div>
  </div>
</article>"""


def show_grid(items, matches=None, scores=None):
    cards = [
        card_html(it, (matches or {}).get(it["id"]), (scores or {}).get(it["id"]))
        for it in items
    ]
    block = f'<div class="lf-grid">{"".join(cards)}</div>'
    # Remove line breaks + indentation: Markdown treats indented lines as code blocks.
    block = re.sub(r"\n\s*", "", block)
    st.markdown(block, unsafe_allow_html=True)


def empty_state(title, text):
    st.markdown(
        f'<div class="lf-empty"><b>{html.escape(title)}</b>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def crest_data_uri(path, mtime, size=160):
    """Trim the empty margin around a wide logo and center it in a square."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, img.getpixel((0, 0)))
    mask = ImageChops.difference(img, bg).convert("L").point(lambda p: 255 if p > 20 else 0)
    box = mask.getbbox()
    if box:
        img = img.crop(box)
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
    square = square.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    square.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def brand_block():
    uri = None
    if LOGO_PATH:
        try:
            uri = crest_data_uri(LOGO_PATH, os.path.getmtime(LOGO_PATH))
        except Exception:
            uri = None
    mark = f'<img src="{uri}" alt="LISA Academy logo">' if uri else '<div class="crest">LISA</div>'
    st.markdown(
        f'<div class="lf-brand">{mark}<div><small>LISA Academy North</small>'
        f"<strong>Lost &amp; Found</strong></div></div>",
        unsafe_allow_html=True,
    )


def matches_text(item, words):
    hay = " ".join(
        str(item.get(k) or "")
        for k in ("item_type", "location_found", "notes", "holding_location", "pickup_room")
    ).lower()
    if item.get("slot_number"):
        hay += f" slot {item['slot_number']}"
    return all(w in hay for w in words)


# =========================================================
# 6. PAGE: FIND AN ITEM
# =========================================================
def find_page():
    brand_block()
    st.markdown('<h1 class="lf-title">Lost something?</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lf-sub">Browse what\'s been turned in, or search by color, brand, '
        "or the room you were in.</p>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("photo_key", 0)

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1.4, 1.4], vertical_alignment="bottom")
        query = c1.text_input(
            "Search items", placeholder="e.g. navy hoodie, hydroflask, Room 211"
        )
        category = c2.selectbox("Category", ["All categories"] + CATEGORIES)
        with c3.popover("Search by photo", icon=":material/photo_camera:"):
            method = st.segmented_control(
                "Photo source",
                ["Upload photo", "Use camera"],
                default="Upload photo",
                key="photo_method",
            )
            k = st.session_state.photo_key
            if method == "Use camera":
                st.camera_input("Take a photo of an item like yours", key=f"cam_{k}")
            else:
                st.file_uploader(
                    "Upload a photo of your item, or a similar picture from the web",
                    type=["jpg", "jpeg", "png"],
                    key=f"up_{k}",
                )

    k = st.session_state.photo_key
    photo = st.session_state.get(f"cam_{k}") or st.session_state.get(f"up_{k}")
    items = load_items(returned=False)

    # --- Photo search results ---
    if photo is not None:
        top = st.columns([1, 5, 1.2], vertical_alignment="center")
        top[0].image(photo, width=88)
        with top[1]:
            st.markdown('<p class="lf-h2" style="margin:0">Photo matches</p>', unsafe_allow_html=True)
        if top[2].button("Clear photo", icon=":material/close:"):
            st.session_state.photo_key += 1
            st.rerun()

        with st.spinner("Comparing your photo with turned-in items…"):
            qvecs = query_vectors(photo.getvalue())
            if qvecs is None:
                return
            ranked = []
            for it in items:
                ivecs = item_vectors(it)
                if ivecs:
                    ranked.append((best_score(qvecs, ivecs), it))
        ranked.sort(key=lambda x: x[0], reverse=True)

        good = [(s, it) for s, it in ranked if s >= POSSIBLE_MATCH][:6]
        if good:
            shown = good
            label = lambda s: "strong" if s >= STRONG_MATCH else "possible"
            n = len(shown)
            meta = f'{n} item{"s" if n != 1 else ""} look similar'
        else:
            shown = ranked[:FALLBACK_RESULTS]
            label = lambda s: "low"
            meta = "No close matches. These are the closest items — worth a look."

        if not shown:
            empty_state("Nothing has been turned in yet.", " Check back later, or ask at the front office.")
        else:
            labels = {it["id"]: label(s) for s, it in shown}
            scores = {it["id"]: s for s, it in shown} if SHOW_SCORES else None
            st.markdown(f'<p class="lf-meta">{meta}</p>', unsafe_allow_html=True)
            show_grid([it for _, it in shown], labels, scores)
        pickup_steps()
        return

    # --- Keyword / category search (nothing shows until the student searches) ---
    words = query.lower().split()
    if not words and category == "All categories":
        n = len(items)
        if n:
            empty_state(
                "Search to find your item.",
                f" Type what you lost, pick a category, or search by photo. "
                f"{n} item{'s are' if n != 1 else ' is'} waiting to be claimed.",
            )
        else:
            empty_state("Nothing has been turned in yet.", " Check back later, or ask at the front office.")
        pickup_steps()
        return

    results = [
        it
        for it in items
        if (category == "All categories" or it["item_type"].lower() == category.lower())
        and matches_text(it, words)
    ]
    st.markdown('<p class="lf-h2">Search results</p>', unsafe_allow_html=True)
    n = len(results)
    st.markdown(
        f'<p class="lf-meta">{n} item{"s" if n != 1 else ""} · newest first</p>',
        unsafe_allow_html=True,
    )

    if not results:
        empty_state(
            "No items match that search.",
            " Try fewer words, a different category, or search by photo.",
        )
    else:
        show_grid(results)
    pickup_steps()


def pickup_steps():
    st.markdown(
        """
<div class="lf-steps">
  <div class="lf-step"><div class="n">1</div><div><b>Find your item</b><span>Search above or browse the newest items.</span></div></div>
  <div class="lf-step"><div class="n">2</div><div><b>Note where it is</b><span>Each card shows the office slot or classroom.</span></div></div>
  <div class="lf-step"><div class="n">3</div><div><b>Pick it up</b><span>Bring your student ID to the front office or that room.</span></div></div>
</div>""",
        unsafe_allow_html=True,
    )


# =========================================================
# 7. PAGE: TURN IN AN ITEM
# =========================================================
def register_page():
    brand_block()
    st.session_state.setdefault("reg_key", 0)
    k = st.session_state.reg_key

    # Confirmation screen after a successful submit
    done = st.session_state.get("last_registered")
    if done:
        st.markdown('<h1 class="lf-title">Item added</h1>', unsafe_allow_html=True)
        left, right = st.columns([1, 1.3])
        with left:
            if done["slot"]:
                st.markdown(
                    f"""<div class="lf-tag"><div class="label">Tag for this item</div>
<div class="slot"><small>SLOT</small><span>{done['slot']}</span></div>
<p>Write <b>{done['slot']}</b> on the tag and hang it on hook {done['slot']}.</p></div>""",
                    unsafe_allow_html=True,
                )
            else:
                room = html.escape(done["room"])
                st.markdown(
                    f"""<div class="lf-tag"><div class="label">Kept in classroom</div>
<div class="slot room"><small>PICKUP</small><span>{room}</span></div>
<p>Keep the item in {room}. Students will be sent there.</p></div>""",
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown(
                '<p class="lf-sub" style="margin-top:.4rem">It now shows up for students '
                "searching Lost &amp; Found.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Turn in another item", type="primary", icon=":material/add:"):
                st.session_state.pop("last_registered")
                st.session_state.reg_key += 1
                st.rerun()
        return

    st.markdown('<h1 class="lf-title">Turn in a found item</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lf-sub">Takes about a minute. Fields marked * are required.</p>',
        unsafe_allow_html=True,
    )

    form_col, side_col = st.columns([2, 1], gap="large")

    with form_col:
        # Step 1: photo
        with st.container(border=True):
            step_head(1, "Photo of the item *")
            method = st.segmented_control(
                "Photo source", ["Upload photo", "Use camera"],
                default="Upload photo", key=f"reg_method_{k}", label_visibility="collapsed",
            )
            if method == "Use camera":
                photo = st.camera_input("Take a photo of the found item", key=f"reg_cam_{k}")
            else:
                photo = st.file_uploader(
                    "Upload photo", type=["jpg", "jpeg", "png"], key=f"reg_up_{k}",
                    help="Lay the item on a plain surface for better photo matches.",
                )

        # Step 2: details
        with st.container(border=True):
            step_head(2, "About the item")
            a, b = st.columns(2)
            item_type = a.selectbox("Category *", CATEGORIES, key=f"reg_cat_{k}")
            found_on = b.date_input(
                "Date found *", value=date.today(), max_value=date.today(),
                format="MM/DD/YYYY", key=f"reg_date_{k}",
            )
            where = st.text_input(
                "Where was it found? *", placeholder="e.g. Gymnasium, Library, Room 211",
                key=f"reg_where_{k}",
            )
            notes = st.text_area(
                "Description",
                placeholder="Color, brand, stickers, name tag — anything that helps the owner recognize it",
                key=f"reg_notes_{k}",
            )

        # Step 3: storage
        with st.container(border=True):
            step_head(3, "Where will it be kept?")
            keep = st.radio(
                "Storage location",
                ["Front Office", "My classroom"],
                captions=["Gets the next free bin / hook slot", "Student picks it up from your room"],
                horizontal=True, label_visibility="collapsed", key=f"reg_keep_{k}",
            )
            pickup_room = ""
            if keep == "My classroom":
                pickup_room = st.text_input(
                    "Your room *", placeholder="e.g. Room 211", key=f"reg_room_{k}"
                )

        submitted = st.button(
            "Add to Lost & Found", type="primary", key="submit_item", icon=":material/check:"
        )

    with side_col:
        if keep == "Front Office":
            slot = next_free_slot()
            st.markdown(
                f"""<div class="lf-tag"><div class="label">Tag for this item</div>
<div class="slot"><small>SLOT</small><span>{slot}</span></div>
<p>Write <b>{slot}</b> on the tag and hang it on hook {slot}.</p>
<p class="muted">The slot is reserved when you submit, so two people can't get the same number.</p></div>""",
                unsafe_allow_html=True,
            )
        else:
            room = html.escape(pickup_room.strip() or "your room")
            st.markdown(
                f"""<div class="lf-tag"><div class="label">Classroom pickup</div>
<div class="slot room"><small>PICKUP</small><span>{room if pickup_room.strip() else "—"}</span></div>
<p>Students will be sent to {room}.</p></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="lf-note">Phones, keys, money, or IDs? Take them straight to the front desk instead.</div>',
            unsafe_allow_html=True,
        )

    if submitted:
        problems = []
        if photo is None:
            problems.append("Add a photo of the item.")
        if not where.strip():
            problems.append("Enter where the item was found.")
        if keep == "My classroom" and not pickup_room.strip():
            problems.append("Enter the room where students can pick it up.")
        if problems:
            with form_col:
                st.error("  \n".join(problems))
            return
        save_item(photo, item_type, where.strip(), found_on, notes.strip(), keep, pickup_room.strip())


def step_head(n, text):
    st.markdown(
        f'<div class="lf-stephead"><span class="n">{n}</span><b>{html.escape(text)}</b></div>',
        unsafe_allow_html=True,
    )


def save_item(photo, item_type, where, found_on, notes, keep, pickup_room):
    raw = photo.getvalue()
    vector = photo_vector(raw)
    ext = os.path.splitext(getattr(photo, "name", "") or "")[1].lower() or ".jpg"

    conn = db()
    try:
        c = conn.cursor()
        c.execute("BEGIN IMMEDIATE")  # locks the DB so two people can't take the same slot
        if keep == "Front Office":
            slot, holding, room = lowest_free_slot(c), "Front Office", None
        else:
            slot, holding, room = None, "Classroom", pickup_room
        c.execute(
            "INSERT INTO items (slot_number, holding_location, item_type, location_found, "
            "date_found, notes, image_path, pickup_room) VALUES (?, ?, ?, ?, ?, ?, '', ?)",
            (slot, holding, item_type, where, found_on.strftime("%Y-%m-%d"), notes, room),
        )
        new_id = c.lastrowid
        img_path = os.path.join(UPLOAD_DIR, f"item_{new_id}{ext}")
        with open(img_path, "wb") as f:
            f.write(raw)
        vec_path = None
        if vector is not None:
            vec_path = os.path.join(VECTOR_DIR, f"vector_{new_id}.npy")
            np.save(vec_path, vector)
        c.execute(
            "UPDATE items SET image_path = ?, vector_path = ? WHERE id = ?",
            (img_path, vec_path, new_id),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"The item wasn't saved. Try again. ({e})")
        return
    finally:
        conn.close()

    st.session_state.last_registered = {"slot": slot, "room": pickup_room}
    st.rerun()


# =========================================================
# 8. PAGE: STAFF CONSOLE
# =========================================================
def staff_page():
    brand_block()
    if not st.session_state.get("is_staff"):
        staff_login()
        return

    head, actions = st.columns([2, 1.4], vertical_alignment="bottom")
    with head:
        st.markdown('<h1 class="lf-title">Inventory</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lf-sub" style="margin:0">Manage unclaimed items and record pickups.</p>',
            unsafe_allow_html=True,
        )
    with actions:
        a1, a2, a3 = st.columns(3)
        with a1.popover("QR poster", icon=":material/qr_code_2:"):
            qr = "https://api.qrserver.com/v1/create-qr-code/?size=600x600&data=" + urllib.parse.quote(APP_URL)
            st.image(qr, width=220)
            st.caption("Scan to open Lost & Found on a phone.")
            st.link_button("Open printable QR", qr)
        a2.page_link(register_pg, label="Turn in item", icon=":material/add:")
        if a3.button("Log out", icon=":material/logout:"):
            st.session_state.is_staff = False
            st.rerun()

    active = load_items(returned=False)
    returned = load_items(returned=True)
    month = date.today().strftime("%Y-%m")
    returned_month = sum(1 for r in returned if str(r["returned_at"] or "").startswith(month))
    old = sum(1 for it in active if days_waiting(it) >= DONATE_AFTER_DAYS)

    st.markdown(
        f"""<div class="lf-metrics">
<div class="lf-metric"><span>Unclaimed</span><b>{len(active)}</b></div>
<div class="lf-metric"><span>Returned this month</span><b>{returned_month}</b></div>
<div class="lf-metric{' warn' if old else ''}"><span>Waiting {DONATE_AFTER_DAYS}+ days</span><b>{old}</b></div>
<div class="lf-metric"><span>Next free office slot</span><b>#{next_free_slot()}</b></div>
</div>""",
        unsafe_allow_html=True,
    )

    # Undo banner for the last return
    undo = st.session_state.get("undo_ids")
    if undo:
        u1, u2 = st.columns([5, 1], vertical_alignment="center")
        n = len(undo)
        u1.success(f"Marked {n} item{'s' if n != 1 else ''} returned.")
        if u2.button("Undo", icon=":material/undo:"):
            undo_return(undo)
            st.session_state.undo_ids = None
            st.rerun()

    tab_active, tab_history = st.tabs([f"Active ({len(active)})", "Returned history"])

    with tab_active:
        f1, f2 = st.columns([1.3, 2], vertical_alignment="bottom")
        where = f1.segmented_control(
            "Show items from", ["All", "Front Office", "Classrooms"], default="All", key="staff_where"
        )
        text = f2.text_input("Filter by room or keyword", placeholder="e.g. 211, Stanley, jacket")
        words = text.lower().split()
        rows = [
            it for it in active
            if (where in (None, "All")
                or (where == "Front Office" and it["holding_location"] == "Front Office")
                or (where == "Classrooms" and it["holding_location"] != "Front Office"))
            and matches_text(it, words)
        ]

        if not active:
            empty_state("No unclaimed items.", " Items you turn in will show up here.")
        elif not rows:
            empty_state("No items match this filter.", " Clear the filter to see everything.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "id": it["id"],
                        "Photo": thumb(it["image_path"], 96),
                        "Item": item_title(it),
                        "Category": it["item_type"],
                        "Found in": it["location_found"],
                        "Found": nice_date(it["date_found"]),
                        "Kept at": pickup_text(it)[0].replace("Pick up from ", ""),
                        "Waiting": (
                            f"{days_waiting(it)} days · consider donating"
                            if days_waiting(it) >= DONATE_AFTER_DAYS
                            else f"{days_waiting(it)} days"
                        ),
                    }
                    for it in rows
                ]
            )
            event = st.dataframe(
                df,
                hide_index=True,
                column_order=["Photo", "Item", "Category", "Found in", "Found", "Kept at", "Waiting"],
                column_config={"Photo": st.column_config.ImageColumn("Photo", width="small")},
                on_select="rerun",
                selection_mode="multi-row",
                key="active_table",
            )
            picked = [int(df.iloc[i]["id"]) for i in event.selection.rows]
            if picked:
                n = len(picked)
                if st.button(
                    f"Mark {n} item{'s' if n != 1 else ''} returned…",
                    type="primary", icon=":material/assignment_turned_in:",
                ):
                    confirm_return(picked)
            else:
                st.caption("Select rows to mark items returned. Their office slots are freed automatically.")

    with tab_history:
        if not returned:
            empty_state("No returns recorded yet.", " Returned items will be listed here.")
        else:
            q = st.text_input("Search returned items", placeholder="e.g. Alex, Stanley, Room 211")
            words = q.lower().split()
            hist = [
                r for r in returned
                if all(w in f"{r['item_type']} {r['location_found']} {r['notes'] or ''} {r['claimed_by'] or ''}".lower() for w in words)
            ]
            if not hist:
                empty_state("No returned items match.", " Try a student name or item type.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Photo": thumb(r["image_path"], 96),
                                "Item": item_title(r),
                                "Claimed by": r["claimed_by"] or "Not recorded",
                                "Kept at": f"Office slot {r['slot_number']}" if r["slot_number"] else (r["pickup_room"] or r["location_found"]),
                                "Found": nice_date(r["date_found"]),
                                "Returned": nice_date(r["returned_at"]),
                            }
                            for r in hist
                        ]
                    ),
                    hide_index=True,
                    column_config={"Photo": st.column_config.ImageColumn("Photo", width="small")},
                )


@st.dialog("Mark returned")
def confirm_return(item_ids):
    n = len(item_ids)
    st.write(f"Record pickup for {n} item{'s' if n != 1 else ''}. Office slots will be freed.")
    who = st.text_input("Picked up by", placeholder="Student name or ID (optional)")
    c1, c2 = st.columns(2)
    if c1.button("Mark returned", type="primary"):
        mark_returned(item_ids, who.strip() or "Not recorded")
        st.session_state.undo_ids = item_ids
        st.rerun()
    if c2.button("Cancel"):
        st.rerun()


def staff_login():
    passcode = get_staff_passcode()
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<h1 class="lf-title">Staff sign in</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lf-sub">For teachers and office staff managing Lost &amp; Found.</p>',
            unsafe_allow_html=True,
        )
        if passcode is None:
            st.error(
                "No staff passcode is set. Add STAFF_PASSCODE in the app's Secrets "
                "(Streamlit Cloud: App settings → Secrets)."
            )
            return
        with st.form("login", border=True):
            code = st.text_input("Staff passcode", type="password")
            ok = st.form_submit_button("Sign in", type="primary")
        if ok:
            if code == passcode:
                st.session_state.is_staff = True
                st.rerun()
            else:
                st.error("That passcode isn't right. Check with the front office.")


# =========================================================
# 9. NAVIGATION
# =========================================================
find_pg = st.Page(find_page, title="Find an item", icon=":material/search:", url_path="find", default=True)
register_pg = st.Page(register_page, title="Turn in an item", icon=":material/add_box:", url_path="turn-in")
staff_pg = st.Page(staff_page, title="Staff", icon=":material/lock:", url_path="staff")

st.navigation([find_pg, register_pg, staff_pg], position="top").run()
