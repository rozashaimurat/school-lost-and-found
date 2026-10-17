"""
School Lost & Found Hub - LISA Academy North
Redesigned UI (requires Streamlit >= 1.46 for top navigation).
"""
import base64
import html
import io
import os
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageCms, ImageOps

# =========================================================
# 1. SETTINGS
# =========================================================
# Items and photos are stored in Supabase (a free online database), so they are
# NOT lost when Streamlit restarts. Keys go in Streamlit Secrets:
#   SUPABASE_URL = "https://xxxx.supabase.co"
#   SUPABASE_KEY = "your secret / service_role key"
PHOTO_BUCKET = "item-photos"
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
# Color check: the AI "shape" score above knows a jacket is a jacket, but it barely
# notices color. So we also compare colors (0 = totally different, 1 = same colors).
# In our tests the same item scored 0.58-0.92 (even in different lighting or on a
# different background) and a black jacket vs. a green jacket scored 0.34.
COLOR_STRONG = 0.55   # needed for "Strong match"
COLOR_MATCH = 0.45    # needed to show an item at all
# If nothing reaches POSSIBLE_MATCH, still show this many closest items
# (labeled "Worth a look") so a real match is never hidden completely —
# but only items whose colors match.
FALLBACK_RESULTS = 3
SHOW_SCORES = False

DONATE_AFTER_DAYS = 30
# From this many days on, items show a "last chance" warning (hallway screen + search cards)
LAST_CHANCE_DAYS = 25

# Word filter for descriptions and room fields (first layer only; staff review is the real check).
# Words are matched after common letter swaps (@ -> a, 0 -> o, $ -> s ...).
# Add words to this list if you see new ones get through.
BLOCKED_EXACT = {
    "ass", "sex", "tit", "tits", "cum", "hoe", "hoes", "dick", "dicks", "dickhead",
    "cock", "cocks", "nudes", "fag", "fags", "wtf", "stfu", "kys",
}
BLOCKED_PREFIXES = (
    "fuck", "fuk", "shit", "bitch", "asshole", "bastard", "cunt", "pussy",
    "slut", "whore", "nigg", "faggot", "retard", "porn", "penis",
    "vagina", "boob", "dildo", "horny",
)

# Hallway screen: seconds each item stays on screen
SLIDE_SECONDS = 10

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
.lf-last { padding: 8px 12px; border-radius: 10px; background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; font-size: 13px; font-weight: 700; }
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
# 3. DATABASE (SUPABASE)
# =========================================================
from supabase import create_client  # noqa: E402


@st.cache_resource
def sb():
    """One shared connection to Supabase."""
    try:
        url, key = st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error(
            "The database isn't connected yet. Add SUPABASE_URL and SUPABASE_KEY "
            "in the app's Secrets (Streamlit Cloud: App settings → Secrets)."
        )
        st.stop()
    return create_client(url, key)


def next_free_slot():
    """Preview of the lowest empty office slot (the real one is assigned by the database)."""
    rows = (
        sb().table("items").select("slot_number")
        .eq("is_returned", False).eq("holding_location", "Front Office")
        .not_.is_("slot_number", "null").execute().data
    )
    taken = {r["slot_number"] for r in rows}
    slot = 1
    while slot in taken:
        slot += 1
    return slot


def load_items(returned=False, status="approved"):
    """Public pages only ever see status='approved'. Staff review uses status='pending'."""
    q = sb().table("items").select("*").eq("is_returned", returned).eq("status", status)
    if returned:
        q = q.order("returned_at", desc=True)
    else:
        q = q.order("date_found", desc=True).order("id", desc=True)
    return q.execute().data or []


def approve_item(item_id):
    sb().table("items").update(
        {"status": "approved", "reported": False, "report_reason": None}
    ).eq("id", item_id).execute()


def reject_item(item_id):
    """Delete the item and its photo, so inappropriate images are not kept anywhere."""
    rows = sb().table("items").select("image_path").eq("id", item_id).execute().data
    sb().table("items").delete().eq("id", item_id).execute()
    if rows and rows[0].get("image_path"):
        try:
            sb().storage.from_(PHOTO_BUCKET).remove([rows[0]["image_path"]])
        except Exception:
            pass


def report_item(item_id, reason):
    """Hide a reported item right away; it goes back to the staff review list."""
    sb().table("items").update(
        {"status": "pending", "reported": True, "report_reason": reason}
    ).eq("id", item_id).execute()


def has_blocked_words(*texts):
    swaps = str.maketrans({"@": "a", "4": "a", "0": "o", "1": "i", "!": "i", "3": "e", "$": "s", "5": "s", "7": "t"})
    for text in texts:
        t = (text or "").lower().translate(swaps)
        t = re.sub(r"(.)\1{2,}", r"\1", t)  # "fuuuuck" -> "fuck"
        words, letters = [], ""
        for w in re.findall(r"[a-z]+", t):
            if len(w) == 1:  # join spaced-out letters: "f u c k" / "f.u.c.k" -> "fuck"
                letters += w
                continue
            if letters:
                words.append(letters)
                letters = ""
            words.append(w)
        if letters:
            words.append(letters)
        for word in words:
            if word in BLOCKED_EXACT or word.startswith(BLOCKED_PREFIXES):
                return True
    return False


def mark_returned(item_ids, claimed_by):
    sb().table("items").update(
        {"is_returned": True, "claimed_by": claimed_by, "returned_at": datetime.now(timezone.utc).isoformat()}
    ).in_("id", list(item_ids)).execute()


def undo_return(item_ids):
    sb().table("items").update(
        {"is_returned": False, "claimed_by": None, "returned_at": None}
    ).in_("id", list(item_ids)).execute()


@st.cache_data(show_spinner=False, max_entries=1000)
def photo_bytes(path):
    """Download a photo from Supabase Storage once, then keep it in memory."""
    return sb().storage.from_(PHOTO_BUCKET).download(path)


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


@st.cache_data(show_spinner=False, max_entries=1000)
def stored_vectors(image_path):
    """Same multi-crop vectors for a saved item photo (cached, so it runs once per item)."""
    return image_vectors(Image.open(io.BytesIO(photo_bytes(image_path))))


def item_vectors(item):
    path = item.get("image_path")
    if not path:
        return []
    try:
        return stored_vectors(path)
    except Exception:
        return []


_LAB = ImageCms.buildTransformFromOpenProfiles(
    ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB"
)


def color_signature(img):
    """A smoothed color histogram of the middle of the photo.
    Uses LAB colors (closer to how people see color) and ignores white backgrounds."""
    img = _center_crop(ImageOps.exif_transpose(img).convert("RGB"), 0.7).resize((96, 96))
    L, a, b = (np.asarray(ch, dtype=float).ravel() for ch in ImageCms.applyTransform(img, _LAB).split())
    L, a, b = L * 100 / 255, a - 128, b - 128
    background = (L > 92) & (np.abs(a) < 6) & (np.abs(b) < 6)  # plain white backdrop
    if background.mean() < 0.9:
        L, a, b = L[~background], a[~background], b[~background]
    hist, _ = np.histogramdd(
        np.stack([L, a, b], axis=1), bins=(5, 8, 8), range=((0, 100), (-40, 40), (-40, 40))
    )
    # Blur the histogram a little so small lighting changes don't matter
    padded = np.pad(hist, 1)
    smooth = sum(
        padded[i:i + 5, j:j + 8, k:k + 8] for i in range(3) for j in range(3) for k in range(3)
    )
    return smooth / smooth.sum()


def color_similarity(c1, c2):
    return float(np.minimum(c1, c2).sum())


@st.cache_data(show_spinner=False, max_entries=1000)
def stored_color(image_path):
    return color_signature(Image.open(io.BytesIO(photo_bytes(image_path))))


def best_score(qvecs, ivecs):
    return max((cosine(q, v) for q in qvecs for v in ivecs), default=0.0)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


# =========================================================
# 5. DISPLAY HELPERS
# =========================================================
@st.cache_data(show_spinner=False, max_entries=2000)
def thumb_data_uri(path, size=480):
    """Cropped thumbnail of a stored photo as a data URI (cached)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes(path)))).convert("RGB")
    img = ImageOps.fit(img, (size, int(size * 0.75)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb(path, size=480):
    if not path:
        return None
    try:
        return thumb_data_uri(path, size)
    except Exception:
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


def last_chance_text(item):
    """Warning for items close to being donated or thrown away, or None."""
    days = days_waiting(item)
    if days < LAST_CHANCE_DAYS:
        return None
    left = DONATE_AFTER_DAYS - days
    if left > 1:
        return f"Last chance! This item will be donated or thrown away in {left} days."
    if left == 1:
        return "Last chance! This item will be donated or thrown away tomorrow."
    return "Last chance! This item will be donated or thrown away very soon."


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
        badge += f'<span class="lf-match low">{e(str(score))}</span>'
    return f"""
<article class="lf-card{best}">
  {photo}
  <div class="lf-body">
    {badge}
    <div class="lf-row"><span class="cat">{e(item['item_type'])}</span><span>{e(nice_date(item['date_found']))}</span></div>
    <h3 class="lf-name">{e(item_title(item))}</h3>
    <p class="lf-found">Found in {e(item['location_found'])}</p>
    {f'<div class="lf-last">{e(last_chance_text(item))}</div>' if last_chance_text(item) else ""}
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

    if st.session_state.pop("report_sent", False):
        st.success("Thanks for reporting. The item is hidden until a staff member checks it.")

    k = st.session_state.photo_key
    photo = st.session_state.get(f"cam_{k}") or st.session_state.get(f"up_{k}")
    items = load_items(returned=False)

    # --- Photo search results ---
    if photo is not None:
        top = st.columns([1, 5, 1.2], vertical_alignment="center")
        try:
            qimg = ImageOps.exif_transpose(Image.open(io.BytesIO(photo.getvalue()))).convert("RGB")
            qimg.thumbnail((176, 176))
            qbuf = io.BytesIO()
            qimg.save(qbuf, format="JPEG", quality=85)
            top[0].markdown(
                '<img alt="Your photo" style="width:88px;height:88px;object-fit:cover;border-radius:10px" '
                f'src="data:image/jpeg;base64,{base64.b64encode(qbuf.getvalue()).decode()}">',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        with top[1]:
            st.markdown('<p class="lf-h2" style="margin:0">Photo matches</p>', unsafe_allow_html=True)
        if top[2].button("Clear photo", icon=":material/close:"):
            st.session_state.photo_key += 1
            st.rerun()

        with st.spinner("Comparing your photo with turned-in items…"):
            qvecs = query_vectors(photo.getvalue())
            if qvecs is None:
                return
            qcolor = color_signature(Image.open(io.BytesIO(photo.getvalue())))
            ranked = []  # (shape score, color score, item)
            for it in items:
                ivecs = item_vectors(it)
                if not ivecs:
                    continue
                try:
                    color = color_similarity(qcolor, stored_color(it["image_path"]))
                except Exception:
                    color = 0.0
                ranked.append((best_score(qvecs, ivecs), color, it))
        # Sort by a mix of both scores; shape matters most
        ranked.sort(key=lambda x: 0.6 * x[0] + 0.4 * x[1], reverse=True)
        same_color = [r for r in ranked if r[1] >= COLOR_MATCH]

        good = [r for r in same_color if r[0] >= POSSIBLE_MATCH][:6]
        if good:
            shown = good
            label = lambda shape, color: (
                "strong" if shape >= STRONG_MATCH and color >= COLOR_STRONG else "possible"
            )
            n = len(shown)
            meta = f'{n} item{"s" if n != 1 else ""} look similar'
        else:
            shown = same_color[:FALLBACK_RESULTS]
            label = lambda shape, color: "low"
            meta = "No close matches. These items have similar colors — worth a look."

        if not ranked:
            empty_state("Nothing has been turned in yet.", " Check back later, or ask at the front office.")
        elif not shown:
            empty_state(
                "No matches for this photo.",
                " Nothing turned in looks like your item. Try another photo, or search by words above.",
            )
        else:
            labels = {it["id"]: label(sh, co) for sh, co, it in shown}
            scores = (
                {it["id"]: f"shape {sh:.2f} · color {co:.2f}" for sh, co, it in shown}
                if SHOW_SCORES else None
            )
            st.markdown(f'<p class="lf-meta">{meta}</p>', unsafe_allow_html=True)
            show_cards_with_report([it for _, _, it in shown], labels, scores)
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
        show_cards_with_report(results)
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


def show_cards_with_report(items, matches=None, scores=None, per_row=3):
    """Cards in rows of three, each with a small 'Report' button underneath."""
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, it in zip(cols, items[start:start + per_row]):
            with col:
                card = card_html(it, (matches or {}).get(it["id"]), (scores or {}).get(it["id"]))
                st.markdown(re.sub(r"\n\s*", "", card), unsafe_allow_html=True)
                if st.button("Report", key=f"report_{it['id']}", type="tertiary", icon=":material/flag:"):
                    report_dialog(it["id"])


@st.dialog("Report this item")
def report_dialog(item_id):
    st.write("Tell staff what's wrong. The item will be hidden until a staff member checks it.")
    reason = st.radio(
        "What's the problem?",
        [
            "Inappropriate photo",
            "Inappropriate description",
            "Shows a person's face or private information",
            "Not a real lost item",
            "Other",
        ],
    )
    if st.button("Send report", type="primary"):
        report_item(item_id, reason)
        st.session_state.report_sent = True
        st.rerun()


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
        pending = done.get("pending")
        st.markdown(
            f'<h1 class="lf-title">{"Thanks! Item sent for review" if pending else "Item added"}</h1>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1, 1.3])
        with left:
            if done["slot"]:
                st.markdown(
                    f"""<div class="lf-tag"><div class="label">Tag for this item</div>
<div class="slot"><small>SLOT</small><span>{done['slot']}</span></div>
<p>{"Bring the item to the front office. It goes on hook " + str(done['slot']) + "." if pending else "Write <b>" + str(done['slot']) + "</b> on the tag and hang it on hook " + str(done['slot']) + "."}</p></div>""",
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
            msg = (
                "A staff member will check the photo and description before it appears "
                "in search and on the hallway screen."
                if pending
                else "It now shows up for students searching Lost &amp; Found and on the hallway screen."
            )
            st.markdown(f'<p class="lf-sub" style="margin-top:.4rem">{msg}</p>', unsafe_allow_html=True)
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
    if st.session_state.get("is_staff"):
        st.caption("Signed in as staff: this item will appear right away.")
    else:
        st.info(
            "Items turned in by students are checked by staff before they appear. "
            "Photograph only the item, with no people in the picture.",
            icon=":material/verified_user:",
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
        if has_blocked_words(where, notes, pickup_room):
            problems.append("Please remove inappropriate language from the description or room fields.")
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
    # Make sure the photo opens, then shrink it so storage stays small
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(photo.getvalue()))).convert("RGB")
    except Exception:
        st.error("That photo couldn't be read. Try a JPG or PNG.")
        return
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)

    status = "approved" if st.session_state.get("is_staff") else "pending"
    holding = "Front Office" if keep == "Front Office" else "Classroom"
    try:
        # add_item runs inside the database and locks it while it picks the lowest
        # free slot, so two people can never get the same slot number.
        row = sb().rpc("add_item", {
            "p_holding": holding,
            "p_item_type": item_type,
            "p_location": where,
            "p_date": found_on.isoformat(),
            "p_notes": notes,
            "p_pickup_room": pickup_room if holding == "Classroom" else None,
            "p_status": status,
        }).execute().data
        row = row[0] if isinstance(row, list) else row
        path = f"item_{row['id']}.jpg"
        try:
            sb().storage.from_(PHOTO_BUCKET).upload(
                path, buf.getvalue(), {"content-type": "image/jpeg", "upsert": "true"}
            )
        except Exception:
            sb().table("items").delete().eq("id", row["id"]).execute()  # don't keep an item with no photo
            raise
        sb().table("items").update({"image_path": path}).eq("id", row["id"]).execute()
    except Exception as e:
        st.error(f"The item wasn't saved. Try again. ({e})")
        return

    st.session_state.last_registered = {
        "slot": row.get("slot_number"), "room": pickup_room, "pending": status == "pending"
    }
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
        st.page_link(hallway_pg, label="Open hallway screen", icon=":material/tv:")
        if a3.button("Log out", icon=":material/logout:"):
            st.session_state.is_staff = False
            st.rerun()

    active = load_items(returned=False)
    returned = load_items(returned=True)
    pending = load_items(returned=False, status="pending")
    month = date.today().strftime("%Y-%m")
    returned_month = sum(1 for r in returned if str(r["returned_at"] or "").startswith(month))
    old = sum(1 for it in active if days_waiting(it) >= DONATE_AFTER_DAYS)

    st.markdown(
        f"""<div class="lf-metrics">
<div class="lf-metric{' warn' if pending else ''}"><span>Needs review</span><b>{len(pending)}</b></div>
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

    labels = [f"Active ({len(active)})", "Returned history"]
    if pending:
        labels.insert(0, f"Needs review ({len(pending)})")
    tabs = st.tabs(labels)
    if pending:
        with tabs[0]:
            review_queue(pending)
        tab_active, tab_history = tabs[1], tabs[2]
    else:
        tab_active, tab_history = tabs

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


def review_queue(pending):
    st.caption(
        "Student uploads and reported items wait here. Nothing in this list appears in "
        "search or on the hallway screen until you approve it. Rejecting deletes the item and its photo."
    )
    for it in pending:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.2, 2.4, 1.2], vertical_alignment="center")
            uri = thumb(it["image_path"], 480)
            c1.markdown(
                f'<img class="lf-photo" style="border-radius:10px" src="{uri}" alt="">' if uri
                else '<div class="lf-nophoto">No photo</div>',
                unsafe_allow_html=True,
            )
            with c2:
                if it.get("reported"):
                    st.warning(f"Reported: {it.get('report_reason') or 'no reason given'}", icon=":material/flag:")
                st.markdown(f"**{item_title(it)}**")
                st.caption(
                    f"{it['item_type']} · found in {it['location_found']} · "
                    f"{nice_date(it['date_found'])} · {pickup_text(it)[0]}"
                )
            with c3:
                if st.button("Approve", key=f"approve_{it['id']}", type="primary", icon=":material/check:"):
                    approve_item(it["id"])
                    st.toast("Approved. The item is now visible.")
                    st.rerun()
                if st.button("Reject", key=f"reject_{it['id']}", icon=":material/delete:"):
                    confirm_reject(it["id"])


@st.dialog("Reject this item?")
def confirm_reject(item_id):
    st.write("The item and its photo will be permanently deleted, and its office slot will be freed.")
    c1, c2 = st.columns(2)
    if c1.button("Delete item", type="primary"):
        reject_item(item_id)
        st.rerun()
    if c2.button("Cancel"):
        st.rerun()


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
# 9. PAGE: HALLWAY SCREEN (live slideshow for hallway TVs)
# =========================================================
HALLWAY_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 100% !important; padding: 2.5vh 3vw 1.5vh !important; }
.hw { height: 94vh; display: flex; flex-direction: column; gap: 2.2vh; color: #0F172A; }
.hw-top { display: flex; align-items: center; justify-content: space-between; gap: 2vw; }
.hw-brand { display: flex; align-items: center; gap: 1.2vw; }
.hw-brand img { width: 6.5vh; height: 6.5vh; border-radius: 1.2vh; background: #fff; border: 1px solid #E2E8F0; }
.hw-brand small { display: block; font-size: 1.8vh; font-weight: 600; color: #475569; }
.hw-brand strong { display: block; font-size: 3.4vh; font-weight: 800; line-height: 1.1; }
.hw-ask { font-size: 4.6vh; font-weight: 800; color: #3730A3; letter-spacing: -0.01em; }
.hw-main { flex: 1; min-height: 0; display: flex; gap: 3vw; align-items: stretch; }
.hw-photo { flex: 1.25; min-width: 0; border-radius: 2.4vh; overflow: hidden; background: #E2E8F0; display: flex; align-items: center; justify-content: center; }
.hw-photo img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hw-photo span { font-size: 3vh; color: #475569; }
.hw-info { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; gap: 1.6vh; }
.hw-cat { font-size: 2.2vh; font-weight: 700; color: #475569; letter-spacing: .06em; text-transform: uppercase; }
.hw-info h1 { margin: 0; font-size: 6vh; font-weight: 800; line-height: 1.08; letter-spacing: -0.02em; color: #0F172A; }
.hw-found { margin: 0; font-size: 3vh; color: #334155; }
.hw-pick { margin-top: 1.5vh; border-radius: 2.4vh; padding: 2.6vh 2.4vw; display: flex; align-items: center; justify-content: space-between; gap: 2vw; }
.hw-pick.office { background: #4338CA; color: #fff; }
.hw-pick.room { background: #FFF7ED; color: #9A3412; border: 0.4vh solid #FDBA74; }
.hw-pick small { display: block; font-size: 2.2vh; font-weight: 700; letter-spacing: .08em; opacity: .85; }
.hw-pick b { display: block; font-size: 5.4vh; font-weight: 800; line-height: 1.1; }
.hw-slot { text-align: center; background: #fff; color: #3730A3; border-radius: 2vh; padding: 1.4vh 2.2vw; }
.hw-slot small { font-size: 2vh; opacity: 1; }
.hw-slot span { display: block; font-size: 11vh; font-weight: 800; line-height: 1; }
.hw-last { display: flex; align-items: center; gap: 1.2vw; padding: 1.8vh 1.8vw; border-radius: 1.8vh; background: #DC2626; color: #fff; font-size: 3vh; font-weight: 800; line-height: 1.2; }
.hw-last .icon { flex-shrink: 0; width: 5vh; height: 5vh; border-radius: 999px; background: #fff; color: #DC2626; display: flex; align-items: center; justify-content: center; font-size: 3.4vh; font-weight: 800; }
.hw-id { margin: 0; font-size: 2.6vh; color: #475569; }
.hw-bottom { display: flex; align-items: center; justify-content: space-between; gap: 3vw; }
.hw-progress { flex: 1; display: flex; align-items: center; gap: 1.5vw; font-size: 2.2vh; font-weight: 700; color: #475569; }
.hw-bar { flex: 1; height: 1vh; border-radius: 1vh; background: #E2E8F0; overflow: hidden; }
.hw-bar div { height: 100%; background: #4338CA; border-radius: 1vh; }
.hw-qr { display: flex; align-items: center; gap: 1.2vw; font-size: 2.2vh; font-weight: 700; color: #0F172A; }
.hw-qr img { width: 11vh; height: 11vh; background: #fff; padding: .6vh; border-radius: 1vh; border: 1px solid #E2E8F0; }
.hw-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 2vh; }
.hw-empty h1 { margin: 0; font-size: 7vh; font-weight: 800; }
.hw-empty p { margin: 0; font-size: 3vh; color: #475569; }
</style>
"""


def hallway_page():
    st.markdown(HALLWAY_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("slide", 0)
    hallway_slide()


@st.fragment(run_every=SLIDE_SECONDS)
def hallway_slide():
    e = html.escape
    items = load_items(returned=False)  # re-read every slide, so new items appear automatically
    crest = crest_data_uri(LOGO_PATH, os.path.getmtime(LOGO_PATH)) if LOGO_PATH else ""
    qr = "https://api.qrserver.com/v1/create-qr-code/?size=300x300&margin=0&data=" + urllib.parse.quote(APP_URL)
    top = (
        '<div class="hw-top"><div class="hw-brand">'
        + (f'<img src="{crest}" alt="">' if crest else "")
        + '<div><small>LISA Academy North</small><strong>Lost &amp; Found</strong></div></div>'
        + '<div class="hw-ask">Is this yours?</div></div>'
    )
    qr_block = f'<div class="hw-qr"><img src="{qr}" alt="QR code"><span>Scan to search<br>all items</span></div>'

    if not items:
        page = (
            f'<div class="hw">{top}<div class="hw-empty"><h1>Nothing in Lost &amp; Found right now</h1>'
            "<p>Lost something? Scan the code to check online or ask at the front office.</p></div>"
            f'<div class="hw-bottom"><div class="hw-progress"></div>{qr_block}</div></div>'
        )
        st.markdown(page, unsafe_allow_html=True)
        return

    i = st.session_state.slide % len(items)
    st.session_state.slide = i + 1
    it = items[i]

    uri = thumb(it["image_path"], 1200)
    photo = f'<img src="{uri}" alt="">' if uri else "<span>No photo</span>"
    if it["holding_location"] == "Front Office" and it["slot_number"]:
        pick = (
            '<div class="hw-pick office"><div><small>PICK UP AT</small><b>Front Office</b></div>'
            f'<div class="hw-slot"><small>SLOT</small><span>{it["slot_number"]}</span></div></div>'
        )
    else:
        room = it.get("pickup_room") or it["location_found"]
        pick = f'<div class="hw-pick room"><div><small>PICK UP FROM</small><b>{e(room)}</b></div></div>'

    warning = last_chance_text(it)
    last = f'<div class="hw-last"><span class="icon">!</span><span>{e(warning)}</span></div>' if warning else ""
    pct = round((i + 1) / len(items) * 100)
    page = (
        f'<div class="hw">{top}<div class="hw-main"><div class="hw-photo">{photo}</div>'
        '<div class="hw-info">'
        f"{last}"
        f'<div class="hw-cat">{e(it["item_type"])} · Turned in {e(nice_date(it["date_found"]))}</div>'
        f"<h1>{e(item_title(it))}</h1>"
        f'<p class="hw-found">Found in {e(it["location_found"])}</p>'
        f"{pick}"
        '<p class="hw-id">Bring your student ID to pick it up.</p>'
        "</div></div>"
        f'<div class="hw-bottom"><div class="hw-progress"><span>{i + 1} of {len(items)}</span>'
        f'<div class="hw-bar"><div style="width:{pct}%"></div></div></div>{qr_block}</div></div>'
    )
    st.markdown(page, unsafe_allow_html=True)


# =========================================================
# 10. NAVIGATION
# =========================================================
find_pg = st.Page(find_page, title="Find an item", icon=":material/search:", url_path="find", default=True)
register_pg = st.Page(register_page, title="Turn in an item", icon=":material/add_box:", url_path="turn-in")
staff_pg = st.Page(staff_page, title="Staff", icon=":material/lock:", url_path="staff")
hallway_pg = st.Page(hallway_page, title="Hallway screen", icon=":material/tv:", url_path="hallway")

st.navigation([find_pg, register_pg, staff_pg, hallway_pg], position="top").run()
