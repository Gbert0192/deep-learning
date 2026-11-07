import base64
import io
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

BRAND = "DermaScan"
st.set_page_config(page_title=BRAND, page_icon="🩺", layout="wide")

IMG_DIR = Path(__file__).parent / "images"
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "model_deploy")
MODEL_PATH = os.path.join(DEPLOY_DIR, "model.keras")
LABELS_PATH = os.path.join(DEPLOY_DIR, "labels.json")
SLIDE_SIZE = (900, 560)

SLIDES = [
    ("nv", "Nevus melanositik", "Tahi lalat biasa: kumpulan sel pigmen yang jinak. Umumnya bulat, simetris, dengan warna cokelat yang merata.", "jinak"),
    ("mel", "Melanoma", "Kanker kulit dari sel pigmen yang dapat menyebar. Waspadai bentuk asimetris, tepi tidak rata, warna beragam, ukuran membesar, dan perubahan dari waktu ke waktu.", "ganas"),
    ("bkl", "Lesi mirip keratosis jinak", "Mencakup lentigo solar, keratosis seboroik, dan keratosis mirip lichen planus. Tampak seperti bercak atau kutil cokelat yang seolah menempel di kulit.", "jinak"),
    ("bcc", "Karsinoma sel basal", "Kanker kulit paling umum. Tumbuh lambat dan jarang menyebar, biasanya berupa benjolan mengilap seperti mutiara atau luka yang tak kunjung sembuh.", "ganas"),
    ("akiec", "Keratosis aktinik / penyakit Bowen", "Bercak kasar bersisik akibat paparan sinar matahari. Berpotensi pra-kanker dan dapat berkembang menjadi kanker sel skuamosa.", "pra"),
    ("vasc", "Lesi vaskular", "Angioma, angiokeratoma, granuloma piogenik, dan perdarahan. Berasal dari pembuluh darah sehingga tampak merah, ungu, atau kebiruan.", "jinak"),
    ("df", "Dermatofibroma", "Benjolan fibrosa kecil dan keras berwarna cokelat kemerahan, sering di tungkai. Jinak dan biasanya tidak perlu diobati.", "jinak"),
]
CATEGORY = {"jinak": "Jinak", "pra": "Pra-kanker", "ganas": "Ganas"}

KEYWORDS = {"nv": ["nevus", "nevi"], "mel": ["melanoma"], "bkl": ["benign", "seboroik", "seborrheic"],
            "bcc": ["basal"], "akiec": ["actinic", "bowen"], "vasc": ["vascular"], "df": ["dermatofibroma"]}
SLIDE_COLORS = ["#1F5AA6", "#174A8C", "#2A6BB8", "#1B4F9C", "#2F7BE0", "#2563B0", "#123F80"]

FEATURES = [
    ("Unggah foto", "Unggah citra lesi kulit yang terang dan fokus, atau ambil langsung lewat kamera."),
    ("Analisis AI", "Model MobileNetV2 hasil transfer learning menganalisis citra dalam hitungan detik."),
    ("Top-3 prediksi", "Hasil menampilkan tiga kemungkinan terbaik beserta tingkat keyakinannya."),
    ("Konsultasi dokter", "Ini skrining awal, bukan diagnosis. Selalu konsultasikan dengan dokter kulit."),
]


# ---------------------------------------------------------------- tampilan
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root { --navy:#0A3470; --blue:#1F5AA6; --blue2:#2F7BE0; --sky:#8CBBF0; --ice:#EAF2FC; --ink:#0F3D7A; }
.stApp { background: linear-gradient(160deg,#0A3470 0%,#124A92 55%,#1B5FA8 100%) fixed;
         font-family:'Montserrat',sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0 3rem 3rem !important; max-width: 100% !important; }
h1,h2,h3 { font-family:'Montserrat',sans-serif !important; color: var(--ink) !important; font-weight:800 !important; }
.topbar { display:flex; justify-content:space-between; align-items:center; padding:1.4rem 0 1rem; color:#fff; }
.brand { font-weight:800; font-size:1.25rem; letter-spacing:1px; text-transform:uppercase; }
.brand i { display:inline-block; width:14px; height:14px; border-radius:50%; background:var(--sky); margin-right:10px; box-shadow:0 0 12px var(--sky); }
.year { font-weight:500; opacity:.85; }
[data-testid="stRadio"] [role="radiogroup"] { gap:.6rem; justify-content:flex-start; margin-bottom:.8rem; }
[data-testid="stRadio"] label { background:rgba(255,255,255,.12); border:1.5px solid rgba(255,255,255,.55);
  border-radius:999px; padding:.35rem 1.5rem; cursor:pointer; }
[data-testid="stRadio"] label:has(input:checked) { background:#fff; border-color:#fff; }
[data-testid="stRadio"] label > span:first-child { display:none; }
[data-testid="stRadio"] label > div:first-child:not(:last-child) { display:none; }
[data-testid="stRadio"] label p { font-weight:700; color:#fff; margin:0; }
[data-testid="stRadio"] label:has(input:checked) p { color:var(--ink); }
.panel, [class*="st-key-panel"] { background:linear-gradient(180deg,#F6FAFF,#DCE9F9); border-radius:28px;
  padding:18px; margin-bottom:1.6rem; box-shadow:0 12px 40px rgba(0,0,0,.28); }
[class*="st-key-panel"] { padding:1.6rem 2rem 2rem; }
.hero-card { position:relative; overflow:hidden; min-height:360px; border-radius:20px; padding:2.6rem 3rem;
  background:linear-gradient(135deg,#1D56A5,#0F3F82); color:#fff; }
.hero-card::before { content:""; position:absolute; inset:0;
  background:radial-gradient(circle at 78% 50%, rgba(140,187,240,.55), transparent 48%); }
.capsule { position:absolute; right:7%; top:50%; width:380px; height:150px; border-radius:999px;
  transform:translateY(-50%) rotate(-35deg);
  background:linear-gradient(90deg,rgba(255,255,255,.92) 0 50%, rgba(111,168,232,.9) 50% 100%);
  box-shadow:0 0 60px rgba(140,187,240,.7), inset 0 0 25px rgba(255,255,255,.6); }
.hero-title { position:relative; margin:.8rem 0 0; padding:0; color:#fff; font-weight:800; font-size:clamp(3.2rem,7vw,6rem); line-height:1; letter-spacing:1px; text-align:left; }
.hero-card p { position:relative; max-width:26rem; margin-top:1.4rem; font-size:1rem; line-height:1.5; opacity:.92; }
.hero-card .tag { position:relative; font-weight:600; opacity:.85; }
.panel-title { font-weight:800; font-size:1.5rem; color:var(--ink); line-height:1.3; margin:.4rem .4rem 1rem; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
.fcard { position:relative; min-height:140px; padding:1.1rem 1.3rem; border-radius:16px; color:#fff;
  background:linear-gradient(135deg,#2A63B0,#123F80); }
.fcard b { font-size:1.05rem; display:block; margin-bottom:.4rem; }
.fcard p { font-size:.82rem; line-height:1.45; opacity:.88; margin:0; max-width:80%; }
.fcard .num { position:absolute; right:1.1rem; bottom:.5rem; font-size:2rem; font-weight:300; }
.slide-card { border-radius:20px; padding:14px; background:linear-gradient(135deg,#2A63B0,#123F80); }
.slide-card img { width:100%; display:block; border-radius:14px; }
.slide-no { font-size:4.2rem; font-weight:300; color:var(--blue2); line-height:1; margin-top:.3rem; }
.slide-no small { font-size:1.2rem; color:var(--ink); opacity:.6; }
.slide-name { font-size:1.9rem; font-weight:800; color:var(--ink); margin:.6rem 0 .4rem; }
.slide-tag { display:inline-block; padding:.25rem .85rem; border-radius:999px; color:#fff; font-size:.75rem; font-weight:700; margin-bottom:.8rem; }
.tag-jinak { background:#2F7BE0; }
.tag-pra { background:#E0952B; }
.tag-ganas { background:#D9534F; }
.slide-text { color:var(--ink); opacity:.85; line-height:1.55; }
.dots { margin:1.2rem 0 1rem; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#B9D1EF; margin-right:7px; }
.dot.on { background:var(--blue2); width:28px; border-radius:8px; }
.stButton > button { background:linear-gradient(135deg,#2F7BE0,#1B4F9C); color:#fff; border:none;
  border-radius:12px; padding:.55rem 1.4rem; font-weight:700; }
.stButton > button:hover { background:linear-gradient(135deg,#5B9BEF,#2F7BE0); color:#fff; }
[data-testid="stFileUploaderDropzone"] { background:#EAF2FC; border:2px dashed var(--blue2); border-radius:18px; }
.stTabs [data-baseweb="tab"] { font-weight:700; color:var(--ink); }
.stTabs [aria-selected="true"] { color:var(--blue2); }
.result { border-radius:20px; padding:1.4rem 1.6rem; color:#fff; background:linear-gradient(135deg,#2A63B0,#123F80); }
.result h2 { color:#fff !important; margin:0 0 .2rem; font-size:1.7rem; }
.result .bignum { font-size:3rem; font-weight:300; line-height:1.1; }
.result .sub { opacity:.8; font-size:.85rem; margin-bottom:.8rem; }
.meter { margin:.6rem 0; }
.meter .lbl { display:flex; justify-content:space-between; font-weight:600; font-size:.9rem; }
.meter .track { background:rgba(255,255,255,.22); border-radius:999px; height:10px; overflow:hidden; margin-top:4px; }
.meter .fill { height:100%; background:var(--sky); border-radius:999px; }
.note { font-size:.8rem; color:var(--ink); opacity:.75; margin-top:1rem; }
</style>
"""


def inject_css():
    css = "\n".join(line.strip() for line in CSS.splitlines() if line.strip())
    st.markdown(css, unsafe_allow_html=True)


def cover_crop(img: Image.Image, size) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS)


def make_placeholder(i: int) -> Image.Image:
    title = SLIDES[i][1]
    base = Image.new("RGB", SLIDE_SIZE, SLIDE_COLORS[i])
    d = ImageDraw.Draw(base)
    for r in range(260, 0, -20):  
        c = tuple(min(255, v + (260 - r) // 3) for v in base.getpixel((5, 5)))
        d.ellipse((SLIDE_SIZE[0] - 330 - r // 2, 60, SLIDE_SIZE[0] - 330 + r, 60 + r * 1.6), fill=c)
    try:
        font = ImageFont.load_default(size=54)
    except TypeError:
        font = ImageFont.load_default()
    d.text((50, SLIDE_SIZE[1] - 120), f"{SLIDES[i][0]}  {title}", fill="#FFFFFF", font=font)
    return base


def match_images():
    files = sorted(p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    used, picked = set(), {}

    def take(code, test):
        for p in files:
            if p not in used and test(p.stem.lower()):
                used.add(p)
                picked[code] = p
                return

    for code, *_ in SLIDES:
        take(code, lambda n, c=code: c in re.split(r"[^a-z0-9]+", n))
    for code, *_ in SLIDES:
        if code not in picked:
            take(code, lambda n, c=code: any(k in n for k in KEYWORDS[c]))
    leftover = [p for p in files if p not in used]
    for code, *_ in SLIDES:
        if code not in picked and leftover:
            picked[code] = leftover.pop(0)
    return picked


@st.cache_data(show_spinner=False)
def load_slides():
    picked = match_images()
    out = []
    for i, (code, *_) in enumerate(SLIDES):
        img = cover_crop(Image.open(picked[code]), SLIDE_SIZE) if code in picked else make_placeholder(i)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


# ---------------------------------------------------------------- backend model
@st.cache_resource(show_spinner="Memuat model... pertama kali bisa 1-2 menit, mohon tunggu.")
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, None
    import tensorflow as tf 
    from keras.models import load_model

    with open(LABELS_PATH) as file:
        labels = json.load(file)
    model = load_model(MODEL_PATH, compile=False)
    return model, labels


def predict_bytes(model, labels, image_bytes):
    import tensorflow as tf

    classes = labels["classes"]
    class_names = labels["class_names"]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    array = tf.image.resize(np.array(image), (labels["img_size"], labels["img_size"]))
    probs = model(tf.expand_dims(array, 0), training=False).numpy()[0]

    top3 = probs.argsort()[::-1][:3]
    best_index = int(top3[0])
    return {
        "label": classes[best_index],
        "name": class_names[classes[best_index]],
        "confidence": float(probs[best_index]),
        "low_confidence": bool(probs[best_index] < labels["confidence_threshold"]),
        "top3": [
            {"label": classes[int(i)], "name": class_names[classes[int(i)]], "confidence": float(probs[i])}
            for i in top3
        ],
    }


# ---------------------------------------------------------------- halaman
def page_home():
    slides = load_slides()
    if "idx" not in st.session_state:  
        st.session_state.idx = 0
    i = st.session_state.idx
    code, title, desc, cat = SLIDES[i]

    st.markdown(
        '<div class="panel"><div class="hero-card"><div class="capsule"></div>'
        '<div class="tag">Skrining kulit berbasis AI</div>'
        '<div class="hero-title">DERMA<br>SCAN</div>'
        '<p>Pengenalan awal tujuh jenis lesi kulit dengan deep learning (MobileNetV2, dataset HAM10000). '
        'Unggah foto, dapatkan hasilnya dalam hitungan detik.</p></div></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="panel_carousel"):
        st.markdown('<div class="panel-title">7 jenis lesi kulit</div>', unsafe_allow_html=True)
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            st.markdown(
                f'<div class="slide-card"><img src="data:image/jpeg;base64,{slides[i]}" alt="{title}"></div>',
                unsafe_allow_html=True,
            )
        with right:
            dots = "".join(f'<span class="dot {"on" if k == i else ""}"></span>' for k in range(len(SLIDES)))
            st.markdown(
                f'<div class="slide-no">{i + 1:02d}<small> / {len(SLIDES)}</small></div>'
                f'<div class="slide-name">{title}</div>'
                f'<span class="slide-tag tag-{cat}">{code.upper()} &middot; {CATEGORY[cat]}</span>'
                f'<div class="slide-text">{desc}</div>'
                f'<div class="dots">{dots}</div>',
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns(2)
            if b1.button("Sebelumnya", key="prev"):
                st.session_state.idx = (i - 1) % len(SLIDES)
                st.rerun()
            if b2.button("Berikutnya", key="next"):
                st.session_state.idx = (i + 1) % len(SLIDES)
                st.rerun()
        st.markdown('<p class="note">Gambar contoh: dataset HAM10000 (Tschandl dkk., 2018), lisensi CC BY-NC 4.0.</p>',
                    unsafe_allow_html=True)

    cards = "".join(
        f'<div class="fcard"><b>{t}</b><p>{d}</p><span class="num">{n:02d}</span></div>'
        for n, (t, d) in enumerate(FEATURES, 1)
    )
    st.markdown(
        f'<div class="panel"><div class="panel-title">Cara kerja {BRAND}: cepat, sederhana, dan terukur</div>'
        f'<div class="grid2">{cards}</div></div>',
        unsafe_allow_html=True,
    )


def page_check():
    with st.container(key="panel_check"):
        st.markdown('<div class="panel-title">Cek kulit: unggah citra lesi, dapatkan prediksi</div>', unsafe_allow_html=True)

        model, labels = load_artifacts()
        if model is None:
            st.error(
                "File model belum ditemukan. Letakkan `model.keras` dan `labels.json` "
                "di folder `model_deploy/` di samping `app.py`, lalu muat ulang halaman."
            )
            return

        left, right = st.columns(2, gap="large")
        with left:
            tab_up, tab_cam = st.tabs(["Unggah foto", "Pakai kamera"])
            with tab_up:
                file = st.file_uploader("Pilih gambar (foto lesi yang terang, fokus, dari dekat)", type=["jpg", "jpeg", "png"])
            with tab_cam:
                cam = st.camera_input("Ambil foto")
        source = file or cam

        if source is None:
            with right:
                st.info("Belum ada foto. Unggah gambar atau ambil foto dulu ya.")
            return

        image_bytes = source.getvalue()
        with st.spinner("Menganalisis..."):
            result = predict_bytes(model, labels, image_bytes)

        with left:
            st.image(image_bytes, caption="Citra yang diunggah", use_container_width=True)

        with right:
            bars = "".join(
                f'<div class="meter"><div class="lbl"><span>{it["name"]}</span><span>{it["confidence"]:.1%}</span></div>'
                f'<div class="track"><div class="fill" style="width:{it["confidence"] * 100:.1f}%"></div></div></div>'
                for it in result["top3"]
            )
            st.markdown(
                f'<div class="result"><h2>{result["name"]}</h2>'
                f'<div class="bignum">{result["confidence"]:.1%}</div>'
                f'<div class="sub">Kode kelas: {result["label"]} &middot; tingkat keyakinan</div>'
                f'<b>Top-3 prediksi</b>{bars}</div>',
                unsafe_allow_html=True,
            )
            if result["low_confidence"]:
                st.warning(f"Keyakinan rendah ({result['confidence']:.1%}), hasil mungkin tidak akurat.")

        st.markdown(
            '<p class="note">Model MobileNetV2 (transfer learning) dilatih dengan dataset HAM10000 '
            "untuk tujuh jenis lesi kulit. Hasil ini bukan diagnosis medis; konsultasikan dengan dokter kulit.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- main
inject_css()
st.markdown(f'<div class="topbar"><div class="brand"><i></i>{BRAND}</div><div class="year">2026</div></div>',
            unsafe_allow_html=True)
menu = st.radio("Menu", ["Beranda", "Cek kulit"], horizontal=True, label_visibility="collapsed")
if menu == "Beranda":
    page_home()
else:
    page_check()