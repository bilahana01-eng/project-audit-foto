import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from PIL import Image, UnidentifiedImageError, ImageFilter
import imagehash
import io
import os
import re
import sqlite3
import zipfile
import hashlib
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# =========================
# CONFIG UI
# =========================
st.set_page_config(page_title="Audit Foto Patroli - Center Focused", layout="wide")
st.title("🕵️ AUDIT FOTO PATROLI (CENTER AUDIT MODE)")
st.caption(
    "Sistem ini hanya mengaudit bagian TENGAH foto untuk menghindari kesalahan deteksi akibat LOGO atau OVERLAY GEO yang selalu sama."
)

# =========================
# STREAMLIT UI
# =========================
uploaded = st.file_uploader("Upload Excel Patroli (.xlsx)", type=["xlsx"])

colA, colB = st.columns([1, 2])
with colA:
    preview_limit = st.number_input("Maks preview gambar", min_value=0, max_value=500, value=200, step=10)
with colB:
    st.info("Logika: Area atas 25% (Logo) dan Bawah 25% (GEO) akan dipotong sebelum proses hashing.")

# =========================
# DATABASE
# =========================
DB_PATH = "audit_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            sha256 TEXT PRIMARY KEY,
            phash  TEXT,
            source_type TEXT,
            source_file TEXT,
            sheet TEXT,
            location TEXT,
            cluster TEXT,
            segment TEXT,
            url TEXT,
            first_seen DATE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_phash ON history(phash)")
    return conn

def db_lookup(conn, sha256_hex: str, phash_str: str):
    exact = conn.execute(
        "SELECT source_file, sheet, location, cluster, segment, url, first_seen FROM history WHERE sha256=?",
        (sha256_hex,)
    ).fetchone()

    ph = conn.execute(
        "SELECT source_file, sheet, location, cluster, segment, url, first_seen FROM history WHERE phash=? LIMIT 1",
        (phash_str,)
    ).fetchone()

    return exact, ph

def db_insert(conn, row: dict):
    conn.execute("""
        INSERT OR IGNORE INTO history
        (sha256, phash, source_type, source_file, sheet, location, cluster, segment, url, first_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["sha256"], row["phash"], row["source_type"], row["source_file"],
        row["sheet"], row["location"], row["cluster"], row["segment"], row["url"], row["first_seen"]
    ))

# =========================
# HASHING (MODIFIED: CENTER CROP)
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def compute_hashes_from_bytes(img_bytes: bytes):
    try:
        img_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img_original.size
        
        # --- PERINTAH: AUDIT BAGIAN TENGAH SAJA ---
        # Potong 25% atas (buang Logo)
        # Potong 25% bawah (buang GEO Overlay)
        # Ambil 10% margin kiri-kanan
        top = int(h * 0.25)
        bottom = int(h * 0.75)
        left = int(w * 0.10)
        right = int(w * 0.90)
        
        # Area inti yang diaudit
        audit_area = img_original.crop((left, top, right, bottom))
        # ------------------------------------------

        # Thumb untuk pHash dari area tengah
        thumb = audit_area.copy()
        thumb.thumbnail((260, 260))
        ph = str(imagehash.phash(thumb))
        
        # SHA256 dari bytes area tengah (agar logo tidak merusak hash)
        img_byte_arr = io.BytesIO()
        audit_area.save(img_byte_arr, format='PNG')
        sh = sha256_bytes(img_byte_arr.getvalue())
        
        return sh, ph, thumb, img_original
    except UnidentifiedImageError:
        return None, None, None, None

# =========================
# METRIK VISUAL (LOGO & GEO DETECTION)
# =========================
def image_entropy_gray(img: Image.Image) -> float:
    g = img.convert("L").resize((256, 256))
    hist = g.histogram()
    total = sum(hist)
    if total == 0: return 0.0
    ent = 0.0
    for h in hist:
        if h:
            p = h / total
            ent -= p * math.log2(p)
    return ent

def edge_density(img: Image.Image) -> float:
    g = img.convert("L").resize((256, 256)).filter(ImageFilter.FIND_EDGES)
    px = list(g.getdata())
    mean = sum(px) / (len(px) or 1)
    return mean / 255.0

def unique_color_ratio(img: Image.Image, k=24) -> float:
    small = img.resize((256, 256))
    q = small.quantize(colors=k, method=2)
    px = list(q.getdata())
    uniq = len(set(px))
    return uniq / float(k)

def bright_ratio(img: Image.Image, thr=240) -> float:
    g = img.convert("L").resize((256, 256))
    px = list(g.getdata())
    return sum(1 for p in px if p >= thr) / (len(px) or 1)

def is_logo_only(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 140 or h < 140: return True, "LogoOnly: kecil"
    ent = image_entropy_gray(img)
    ed = edge_density(img)
    ucr = unique_color_ratio(img, k=24)
    br = bright_ratio(img, thr=240)
    if br >= 0.55 and ucr <= 0.45 and ed <= 0.060:
        return True, "LogoOnly(WhiteBG)"
    if ent <= 4.10 and ucr <= 0.40 and ed <= 0.060:
        return True, "LogoOnly(Flat)"
    return False, "NotLogo"

def _patch_stats(patch_gray: Image.Image):
    px = list(patch_gray.getdata())
    n = len(px) or 1
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    std = math.sqrt(var)
    white_ratio = sum(1 for p in px if p >= 220) / n
    dark_ratio  = sum(1 for p in px if p <= 70) / n
    ed = edge_density(patch_gray.convert("RGB"))
    return mean, std, white_ratio, dark_ratio, ed

def overlay_geo_best(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 240 or h < 240: return False, "NonGEO: kecil"
    rois = [("LB", (0, 0.62, 0.5, 1)), ("MB", (0.2, 0.62, 0.8, 1)), ("RB", (0.5, 0.62, 1, 1))]
    best_score = -1
    for name, (x0, y0, x1, y1) in rois:
        patch = img.crop((int(w*x0), int(h*y0), int(w*x1), int(h*y1))).convert("L").resize((260, 260))
        mean, std, white_ratio, dark_ratio, pedge = _patch_stats(patch)
        score = int(white_ratio>=0.003) + int(dark_ratio>=0.006) + int(std>=14) + int(pedge>=0.030)
        if score > best_score: best_score = score
    if best_score >= 2: return True, "GEO✅"
    return False, "NonGEO"

def classify_for_audit(img: Image.Image) -> tuple[bool, str, str]:
    ok_geo, geo_dbg = overlay_geo_best(img)
    if ok_geo: return True, "", ""
    logo, logo_dbg = is_logo_only(img)
    if logo: return False, "⏭️ SKIP (Logo-Only)", logo_dbg
    return False, "⏭️ SKIP (Non-Patroli)", ""

# =========================
# OPERASI EXCEL & AUDIT (SAMA SEPERTI KODINGANMU)
# =========================
# [Sisa fungsi apply_dup_of_columns, audit_workbook, dan UI Actions tetap sama]
# [Namun pastikan memanggil compute_hashes_from_bytes yang baru di atas]

# ... (Lanjutkan dengan fungsi audit_workbook dan logic download dari kodinganmu)
# Karena keterbatasan panjang teks, pastikan bagian compute_hashes_from_bytes di atas sudah terupdate.
