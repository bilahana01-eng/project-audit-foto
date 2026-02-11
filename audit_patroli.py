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
st.set_page_config(page_title="Audit Foto Patroli", layout="wide")
st.title("🕵️ AUDIT FOTO PATROLI")
st.caption(
    "Audit foto patroli duplicate (Embedded Excel + Google Docs/Drive). "
    "RULE: Foto GEO overlay (kotak gelap + teks putih timestamp/koordinat) => DIAUDIT (VALID/GUGUR). "
    "Logo-only/banner-only => SKIP (tidak ikut audit). Logo perusahaan pada foto GEO BOLEH."
)

uploaded = st.file_uploader("Upload Excel Patroli (.xlsx)", type=["xlsx"])
colA, colB = st.columns([1, 2])
with colA:
    preview_limit = st.number_input("Maks preview gambar", min_value=0, max_value=500, value=120, step=10)
with colB:
    st.info("Reset history ada di sidebar. Duplikat akan ditunjukkan 'duplikat dari foto mana' (lengkap: segment + link + lokasi Excel).")

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
    # upgrade schema: add thumb blob if missing
    try:
        conn.execute("ALTER TABLE history ADD COLUMN thumb_jpg BLOB")
    except sqlite3.OperationalError:
        pass
    return conn

def db_lookup(conn, sha256_hex: str, phash_str: str):
    exact = conn.execute(
        "SELECT source_type, source_file, sheet, location, cluster, segment, url, first_seen, thumb_jpg "
        "FROM history WHERE sha256=?",
        (sha256_hex,)
    ).fetchone()

    ph = conn.execute(
        "SELECT source_type, source_file, sheet, location, cluster, segment, url, first_seen, thumb_jpg "
        "FROM history WHERE phash=? LIMIT 1",
        (phash_str,)
    ).fetchone()

    return exact, ph

def db_insert(conn, row: dict):
    conn.execute("""
        INSERT OR IGNORE INTO history
        (sha256, phash, source_type, source_file, sheet, location, cluster, segment, url, first_seen, thumb_jpg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["sha256"], row["phash"], row["source_type"], row["source_file"],
        row["sheet"], row["location"], row["cluster"], row["segment"], row["url"], row["first_seen"],
        row.get("thumb_jpg", None)
    ))

# =========================
# RESET (SIDEBAR)
# =========================
st.sidebar.subheader("🧹 Reset History Audit")
confirm_reset = st.sidebar.checkbox("Saya yakin mau hapus total history", value=False)

if st.sidebar.button("🗑️ HAPUS TOTAL HISTORY (RESET)"):
    if not confirm_reset:
        st.sidebar.warning("Centang konfirmasi dulu.")
    else:
        try:
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
                st.sidebar.success("✅ History dihapus. App akan refresh.")
                st.rerun()
            else:
                st.sidebar.info("ℹ️ History sudah kosong (file DB tidak ada).")
        except Exception as e:
            st.sidebar.error(f"❌ Gagal hapus DB: {e}")

# =========================
# HASHING + THUMB SERIALIZE
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def pil_to_jpg_bytes(img: Image.Image, max_size=240, quality=70) -> bytes:
    t = img.copy()
    t.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    t.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def jpg_bytes_to_pil(b: bytes) -> Image.Image | None:
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def compute_hashes_from_bytes(img_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return None, None, None, None, None
    thumb = img.copy()
    thumb.thumbnail((240, 240))
    ph = str(imagehash.phash(thumb))
    sh = sha256_bytes(img_bytes)
    thumb_jpg = pil_to_jpg_bytes(img, max_size=240, quality=70)
    return sh, ph, thumb, img, thumb_jpg

# =========================
# LOGO-ONLY DETECTION (ANTI "FAMIKA DOANG")
# =========================
def _down_rgb(img: Image.Image, size=256) -> Image.Image:
    return img.resize((size, size))

def _near_white_ratio(rgb: Image.Image, thr=242) -> float:
    px = list(rgb.getdata())
    n = len(px) or 1
    cnt = 0
    for r, g, b in px:
        if r >= thr and g >= thr and b >= thr:
            cnt += 1
    return cnt / n

def _dominant_colors(rgb: Image.Image, colors=24, min_frac=0.01) -> int:
    pal = rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
    counts = pal.getcolors(256 * 256) or []
    total = 256 * 256
    dom = 0
    for c, _ in counts:
        if (c / total) >= min_frac:
            dom += 1
    return dom

def _edge_ratio(edge_img: Image.Image, thr=40) -> float:
    px = list(edge_img.getdata())
    n = len(px) or 1
    return sum(1 for p in px if p >= thr) / n

def _edge_block_coverage(edge_img: Image.Image, grid=4, thr=40, min_block_ratio=0.015) -> int:
    w, h = edge_img.size
    bw = w // grid
    bh = h // grid
    active = 0
    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * bw
            y0 = gy * bh
            x1 = w if gx == grid - 1 else (gx + 1) * bw
            y1 = h if gy == grid - 1 else (gy + 1) * bh
            patch = edge_img.crop((x0, y0, x1, y1))
            if _edge_ratio(patch, thr=thr) >= min_block_ratio:
                active += 1
    return active

def is_logo_only(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 160 or h < 160:
        return True, "LogoOnly: kecil"

    rgb = _down_rgb(img, 256)
    gray = rgb.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)

    white = _near_white_ratio(rgb, thr=242)
    dom = _dominant_colors(rgb, colors=24, min_frac=0.01)
    blocks = _edge_block_coverage(edge, grid=4, thr=40, min_block_ratio=0.015)
    er = _edge_ratio(edge, thr=40)
    ar = w / max(h, 1)

    if white >= 0.55 and dom <= 8 and blocks <= 6:
        return True, f"LogoOnly: white={white:.2f}, dom={dom}, blocks={blocks}, er={er:.3f}"
    if ar >= 2.1 and white >= 0.40 and dom <= 10 and blocks <= 6:
        return True, f"LogoOnly: ar={ar:.2f}, white={white:.2f}, dom={dom}, blocks={blocks}"

    return False, f"NotLogo: white={white:.2f}, dom={dom}, blocks={blocks}, er={er:.3f}"

# =========================
# GEO OVERLAY DETECTION
# =========================
def patch_edge_density(patch_gray: Image.Image) -> float:
    e = patch_gray.filter(ImageFilter.FIND_EDGES)
    px = list(e.getdata())
    mean = sum(px) / (len(px) or 1)
    return mean / 255.0

def _patch_stats(patch_gray: Image.Image):
    px = list(patch_gray.getdata())
    n = len(px) or 1
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    std = math.sqrt(var)
    white_ratio = sum(1 for p in px if p >= 215) / n
    dark_ratio  = sum(1 for p in px if p <= 95) / n
    ed = patch_edge_density(patch_gray)
    return mean, std, white_ratio, dark_ratio, ed

def overlay_geo_best(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 240 or h < 240:
        return False, "NonGEO: kecil"

    rois = [
        ("LB",  (0.00, 0.62, 0.52, 1.00)),
        ("MB",  (0.18, 0.62, 0.82, 1.00)),
        ("RB",  (0.40, 0.62, 1.00, 1.00)),
        ("RB2", (0.58, 0.72, 1.00, 1.00)),
    ]

    best_dbg = ""
    for name, (x0, y0, x1, y1) in rois:
        X0 = int(w * x0); Y0 = int(h * y0)
        X1 = int(w * x1); Y1 = int(h * y1)

        patch = img.crop((X0, Y0, X1, Y1)).convert("L").resize((260, 260))
        mean, std, white_ratio, dark_ratio, ed = _patch_stats(patch)

        has_dark_box   = dark_ratio  >= 0.010
        has_white_text = white_ratio >= 0.003
        has_contrast   = std >= 12
        mean_ok        = mean <= 210
        edge_ok        = ed >= 0.030

        passed = has_dark_box and has_white_text and has_contrast and mean_ok and edge_ok
        dbg = (f"{name}: mean={mean:.1f}, std={std:.1f}, "
               f"white={white_ratio:.3f}, dark={dark_ratio:.3f}, edge={ed:.3f}, pass={passed}")

        if passed:
            return True, f"GEO✅ ({dbg})"
        best_dbg = dbg

    return False, f"NonGEO ({best_dbg})"

def classify_for_audit(img: Image.Image) -> tuple[bool, str, str]:
    logo, logo_dbg = is_logo_only(img)
    if logo:
        return False, "⏭️ SKIP (Logo-Only)", logo_dbg

    ok_geo, geo_dbg = overlay_geo_best(img)
    if ok_geo:
        return True, "", ""
    return False, "⏭️ SKIP (Non-Patroli)", geo_dbg

# =========================
# GOOGLE LINK HANDLING
# =========================
DOC_ID_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")
DRIVE_FILE_ID_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
GENERIC_ID_RE = re.compile(r"(?:id=)([a-zA-Z0-9_-]+)")

def build_gdocs_export_docx_url(doc_id: str) -> str:
    return f"https://docs.google.com/document/d/{doc_id}/export?format=docx"

def build_drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def http_get_bytes(url: str, timeout=25):
    try:
        r = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return None
        return r.content
    except requests.RequestException:
        return None

def extract_images_from_docx_bytes(docx_bytes: bytes) -> list[bytes]:
    out = []
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as z:
        for name in z.namelist():
            if name.startswith("word/media/"):
                try:
                    out.append(z.read(name))
                except:
                    pass
    return out

def download_images_from_url(url: str) -> list[bytes]:
    if not url or not isinstance(url, str):
        return []
    m = DOC_ID_RE.search(url)
    if m:
        doc_id = m.group(1)
        docx_bytes = http_get_bytes(build_gdocs_export_docx_url(doc_id))
        if not docx_bytes:
            return []
        return extract_images_from_docx_bytes(docx_bytes)
    m = DRIVE_FILE_ID_RE.search(url)
    if m:
        b = http_get_bytes(build_drive_download_url(m.group(1)))
        return [b] if b else []
    m = GENERIC_ID_RE.search(url)
    if m and "google" in url:
        b = http_get_bytes(build_drive_download_url(m.group(1)))
        return [b] if b else []
    return []

# =========================
# EXCEL PARSING
# =========================
def find_header_row_and_cols(ws, max_scan_rows=40):
    target = {"cluster": ["cluster"], "segment": ["segment", "segment name", "segmen"], "link": ["link", "url"]}
    def norm(v): return str(v).strip().lower() if v is not None else ""

    for r in range(1, min(max_scan_rows, ws.max_row) + 1):
        row_vals = [norm(ws.cell(r, c).value) for c in range(1, min(ws.max_column, 30) + 1)]
        if not any(row_vals):
            continue

        col_cluster = col_segment = col_link = None
        for idx, val in enumerate(row_vals, start=1):
            if any(k in val for k in target["cluster"]): col_cluster = idx
            if any(k in val for k in target["segment"]): col_segment = idx
            if any(k in val for k in target["link"]): col_link = idx

        if col_cluster and col_segment and col_link:
            return r, col_cluster, col_segment, col_link

    return 4, 2, 3, 7

def fmt_ref(source_file, sheet, location, cluster, segment, url, first_seen=""):
    # string ringkas tapi jelas (buat dup_of / preview)
    seg = (segment or "").strip()
    clu = (cluster or "").strip()
    u = (url or "").strip()
    fs = (first_seen or "").strip()
    parts = [
        f"{source_file} | {sheet} | {location}",
        f"Cluster: {clu}" if clu else "",
        f"Segment: {seg}" if seg else "",
        f"Link: {u}" if u else "",
        f"Tanggal: {fs}" if fs else "",
    ]
    return " | ".join([p for p in parts if p])

def extract_embedded_images(wb, source_file_name: str):
    items = []
    for ws in wb.worksheets:
        imgs = getattr(ws, "_images", [])
        if not imgs:
            continue

        header_row, col_cluster, col_segment, _ = find_header_row_and_cols(ws)

        for img_obj in imgs:
            try:
                row = img_obj.anchor._from.row + 1
                col = img_obj.anchor._from.col + 1
                cluster = ws.cell(row=row, column=col_cluster).value or "N/A"
                segment = ws.cell(row=row, column=col_segment).value or "N/A"

                raw = img_obj._data()
                sh, ph, thumb, full_img, thumb_jpg = compute_hashes_from_bytes(raw)
                if full_img is None:
                    continue

                audit_ok, skip_status, skip_reason = classify_for_audit(full_img)
                location = f"R{row}C{col}"

                base = {
                    "source_type": "EmbeddedExcel",
                    "source_file": source_file_name,
                    "sheet": ws.title,
                    "location": location,
                    "cluster": str(cluster),
                    "segment": str(segment),
                    "url": "",
                    "sha256": "",
                    "phash": "",
                    "thumb_jpg": None,
                    "status_akhir": "",
                    "skip_reason": "",
                    "dup_type": "",
                    "dup_of": "",
                    "ref_source_file": "",
                    "ref_sheet": "",
                    "ref_location": "",
                    "ref_cluster": "",
                    "ref_segment": "",
                    "ref_url": "",
                    "ref_first_seen": "",
                    "ref_thumb_jpg": None,
                    "thumb": thumb,
                }

                if not audit_ok:
                    base["status_akhir"] = skip_status
                    base["skip_reason"] = skip_reason
                    items.append(base)
                    continue

                base["sha256"] = sh
                base["phash"] = ph
                base["thumb_jpg"] = thumb_jpg
                items.append(base)

            except:
                continue
    return items

def extract_link_images(wb, source_file_name: str, max_workers=12):
    jobs = []
    for ws in wb.worksheets:
        header_row, col_cluster, col_segment, col_link = find_header_row_and_cols(ws)
        for r in range(header_row + 1, ws.max_row + 1):
            url = ws.cell(r, col_link).value
            if not url:
                continue
            url = str(url)
            if "google" not in url:
                continue

            cluster = ws.cell(r, col_cluster).value or "N/A"
            segment = ws.cell(r, col_segment).value or "N/A"
            jobs.append((ws.title, r, col_link, str(cluster), str(segment), url))

    items = []
    if not jobs:
        return items

    def worker(job):
        sheet, r, col_link, cluster, segment, url = job
        img_bytes_list = download_images_from_url(url)

        out = []
        for idx, b in enumerate(img_bytes_list, start=1):
            if not b:
                continue
            sh, ph, thumb, full_img, thumb_jpg = compute_hashes_from_bytes(b)
            if full_img is None:
                continue

            audit_ok, skip_status, skip_reason = classify_for_audit(full_img)
            location = f"R{r}C{col_link}#IMG{idx}"

            base = {
                "source_type": "CloudLink",
                "source_file": source_file_name,
                "sheet": sheet,
                "location": location,
                "cluster": cluster,
                "segment": segment,
                "url": url,
                "sha256": "",
                "phash": "",
                "thumb_jpg": None,
                "status_akhir": "",
                "skip_reason": "",
                "dup_type": "",
                "dup_of": "",
                "ref_source_file": "",
                "ref_sheet": "",
                "ref_location": "",
                "ref_cluster": "",
                "ref_segment": "",
                "ref_url": "",
                "ref_first_seen": "",
                "ref_thumb_jpg": None,
                "thumb": thumb,
            }

            if not audit_ok:
                base["status_akhir"] = skip_status
                base["skip_reason"] = skip_reason
                out.append(base)
                continue

            base["sha256"] = sh
            base["phash"] = ph
            base["thumb_jpg"] = thumb_jpg
            out.append(base)

        return out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for f in as_completed(futures):
            try:
                items.extend(f.result())
            except:
                pass

    return items

# =========================
# AUDIT LOGIC (PAIRING DUPLICATE + REF DETAIL)
# =========================
def audit_workbook(xlsx_path: str):
    wb = load_workbook(xlsx_path, data_only=True)
    source_file_name = os.path.basename(xlsx_path)

    df = pd.DataFrame(extract_embedded_images(wb, source_file_name) + extract_link_images(wb, source_file_name))
    if df.empty:
        return df

    audited_mask = (df["status_akhir"].astype(str).str.strip() == "") & (df["sha256"].astype(str).str.strip() != "")
    df_audit = df[audited_mask].copy()
    df_skip = df[~audited_mask].copy()

    if df_audit.empty:
        return df

    # ------- INTERNAL PAIRING (exact & phash)
    first_sha = {}
    first_ph = {}

    df_audit["dup_internal_exact"] = False
    df_audit["dup_internal_phash"] = False

    for idx, r in df_audit.iterrows():
        sh = r["sha256"]
        ph = r["phash"]

        # exact
        if sh in first_sha:
            ref_idx = first_sha[sh]
            ref = df_audit.loc[ref_idx]

            df_audit.at[idx, "dup_internal_exact"] = True
            df_audit.at[idx, "dup_type"] = "INTERNAL_EXACT"
            df_audit.at[idx, "dup_of"] = fmt_ref(
                ref["source_file"], ref["sheet"], ref["location"],
                ref["cluster"], ref["segment"], ref["url"], ""
            )
            df_audit.at[idx, "ref_source_file"] = ref["source_file"]
            df_audit.at[idx, "ref_sheet"] = ref["sheet"]
            df_audit.at[idx, "ref_location"] = ref["location"]
            df_audit.at[idx, "ref_cluster"] = ref["cluster"]
            df_audit.at[idx, "ref_segment"] = ref["segment"]
            df_audit.at[idx, "ref_url"] = ref["url"]
            df_audit.at[idx, "ref_thumb_jpg"] = ref.get("thumb_jpg", None)

        else:
            first_sha[sh] = idx

        # similar (phash)
        if ph in first_ph:
            ref_idx = first_ph[ph]
            ref = df_audit.loc[ref_idx]
            df_audit.at[idx, "dup_internal_phash"] = True

            # kalau belum punya dup_type dari exact, isi dari phash
            if not str(df_audit.at[idx, "dup_type"]).strip():
                df_audit.at[idx, "dup_type"] = "INTERNAL_SIMILAR"
                df_audit.at[idx, "dup_of"] = fmt_ref(
                    ref["source_file"], ref["sheet"], ref["location"],
                    ref["cluster"], ref["segment"], ref["url"], ""
                )
                df_audit.at[idx, "ref_source_file"] = ref["source_file"]
                df_audit.at[idx, "ref_sheet"] = ref["sheet"]
                df_audit.at[idx, "ref_location"] = ref["location"]
                df_audit.at[idx, "ref_cluster"] = ref["cluster"]
                df_audit.at[idx, "ref_segment"] = ref["segment"]
                df_audit.at[idx, "ref_url"] = ref["url"]
                df_audit.at[idx, "ref_thumb_jpg"] = ref.get("thumb_jpg", None)
        else:
            first_ph[ph] = idx

    # ------- HISTORY CHECK (lintas bulan)
    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")
    df_audit["first_seen"] = today

    hist_status = []
    hist_detail = []

    for idx, row in df_audit.iterrows():
        exact, ph = db_lookup(conn, row["sha256"], row["phash"])

        if exact:
            # exact tuple: source_type, source_file, sheet, location, cluster, segment, url, first_seen, thumb
            hist_status.append("REUPLOAD_EXACT")
            hist_detail.append(f"Pernah terbit {exact[7]} | {exact[1]} | {exact[2]} | {exact[3]}")

            df_audit.at[idx, "dup_type"] = "HISTORY_EXACT"
            df_audit.at[idx, "ref_source_file"] = exact[1]
            df_audit.at[idx, "ref_sheet"] = exact[2]
            df_audit.at[idx, "ref_location"] = exact[3]
            df_audit.at[idx, "ref_cluster"] = exact[4]
            df_audit.at[idx, "ref_segment"] = exact[5]
            df_audit.at[idx, "ref_url"] = exact[6]
            df_audit.at[idx, "ref_first_seen"] = exact[7]
            df_audit.at[idx, "ref_thumb_jpg"] = exact[8]

            df_audit.at[idx, "dup_of"] = fmt_ref(
                exact[1], exact[2], exact[3], exact[4], exact[5], exact[6], exact[7]
            )

        elif ph:
            hist_status.append("REUPLOAD_SIMILAR_PHASH")
            hist_detail.append(f"Mirip foto lama {ph[7]} | {ph[1]} | {ph[2]} | {ph[3]}")

            df_audit.at[idx, "dup_type"] = "HISTORY_SIMILAR"
            df_audit.at[idx, "ref_source_file"] = ph[1]
            df_audit.at[idx, "ref_sheet"] = ph[2]
            df_audit.at[idx, "ref_location"] = ph[3]
            df_audit.at[idx, "ref_cluster"] = ph[4]
            df_audit.at[idx, "ref_segment"] = ph[5]
            df_audit.at[idx, "ref_url"] = ph[6]
            df_audit.at[idx, "ref_first_seen"] = ph[7]
            df_audit.at[idx, "ref_thumb_jpg"] = ph[8]

            df_audit.at[idx, "dup_of"] = fmt_ref(
                ph[1], ph[2], ph[3], ph[4], ph[5], ph[6], ph[7]
            )

        else:
            hist_status.append("NEW")
            hist_detail.append("")
            if not str(df_audit.at[idx, "dup_type"]).strip():
                df_audit.at[idx, "dup_type"] = "NONE"
                df_audit.at[idx, "dup_of"] = ""

    df_audit["history_status"] = hist_status
    df_audit["history_detail"] = hist_detail

    # ------- FINAL DECISION
    def decide(r):
        if r["history_status"] == "REUPLOAD_EXACT":
            return "❌ GUGUR (Pernah Terbit - Exact)"
        if r["history_status"] == "REUPLOAD_SIMILAR_PHASH":
            return "⚠️ CEK MANUAL (Mirip Foto Lama)"
        if r["dup_internal_exact"]:
            return "❌ GUGUR (Duplikat di File Ini - Exact)"
        if r["dup_internal_phash"]:
            return "⚠️ CEK MANUAL (Duplikat Mirip di File Ini)"
        return "✅ VALID"

    df_audit["status_akhir"] = df_audit.apply(decide, axis=1)

    # ------- SIMPAN KE DB: simpan hanya yang NEW dan bukan duplikat exact internal (biar referensi rapi)
    for idx, r in df_audit.iterrows():
        if r["history_status"] != "NEW":
            continue
        if r["dup_internal_exact"]:
            continue
        db_insert(conn, {
            "sha256": r["sha256"],
            "phash": r["phash"],
            "source_type": r["source_type"],
            "source_file": r["source_file"],
            "sheet": r["sheet"],
            "location": r["location"],
            "cluster": r["cluster"],
            "segment": r["segment"],
            "url": r["url"],
            "first_seen": r["first_seen"],
            "thumb_jpg": r.get("thumb_jpg", None)
        })

    conn.commit()
    conn.close()

    # gabung lagi
    out = pd.concat([df_audit, df_skip], ignore_index=True)
    return out

# =========================
# RUN UI
# =========================
if uploaded:
    tmp_path = "temp_upload.xlsx"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🚀 MULAI AUDIT"):
        with st.status("Meng-audit foto…", expanded=True) as status:
            df = audit_workbook(tmp_path)

            if df.empty:
                status.update(label="Tidak ada foto yang terbaca.", state="error")
                st.warning("Tidak ditemukan foto embedded atau foto dari link Google yang bisa diproses.")
            else:
                status.update(label="Audit selesai.", state="complete")

                audited_only = df[(df["status_akhir"].astype(str).str.startswith("✅")) |
                                  (df["status_akhir"].astype(str).str.contains("GUGUR")) |
                                  (df["status_akhir"].astype(str).str.contains("CEK MANUAL"))]

                st.subheader("Ringkasan (HANYA yang diaudit)")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Diaudit", len(audited_only))
                c2.metric("VALID", int((audited_only["status_akhir"] == "✅ VALID").sum()))
                c3.metric("GUGUR", int(audited_only["status_akhir"].astype(str).str.contains("GUGUR").sum()))
                c4.metric("CEK MANUAL", int(audited_only["status_akhir"].astype(str).str.contains("CEK MANUAL").sum()))
                c5.metric("SKIP", int(df["status_akhir"].astype(str).str.startswith("⏭️ SKIP").sum()))

                st.subheader("Laporan (Termasuk SKIP)")
                report = df.drop(columns=["thumb", "thumb_jpg", "ref_thumb_jpg"], errors="ignore")
                st.dataframe(report, use_container_width=True)

                out_x = io.BytesIO()
                report.to_excel(out_x, index=False)
                today = datetime.now().strftime("%Y-%m-%d")
                st.download_button(
                    "📥 Download Laporan (Excel)",
                    data=out_x.getvalue(),
                    file_name=f"Laporan_Audit_Foto_{today}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.subheader("Preview (dibatasi) + Referensi Duplikat (Segment/Link/Lokasi)")
                shown = 0
                cols = st.columns(4)

                def prio(s):
                    s = str(s)
                    if "GUGUR" in s or "CEK MANUAL" in s:
                        return 0
                    if s.startswith("✅"):
                        return 1
                    if s.startswith("⏭️ SKIP"):
                        return 2
                    return 3

                df_view = df.copy()
                df_view["_p"] = df_view["status_akhir"].map(prio)
                df_view = df_view.sort_values("_p").drop(columns=["_p"])

                for _, r in df_view.iterrows():
                    if preview_limit == 0 or shown >= preview_limit:
                        break
                    if r.get("thumb") is None:
                        continue

                    with cols[shown % 4]:
                        st.image(r["thumb"], caption=f'{r["sheet"]} | {r["location"]}\n{r["status_akhir"]}')

                        if r.get("skip_reason"):
                            st.caption(r["skip_reason"])

                        # dup info lengkap
                        if str(r.get("dup_type", "")).strip() and str(r.get("dup_type", "")).strip() not in ("NONE", ""):
                            st.caption(f"🔁 {r.get('dup_type')} dari:")
                            st.caption(str(r.get("dup_of", "")))

                        # show reference thumb (history or internal)
                        ref_img = jpg_bytes_to_pil(r.get("ref_thumb_jpg", None))
                        if ref_img is not None:
                            st.image(ref_img, caption="📌 Foto referensi (yang jadi acuan)")

                        if r.get("history_detail"):
                            st.caption(r["history_detail"])

                    shown += 1

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

st.divider()
st.caption(f"DB history disimpan lokal: {DB_PATH} (jangan dihapus kalau mau deteksi lintas bulan)")
