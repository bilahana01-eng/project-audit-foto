import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from PIL import Image
import imagehash
import io
import os
import sqlite3
import requests
import re
from concurrent.futures import ThreadPoolExecutor

# --- KONFIGURASI ---
st.set_page_config(page_title="Audit Foto Duplicate", layout="wide")
st.title("💾 AUDIT FOTO DUPLICATE")
st.markdown("### Deteksi Otomatis Duplikasi")

# --- DATABASE ENGINE ---
def get_db_connection():
    conn = sqlite3.connect('audit_history_master.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS history 
                 (hash TEXT PRIMARY KEY, info TEXT, date TEXT)''')
    return conn

# --- OPTIMASI DOWNLOADER (MULTITHREADING READY) ---
def fast_download(url):
    if not url or not isinstance(url, str) or "google.com" not in url:
        return None
    try:
        # Ekstrak ID File
        file_id_match = re.search(r'/(?:d|document/d|file/d)/([a-zA-Z0-9-_]+)', url)
        if not file_id_match: return None
        file_id = file_id_match.group(1)

        # Gunakan link ekspor jika itu Google Docs, atau direct link jika itu file
        # Ini rahasia agar download GDocs tidak lama/gagal
        direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Timeout diperketat agar tidak nunggu kelamaan
        response = requests.get(direct_url, timeout=5, stream=True)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            img.thumbnail((200, 200)) # Kecilkan agar hemat RAM & Cepat
            return img
    except:
        return None
    return None

# --- CORE AUDIT ENGINE ---
def run_deep_audit(file_path):
    wb = load_workbook(file_path, data_only=True)
    results = []
    
    for sheet in wb.worksheets:
        # A. AMBIL FOTO NEMPEL (Cepat karena lokal)
        if hasattr(sheet, '_images'):
            for img_obj in sheet._images:
                try:
                    row = img_obj.anchor._from.row + 1
                    img = Image.open(io.BytesIO(img_obj._data())).convert('RGB')
                    img.thumbnail((200, 200))
                    h = str(imagehash.phash(img))
                    results.append({
                        "Source": "Embedded", "Location": f"Baris {row}",
                        "Cluster": sheet.cell(row=row, column=2).value,
                        "Segment": sheet.cell(row=row, column=3).value,
                        "Hash": h, "Img_Obj": img
                    })
                except: continue

        # B. AMBIL FOTO LINK (Gunakan Threading agar Cepat)
        urls_to_process = []
        for r in range(4, sheet.max_row + 1): # Start baris 4 (berdasarkan data kamu)
            url = sheet.cell(row=r, column=7).value # Kolom G
            if url: urls_to_process.append((r, url, sheet.cell(row=r, column=2).value, sheet.cell(row=r, column=3).value))

        # Proses download 5 link sekaligus (Concurrent)
        def process_url(item):
            r_idx, url, cluster, segment = item
            img = fast_download(url)
            if img:
                return {
                    "Source": "Cloud Link", "Location": f"Baris {r_idx}",
                    "Cluster": cluster, "Segment": segment,
                    "Hash": str(imagehash.phash(img)), "Img_Obj": img
                }
            return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            cloud_results = list(executor.map(process_url, urls_to_process))
            results.extend([res for res in cloud_results if res])

    return results

# --- UI LOGIC ---
uploaded = st.file_uploader("Upload Excel Patroli", type=["xlsx"])

if uploaded:
    if st.button("🚀 JALANKAN AUDIT CEPAT"):
        with st.status("Sedang memproses data...", expanded=True) as status:
            with open("temp.xlsx", "wb") as f: f.write(uploaded.getbuffer())
            
            raw_data = run_deep_audit("temp.xlsx")
            
            if raw_data:
                df = pd.DataFrame(raw_data)
                df['is_internal_dup'] = df.duplicated('Hash', keep='first')
                
                # Cek History
                conn = get_db_connection()
                def check_hist(h):
                    res = conn.execute("SELECT info FROM history WHERE hash=?", (h,)).fetchone()
                    return f"⚠️ Duplikat ({res[0]})" if res else "✅ NEW"
                
                df['History_Status'] = df['Hash'].apply(check_hist)
                df['Final_Result'] = df.apply(lambda r: "❌ GUGUR" if "⚠️" in r['History_Status'] or r['is_internal_dup'] else "✅ VALID", axis=1)
                
                status.update(label="Audit Selesai!", state="complete")
                
                st.success(f"Berhasil mengaudit {len(df)} foto!")
                st.dataframe(df.drop(columns=['Img_Obj', 'Hash']), use_container_width=True)
                
                # Galeri Ringkas
                cols = st.columns(4)
                for idx, r in df.iterrows():
                    with cols[idx % 4]:
                        st.image(r['Img_Obj'], caption=f"{r['Location']} - {r['Final_Result']}")

    if os.path.exists("temp.xlsx"): os.remove("temp.xlsx")
