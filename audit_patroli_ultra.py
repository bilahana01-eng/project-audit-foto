import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image
import imagehash
import io
import os
import sqlite3
import requests
import re

# --- KONFIGURASI ---
st.set_page_config(page_title="Audit Foto Ultra", layout="wide")
st.title("🛡️ Sistem Audit Foto Patroli Terintegrasi")
st.markdown("Sistem otomatis deteksi duplikasi: **Embedded Image**, **Google Drive**, & **Google Docs**.")

# --- DATABASE ENGINE ---
def get_db_connection():
    conn = sqlite3.connect('audit_history_master.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS history 
                 (hash TEXT PRIMARY KEY, info TEXT, date TEXT)''')
    return conn

def cek_global_history(p_hash):
    with get_db_connection() as conn:
        return conn.execute("SELECT info, date FROM history WHERE hash=?", (p_hash,)).fetchone()

# --- INTELLIGENT DOWNLOADER ---
def smart_download(url):
    try:
        # Regex untuk mengekstrak ID dari berbagai format GDrive/GDocs
        regex = r"(?<=/d/|id=)([a-zA-Z0-9-_]+)"
        match = re.search(regex, url)
        if not match: return None
        
        file_id = match.group(1)
        direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Request dengan Timeout & Stream agar tidak hang
        response = requests.get(direct_url, timeout=7, stream=True)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img.thumbnail((300, 300)) # Optimasi RAM
            return img
    except:
        return None
    return None

# --- CORE AUDIT ENGINE ---
def run_deep_audit(file_path):
    wb = load_workbook(file_path, data_only=True)
    results = []
    
    for sheet in wb.worksheets:
        # A. PROSES FOTO YANG MENEMPEL (EMBEDDED)
        if hasattr(sheet, '_images'):
            for img_obj in sheet._images:
                try:
                    # Ambil lokasi baris & kolom foto tersebut berada
                    row_idx = img_obj.anchor._from.row + 1
                    col_idx = img_obj.anchor._from.col + 1
                    
                    # Ambil metadata di sekitar foto (Asumsi: B=Cluster, C=Segment)
                    cluster = sheet.cell(row=row_idx, column=2).value or "Unknown"
                    segment = sheet.cell(row=row_idx, column=3).value or "Unknown"
                    
                    img = Image.open(io.BytesIO(img_obj._data())).convert('RGB')
                    h = str(imagehash.phash(img))
                    
                    results.append({
                        "Source": "Embedded",
                        "Location": f"Sheet {sheet.title} | Baris {row_idx}",
                        "Cluster": cluster, "Segment": segment,
                        "Hash": h, "Img_Obj": img
                    })
                except: continue

        # B. PROSES FOTO DARI LINK (GDRIVE/GDOCS)
        # Scan kolom LINK (Kolom G = indeks 7)
        for r in range(1, sheet.max_row + 1):
            cell_val = sheet.cell(row=r, column=7).value
            if cell_val and "google.com" in str(cell_val):
                with st.status(f"Downloading Link Baris {r}...", expanded=False):
                    img = smart_download(str(cell_val))
                    if img:
                        h = str(imagehash.phash(img.convert('RGB')))
                        results.append({
                            "Source": "Cloud Link",
                            "Location": f"Baris {r} (Kol G)",
                            "Cluster": sheet.cell(row=r, column=2).value or "N/A",
                            "Segment": sheet.cell(row=r, column=3).value or "N/A",
                            "Hash": h, "Img_Obj": img
                        })
    return results

# --- UI LOGIC ---
uploaded = st.file_uploader("Upload File Patroli (.xlsx)", type=["xlsx"])

if uploaded:
    if st.button("🚀 JALANKAN FULL AUDIT"):
        with open("temp.xlsx", "wb") as f: f.write(uploaded.getbuffer())
        
        raw_data = run_deep_audit("temp.xlsx")
        
        if raw_data:
            df = pd.DataFrame(raw_data)
            
            # 1. Cek Duplikasi Internal (Dalam satu file)
            df['is_internal_dup'] = df.duplicated('Hash', keep='first')
            
            # 2. Cek Duplikasi History (Bulan-bulan sebelumnya)
            hist_check = []
            for h in df['Hash']:
                found = cek_global_history(h)
                hist_check.append(f"⚠️ Dulu di {found[0]}" if found else "✅ NEW")
            df['History_Status'] = hist_check

            # 3. Final Decision
            def judge(row):
                if "⚠️" in row['History_Status']: return "❌ GUGUR (HISTORY)"
                if row['is_internal_dup']: return "❌ GUGUR (DUPLIKAT INTERNAL)"
                return "✅ VALID"
            
            df['Final_Result'] = df.apply(judge, axis=1)
            
            st.divider()
            st.success(f"Audit Selesai! {len(df)} entitas foto diproses.")

            # DOWNLOAD HASIL
            res_excel = df.drop(columns=['Img_Obj', 'Hash'])
            towrite = io.BytesIO()
            res_excel.to_excel(towrite, index=False)
            st.download_button("📥 Download Laporan Audit", towrite.getvalue(), "Hasil_Audit_Lengkap.xlsx")

            # DISPLAY TABS
            t1, t2 = st.tabs(["📊 Tabel Analisis", "🖼️ Galeri Audit"])
            with t1:
                st.dataframe(res_excel, use_container_width=True)
            with t2:
                for _, r in df.iterrows():
                    color = "red" if "GUGUR" in r['Final_Result'] else "green"
                    st.markdown(f"### :{color}[{r['Final_Result']}]")
                    st.write(f"**Cluster:** {r['Cluster']} | **Sumber:** {r['Source']}")
                    st.image(r['Img_Obj'], width=350)
                    st.divider()
        else:
            st.warning("Tidak ada data foto atau link yang dapat diidentifikasi.")
