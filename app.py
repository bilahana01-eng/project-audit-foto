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
st.set_page_config(page_title="Audit Foto Patroli Multi-Source", layout="wide")
st.title("🔍 Audit Foto Patroli Fiber Optic (Bulan 1-4+)")
st.write("Mendukung foto langsung di Excel & Link Google Drive.")

# --- DATABASE (Ingatan Lintas Bulan) ---
def get_db_connection():
    conn = sqlite3.connect('audit_history.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS foto_history 
                 (hash TEXT PRIMARY KEY, cluster TEXT, segment TEXT, tanggal TEXT)''')
    return conn

def cek_history(p_hash):
    with get_db_connection() as conn:
        return conn.execute("SELECT cluster, tanggal FROM foto_history WHERE hash=?", (p_hash,)).fetchone()

def simpan_history(df):
    with get_db_connection() as conn:
        for _, r in df.iterrows():
            try:
                conn.execute("INSERT OR IGNORE INTO foto_history VALUES (?,?,?,?)", 
                             (r['Hash'], r['Cluster'], r['Segment Name'], r['Tanggal Patroli']))
            except: pass

# --- FUNGSI DOWNLOAD GOOGLE DRIVE ---
def download_gdrive_img(url):
    try:
        # Mengubah link gdrive biasa menjadi link direct download
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[-1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if file_id:
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            response = requests.get(direct_url, timeout=10)
            return Image.open(io.BytesIO(response.content))
    except:
        return None
    return None

# --- PROSES EXCEL ---
def proses_excel(file):
    wb = load_workbook(file, data_only=True)
    results = []
    
    for sheet in wb.worksheets:
        # 1. AMBIL FOTO TERTANAM (PICTURE LANGSUNG)
        if hasattr(sheet, '_images'):
            for img_obj in sheet._images:
                row = img_obj.anchor._from.row + 1
                col = img_obj.anchor._from.col + 1
                
                # Metadata (A=Cluster, B=Segment, C=Tanggal)
                c_val = sheet.cell(row=row, column=1).value
                s_val = sheet.cell(row=row, column=2).value
                t_val = sheet.cell(row=row, column=3).value
                
                img = Image.open(io.BytesIO(img_obj._data()))
                h = str(imagehash.phash(img))
                hist = cek_history(h)
                
                results.append({
                    "Cluster": str(c_val) if c_val else "N/A",
                    "Segment Name": str(s_val) if s_val else "N/A",
                    "Tanggal Patroli": str(t_val) if t_val else "N/A",
                    "Posisi": f"Baris {row}, Kol {get_column_letter(col)}",
                    "Tipe": "Embedded Image",
                    "Hash": h,
                    "Hist_Info": f"⚠️ Dulu di {hist[0]} ({hist[1]})" if hist else "✅ NEW",
                    "Img_Obj": img
                })

        # 2. AMBIL FOTO DARI LINK GDRIVE (DI KOLOM E/5 - Sesuaikan jika beda)
        # Scan baris 1 sampai 1000 (atau sesuai data kamu)
        for r_idx in range(1, sheet.max_row + 1):
            cell_url = sheet.cell(row=r_idx, column=5).value # Asumsi Link di Kolom E
            if isinstance(cell_url, str) and "drive.google.com" in cell_url:
                with st.spinner(f"Mendownload foto baris {r_idx}..."):
                    img = download_gdrive_img(cell_url)
                    if img:
                        h = str(imagehash.phash(img))
                        hist = cek_history(h)
                        results.append({
                            "Cluster": str(sheet.cell(row=r_idx, column=1).value) or "N/A",
                            "Segment Name": str(sheet.cell(row=r_idx, column=2).value) or "N/A",
                            "Tanggal Patroli": str(sheet.cell(row=r_idx, column=3).value) or "N/A",
                            "Posisi": f"Baris {r_idx}, Kol E (Link)",
                            "Tipe": "GDrive Link",
                            "Hash": h,
                            "Hist_Info": f"⚠️ Dulu di {hist[0]} ({hist[1]})" if hist else "✅ NEW",
                            "Img_Obj": img
                        })

    return results

# --- UI ---
up = st.file_uploader("Upload Excel Patroli (Bulan 1-4+)", type=["xlsx"])

if up:
    with st.spinner('Menganalisis foto & link...'):
        with open("temp.xlsx", "wb") as f: f.write(up.getbuffer())
        data = proses_excel("temp.xlsx")
    
    if data:
        df = pd.DataFrame(data)
        dup_internal = df.duplicated('Hash', keep=False)
        
        def set_status(i, r):
            if "Dulu" in r['Hist_Info']: return "❌ DUPLICATE (HISTORY MONTH)"
            if dup_internal[i]: return "❌ DUPLICATE (INTERNAL FILE)"
            return "✅ REAL PICT"
            
        df['Status Audit'] = [set_status(i, r) for i, r in df.iterrows()]
        
        st.success(f"Analisis Selesai! {len(df)} foto terdeteksi.")
        
        if st.button("💾 Simpan ke Database History (Untuk Bulan Depan)"):
            simpan_history(df)
            st.success("Data masuk ke memori audit!")

        # Download Report
        out = df.drop(columns=['Img_Obj'])
        towrite = io.BytesIO()
        out.to_excel(towrite, index=False)
        st.download_button("📥 Download Laporan Hasil Audit", towrite.getvalue(), "Audit_Bulanan.xlsx")

        t1, t2 = st.tabs(["📊 Tabel Audit", "🚩 Galeri Foto"])
        with t1: st.dataframe(out, use_container_width=True)
        with t2:
            for _, r in df.iterrows():
                st.write(f"**{r['Cluster']} - {r['Status Audit']}** ({r['Posisi']})")
                st.image(r['Img_Obj'], width=300)
                st.divider()

    if os.path.exists("temp.xlsx"): os.remove("temp.xlsx")
