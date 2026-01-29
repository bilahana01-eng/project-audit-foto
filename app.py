import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image
import imagehash
import io
import os
import sqlite3

# --- KONFIGURASI ---
st.set_page_config(page_title="Audit Foto Patroli", layout="wide")
st.title("🔍 Audit Foto Patroli Fiber Optic")

# --- DATABASE ---
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

# --- PROSES EXCEL ---
def proses_excel(file):
    wb = load_workbook(file, data_only=True)
    results = []
    for sheet in wb.worksheets:
        if not hasattr(sheet, '_images'): continue
        for img_obj in sheet._images:
            row = img_obj.anchor._from.row + 1
            col = img_obj.anchor._from.col + 1
            
            # Ambil data: Cluster(A), Segment(B), Tanggal(C)
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
                "Hash": h,
                "Hist_Info": f"⚠️ Dulu di {hist[0]} ({hist[1]})" if hist else "✅ NEW",
                "Img": img
            })
    return results

# --- UI ---
up = st.file_uploader("Upload Excel Patroli (.xlsx)", type=["xlsx"])

if up:
    with st.spinner('Menganalisis...'):
        with open("temp.xlsx", "wb") as f: f.write(up.getbuffer())
        data = proses_excel("temp.xlsx")
    
    if data:
        df = pd.DataFrame(data)
        dup_internal = df.duplicated('Hash', keep=False)
        
        def set_status(i, r):
            if "Dulu" in r['Hist_Info']: return "❌ DUPLICATE (HISTORY)"
            if dup_internal[i]: return "❌ DUPLICATE (INTERNAL)"
            return "✅ REAL PICT"
            
        df['Status Audit'] = [set_status(i, r) for i, r in df.iterrows()]
        
        st.success(f"Selesai! {len(df)} foto ditemukan.")
        
        if st.button("💾 Simpan ke History"):
            simpan_history(df)
            st.success("Tersimpan!")

        # Download
        out = df.drop(columns=['Img'])
        towrite = io.BytesIO()
        out.to_excel(towrite, index=False)
        st.download_button("📥 Download Laporan", towrite.getvalue(), "Hasil_Audit.xlsx")

        t1, t2 = st.tabs(["📊 Tabel", "🚩 Galeri Duplikat"])
        with t1: st.dataframe(out, use_container_width=True)
        with t2:
            dups = df[df['Status Audit'].str.contains("❌")]
            for h, g in dups.groupby('Hash'):
                st.divider()
                st.error(f"Hash: {h}")
                rows = st.columns(len(g))
                for i, (_, r) in enumerate(g.iterrows()):
                    rows[i].image(r['Img'], caption=f"{r['Cluster']} | {r['Posisi']}")
    
    if os.path.exists("temp.xlsx"): os.remove("temp.xlsx")
