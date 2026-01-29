import streamlit as st
import pandas as pd
import sqlite3
import imagehash
from PIL import Image
import requests
import io
import os
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Audit Foto Link GDrive", layout="wide")
st.title("🔍 Audit Foto Patroli (Versi Link GDrive)")
st.write("Deteksi duplikasi lintas bulan menggunakan link Google Drive.")

# --- DATABASE LOGIC (INGATAN LINTAS BULAN) ---
def get_db():
    conn = sqlite3.connect('audit_history.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS history 
                 (hash TEXT PRIMARY KEY, cluster TEXT, segment TEXT, tanggal TEXT, link TEXT)''')
    return conn

def cek_history(p_hash):
    with get_db() as conn:
        return conn.execute("SELECT cluster, tanggal, link FROM history WHERE hash=?", (p_hash,)).fetchone()

def simpan_history(df):
    with get_db() as conn:
        for _, r in df.iterrows():
            try:
                conn.execute("INSERT OR IGNORE INTO history VALUES (?,?,?,?,?)", 
                             (r['Hash'], r['Cluster'], r['Segment Name'], r['Tanggal Patroli'], r['Link Foto']))
            except: pass

# --- HELPER: KONVERSI LINK GDRIVE KE DIRECT DOWNLOAD ---
def format_gdrive_link(link):
    file_id_match = re.search(r'd/([^/]+)', link)
    if file_id_match:
        return f'https://drive.google.com/uc?export=download&id={file_id_match.group(1)}'
    return link

# --- PROSES DOWNLOAD & HASH ---
def proses_audit_link(df):
    results = []
    progress_bar = st.progress(0)
    total = len(df)
    
    for i, row in df.iterrows():
        link_asli = str(row.get('Link Foto', ''))
        direct_link = format_gdrive_link(link_asli)
        
        try:
            # Download gambar
            response = requests.get(direct_link, timeout=10)
            img = Image.open(io.BytesIO(response.content))
            
            # Hashing
            h = str(imagehash.phash(img))
            hist = cek_history(h)
            
            results.append({
                "Cluster": row.get('Cluster', 'N/A'),
                "Segment Name": row.get('Segment Name', 'N/A'),
                "Tanggal Patroli": row.get('Tanggal Patroli', 'N/A'),
                "Link Foto": link_asli,
                "Hash": h,
                "Hist_Info": f"⚠️ Duplikat Bulan Lalu di {hist[0]} ({hist[1]})" if hist else "✅ NEW",
                "Img_Obj": img
            })
        except Exception as e:
            st.error(f"Gagal akses link baris {i+1}: {e}")
            
        progress_bar.progress((i + 1) / total)
    return results

# --- UI UTAMA ---
uploaded_file = st.file_uploader("Upload Excel (Kolom: Cluster, Segment Name, Tanggal Patroli, Link Foto)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    
    if st.button("🚀 Mulai Audit Foto"):
        data_hasil = proses_audit_link(df_input)
        
        if data_hasil:
            res_df = pd.DataFrame(data_hasil)
            
            # Cek Duplikat Internal (dalam file yang sama)
            is_dup_internal = res_df.duplicated('Hash', keep=False)
            
            def final_status(idx, r):
                if "Duplikat Bulan Lalu" in r['Hist_Info']: return "❌ DUPLICATE (HISTORY)"
                if is_dup_internal[idx]: return "❌ DUPLICATE (INTERNAL)"
                return "✅ REAL PICT"
            
            res_df['Status Audit'] = [final_status(i, r) for i, r in res_df.iterrows()]
            
            st.success(f"Audit Selesai! {len(res_df)} foto diproses.")
            
            # Tombol Simpan
            if st.button("💾 Simpan Hash ke Database (Untuk Audit Bulan Depan)"):
                simpan_history(res_df)
                st.success("Database diperbarui!")

            # Download Excel
            output = res_df.drop(columns=['Img_Obj'])
            towrite = io.BytesIO()
            output.to_excel(towrite, index=False)
            st.download_button("📥 Download Hasil Audit", towrite.getvalue(), "Hasil_Audit_GDrive.xlsx")

            # Galeri
            t1, t2 = st.tabs(["📊 Data Tabel", "🚩 Temuan Duplikat"])
            with t1: st.dataframe(output, use_container_width=True)
            with t2:
                dups = res_df[res_df['Status Audit'].str.contains("❌")]
                for h, g in dups.groupby('Hash'):
                    st.divider()
                    st.error(f"Hash Identik: {h}")
                    cols = st.columns(len(g))
                    for i, (_, r) in enumerate(g.iterrows()):
                        cols[i].image(r['Img_Obj'], caption=f"{r['Cluster']} | {r['Status Audit']}")
