import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image
import imagehash
import io
import os
import sqlite3

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Audit Foto Patroli Fiber Optic", layout="wide")

st.title("🔍 Audit Foto Patroli Fiber Optic")
st.write("Sistem deteksi duplikasi foto (Internal File & History Lintas Bulan).")

# --- FUNGSI DATABASE (UNTUK INGATAN LINTAS BULAN) ---
def init_db():
    conn = sqlite3.connect('audit_history.db')
    c = conn.cursor()
    # Membuat tabel jika belum ada
    c.execute('''CREATE TABLE IF NOT EXISTS foto_history 
                 (hash TEXT, cluster TEXT, segment TEXT, tanggal TEXT, sheet TEXT, posisi TEXT)''')
    conn.commit()
    return conn

def cek_duplikat_history(p_hash):
    conn = sqlite3.connect('audit_history.db')
    c = conn.cursor()
    c.execute("SELECT cluster, segment, tanggal FROM foto_history WHERE hash = ?", (p_hash,))
    result = c.fetchone()
    conn.close()
    return result

def simpan_ke_history(df):
    conn = sqlite3.connect('audit_history.db')
    # Ambil data yang perlu saja untuk disimpan di history
    history_df = df[['Hash', 'Cluster', 'Segment Name', 'Tanggal Patroli', 'Sheet', 'Posisi']]
    history_df.columns = ['hash', 'cluster', 'segment', 'tanggal', 'sheet', 'posisi']
    
    # Hanya simpan hash yang belum ada di database agar tidak menumpuk duplikat di DB
    for _, row in history_df.iterrows():
        if not cek_duplikat_history(row['hash']):
            row_df = pd.DataFrame([row])
            row_df.to_sql('foto_history', conn, if_exists='append', index=False)
    conn.close()

# --- FUNGSI EKSTRAKSI DATA EXCEL ---
def get_images_and_data(file_path):
    wb = load_workbook(file_path, data_only=True)
    report_data = []
    
    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        for image in sheet._images:
            row = image.anchor._from.row + 1
            col = image.anchor._from.col + 1
            col_letter = get_column_letter(col)
            
            # Ambil data Cluster (A), Segment (B), Tanggal (C)
            cluster = sheet.cell(row=row, column=1).value 
            segment = sheet.cell(row=row, column=2).value
            tanggal = sheet.cell(row=row, column=3).value
            
            img_data = io.BytesIO(image._data())
            img = Image.open(img_data)
            p_hash = str(imagehash.phash(img))
            
            # Cek apakah hash ini sudah pernah ada di database bulan-bulan sebelumnya
            match_history = cek_duplikat_history(p_hash)
            
            status_history = "✅ NEW"
            if match_history:
                status_history = f"⚠️ DUPLIKAT HISTORY (Bulan Lalu di {match_history[0]})"
            
            report_data.append({
                "Cluster": cluster if cluster else "N/A",
                "Segment Name": segment if segment else "N/A",
                "Tanggal Patroli": str(tanggal) if tanggal else "N/A",
                "Sheet": sheetname,
                "Posisi": f"Baris {row}, Kolom {col_letter}",
                "Hash": p_hash,
                "Status History": status_history,
                "Image_Object": img
            })
    return report_data

# --- ALUR UTAMA (UI) ---
init_db() # Jalankan database saat aplikasi start
uploaded_file = st.file_uploader("Upload File Excel Patroli (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner('Menganalisis data dan foto lintas database...'):
        with open("temp_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        data_foto = get_images_and_data("temp_file.xlsx")
        
    if data_foto:
        df = pd.DataFrame(data_foto)
        
        # 1. Cek Duplikat di dalam file itu sendiri
        is_duplicate_internal = df.duplicated('Hash', keep=False)
        
        # 2. Gabungkan logika status
        def tentukan_status(row):
            if "DUPLIKAT HISTORY" in row['Status History']:
                return "❌ DUPLICATE (HISTORY)"
            elif is_duplicate_internal[df.index[df['Hash'] == row['Hash']][0]]:
                return "❌ DUPLICATE (INTERNAL)"
            return "✅ REAL PICT"

        df['Status Audit'] = df.apply(tentukan_status, axis=1)
        
        st.success(f"Analisis Selesai: {len(data_foto)} foto diproses.")

        # Tombol Simpan ke History
        if st.button("💾 Simpan Data Foto ini ke Database History"):
            simpan_ke_history(df)
            st.toast("Data berhasil disimpan untuk audit bulan depan!")

        # Fitur Download
        output_df = df.drop(columns=['Image_Object'])
        towrite = io.BytesIO()
        output_df.to_excel(towrite, index=False)
        towrite.seek(0)
        
        st.download_button(
            label="📥 Download Laporan Audit (.xlsx)",
            data=towrite,
            file_name="Laporan_Audit_Fiber_Optic.xlsx",
            mime="application/vnd.ms-excel"
        )

        tab1, tab2 = st.tabs(["📊 Tabel Hasil Audit", "🚩 Galeri Temuan"])
        
        with tab1:
            st.dataframe(df.drop(columns=['Image_Object']), use_container_width=True)

        with tab2:
            duplikat_only = df[df['Status Audit'].str.contains("❌")].sort_values('Hash')
            if not duplikat_only.empty:
                for h, group in duplikat_only.groupby('Hash
