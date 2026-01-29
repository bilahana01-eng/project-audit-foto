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
    # Filter data yang valid untuk disimpan
    history_df = df[['Hash', 'Cluster', 'Segment Name', 'Tanggal Patroli', 'Sheet', 'Posisi']]
    history_df.columns = ['hash', 'cluster', 'segment', 'tanggal', 'sheet', 'posisi']
    
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
        if not hasattr(sheet, '_images'):
            continue
            
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

# --- ALUR UTAMA ---
init_db()
uploaded_file = st.file_uploader("Upload File Excel Patroli (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner('Menganalisis data dan foto...'):
        with open("temp_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_foto = get_images_and_data("temp_file.xlsx")
        
    if data_foto:
        df = pd.DataFrame(data_foto)
        
        # Cek Duplikat Internal
        is_duplicate_internal = df.duplicated('Hash', keep=False)
        
        def tentukan_status(index, row):
            if "DUPLIKAT HISTORY" in row['Status History']:
                return "❌ DUPLICATE (HISTORY)"
            elif is_duplicate_internal[index]:
                return "❌ DUPLICATE (INTERNAL)"
            return "✅ REAL PICT"

        df['Status Audit'] = [tentukan_status(i, r) for i, r in df.iterrows()]
        
        st.success(f"Analisis Selesai: {len(data_foto)} foto diproses.")

        if st.button("💾 Simpan Data ke Database History"):
            simpan_ke_history(df)
            st.success("Data disimpan untuk audit bulan depan!")

        # Download Button
        output_df = df.drop(columns=['Image_Object'])
        towrite = io.BytesIO()
        output_df.to_excel(towrite, index=False)
        towrite.seek(0)
        st.download_button("📥 Download Laporan LENGKAP", towrite, "Laporan_Audit.xlsx")

        tab1, tab2 = st.tabs(["📊 Tabel Hasil Audit", "🚩 Galeri Temuan"])
        
        with tab1:
            st.dataframe(df.drop(columns=['Image_Object']), use_container_width=True)

        with tab2:
            duplikat_only = df[df['Status Audit'].str.contains("❌")]
            if not duplikat_only.empty:
                for h, group in duplikat_only.groupby('Hash'):
                    st.divider()
                    st.warning(f"Grup Foto Identik - Hash: {h}")
                    cols = st.columns(len(group))
                    for i, (idx, row) in enumerate(group.iterrows()):
                        with cols[i]:
                            st.image(row['Image_Object'], caption=f"{row['Cluster']} | {row['Posisi']}")
            else:
                st.success("Tidak ada duplikasi ditemukan!")
    
    if
