import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from PIL import Image
import imagehash
import io
import os

st.set_page_config(page_title="Audit Foto Anti-Duplikasi", layout="wide")

st.title("🔍 Sistem Audit Foto Fiber Optic")
st.write("Upload file Excel berisi koordinat dan foto untuk memulai audit.")

# --- Fungsi Core ---
def get_images_from_excel(file_path):
    """Mengekstrak gambar dan lokasinya dari file Excel"""
    wb = load_workbook(file_path)
    report_data = []
    
    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        # Mencari objek gambar di dalam sheet
        for image in sheet._images:
            # Mendapatkan lokasi cell (misal: 'A1')
            row = image.anchor._from.row + 1
            col = image.anchor._from.col + 1
            
            # Membaca data gambar ke PIL
            img_data = io.BytesIO(image._data())
            img = Image.open(img_data)
            
            # Hitung Hash (Perceptual Hash)
            p_hash = str(imagehash.phash(img))
            
            report_data.append({
                "Sheet": sheetname,
                "Baris": row,
                "Kolom": col,
                "Hash": p_hash,
                "Image_Object": img
            })
    return report_data

# --- UI Sidebar ---
uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner('Sedang mengekstrak foto... Mohon tunggu...'):
        # Simpan sementara untuk dibaca openpyxl
        with open("temp_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        data_foto = get_images_from_excel("temp_file.xlsx")
        
    if data_foto:
        df = pd.DataFrame(data_foto)
        
        # --- Logika Audit Duplikasi ---
        # Mencari hash yang sama
        duplikat = df[df.duplicated('Hash', keep=False)].sort_values('Hash')
        
        st.success(f"Berhasil memproses {len(data_foto)} foto.")
        
        # --- Tampilan Hasil ---
        col1, col2 = st.tabs(["📊 Ringkasan Audit", "🚩 Temuan Duplikat"])
        
        with col1:
            st.metric("Total Foto", len(data_foto))
            st.metric("Terdeteksi Duplikat", len(duplikat))
            st.dataframe(df.drop(columns=['Image_Object']))

        with col2:
            if not duplikat.empty:
                st.warning("Daftar Foto yang Terindikasi Sama (Visual/Copy-Paste):")
                
                # Menampilkan foto yang mirip berdampingan
                for h, group in duplikat.groupby('Hash'):
                    st.divider()
                    st.write(f"**Grup Duplikat (Hash: {h})**")
                    cols = st.columns(len(group))
                    for i, (idx, row) in enumerate(group.iterrows()):
                        with cols[i]:
                            st.image(row['Image_Object'], caption=f"Sheet: {row['Sheet']} | Baris: {row['Baris']}")
            else:
                st.balloons()
                st.success("Tidak ditemukan duplikasi foto!")
    
    os.remove("temp_file.xlsx") # Hapus file temp
