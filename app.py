import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image
import imagehash
import io
import os

st.set_page_config(page_title="Audit Foto Patroli Fiber Optic", layout="wide")

st.title("🔍 Audit Foto Patroli Fiber Optic")
st.write("Sistem deteksi duplikasi foto untuk validasi patroli lapangan.")

def get_images_and_data(file_path):
    wb = load_workbook(file_path, data_only=True)
    report_data = []
    
    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        # Mengambil semua gambar yang ada di sheet
        for image in sheet._images:
            # Koordinat baris dan kolom
            row = image.anchor._from.row + 1
            col = image.anchor._from.col + 1
            col_letter = get_column_letter(col)
            
            # --- PENGAMBILAN DATA DARI EXCEL ---
            # Asumsi: Cluster di Kolom A (1), Segment di Kolom B (2), Tanggal di Kolom C (3)
            # Kamu bisa mengubah nomor kolom di bawah ini sesuai file asli kamu
            cluster = sheet.cell(row=row, column=1).value 
            segment = sheet.cell(row=row, column=2).value
            tanggal = sheet.cell(row=row, column=3).value
            
            # Ekstraksi foto untuk di-hash
            img_data = io.BytesIO(image._data())
            img = Image.open(img_data)
            p_hash = str(imagehash.phash(img))
            
            report_data.append({
                "Cluster": cluster if cluster else "N/A",
                "Segment Name": segment if segment else "N/A",
                "Tanggal Patroli": str(tanggal) if tanggal else "N/A",
                "Sheet": sheetname,
                "Posisi": f"Baris {row}, Kolom {col_letter}",
                "Hash": p_hash,
                "Image_Object": img
            })
    return report_data

uploaded_file = st.file_uploader("Upload File Excel Patroli (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner('Menganalisis data dan foto...'):
        with open("temp_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        data_foto = get_images_and_data("temp_file.xlsx")
        
    if data_foto:
        df = pd.DataFrame(data_foto)
        
        # --- LOGIKA AUDIT ---
        # Cek duplikat berdasarkan Hash foto
        is_duplicate = df.duplicated('Hash', keep=False)
        df['Status Audit'] = is_duplicate.map({True: "❌ DUPLICATE", False: "✅ REAL PICT"})
        
        st.success(f"Analisis Selesai: {len(data_foto)} foto diproses.")

        # --- FITUR DOWNLOAD EXCEL HASIL AUDIT ---
        output_df = df.drop(columns=['Image_Object']) # Hapus objek gambar agar file excel ringan
        towrite = io.BytesIO()
        output_df.to_excel(towrite, index=False)
        towrite.seek(0)
        
        st.download_button(
            label="📥 Download Laporan Hasil Audit (.xlsx)",
            data=towrite,
            file_name="Laporan_Audit_Patroli_IOH.xlsx",
            mime="application/vnd.ms-excel"
        )

        # --- TAMPILAN DASHBOARD ---
        tab1, tab2 = st.tabs(["📊 Tabel Hasil Audit", "🚩 Galeri Temuan Duplikat"])
        
        with tab1:
            st.dataframe(df.drop(columns=['Image_Object']), use_container_width=True)

        with tab2:
            duplikat_only = df[df['Status Audit'] == "❌ DUPLICATE"].sort_values('Hash')
            if not duplikat_only.empty:
                for h, group in duplikat_only.groupby('Hash'):
                    st.divider()
                    st.warning(f"Grup Foto Identik (Hash: {h})")
                    cols = st.columns(len(group))
                    for i, (idx, row) in enumerate(group.iterrows()):
                        with cols[i]:
                            st.image(row['Image_Object'], 
                                     caption=f"{row['Cluster']} | {row['Segment Name']} | {row['Posisi']}")
            else:
                st.balloons()
                st.success("Tidak ada duplikasi foto. Semua data valid!")
    
    if os.path.exists("temp_file.xlsx"):
        os.remove("temp_file.xlsx")
