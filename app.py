import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from PIL import Image
import imagehash
import io
import os

st.set_page_config(page_title="Audit Foto Patroli Fiber Optic", layout="wide")

st.title("🔍 Audit Foto Patroli Fiber Optic")
st.write("Sistem deteksi duplikasi foto untuk validasi patroli lapangan berdasarkan visual dan data Excel.")

def get_images_and_data(file_path):
    wb = load_workbook(file_path, data_only=True)
    report_data = []
    
    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        # Mencari semua gambar di dalam sheet
        for image in sheet._images:
            # Dapatkan posisi Anchor gambar
            row = image.anchor._from.row + 1
            col_idx = image.anchor._from.col + 1
            col_letter = sheet.cell(row=1, column=col_idx).column_letter
            
            # --- PENGAMBILAN DATA BERDASARKAN BARIS FOTO ---
            # Asumsi Struktur Excel: Kolom A=Cluster, Kolom B=Segment Name, Kolom C=Tanggal (Sesuaikan jika berbeda)
            cluster = sheet.cell(row=row, column=1).value 
            segment = sheet.cell(row=row, column=2).value
            tgl_patroli = sheet.cell(row=row, column=3).value 
            
            # Ekstraksi foto ke memori
            img_data = io.BytesIO(image._data())
            img = Image.open(img_data)
            
            # Perceptual Hashing untuk deteksi kemiripan visual
            p_hash = str(imagehash.phash(img))
            
            report_data.append({
                "Cluster": cluster if cluster else "N/A",
                "Segment Name": segment if segment else "N/A",
                "Tanggal Patroli": str(tgl_patroli) if tgl_patroli else "N/A",
                "Sheet": sheetname,
                "Baris": row,
                "Kolom": col_letter,
                "Hash_ID": p_hash,
                "Image_Object": img
            })
    return report_data

uploaded_file = st.file_uploader("Upload File Excel Patroli (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner('Menganalisis data dan foto...'):
        # Simpan file sementara
        with open("temp_audit.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        results = get_images_and_data("temp_audit.xlsx")
        
    if results:
        df = pd.DataFrame(results)
        
        # LOGIKA AUDIT: Jika Hash sama, maka status DUPLICATE
        is_duplicate = df.duplicated('Hash_ID', keep=False)
        df['Status Audit'] = is_duplicate.map({True: "🔴 DUPLICATE", False: "🟢 REAL PICT"})
        
        st.success(f"Analisis Selesai: {len(results)} foto diproses.")

        # --- FITUR DOWNLOAD HASIL AUDIT ---
        # Hapus objek gambar agar file excel hasil ringan
        excel_export = df.drop(columns=['Image_Object', 'Hash_ID'])
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            excel_export.to_excel(writer, index=False, sheet_name='Hasil Audit')
        
        st.download_button(
            label="📥 Download Laporan Audit (.xlsx)",
            data=buffer.getvalue(),
            file_name="Laporan_Audit_Patroli_Fiber.xlsx",
            mime="application/vnd.ms-excel"
        )

        # --- DASHBOARD TAMPILAN ---
        tab1, tab2 = st.tabs(["📊 Tabel Hasil Audit", "🚩 Detail Duplikasi"])
        
        with tab1:
            st.dataframe(df.drop(columns=['Image_Object']), use_container_width=True)

        with tab2:
            dupes = df[df['Status Audit'] == "🔴 DUPLICATE"].sort_values('Hash_ID')
            if not dupes.empty:
                for h_id, group in dupes.groupby('Hash_ID'):
                    with st.expander(f"Temuan Grup Duplikat (ID: {h_id})"):
                        cols = st.columns(len(group))
                        for i, (idx, row) in enumerate(group.iterrows()):
                            with cols[i]:
                                st.image(row['Image_Object'], use_container_width=True)
                                st.caption(f"📍 {row['Cluster']} - {row['Segment Name']}\n📅 {row['Tanggal Patroli']}\nPosisi: {row['Sheet']} Cell {row['Kolom']}{row['Baris']}")
            else:
                st.balloons()
                st.success("Tidak ada duplikasi ditemukan. Semua foto valid!")
    
    os.remove("temp_audit.xlsx")
