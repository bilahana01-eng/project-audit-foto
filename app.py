import streamlit as st
import pandas as pd
from openpyxl import load_workbook
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
        # Mengambil semua gambar
        for image in sheet._images:
            row = image.anchor._from.row + 1
            col = image.anchor._from.col + 1
            
            # Mengambil Tanggal Patroli (Asumsi tanggal ada di kolom tertentu, misal kolom A di baris yang sama)
            # Kamu bisa menyesuaikan koordinat cell tanggalnya di sini
            tgl_patroli = sheet.cell(row=row, column=1).value 
            
            # Ekstraksi foto
            img_data = io.BytesIO(image._data())
            img = Image.open(img_data)
            p_hash = str(imagehash.phash(img))
            
            report_data.append({
                "Sheet": sheetname,
                "Baris": row,
                "Kolom": col,
                "Tanggal Patroli": str(tgl_patroli) if tgl_patroli else "Tidak Terdeteksi",
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
        
        # Logika Audit
        # Cek duplikat berdasarkan Hash
        is_duplicate = df.duplicated('Hash', keep=False)
        df['Status Audit'] = is_duplicate.map({True: "DUPLICATE", False: "REAL PICT"})
        
        st.success(f"Analisis Selesai: {len(data_foto)} foto diproses.")

        # --- FITUR DOWNLOAD EXCEL ---
        # Menyiapkan file excel hasil audit tanpa objek gambar agar ringan
        output_df = df.drop(columns=['Image_Object'])
        towrite = io.BytesIO()
        output_df.to_excel(towrite, index=False, header=True)
        towrite.seek(0)
        
        st.download_button(
            label="📥 Download Hasil Audit (.xlsx)",
            data=towrite,
            file_name="Hasil_Audit_Patroli.xlsx",
            mime="application/vnd.ms-excel"
        )

        # --- TAMPILAN DASHBOARD ---
        tab1, tab2 = st.tabs(["📊 Data Audit", "🚩 Galeri Duplikat"])
        
        with tab1:
            st.dataframe(output_df, use_container_width=True)

        with tab2:
            duplikat_only = df[df['Status Audit'] == "DUPLICATE"].sort_values('Hash')
            if not duplikat_only.empty:
                for h, group in duplikat_only.groupby('Hash'):
                    st.divider()
                    st.warning(f"Temuan Duplikat - Hash: {h}")
                    cols = st.columns(len(group))
                    for i, (idx, row) in enumerate(group.iterrows()):
                        with cols[i]:
                            st.image(row['Image_Object'], caption=f"Sheet: {row['Sheet']} | Baris: {row['Baris']}")
            else:
                st.success("Hebat! Semua foto adalah Real Pict.")
    
    os.remove("temp_file.xlsx")
