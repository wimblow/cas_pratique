import streamlit as st
from skimage import io
import os
from prat.services.yolo_service import YoloService
from prat.services.img_processing import ImageProcessing

yolo = YoloService()
ip = ImageProcessing()




# st.set_page_config(layout="wide")

st.markdown("<h1>Détection d'ouverture de porte</h1>", unsafe_allow_html=True)

folder_path = "prat/data/"
filenames = os.listdir(folder_path)

filenames.insert(0, "<select file>")
img_data = st.sidebar.selectbox(
    "Sélectionnez un fichier", filenames, index=0
    )


if img_data != f"<select file>":
    if st.sidebar.button("Classify"):
        img = io.imread(f"{folder_path}/{img_data}")
        data = yolo.yolo_detect(img)
        print(data)
        for count, i in enumerate(data):
            image = ip.box(img.copy(), i)

            col1, col2 = st.columns(2)  
            with col1:
                st.image(image, use_column_width=True)
                st.write("-----")
            with col2:
                st.markdown("<h2>Classification</h2>", unsafe_allow_html=True)

                st.markdown(f"<h3 style=text-align:center'>Classe : <span style='color:red'>{data[count]['name']}</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style=text-align:center'>Confiance : <span style='color:red'>{round(data[count]['confidence'], 2)}</span></h3>", unsafe_allow_html=True)

