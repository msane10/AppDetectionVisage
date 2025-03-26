import cv2
import streamlit as st
<<<<<<< HEAD
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
=======
>>>>>>> 1f81e7b (Mise à jour des fichiers requirement et gitignore a jour)
import time
import os
import zipfile
from io import BytesIO
import tempfile

# Charger le classificateur de cascade de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialiser les valeurs par défaut dans session_state
if 'color' not in st.session_state:
    st.session_state.color = "#00FF00"
if 'scale_factor' not in st.session_state:
    st.session_state.scale_factor = 1.3
if 'min_neighbors' not in st.session_state:
    st.session_state.min_neighbors = 5
if 'detecting' not in st.session_state:
    st.session_state.detecting = False
if 'saved_images' not in st.session_state:
    st.session_state.saved_images = []

# Définition de la fonction principale pour la détection des visages
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=st.session_state.scale_factor,
                                          minNeighbors=st.session_state.min_neighbors)
    color = tuple(int(st.session_state.color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    color = color[::-1]
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame, faces

# Fonction pour enregistrer l'image
def save_image(frame):
    filename = f"detected_faces_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    if os.path.exists(filename):
        st.session_state.saved_images.append(filename)
        st.success(f"✅ Image enregistrée sous {filename}")

# Fonction pour compresser les images enregistrées en un fichier ZIP
def create_zip_of_images():
    if not st.session_state.saved_images:
        return None

    try:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for image_file in st.session_state.saved_images:
                zip_file.write(image_file, os.path.basename(image_file))
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier ZIP: {e}")
        return None

# Fonction principale de l'application Streamlit
def app():
    st.title("Détection de visages avec Viola-Jones")
    st.markdown("""
    ### Instructions :
    1. Ajustez les paramètres avant de démarrer la détection.
    2. Cliquez sur **Démarrer la détection** pour activer la webcam.
    3. Cliquez sur **Enregistrer l'image** pour sauvegarder l'image actuelle immédiatement.
    4. Cliquez sur **Arrêter la détection** pour fermer la webcam.
    """)

    st.session_state.color = st.color_picker("Choisissez la couleur des rectangles", st.session_state.color)
    st.session_state.scale_factor = st.slider("Ajustez le scaleFactor", 1.1, 2.0, st.session_state.scale_factor, 0.1)
    st.session_state.min_neighbors = st.slider("Ajustez le minNeighbors", 1, 10, st.session_state.min_neighbors)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Démarrer la détection"):
            st.session_state.detecting = True
            st.rerun()
    with col2:
        if st.button("Enregistrer l'image") and 'frame' in st.session_state:
            save_image(st.session_state.frame)
    with col3:
        if st.button("Arrêter la détection"):
            st.session_state.detecting = False
            st.rerun()

    # Conteneur pour la vidéo en bas
    video_container = st.empty()

    if st.session_state.detecting:
<<<<<<< HEAD
        try:
            webrtc_streamer(key="webcam", video_transformer_factory=FaceDetection)
        except Exception as e:
            st.error(f"Erreur lors de l'activation de la webcam : {e}")
=======
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur : Impossible de capturer l'image")
                break

            frame, _ = detect_faces(frame)
            st.session_state.frame = frame
            video_container.image(frame, channels="BGR")  # Affichage en bas

        cap.release()
        cv2.destroyAllWindows()
>>>>>>> 1f81e7b (Mise à jour des fichiers requirement et gitignore a jour)

    # Vérification avant d'afficher le bouton de téléchargement
    zip_buffer = create_zip_of_images()
    if zip_buffer:
        # Créer un fichier ZIP temporaire et le télécharger
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
                temp_zip.write(zip_buffer.read())
                temp_zip_path = temp_zip.name

            with open(temp_zip_path, 'rb') as f:
                st.download_button(
                    label="📥 Télécharger toutes les images enregistrées",
                    data=f,
                    file_name="images.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des images: {e}")
    else:
        st.warning("⚠️ Aucune image enregistrée à télécharger.")

if __name__ == "__main__":
    app()
