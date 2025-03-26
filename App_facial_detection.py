import cv2
import streamlit as st
import time
import os
from PIL import Image

# Charger le classificateur de cascade de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser les valeurs par défaut dans session_state
if 'color' not in st.session_state:
    st.session_state.color = "#00FF00"
if 'scale_factor' not in st.session_state:
    st.session_state.scale_factor = 1.3
if 'min_neighbors' not in st.session_state:
    st.session_state.min_neighbors = 5
if 'save_image' not in st.session_state:
    st.session_state.save_image = False
if 'detecting' not in st.session_state:
    st.session_state.detecting = False
if 'last_saved_image' not in st.session_state:
    st.session_state.last_saved_image = None
if 'frame' not in st.session_state:  # Nouveau: pour stocker le dernier frame
    st.session_state.frame = None


# Définition de la fonction principale
def detect_faces():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        st.error("Erreur : Impossible d'ouvrir la webcam")
        return

    st.session_state.detecting = True
    stframe = st.empty()

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur : Impossible de capturer l'image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=st.session_state.scale_factor,
                                            minNeighbors=st.session_state.min_neighbors)

        # Mettre à jour la couleur choisie
        color = tuple(int(st.session_state.color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))  # HEX -> BGR
        color = color[::-1]  # Convertir en BGR (OpenCV utilise BGR au lieu de RGB)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        st.session_state.frame = frame  # Stocker le frame actuel
        stframe.image(frame, channels="BGR")

        # Enregistrer l'image si demandé
        if st.session_state.save_image:
            filename = f"detected_faces_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            st.session_state.last_saved_image = filename
            st.session_state.save_image = False
            break

    cap.release()
    cv2.destroyAllWindows()
    st.session_state.detecting = False


# Interface Streamlit
def app():
    st.title("Détection de visages avec Viola-Jones")
    st.markdown("""
    ### Instructions :
    1. Ajustez les paramètres avant de démarrer la détection.
    2. Cliquez sur **Démarrer la détection** pour activer la webcam.
    3. Cliquez sur **Enregistrer l'image** pour sauvegarder les visages détectés.
    4. Utilisez **Télécharger l'image** pour récupérer l'image enregistrée.
    5. Cliquez sur **Arrêter la détection** pour fermer la webcam.
    """)

    st.session_state.color = st.color_picker("Choisissez la couleur des rectangles", st.session_state.color)
    st.session_state.scale_factor = st.slider("Ajustez le scaleFactor", 1.1, 2.0, st.session_state.scale_factor, 0.1)
    st.session_state.min_neighbors = st.slider("Ajustez le minNeighbors", 1, 10, st.session_state.min_neighbors)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Démarrer la détection"):
            if not st.session_state.detecting:
                detect_faces()
    with col2:
        if st.button("Enregistrer l'image"):
            if st.session_state.detecting:
                st.session_state.save_image = True
            else:
                st.warning("Veuillez d'abord démarrer la détection")
    with col3:
        if st.button("Arrêter la détection"):
            st.session_state.detecting = False
    
    # Section téléchargement
    if st.session_state.last_saved_image and os.path.exists(st.session_state.last_saved_image):
        st.markdown("---")
        st.subheader("Télécharger l'image enregistrée")
        with open(st.session_state.last_saved_image, "rb") as file:
            st.download_button(
                label="Télécharger l'image",
                data=file,
                file_name=st.session_state.last_saved_image,
                mime="image/jpeg"
            )
        
        if st.button("Supprimer l'image"):
            os.remove(st.session_state.last_saved_image)
            st.session_state.last_saved_image = None
            st.experimental_rerun()


if __name__ == "__main__":
    app()
