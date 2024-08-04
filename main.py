import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pages.nav.home as home
import pages.nav.predict as predict
import pages.nav.about as about
from streamlit_option_menu import option_menu



# Configuração da página principal
st.set_page_config(page_title="Classificado de imagens CIFAR-10 Classificador", page_icon="🚀")

# Aplicar o CSS customizado
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/css/style.css")

# Função para carregar o modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models\modelo_cifar10 .h5')

model = load_model()

# Função para navegação entre páginas
def run():
    with st.sidebar:
        choice = option_menu("Menu", ["Home", "Fazer Previsão", "Sobre o Projeto"],
                             icons=['house', 'upload', 'info-circle'],
                             menu_icon="cast", default_index=0,
                         
        )

    if choice == "Home":
        home.show()
    elif choice == "Fazer Previsão":
        predict.show(model)
    elif choice == "Sobre o Projeto":
        about.show()

if __name__ == '__main__':
    run()


