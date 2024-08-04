import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = [
    'Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo',
    'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão'
]

custom_messages = {
    'Avião': "Parece que esta imagem é de um avião, voando alto!",
    'Automóvel': "É um automóvel! Pronto para uma viagem na estrada.",
    'Pássaro': "Um lindo pássaro! Que maravilha da natureza.",
    'Gato': "Este é um gato, sempre curioso e brincalhão.",
    'Cervo': "Um cervo majestoso, símbolo de elegância.",
    'Cachorro': "É um cachorro! O melhor amigo do homem.",
    'Sapo': "Um sapo! Sempre pulando por aí.",
    'Cavalo': "Um cavalo poderoso e elegante.",
    'Navio': "Este é um navio, navegando pelos mares.",
    'Caminhão': "Um caminhão, pronto para transportar cargas."
}


def predict_image(img, model):
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    return predicted_class_name, confidence

def show(model):
    st.title("Fazer Previsão da Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagem enviada', use_column_width=True)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_resized = img.resize((32, 32))

        if st.button("Fazer Previsão", key="predict_button"):
            with st.spinner('Fazendo previsão...'):
                # Inicializa a barra de progresso
                progress_bar = st.progress(0)
                
                # Atualiza a barra de progresso
                progress_bar.progress(50)
                
                # Faz a previsão
                predicted_class, confidence = predict_image(img_resized, model)
                message = custom_messages[predicted_class]


                # Exibe a mensagem e a barra de progresso completa
                st.success(f"O animal é: {predicted_class} com {confidence:.2f}% de confiança. {message}")
            

                progress_bar.progress(100)
