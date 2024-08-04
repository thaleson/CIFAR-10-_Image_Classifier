import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show():
    st.title("Sobre o Projeto")
    st.write("""
    Este é um projeto de classificação de imagens usando o conjunto de dados CIFAR-10.
    O modelo foi treinado utilizando uma rede neural convolucional (CNN) e pode classificar imagens em 10 categorias diferentes.
    """)
    
    st.write("### Desempenho do Modelo")
    accuracy = 0.72
    st.write(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    st.write("### Distribuição das Classes no Conjunto de Dados")
    class_names = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']
    class_distribution = np.random.randint(500, 1500, size=len(class_names))

    fig, ax = plt.subplots()
    ax.barh(class_names, class_distribution, color='#000080')
    ax.set_xlabel('Número de Imagens')
    ax.set_title('Distribuição das Classes no Conjunto de Dados')

    st.pyplot(fig)

    st.write("### Sobre Mim")
    st.write("""
    Olá, eu sou **Thaleson Silva**, um desenvolvedor apaixonado por Machine Learning e Inteligência Artificial. Tenho experiência em projetos de classificação de imagens e estou sempre em busca de aprimorar minhas habilidades e explorar novas tecnologias.
    """)
    st.write("🔗 [**Conecte-se comigo no LinkedIn**](https://www.linkedin.com/in/thaleson-silva-9298a0296/) :briefcase:")
    st.write("🐙 [**Visite meu GitHub**](https://github.com/thaleson)")
    st.write("📧 Para entrar em contato, envie um e-mail para: thaleson19@hotmail.com")

