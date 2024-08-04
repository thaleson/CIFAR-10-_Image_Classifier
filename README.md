# 🚀 **IA predict**: Classificação de Imagens com CIFAR-10

Bem-vindo ao projeto de classificação de imagens usando o dataset CIFAR-10! Este projeto utiliza TensorFlow e Streamlit para criar uma aplicação web interativa para prever a classe de imagens.

## 🛠️ **Tecnologias Utilizadas**

- **Python** 🐍
- **TensorFlow** 📈
- **Keras** 🔍
- **Streamlit** 🌟
- **NumPy** 🔢
- **Pandas** 🐼
- **Matplotlib** 📊
- **Altair** 📈
- **Seaborn** 🌈

## 🌟 **Funcionalidades**

- **Página Inicial** 🏠: Visão geral do projeto e informações sobre como usá-lo.
- **Fazer Previsão** 🔮: Faça upload de uma imagem para classificar e veja a previsão do modelo.
- **Sobre o Projeto** ℹ️: Detalhes sobre o desenvolvimento do projeto e informações do autor.

## 🚀 **Como Rodar o Projeto**

### Pré-requisitos

Certifique-se de ter o Python 3.x instalado e siga estes passos para instalar as dependências:

1. Clone o repositório:

   ```bash
   git clone https://github.com/thaleson/CIFAR-10-_Image_Classifier
   cd seu_repositorio
   ```

2. Crie um ambiente virtual e ative-o:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Windows use `venv\Scripts\activate`
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute a aplicação:

   ```bash
   streamlit run main.py
   ```

## 📂 **Estrutura do Projeto**

- `main.py`: Arquivo principal que executa a aplicação Streamlit.
- `pages/`: Diretório contendo as páginas da aplicação.
  - `nav/`: Contém os módulos de navegação:
    - `home.py`: Página inicial.
    - `predict.py`: Página de previsão.
    - `about.py`: Página sobre o projeto.
- `models/`: Diretório para armazenar modelos treinados.
- `static/css/`: Diretório para arquivos CSS.
- `requirements.txt`: Lista de dependências do projeto.
- `.gitignore`: Arquivos e diretórios a serem ignorados pelo Git.

## 📸 **Capturas de Tela**

![Página Inicial](static/screenshots/home.png)
![Página de Previsão](static/screenshots/predict.png)
![Sobre o Projeto](static/screenshots/about.png)

## 📄 **Licença**

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 📫 **Contato**

Thaleson Silva - [LinkedIn](https://www.linkedin.com/in/thaleson-silva-9298a0296/)

GitHub: [thaleson](https://github.com/thaleson)

---

Feito com 💙 por Thaleson Silva
