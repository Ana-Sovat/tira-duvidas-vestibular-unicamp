import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="ChatBot"
    )

    st.title("Tira-Dúvidas Automático")

    st.header("Você tem alguma dúvida sobre o funcionamento do Vestibular Unicamp 2024?")
    st.header("Pergunte ao nosso chatbot!")
    st.subheader("Você pode perguntar sobre o processo de inscrição, vagas, matrícula, datas e conteúdo das provas...")

    pergunta = st.chat_input("Qual a sua dúvida?")

    if pergunta:
        with st.chat_message("user"):
            st.write(f"{pergunta}")
        with st.chat_message("assistant"):
            st.write("Não sei, ainda estou estudando!")



if __name__ == "__main__":
    run()
