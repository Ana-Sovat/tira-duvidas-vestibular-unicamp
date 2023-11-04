import streamlit as st
from streamlit.logger import get_logger

import pandas as pd

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

#lê os dados
tabela = pd.read_csv("Vestibular.csv",index_col=0)

#cria os embeddings e coloca no chroma
chroma_unicamp = Chroma.from_texts(texts=tabela.Textos.to_list(),embedding=OpenAIEmbeddings())

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

modelo = ChatOpenAI()

prompt_template_unicamp = """Você é um assistente virtual que tem por objetivo sanar dúvidas sobre o Vestibular Unicamp 2024.
As respostas para as perguntas devem ser baseadas nas informações contidas no documento oficial da universidade (Resolução GR-031/2023, de 13/07/2023).
Se uma pergunta não puder ser respondida com base no seu conhecimento, você deve dizer que não possui informações suficientes.

{context}

Pergunta: {question}"""

template = PromptTemplate(template=prompt_template_unicamp, input_variables=["context","question"])

respondedor_de_perguntas = RetrievalQA.from_chain_type(llm=modelo,retriever=chroma_unicamp.as_retriever(),chain_type_kwargs={"prompt": template},return_source_documents=True)

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

        resposta = respondedor_de_perguntas(pergunta)['result']

        with st.chat_message("assistant"):
            st.write(resposta)



if __name__ == "__main__":
    run()
