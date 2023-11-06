__import__('pysqlite3')

import sys
import os

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit.logger import get_logger

import pandas as pd

import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

#lê os dados
tabela = pd.read_csv("Vestibular.csv",index_col=0)

#pega a chave
os.environ['OPENAI_API_KEY'] = st.secrets["API_SECRET_KEY"]

#cria os embeddings e coloca no chroma
chroma_unicamp = Chroma.from_texts(texts=tabela.Textos.to_list(),embedding=OpenAIEmbeddings())

#cria a conexão com o modelo
modelo = ChatOpenAI()

#cria o template do prompt, para incorporar o contexto
prompt_template_unicamp = """Você é um assistente virtual que tem por objetivo tirar dúvidas dos estudantes ou público geral sobre o Vestibular Unicamp 2024.
As respostas para as perguntas feitas devem ser baseadas nas informações contidas na Resolução GR-031/2023, de 13/07/2023.
Se uma pergunta não puder ser respondida com base no seu conhecimento, você deve dizer que não possui informações suficientes.
Responda de forma simples, clara e amigável.

Norma publicada: {context}

Pergunta: {question}"""

template = PromptTemplate(template=prompt_template_unicamp, input_variables=["context","question"])

#cria o chatbot
respondedor_de_perguntas = RetrievalQA.from_chain_type(llm=modelo,retriever=chroma_unicamp.as_retriever(),chain_type_kwargs={"prompt": template},return_source_documents=True)

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Tira-Dúvidas Automático"
    )

    st.title("Tira-Dúvidas Automático")

    st.header("Você tem alguma dúvida sobre o funcionamento do Vestibular Unicamp 2024?")
    st.header("Pergunte ao nosso chatbot!")
    st.subheader("Você pode perguntar sobre o processo de inscrição, vagas, matrícula, datas e conteúdo das provas...")

    pergunta = st.chat_input("Qual a sua dúvida?")

    if pergunta:
        with st.chat_message("user"):
            st.write(f"{pergunta}")

        #envia a pergunta ao ChatGPT
        resposta = respondedor_de_perguntas(pergunta)['result']

        with st.chat_message("assistant"):
            st.write(resposta)
            #st.write("Não sei, ainda estou estudando.")



if __name__ == "__main__":
    run()
