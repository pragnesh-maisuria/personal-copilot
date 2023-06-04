import os
import utils
import streamlit as st
from dotenv import load_dotenv
from htmlchattemolete import css
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd


def configure():
    load_dotenv()


def process_pdf():
    st.subheader("Ask questions to your PDFs :books:")
    pdfs = st.file_uploader(
        "Upload your PDF and click on 'Process", accept_multiple_files=True
    )

    if st.button("Process"):
        with st.spinner("Processing your PDFs"):
            # get text from pdf files
            rawtext = utils.get_text_from_file(pdfs)

            # split the text into chunk
            chunks = utils.get_chunks_from_text(rawtext)

            # create the embedding and vector store
            vectorstore = utils.get_vectors_from_chunks(chunks)

            # create conversation chain
            st.session_state.conversation = utils.create_conversation_chain(vectorstore)

    user_question = st.text_input("Ask a question about your pdfs:")
    if user_question is not None and user_question != "":
        utils.handle_user_question(user_question)


def process_csv():
    st.subheader("Ask your data 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    user_question = st.text_input("Ask a question about your csv:")
    if csv_file is not None and user_question != "":
        df = pd.read_csv(csv_file)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


def main():
    configure()
    st.set_page_config(page_title="Personal Copilot", page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)

    PAGES = {"PDF Chat": process_pdf, "CSV Chat": process_csv}

    # intializing the global session variable
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Pragnesh's _:red[Copilot]_ 🤖")

    with st.sidebar:
        st.title("Navigation")
        selected_page = st.radio("Select a page", list(PAGES.keys()))
    page = PAGES[selected_page]
    page()


if __name__ == "__main__":
    main()
