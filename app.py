import os
import utils
import streamlit as st
from dotenv import load_dotenv
from htmlchattemolete import css
import pandas as pd
import json
from agent import query_agent, create_agent


def configure():
    load_dotenv()


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)


def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.set_index(df.columns[0], inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


def process_pdf():
    st.subheader("Ask questions to your PDFs :books:")
    pdfs = st.file_uploader(
        "Upload your PDF and click on 'Process", accept_multiple_files=True
    )
    try:
        if st.button("Process"):
            with st.spinner("Processing your PDFs"):
                # get text from pdf files
                rawtext = utils.get_text_from_file(pdfs)

                # split the text into chunk
                chunks = utils.get_chunks_from_text(rawtext)

                # create the embedding and vector store
                vectorstore = utils.get_vectors_from_chunks(chunks)

                # create conversation chain
                st.session_state.conversation = utils.create_conversation_chain(
                    vectorstore
                )

        user_question = st.text_input("Ask a question about your pdfs:")
        if user_question is not None and user_question != "":
            utils.handle_user_question(user_question)
    except:
        st.error(
            "Oops, there was an error :( Please try again with a different question or upload a different file."
        )


def process_csv():
    st.subheader("Ask your data 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    user_question = st.text_input("Ask a question about your csv:")
    if st.button("Submit Query", type="primary"):
        if csv_file is not None and user_question != "":
            with st.spinner(text="In progress..."):
                try:
                    agent = create_agent(csv_file)
                    response = query_agent(agent=agent, query=user_question)
                    decoded_response = decode_response(response)
                    write_response(decoded_response)
                except:
                    st.error(
                        "Oops, there was an error :( Please try again with a different question."
                    )


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
