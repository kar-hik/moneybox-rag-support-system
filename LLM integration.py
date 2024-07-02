import sys
import os
import streamlit as st

# Add the current directory to the Python path
sys.path.append(os.path.dirname(__file__))

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains.question_answering import load_qa_chain


def main():
    st.set_page_config(page_title="Web Content Q&A")
    st.header("Web Content Q&A 💬")

    # Text input for the URL
    url_input = st.text_input("Paste the URL of the webpage:")

    if url_input:
        # Show the URL after it's entered
        st.text(f"You entered: {url_input}")

        # Read the webpage content from the file
        file_path = 'scraped_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if content:
                # Create embeddings and knowledge base
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(content)

                embeddings = CohereEmbeddings(cohere_api_key="feqm6L1rQdncmas9yquTodYg04XgLhJ5iVTFeAxU")
                knowledge_base = FAISS.from_texts(chunks, embeddings)

                st.text("Webpage content scraped and knowledge base created.")

                # Disable the URL input and enable question input
                user_question = st.text_input("Ask a question about the webpage content:")

                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    llm = Cohere(cohere_api_key="feqm6L1rQdncmas9yquTodYg04XgLhJ5iVTFeAxU", temperature=0.5)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)

                    st.write(response)
        else:
            st.text("Scraped data file not found. Please run the scraping function first.")


if __name__ == '__main__':
    main()
