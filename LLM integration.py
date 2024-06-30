import streamlit as st


def main():
    st.set_page_config(page_title="Web Content Q&A")
    st.header("Web Content Q&A 💬")

    # Text input for the URL
    url_input = st.text_input("Paste the URL of the webpage:")

    if url_input:
        # Show the URL after it's entered
        st.text(f"You entered: {url_input}")

        # Disable the URL input and enable question input
        st.text_area("Ask your questions about the webpage content:")

        # Button to send the question (optional)
        if st.button("Send"):
            # Handle sending the question or further processing
            st.text("Question sent!")


if __name__ == '__main__':
    main()
