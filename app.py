import os
import base64
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.utils import get_pdf_text, chunk_text, translate_text_chunk, save_text_to_file

def download_link(text, filename, text_type):
    b64 = base64.b64encode(text.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:text/{text_type};base64,{b64}" download="{filename}">Download {filename}</a>'

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI_API_VERSION"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]

# Initialize the language models
model_gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, safety_settings=None)
model_gpt4o = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
output_parser = StrOutputParser()

# Streamlit UI
st.title("Machine Translation")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
target_language = st.text_input("Enter the language you want to convert to: ")

if 'translated_text_gemini' not in st.session_state:
    st.session_state['translated_text_gemini'] = None
if 'satisfaction' not in st.session_state:
    st.session_state['satisfaction'] = "Select an option"

if uploaded_file is not None:
    st.write("Extracting text from PDF...")
    pdf_text = get_pdf_text(uploaded_file)
    st.write("Text extracted successfully!")

    if st.button("Translate with Gemini"):
        st.write("Translating text with Gemini...")

        # Chunk the text for translation
        text_chunks = chunk_text(pdf_text)

        translated_chunks_gemini = []

        for i, chunk in enumerate(text_chunks):
            st.write(f"Chunk {i + 1}:")
            st.write(chunk)

            st.write(f"Translating Chunk {i + 1} with Gemini...")
            translated_chunk_gemini = translate_text_chunk(chunk, target_language, model_gemini)
            translated_chunks_gemini.append(translated_chunk_gemini)
            st.write(f"Translated Chunk {i + 1} with Gemini:")
            st.write(translated_chunk_gemini)

        # Combine the translated chunks
        st.session_state['translated_text_gemini'] = '\n'.join(translated_chunks_gemini)

        st.write("Complete Translation with Gemini:")
        st.write(st.session_state['translated_text_gemini'])
        st.write("Translation complete with Gemini!")

        # Provide download link for the translated text
        st.markdown(download_link(st.session_state['translated_text_gemini'], 'translated_text_gemini.txt', 'plain'), unsafe_allow_html=True)

# '''
# Streamlit restarts the app if things are not stored in session_state(reruns the code)
# Important Comments -
# To manage this state and avoid re-execution of the entire script, you can use Streamlit's session_state.
# '''
if st.session_state['translated_text_gemini']:
    st.session_state['satisfaction'] = st.selectbox(
        "Are you satisfied with the Gemini translation, or do you want to try with GPT-4?",
        ["Select an option", "Yes", "No"]
    )

    if st.session_state['satisfaction'] == "No":
        st.write("Translating text with GPT-4...")

        text_chunks = chunk_text(pdf_text)
        translated_chunks_gpt4o = []

        for i, chunk in enumerate(text_chunks):
            st.write(f"Translating Chunk {i + 1} with GPT-4...")
            translated_chunk_gpt4o = translate_text_chunk(chunk, target_language, model_gpt4o)
            translated_chunks_gpt4o.append(translated_chunk_gpt4o)
            st.write(f"Translated Chunk {i + 1} with GPT-4:")
            st.write(translated_chunk_gpt4o)

        # Combine the translated chunks
        translated_text_gpt4o = '\n'.join(translated_chunks_gpt4o)

        st.write("Complete Translation with GPT-4:")
        st.write(translated_text_gpt4o)
        st.write("Translation complete with GPT-4!")

        # Provide download link for the translated text
        st.markdown(download_link(translated_text_gpt4o, 'translated_text_gpt4o.txt', 'plain'), unsafe_allow_html=True)

    if st.session_state['satisfaction']=='Yes':
        st.write("Thanks for using the app!!")
