import os
import base64
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=5000):
    # Split text into chunks of specified size
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def translate_text_chunk(text_chunk, target_language, model):
    prompt_template = f"Act as a machine translator,translate the following text to {target_language}:\n\n{{text_chunk}}"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | output_parser
    
    result = chain.invoke({"text_chunk": text_chunk})
    return result

def save_text_to_file(text, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)

