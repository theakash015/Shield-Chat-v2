import re
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
import logging
import pypandoc
import pdfkit
from paddleocr import PaddleOCR
import fitz  
import asyncio

llm_groq = ChatGroq(
            model_name='llama3-70b-8192'
    )

# Initialize anonymizer
anonymizer = PresidioReversibleAnonymizer(analyzed_fields=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'IBAN_CODE', 'CREDIT_CARD', 'CRYPTO', 'IP_ADDRESS', 'LOCATION', 'DATE_TIME', 'NRP', 'MEDICAL_LICENSE', 'URL', 'US_BANK_NUMBER', 'US_DRIVER_LICENSE', 'US_ITIN', 'US_PASSPORT', 'US_SSN'], faker_seed=18)

def extract_text_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text

def has_sufficient_selectable_text(page, threshold=50):
    text = page.extract_text()
    if len(text.strip()) > threshold:
        return True
    return False

async def get_text(file_path):
    text = ""
    try:
        logging.info("Starting OCR process for file: %s", file_path)
        extension = file_path.split(".")[-1].lower()
        allowed_extension = ["jpg", "jpeg", "png", "pdf", "docx"]
        if extension not in allowed_extension:
            error = "Not a valid File. Allowed Format are jpg, jpeg, png, pdf, docx"
            logging.error(error)
            return {"error": error}
        
        if extension == "docx":
            file_path = convert_docx_to_pdf(file_path)
        
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(file_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text += line[1][0] + " "
        logging.info("OCR process completed successfully for file: %s", file_path)
    except Exception as e:
        logging.error("Error occurred during OCR process for file %s: %s", file_path, e)
        text = "Error occurred during OCR process."
    logging.info("Extracted text: %s", text)
    return text

def convert_docx_to_pdf(input_path):
    html_path = input_path.replace('.docx', '.html')
    output_path = ".".join(input_path.split(".")[:-1]) + ".pdf"
    pypandoc.convert_file(input_path, 'html', outputfile=html_path)
    pdfkit.from_file(html_path, output_path)
    logging.info("DOCX Format Handled")
    return output_path

async def extract_text_from_mixed_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    pdf_text = ""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if not has_sufficient_selectable_text(page):
            logging.info(f"Page {i+1} has insufficient selectable text, performing OCR.")
            pdf_document = fitz.open(file_path)
            pdf_page = pdf_document.load_page(i)
            pix = pdf_page.get_pixmap()
            image_path = f"page_{i+1}.png"
            pix.save(image_path)
            result = ocr.ocr(image_path, cls=True)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    text += line[1][0] + " "
        pdf_text += text
    return pdf_text

@cl.on_chat_start
async def on_chat_start():
    
    files = None # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            # accept=["application/pdf"],
            accept=["application/pdf", "image/jpeg", "image/png", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=100,
            timeout=180, 
        ).send()

    file = files[0] # Get the first uploaded file
    
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Extract text from PDF, checking for selectable and handwritten text
    if file.name.endswith('.pdf'):
        pdf_text = await extract_text_from_mixed_pdf(file.path)
    else:
        pdf_text = await get_text(file.path)

    # Anonymize the text
    anonymized_text = anonymizer.anonymize(
        pdf_text
    )
    
    # with splitting into chunks
    # {
    # # Split the sanitized text into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_text(anonymized_text)

    # # Create metadata for each chunk
    # metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # # Create a Chroma vector store
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # docsearch = await cl.make_async(Chroma.from_texts)(
    #     texts, embeddings, metadatas=metadatas
    # )
    # }
    
    # without splitting into chunks
    # {
    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        [anonymized_text], embeddings, metadatas=[{"source": "0-pl"}]
    )
    # }
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
        
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    # Callbacks happen asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = anonymizer.deanonymize(
        "ok"+res["answer"]
    )  
    text_elements = [] 
            
    # Return results
    await cl.Message(content=answer, elements=text_elements).send()

