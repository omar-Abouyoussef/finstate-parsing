import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load summarization model locally
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    return summarizer

# Load QA model locally
@st.cache_resource
def load_qa_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

# Streamlit UI
st.set_page_config(page_title="PDF Summary & Q&A", layout="wide")
st.title("Financial PDF Summary + Q&A")

uploaded_files = st.file_uploader("Upload financial PDF(s)", type=["pdf"], accept_multiple_files=True)
query = st.text_input("Ask a question about the uploaded content:")

if uploaded_files:
    all_text = ""
    raw_docs = []

    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            all_text += text
            raw_docs.append(Document(page_content=text, metadata={"filename": file.name}))

    # Show summary
    st.subheader("📝 Summary")
    summarizer = load_summarizer()
    if len(all_text) > 1000:
        summary = summarizer(all_text[:3000], max_length=512, min_length=30, do_sample=False)[0]["summary_text"]
    else:
        summary = summarizer(all_text, max_length=512, min_length=30, do_sample=False)[0]["summary_text"]
    st.write(summary)

    # Prepare for QA
    st.subheader("📌 Q&A")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa_llm = load_qa_model()
    qa = RetrievalQA.from_chain_type(llm=qa_llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    if query:
        answer = qa.run(query)
        st.write(answer)

    st.subheader("📁 Uploaded Files")
    for file in uploaded_files:
        st.write(file.name)

else:
    st.info("Upload at least one financial PDF to get started.")
