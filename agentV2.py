import streamlit as st
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.llms import HuggingFaceHub
import os
import json
import math
import csv
import re
from datetime import datetime

# Hugging Face API token from environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")


# Metadata extractor using Hugging Face Inference API
meta_llm = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta",
                           token=hf_token)

def get_rightmost_value(row):
    """Get the rightmost non-empty value from a row, skipping the first two columns."""
    # Skip the first two columns (label and notes) and get the rightmost value
    for value in reversed(row[2:]):
        if value and value.strip():
            return value.strip()
    return None

def parse_date(date_str):
    """Parse date string in format 'MMM. DD, YYYY'."""
    try:
        return datetime.strptime(date_str, '%b. %d, %Y')
    except:
        return None

def get_latest_date_column(rows):
    """Find the column index of the later date in the header row."""
    if not rows or len(rows) < 2:
        return -1
    
    header = rows[0]
    dates = []
    
    # Find all date columns
    for i, cell in enumerate(header):
        if isinstance(cell, str):
            date = parse_date(cell)
            if date:
                dates.append((i, date))
    
    # If we found dates, return the column with the later date
    if len(dates) >= 2:
        dates.sort(key=lambda x: x[1], reverse=True)  # Sort by date, most recent first
        return dates[0][0]  # Return column index of most recent date
    
    return -1

def get_value_from_latest_date(row, latest_date_col):
    """Get the value from the column with the latest date."""
    if latest_date_col >= 0 and len(row) > latest_date_col:
        value = row[latest_date_col].strip()
        return value if value else None
    return None

def extract_metadata_from_structure(file_path: str) -> dict:
    """Extract metadata from directory structure and CSV files."""
    # Get the base directory (Statements)
    base_dir = os.path.join(os.getcwd(), "Statements")
    
    # Initialize metadata with default values
    metadata = {
        "Country": None,
        "Company": None,
        "Total Assets": None,
        "Total Liabilities": None,
        "Total Equity": None,
        "Retained Earnings": None
    }
    
    # Search through the Statements directory to find matching files
    for country in os.listdir(base_dir):
        country_path = os.path.join(base_dir, country)
        if not os.path.isdir(country_path) or country.startswith('.'):
            continue
            
        for company in os.listdir(country_path):
            company_path = os.path.join(country_path, company)
            if not os.path.isdir(company_path) or company.startswith('.'):
                continue
                
            # Check if this company's directory contains our file
            if os.path.exists(os.path.join(company_path, "markdown", file_path)):
                metadata["Country"] = country
                metadata["Company"] = company
                break
        if metadata["Country"]:
            break
    
    # Get the corresponding parsed tables directory
    if metadata["Country"] and metadata["Company"]:
        company_parsed_dir = os.path.join(os.getcwd(), "parsed_tables", metadata["Country"], metadata["Company"])
        print(f"\nSearching for CSVs in: {company_parsed_dir}")
        if os.path.exists(company_parsed_dir):
            for root, dirs, files in os.walk(company_parsed_dir):
                for filename in files:
                    if filename.endswith(".csv"):
                        csv_path = os.path.join(root, filename)
                        try:
                            with open(csv_path, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                rows = list(reader)
                                # Extract financial metrics
                                latest_date_col = get_latest_date_column(rows)
                                print(f"Latest date column: {latest_date_col}")  # Debug print
                                for row in rows:
                                    row_str = str(row[0]).lower() if row else ""
                                    if "total assets" in row_str and metadata["Total Assets"] is None:
                                        print(f"Found row with Total assets: {row}")
                                        value = get_value_from_latest_date(row, latest_date_col)
                                        print(f"Value found: {value}")
                                        if value:
                                            metadata["Total Assets"] = value
                                            print(f"Found and stored Total assets: {value}")
                                    elif "total liabilities" in row_str and metadata["Total Liabilities"] is None:
                                        print(f"Found row with Total liabilities: {row}")
                                        value = get_value_from_latest_date(row, latest_date_col)
                                        print(f"Value found: {value}")
                                        if value:
                                            metadata["Total Liabilities"] = value
                                            print(f"Found and stored Total liabilities: {value}")
                                    elif "total equity" in row_str and metadata["Total Equity"] is None:
                                        print(f"Found row with Total equity: {row}")
                                        value = get_value_from_latest_date(row, latest_date_col)
                                        if value:
                                            metadata["Total Equity"] = value
                                            print(f"Found and stored Total equity: {value}")
                                    elif "retained earnings" in row_str and metadata["Retained Earnings"] is None:
                                        print(f"Found row with Retained earnings: {row}")
                                        value = get_value_from_latest_date(row, latest_date_col)
                                        if value:
                                            metadata["Retained Earnings"] = value
                                            print(f"Found and stored Retained earnings: {value}")
                        except Exception as e:
                            print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Directory not found: {company_parsed_dir}")
    
    return metadata

# Load and preprocess markdown files
def load_markdown_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        # Get the file path from the uploaded file
        file_path = file.name
        metadata = extract_metadata_from_structure(file_path)
        print(f"\nMetadata for {file_path}:")
        print(metadata)
        docs.append({"content": content, "metadata": metadata})
    return docs


def safe_eval_math(expression: str) -> str:
    try:
        allowed_names = {
            k: getattr(math, k)
            for k in dir(math)
            if not k.startswith("__")
        }

        allowed_names.update({
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
        })

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Setup RAG agent
@st.cache_resource(show_spinner=False)
def setup_agentic_rag(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for doc in docs:
        for chunk in splitter.split_text(doc["content"]):
            chunks.append(Document(page_content=chunk, metadata=doc["metadata"]))

    embedding = HuggingFaceInferenceAPIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",api_key=hf_token)
    vectordb = FAISS.from_documents(chunks, embedding)
    retriever = vectordb.as_retriever(search_type="similarity", k=4)

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="financial_search",
        description="Search financial statements to find relevant information."
    )

    calc_tool = Tool(
        name="calculator",
        func=safe_eval_math,
        description="Use this tool to perform simple math or financial calculations from numbers you extract."
    )
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.2}
    )

    agent = initialize_agent(
        tools=[retriever_tool, calc_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

# Streamlit UI
st.set_page_config(page_title="Agentic RAG for Financial Markdown")
st.title("📊 Agentic RAG on Financial Statements")

uploaded_files = st.file_uploader("Upload Markdown Files", type=["md"], accept_multiple_files=True)

if uploaded_files:
    st.success("Files uploaded successfully.")
    docs = load_markdown_files(uploaded_files)
    # agent = setup_agentic_rag(docs)

    user_query = st.text_input("Ask a financial question:", placeholder="E.g. Compare revenue growth between the two companies")

    # if user_query:
        # with st.spinner("Thinking..."):
            # response = agent.run(user_query)
        # st.markdown("### 📌 Answer")
        # st.write(response) 