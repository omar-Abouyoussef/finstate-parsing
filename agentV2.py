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


# Hugging Face API token
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

# Metadata extractor using Hugging Face Inference API
meta_llm = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta",
                           token=hf_token)

# def extract_metadata(text: str):
#     prompt = f"""<|system|>
# You are a financial document analyzer. Extract metadata as JSON.

# <|user|>
# Extract company name, report date, and currency from this financial document:

# {text[:1000]}

# Return only JSON format like: {{"Company": "...", "Date": "...", "Currency": "..."}}

# <|assistant|>
# """
    
#     try:
#         response = meta_llm.text_generation(
#             prompt, 
#             max_new_tokens=150,
#             temperature=0.1,
#             do_sample=False,
#             repetition_penalty=1.1
#         )
        
#         print(f"API Response: {response}")
        
#         # Find JSON in response
#         json_start = response.find('{')
#         json_end = response.rfind('}') + 1
        
#         if json_start != -1 and json_end > json_start:
#             json_str = response[json_start:json_end]
#             metadata_json = json.loads(json_str)
#             return metadata_json
#         else:
#             raise ValueError("No JSON found")
            
#     except Exception as e:
#         print(f"API Error: {e}")
#         # Return default metadata if API fails
#         import re
        
#         # Try basic regex extraction as fallback
#         company_match = re.search(r'(?:company|corp|inc|ltd)[:\s]+([^\n\.]+)', text[:1000], re.IGNORECASE)
#         company = company_match.group(1).strip() if company_match else "Unknown Company"
        
#         return {
#             "Company": company,
#             "Date": "Unknown",
#             "Currency": "USD"
#         }

def extract_metadata(text: str):
    prompt = f"""<|system|>
    You are a financial document analyzer. Extract metadata as JSON.

    <|user|>
    Extract company name, report date, and currency from this financial document:

    {text[:1000]}

    Return only JSON format like: {{"Company": "...", "Date": "...", "Currency": "..."}}

    <|assistant|>
    """

    
    response = meta_llm.text_generation(
            prompt, 
        )
    print(response)
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        metadata_json = json.loads(response[json_start:json_end])
        return metadata_json
    except:
        return {"Company": None, "Date": None, "Currency": None}

# Load and preprocess markdown files
def load_markdown_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        metadata = extract_metadata(content)
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
