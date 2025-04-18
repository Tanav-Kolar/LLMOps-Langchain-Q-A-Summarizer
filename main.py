#1. main.py — LangChain Q&A pipeline with feedback logging

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Groq
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import sqlite3
import datetime
import os
from dotenv import load_dotenv


def get_db_path():
    if os.getenv("HF_SPACE"):
        return "/tmp/feedback.db"
    else:
        return "feedback/feedback.db"

# Load documents
loader = TextLoader("data/sample.txt")
documents = loader.load()

# Split and embed
txt_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = txt_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# LangChain QA chain
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=Groq(), retriever=retriever)

# Feedback logger setup
def init_db():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    response TEXT,
                    feedback TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def log_feedback(query, response, feedback):
    conn = sqlite3.connect("feedback/feedback.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback_log (query, response, feedback, timestamp) VALUES (?, ?, ?, ?)",
              (query, response, feedback, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Example CLI test
if __name__ == "__main__":
    init_db()
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.run(query)
        print("Answer:", response)
        fb = input("Was this helpful? (up/down): ")
        log_feedback(query, response, fb)