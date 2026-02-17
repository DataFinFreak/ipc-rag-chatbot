import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================
# 1️⃣ SET API KEY
# ==============================

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# ==============================
# 2️⃣ LOAD DOCUMENT
# ==============================

loader = PyPDFLoader("THE_INDIAN_PENAL_CODE.pdf")
documents = loader.load()

text = ""
for doc in documents:
    text += doc.page_content + "\n"

# ==============================
# 3️⃣ CHUNKING
# ==============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(text)

# ==============================
# 4️⃣ EMBEDDINGS + VECTOR STORE
# ==============================

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding,
    persist_directory="IPC_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ==============================
# 5️⃣ LLM SETUP
# ==============================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==============================
# 6️⃣ PROMPT
# ==============================

prompt = ChatPromptTemplate.from_template("""
You are a legal assistant specializing in Indian Penal Code.

Use the context below to answer clearly and mention relevant IPC sections.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
""")

# ==============================
# 7️⃣ RAG PIPELINE
# ==============================

retriever_chain = RunnableLambda(lambda x: retriever.invoke(x["question"]))

rag_chain = (
    {
        "context": retriever_chain,
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==============================
# 8️⃣ CONVERSATION LOOP
# ==============================

chat_history = ""

def conversational_RAG(question):
    global chat_history

    response = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    chat_history += f"\nUser: {question}"
    chat_history += f"\nAssistant: {response}"

    return response


if __name__ == "__main__":
    print("IPC Legal Assistant 🤖⚖️")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Ask your IPC question: ")

        if user_input.lower() == "quit":
            print("Goodbye 👋")
            break

        answer = conversational_RAG(user_input)
        print("\nAnswer:\n", answer)
