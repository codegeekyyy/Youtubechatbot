from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo"

try:
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id)
    transcript_text = " ".join([chunk.text for chunk in transcript_list])
except TranscriptsDisabled:
    print("No caption available for this video.")
    transcript_text = ""

# Step 1b - Split transcript
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript_text)

# Step 1c - Convert to Documents
docs = [Document(page_content=chunk) for chunk in chunks]

# Step 1d - Embeddings + Vector Store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(docs, embeddings)

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Step 3 - LLM + Prompt
llm = OllamaLLM(model="gemma3:4b")

prompt = PromptTemplate(
    template="""
You are a helpful assistant. ANSWER ONLY from the provided transcript context.
If the context is insufficient, just say 'I don't know.'

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# Helper: format retrieved docs into a single string
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Step 4 - Chain setup
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# Step 5 - Run the chain
res = main_chain.invoke("can you summarize the video?")
print("\nFinal Answer:\n", res)
