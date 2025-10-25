import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥", layout="wide")
st.title("🎬 YouTube Video Chatbot using LangChain + Ollama")

# Input field for YouTube video URL
video_url = st.text_input("Enter YouTube video link:")
question = st.text_input("Ask a question about the video:")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -------------------------
# Backend logic
# -------------------------
def load_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1]
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id)
        transcript_text = " ".join([chunk.text for chunk in transcript_list])
        return transcript_text
    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error fetching transcript: {e}")
        return None

def build_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    st.write("🔍 Generating embeddings, please wait...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# -------------------------
# Build FAISS vector store
# -------------------------
if st.button("📥 Process Video"):
    if video_url:
        with st.spinner("Fetching and processing transcript..."):
            transcript_text = load_transcript(video_url)
            if transcript_text:
                st.session_state.vector_store = build_vector_store(transcript_text)
                st.success("✅ Transcript processed and vector store created!")
    else:
        st.warning("Please enter a YouTube video link.")

# -------------------------
# Chat interface
# -------------------------
if question and st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = OllamaLLM(model="gemma3:4b")

    prompt = PromptTemplate(
        template="""You are a helpful assistant. 
Answer ONLY from the provided transcript context. 
If the context is insufficient, reply with "I don't know." 

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    with st.spinner("💭 Generating answer..."):
        res = main_chain.invoke(question)

    st.markdown("### 🧠 Answer:")
    st.write(res)
