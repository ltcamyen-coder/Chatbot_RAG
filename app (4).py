import google.generativeai as genai
genai.configure(api_key="API_KEY")

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API key (thay bằng key thật của bạn từ Google AI Studio)
genai.configure(api_key="API_KEY")

# ================== CẤU HÌNH ==================
JSON_FILE = "/content/drive/RAG/all_procedures_normalized.json"  # Đường dẫn file JSON (sau chunk rule-based)
CHROMA_DB_PATH = "chroma_db"  # Thư mục lưu vector DB
COLLECTION_NAME = "RAG_procedure_children_under_6"
GEMINI_MODEL = "gemini-2.5-flash"  # Hoặc "gemini-1.5-pro"

@st.cache_resource
def get_embedding_function():
    EMBEDDING_MODEL = "BAAI/bge-m3"  # Model embedding tiếng Việt
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return embedding_function

@st.cache_resource
def build_vector_db(_json_file):
    if not os.path.exists(_json_file):
        st.error(f"Không tìm thấy file JSON: {_json_file}. Hãy kiểm tra đường dẫn!")
        return None

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = get_embedding_function()

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        st.success("Tải collection Chroma đã tồn tại.")
    except:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        st.info("Tạo collection Chroma mới.")

    # Load dữ liệu từ JSON
    with open(_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    documents = []
    metadatas = []

    for item in data:
        ids.append(item["id"])
        documents.append(item["content_text"])
        meta = item.get("metadata", {}).copy()
        meta.update({
            "url": item.get("url", "unknown"),
            "title": item.get("title", "unknown"),
            "hierarchy": item.get("hierarchy", "unknown"),  # Từ rule-based (ví dụ: "Trình tự thực hiện")
            "chunk_type": item.get("chunk_type", "text"),
            "domain": item.get("source_domain", "dichvucong.gov.vn")
        })
        metadatas.append(meta)

    if len(ids) > 0:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        st.success(f"Đã thêm {len(data)} chunks vào vector DB!")

    return collection

# Load vector DB (cache để tránh rebuild mỗi lần)
collection = build_vector_db(JSON_FILE)
if collection is None:
    st.stop()

def query_rag(query: str, chat_history: list, top_k: int):
    # Retrieval với top_k động
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['hierarchy']}]\\n{doc}\\n(Nguồn: {meta['url']})")

    context = "\\n\\n".join(context_parts)

    prompt = f"""
    Bạn là trợ lý tư vấn thủ tục hành chính cho trẻ em dưới 6 tuổi tại Việt Nam. Trả lời ngắn gọn, chính xác, dễ hiểu, có dẫn nguồn. Chỉ trả lời dựa trên context, không thêm thông tin ngoài.

    Context:
    {context}

    Câu hỏi: {query}

    Trả lời bằng tiếng Việt, có đánh số nếu là danh sách, và trích dẫn nguồn rõ ràng (tên block, URL):
    """

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt, stream=True)

    return response

# Giao diện chính
st.set_page_config(page_title="Chatbot tư vấn thủ tục hành chính trẻ em dưới 6 tuổi", layout="centered")
st.title("Chatbot tư vấn thủ tục hành chính cho trẻ em dưới 6 tuổi")

# Sidebar với top-k slider và thông tin
with st.sidebar:
    st.header("Cài đặt")
    top_k = st.slider("Top-k retrieval (số chunks lấy về)", min_value=1, max_value=10, value=3, step=1)
    st.header("Thông tin")
    st.write(f"Vector DB: {COLLECTION_NAME}")
    st.write(f"Số chunk: {collection.count() if collection else 0}")
    st.write(f"Model LLM: {GEMINI_MODEL}")
    st.write("Embedding: BAAI/bge-m3 (tối ưu tiếng Việt)")
    st.caption("Dữ liệu load từ file JSON. Nếu lỗi, kiểm tra đường dẫn JSON_FILE.")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input từ user
if prompt := st.chat_input("Hỏi về thủ tục hành chính cho trẻ em dưới 6 tuổi (ví dụ: Đăng ký khai sinh cần gì?)"):
    # Thêm tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi RAG với top_k từ slider và stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            response = query_rag(prompt, st.session_state.messages, top_k)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + " ")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Lỗi khi gọi Gemini: {str(e)}"
            message_placeholder.error(full_response)

    # Lưu response vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})
# Dán toàn bộ code trên vào đây
