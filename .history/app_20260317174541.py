import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 0. 全局页面配置（必须放在第一行，开启宽屏和专属 Title）
# ==========================================
st.set_page_config(
    page_title="Mason 的智能审计与对账引擎",
    page_icon="💼",
    layout="wide", # 开启宽屏模式，更适合阅读长篇财务底稿
    initial_sidebar_state="expanded"
)

# 加载环境变量
load_dotenv()

# 初始化模型
llm = ChatOpenAI(model="glm-4-flash", temperature=0.1) 
embeddings = OpenAIEmbeddings(model="embedding-3", chunk_size=50) 
DB_PATH = "local_faiss_db"

# ==========================================
# 1. 主界面欢迎区 UI
# ==========================================
st.title("💼 智能审计与财务对账引擎")
st.markdown("""
<style>
.small-font { font-size:14px !important; color: #888888; }
</style>
<p class="small-font">Powered by GLM-4 & FAISS Vector Database | 专注解决海量底稿检索与跨文档数据溯源</p>
""", unsafe_allow_html=True)
st.divider() # 加一条优雅的分隔线

# ==========================================
# 2. 侧边栏：知识库管理 (带永久记忆)
# ==========================================
with st.sidebar:
    st.header("📁 审计底稿与知识库管理")
    
    if "vector_store" not in st.session_state:
        if os.path.exists(DB_PATH):
            st.session_state.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            st.toast("🧠 历史知识库已加载完毕！", icon="✅") # 用右上角轻提示代替大块的 success
    
    uploaded_file = st.file_uploader("上传审计报告、财务报表或相关法规 (PDF/TXT)", type=["txt", "pdf"])
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"正在解析并向量化《{uploaded_file.name}》..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = text_splitter.split_documents(docs)

                if "vector_store" not in st.session_state:
                    st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                else:
                    st.session_state.vector_store.add_documents(split_docs)
                
                st.session_state.vector_store.save_local(DB_PATH)
                st.session_state.processed_files.append(uploaded_file.name) 
                st.success(f"解析成功！已并入全局检索库。")
        
    if st.session_state.processed_files:
        st.markdown("### 📚 检索范围池：")
        for file_name in st.session_state.processed_files:
            st.markdown(f"- 📄 `{file_name}`")

# ==========================================
# 3. 聊天与问答逻辑（定制头像版）
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好，Mason。我是您的专属智能审计助手，已准备好为您检索和比对文档数据，请随时提问。"}]

# 定制聊天头像
avatars = {"user": "🧑‍💼", "assistant": "🤖"}

for message in st.session_state.messages:
    st.chat_message(message["role"], avatar=avatars[message["role"]]).write(message["content"])

user_input = st.chat_input("输入查询指令，例如：提取本文档中关于坏账准备的计提标准...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user", avatar=avatars["user"]).write(user_input)
    
    with st.spinner("⚡ 引擎检索中，请稍候..."):
        if "vector_store" in st.session_state:
            prompt = ChatPromptTemplate.from_template("""
            你是一个资深的审计专家和智能分析助手。请严格根据以下【参考文档】的内容回答问题。
            你的回答需要严谨、专业。如果文档中没有相关信息，请直接回答“当前知识库中未检索到相关数据”，绝对不要自己编造。
            
            【参考文档】：
            {context}
            
            【用户问题】：
            {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({"input": user_input})
            ai_answer = response["answer"]
            source_docs = response["context"] 
            
            source_pages = set() 
                if "page" in doc.metadata:
                    source_pages.add(str(doc.metadata["page"] + 1))
            
            if source_pages:
                sorted_pages = sorted(list(source_pages), key=int)
                ai_answer += f"\n\n> 🔍 **溯源定位：** 参考底稿第 {', '.join(sorted_pages)} 页"

        else:
            response = llm.invoke(user_input)
            ai_answer = response.content

        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        st.chat_message("assistant", avatar=avatars["assistant"]).write(ai_answer)