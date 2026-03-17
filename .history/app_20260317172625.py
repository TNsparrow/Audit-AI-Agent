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

# 加载环境变量
load_dotenv()

st.title("🚀 企业级智能文档分析与对账助手")

# 1. 初始化模型
# 智谱的大模型用来聊天和总结
llm = ChatOpenAI(model="glm-4-flash", temperature=0.1) 
# 智谱的向量模型用来把文字变成数字
embeddings = OpenAIEmbeddings(model="embedding-3", chunk_size=50)

# 2. 侧边栏：文档上传与处理逻辑
# 2. 侧边栏：文档上传与处理逻辑
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["txt", "pdf"])
    
    # 核心修复：判断是否传了文件，并且当前文件是不是一本“新书”
    if uploaded_file:
        if "processed_file_name" not in st.session_state or st.session_state.processed_file_name != uploaded_file.name:
            with st.spinner(f"发现新文件 {uploaded_file.name}，正在拼命阅读中..."):
                # 保存临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 读取文件
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()

                # 切割文本
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = text_splitter.split_documents(docs)

                # 建立新的向量数据库，并覆盖旧的记忆
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                # 记住这本新书的名字
                st.session_state.processed_file_name = uploaded_file.name 
                st.success(f"《{uploaded_file.name}》学习完毕！可以提问了。")
        else:
            # 如果名字没变，说明已经读过了
            st.success(f"当前知识库：《{uploaded_file.name}》")


# 3. 聊天与问答逻辑（带历史记忆版）
# ==========================================

# 如果脑子里还没有“聊天记录”这个本子，就建一个空的
if "messages" not in st.session_state:
    st.session_state.messages = []

# 每次网页刷新时，先把本子里的历史记录循环打印出来
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 接收用户的新提问
user_input = st.chat_input("请基于上传的文档提问，例如：总结核心数据...")

if user_input:
    # 1. 把用户的话写进本子，并显示在屏幕上
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.spinner("AI 正在翻阅文档寻找答案..."):
        # 2. 判断有没有上传文档
     