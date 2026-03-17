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
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["txt", "pdf"])
    
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("正在拼命阅读和理解文档中..."):
            # 将上传的文件保存到临时路径，方便后续加载
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # 读取文件内容
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            docs = loader.load()

            # 把长文档切成小块（每块 500 字），方便 AI 精准检索
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)

            # 把文字转换成向量，存入本地 FAISS 数据库，并保存在网页的 session 中
            st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
            st.success("文档学习完毕！现在可以向我提问了。")

# 3. 聊天与问答逻辑
user_input = st.chat_input("请基于上传的文档提问，例如：总结核心数据...")

if user_input:
    st.chat_message("user").write(user_input)
    
    with st.spinner("AI 正在翻阅文档寻找答案..."):
        # 检查是否上传了文档
        if "vector_store" in st.session_state:
            # 创建一个带上下文的提示词模板
            prompt = ChatPromptTemplate.from_template("""
            你是一个专业的智能分析助手。请严格根据以下【参考文档】的内容回答问题。
            如果文档中没有相关信息，请直接回答“文档中未提及”，不要自己编造。
            
            【参考文档】：
            {context}
            
            【用户问题】：
            {input}
            """)
            
            # 把检索器和回答链条组装起来
            document_chain = create_stuff_documents_chain(llm, prompt)
            # 把检索的文本块从默认的 4 个扩大到 10 个（甚至你可以改成 15）
retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # 执行问答
            response = retrieval_chain.invoke({"input": user_input})
            st.chat_message("assistant").write(response["answer"])
        else:
            # 如果没上传文档，就正常聊天
            response = llm.invoke(user_input)
            st.chat_message("assistant").write(response.content)