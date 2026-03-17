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

# ==========================================
# 2. 侧边栏：文档上传与处理逻辑（多文档合并版）
# ==========================================

# ==========================================
# 3. 聊天与问答逻辑（带历史记忆与溯源版）
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
        if "vector_store" in st.session_state:
            # 组装 RAG 提示词
            prompt = ChatPromptTemplate.from_template("""
            你是一个专业的智能分析助手。请严格根据以下【参考文档】的内容回答问题。
            如果文档中没有相关信息，请直接回答“文档中未提及”，不要自己编造。
            
            【参考文档】：
            {context}
            
            【用户问题】：
            {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            # ⚠️ 就是这里！必须先定义 retriever，它才能去干活
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # 获取 AI 的回答以及它参考的文档片段
            response = retrieval_chain.invoke({"input": user_input})
            ai_answer = response["answer"]
            source_docs = response["context"] # 这里面存着 AI 参考的文本块
            
            # --- 新增的溯源逻辑开始 ---
            source_pages = set() # 用集合来去重，避免重复显示同一页
            for doc in source_docs:
                if "page" in doc.metadata:
                    # PyPDFLoader 的页码是从 0 开始的，所以我们 +1 变成人类习惯的页码
                    source_pages.add(str(doc.metadata["page"] + 1))
            
            # 如果找到了页码，就把它们拼接在回答的最后面
            if source_pages:
                # 把页码按数字大小排个序，看起来更专业
                sorted_pages = sorted(list(source_pages), key=int)
                ai_answer += f"\n\n> 🔍 **参考依据：** 本文档的第 {', '.join(sorted_pages)} 页"
            # --- 新增的溯源逻辑结束 ---

        else:
            # 如果没传文档，就当成普通聊天
            response = llm.invoke(user_input)
            ai_answer = response.content

        # 3. 把 AI 的回答写进本子，并显示在屏幕上
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        st.chat_message("assistant").write(ai_answer)