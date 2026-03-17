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
# 2. 侧边栏：文档上传与处理逻辑（带永久记忆落盘版）
# ==========================================

# 定义本地数据库的保存路径（就会存在你当前的代码文件夹里）
DB_PATH = "local_faiss_db"

with st.sidebar:
    st.header("📁 知识库管理")
    
    # 💡 核心新增：每次打开网页，先看看硬盘里有没有以前存过的脑子
    if "vector_store" not in st.session_state:
        if os.path.exists(DB_PATH):
            # allow_dangerous_deserialization=True 是必须加的安全确认参数，因为这是我们自己建的本地库，绝对安全
            st.session_state.vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("🧠 已成功唤醒历史记忆数据库！")
    
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["txt", "pdf"])
    
    # 记录当前已处理的文件（如果不想要每次刷新都清空列表，可以用更复杂的逻辑，这里保持极简）
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"正在将《{uploaded_file.name}》刻入永久记忆..."):
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

                # 合并或新建数据库
                if "vector_store" not in st.session_state:
                    st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                else:
                    st.session_state.vector_store.add_documents(split_docs)
                
                # 💡 核心新增：每次更新完数据库，立刻【保存到硬盘】！
                st.session_state.vector_store.save_local(DB_PATH)
                
                st.session_state.processed_files.append(uploaded_file.name) 
                st.success(f"《{uploaded_file.name}》已永久刻入知识库！")
        
    if st.session_state.processed_files:
        st.markdown("### 📚 本次新增的文档：")
        for file_name in st.session_state.processed_files:
            st.markdown(f"- `{file_name}`")
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