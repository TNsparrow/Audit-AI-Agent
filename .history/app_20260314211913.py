import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 搭建网页 UI
st.title("🚀 我的企业级文档分析助手")

# 1. 增加左侧边栏：用于上传文件
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("请上传需要分析的文件 (支持 TXT/PDF)", type=["txt", "pdf"])
    
    if uploaded_file:
        st.success(f"文件 {uploaded_file.name} 上传成功！(解析逻辑下一步添加)")

# 2. 初始化大模型 (请确保模型名字是你刚才测试成功的那个，比如 glm-4-flash)
llm = ChatOpenAI(
    model="glm-4-flash", 
    temperature=0.7
)

# 3. 聊天界面
user_input = st.chat_input("有什么想问我的吗？")

if user_input:
    st.chat_message("user").write(user_input)
    
    with st.spinner("AI 思考中..."):
        response = llm.invoke(user_input)
        st.chat_message("assistant").write(response.content)