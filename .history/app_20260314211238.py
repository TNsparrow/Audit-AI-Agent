import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 自动加载 .env 文件里的 API 密钥和地址
load_dotenv()

# 搭建网页 UI
st.title("🚀 我的第一个 AI 大模型应用")
st.write("用 Vibe Coding 搞定的极简对话雏形！")

# 链接你的大模型 API
# 注意：把 "你的模型名称" 换成你实际使用的模型，比如 "deepseek-chat" 或 "glm-4"
llm = ChatOpenAI(
    model="glm-4-flash", # 智谱的免费模型
    temperature=0.7
)

# 创建一个像微信一样的底部输入框
user_input = st.chat_input("有什么想问我的吗？")

# 如果用户发送了消息，就调用大模型
if user_input:
    # 显示用户的提问
    st.chat_message("user").write(user_input)
    
    # 显示 AI 的回复
    with st.spinner("AI 思考中..."):
        response = llm.invoke(user_input)
        st.chat_message("assistant").write(response.content)