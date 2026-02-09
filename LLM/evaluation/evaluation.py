import os
from langchain_openai import ChatOpenAI
import langsmith

# 设置 LangSmith 端点（使用云版本时请勿修改）
# LangSmith API 端点负责在 Web 仪表板中存储所有评估指标
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 配置 LangSmith API 密钥
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"

# 配置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 初始化 LangSmith 客户端
client = langsmith.Client()