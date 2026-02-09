# 1 在 LangSmith 中创建数据集
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Q&A dataset about LangSmith documentation."
)

# 问答示例，答案作为'标准答案'  
qa_examples = [
    (
        "What is LangChain?",
        "LangChain is an open-source framework for building applications using large language models. It is also the name of the company building LangSmith.",
    ),
    (
        "How might I query for all runs in a project?",
        "You can use client.list_runs(project_name='my-project-name') in Python, or client.ListRuns({projectName: 'my-project-name'}) in TypeScript.",
    ),
    (
        "What's a langsmith dataset?",
        "A LangSmith dataset is a collection of examples. Each example contains inputs and optional expected outputs or references for that data point.",
    ),
    (
        "How do I move my project between organizations?",
        "LangSmith doesn't directly support moving projects between organizations.",
    ),
]

# 将示例添加到数据集
# 输入键为 'question'，输出键为 'answer'
# 这些键必须与 RAG 链期望和生成的内容匹配  
for question, answer in qa_examples:
    client.create_example(
        inputs={"question": question},
        outputs={"answer": answer},
        dataset_id=dataset.id,)

# 2 接下来构建使用 LangChain 和 LangSmith 文档的 RAG 管道问答系统。
# 该系统包含四个主要步骤：
        # 加载文档（抓取 LangSmith 文档）；
        # 创建检索器（嵌入文档并存储在 ChromaDB 中以查找相关片段）；
        # 生成答案（使用 ChatOpenAI 和提示基于检索内容回答）；
        # 组装链（使用 LangChain 表达语言将所有组件合并为单一管道）。
# 2.1 首先加载和处理文档以创建知识库
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings

# 从网络加载文档
api_loader = RecursiveUrlLoader("https://docs.smith.langchain.com")
raw_documents = api_loader.load()
# 将 HTML 转换为干净文本并分割为可管理的片段
doc_transformer = Html2TextTransformer()
transformed = doc_transformer.transform_documents(raw_documents)
text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=2000, chunk_overlap=200)
documents = text_splitter.split_documents(transformed)

# 2.2 创建向量存储检索器
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2.3 随后定义链的生成部分并组装完整的 RAG 管道
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 定义发送给 LLM 的提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful documentation Q&A assistant, trained to answer"
            " questions from LangSmith's documentation."
            " LangChain is a framework for building applications using large language models."
            "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages.",
        ),
        ("system", "{context}"),  # 检索文档的占位符
        ("human", "{question}"),  # 用户问题的占位符
    ]
).partial(time=str(datetime.now()))

# 初始化 LLM，使用大上下文窗口和低温度参数以获得更准确的响应
model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

# 定义生成链，将提示传递给模型再传递给输出解析器
response_generator = prompt | model | StrOutputParser()



# 3 准备好数据集和 RAG 链后，可以执行评估。
# 此次使用内置的"qa"评估器替代"exact_match"。
# 该评估器使用 LLM 根据数据集中的参考答案对生成答案的正确性进行评分
# 配置评估使用"qa"评估器，基于参考答案对"正确性"进行评分
eval_config = RunEvalConfig(
    evaluators=["qa"],
)

# 4 在数据集上运行 RAG 链并应用评估器
# 评估执行将触发测试运行，输出中的链接可用于在 LangSmith 仪表板中实时查看结果
# 运行完成后，LangSmith仪表板提供结果分析界面。除了聚合分数外，还可筛选失败案例进行调试。
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=rag_chain,
    evaluation=eval_config,
    verbose=True,
    project_metadata={"version": "1.0.0", "model": "gpt-3.5-turbo"},
)

# 5 反馈优化 - 通过筛选正确性分数为 0 的示例，可以隔离问题案例。
# 例如，发现模型因检索到不相关文档而产生幻觉答案的情况。
# 可以形成假设："如果信息不在上下文中，模型需要明确被告知不要回答"。
# 修改提示并重新运行评估来测试这一假设：
# 仪表板显示的结果表明新链性能更优，通过了测试集中的所有示例。
# 这种"测试-分析-改进"的迭代循环是改进大语言模型应用的强大方法论。
# 定义改进的提示模板
prompt_v2 = ChatPromptTemplate.from_messages(
     [
         (
             "system",
             "You are a helpful documentation Q&A assistant, trained to answer"  
             " questions from LangSmith's documentation."  
             "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages.",
         ),
         ("system", "{context}"),
         ("human", "{question}"),
         # 新增防止幻觉的指令
         (
             "system",
             "Respond as best as you can. If no documents are retrieved or if you do not see an answer in the retrieved documents,"  
             " admit you do not know or that you don't see it being supported at the moment.",
         ),
     ]
 ).partial(time=lambda: str(datetime.now()))