from typing import List, TypedDict,Dict,Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph, START
from langchain_ollama import ChatOllama
import os

os.environ["TAVILY_API_KEY"]="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" # 搜索工具TAVILY的key


class Evaulate(TypedDict):
    """对文档的相关性评估结果."""
    score: Annotated[str,  "对问题与文档的相关性评估结果，该结果只可以是yes或no"]


# --- 1. 状态定义 ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str  # 'Yes'或'No'标记
    documents: List[Document]

# --- 2. 组件初始化 ---
model ='qwen3-1.7b' # 'Qwen3-4b'  # 'qwen3-1.7b'
grader_llm = ChatOllama(model=model, temperature=0)  # 文档相关性评估智能体
generator_llm = ChatOllama(model=model, temperature=0.) # 重写查询智能体
# 创建Tavily搜索工具实例，设置最大结果数为3
web_tool = TavilySearch(k=3)

# --- 3. 节点定义 ---
def grade_documents(state):
    """
    自愈核心节点：文档相关性审查，以过滤低质量文档。
    即对相关性文档进行保留，如果不相关则进行网络搜索。
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    print(state)
    # 二分类结构化输出
    structured_llm = grader_llm.with_structured_output(Evaulate)
    prompt = PromptTemplate(
        # You are a grader assessing relevance.
        # Return JSON with key 'score' as 'yes' or 'no'
        template="""你是一个评估相关性的评分员，请评估下面这个文档是否与问题相对应。
        如果该文档中的内容较好地回答了这个问题，则认为文档与问题是强相关，返回yes；
        如果该文档中的信息与该问题不对应或无法对应，则认为两者相关性较低，返回no。
        问题和文档如下:
            问题: {question}  
            文档: {context}   
        NOTE
            最后输出的JSON格式数据必须严格按照Evaulate函数中的定义和要求；
            示例：{{"score":"yes"或"no"}}
            
        """,
        input_variables=["context", "question"],)
    chain = prompt | structured_llm

    filtered_docs = []
    web_search = "No"
    # 获取相关性文档。获取当前问题与其所有的答案文档的相关性
    for d in documents:
        grade = chain.invoke({"question": question, "context": d.page_content})
        print('grade',grade)
        if grade.get('score') == 'yes':
            filtered_docs.append(d)
    exit()
    # 当没有找到相关文档时，则去网络搜索
    if len(filtered_docs)==0:
        # 丢失上下文时触发回退
        web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def decide_to_generate(state):
    '''
    条件分支。决定去网上查询还是直接生成答案
    '''
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"

def transform_query(state):
    """
    自我纠正：重写查询以提升网页搜索效果.
    即将输入查询改写成适合 Web 搜索的关键词.
    return：Dict[str]. 重写后的问题
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    # 简易重写链
    prompt = PromptTemplate(template="Rewrite this for web search: {question}"
                                     "NOTE: "
                                     "Don't say prologue and transitional statement;"
                                     "Just output the final version for web search.",
                            input_variables=["question"])
    chain = prompt | generator_llm
    better_q = chain.invoke({"question": question}).content
    return {"question": better_q}

def web_search_node(state):
    '''
    进行网络搜索。
    最终的输出是网络搜索结果
    '''
    print("---WEB SEARCH---")
    docs = web_tool.invoke({"query": state["question"]})
    # 追加网页结果到已有文档。后续对这个改进，可以相当于文档增量，即增量故障图谱
    web_results = [Document(page_content=d["content"]) for d in docs['results']]
    # scores=[d["score"] for d in docs['results']]
    # print(scores)
    # return {"documents": state["documents"] + web_results} # 增量故障图谱
    return {"documents": web_results}

def generate(state):
    print("---GENERATE---")
    # 这里接标准RAG生成链。即根据查询和所有的答案进行综合输出。
    # generation = rag_chain.invoke(...)
    return {"generation": "Final Answer Placeholder"}


# --- 4. 图构建 ---
def create_grag():
    workflow = StateGraph(GraphState)
    # 添加节点
    # workflow.add_node("retrieve", lambda x: {"documents": []})  # 检索占位
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate", generate)
    # 添加边
    workflow.add_edge(START, "grade_documents")
    workflow.add_conditional_edges("grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query",
         "generate": "generate"})

    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    app = workflow.compile()
    return app


if __name__ == '__main__':
    # query="Compare HyDE and standard retrieval methods"
    # documents=['123','Several studies have shown that HyDE improves the retrieval performance compared to the traditional embedding model. But this comes at the cost of increased']
    query="12乘以12等于几"
    documents=[Document(page_content='123213213ho arer you3435435'),
        Document(page_content='***&%%……&（）*（）%#@！'),
        Document(page_content='12的立方等于144*12'),
               # Document(page_content='12*12等于144, 12乘以12为144')
               ]
    initial_state  = {'question': query,
                      'generation': "",
                      'web_search': "No",
                      'documents': documents}
    app=create_grag()
    print(app,)
    result=app.invoke(initial_state)
    print(result)
