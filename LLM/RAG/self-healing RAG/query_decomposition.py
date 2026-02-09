from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import ChatOllama

#
# 定义输出结构
class SubQueries(BaseModel):
    """待检索的子问题集合"""
    questions: List[str] = Field(description="List of atomic sub-questions.") #


def plan_query(query: str,model = 'qwen3-1.7b'):
    '''
    query：用户输入的查询
    return：List[str], 待检索的子问题集合
    '''
    # 配置规划用的LLM
    llm = ChatOllama(model=model, temperature=0.00)
    system_prompt = """You are an expert complex query decomposer. 
    If the user's query is a complex query containing multiple layers of meaning, 
    break down the complex query into main, simple, atomic sub-queries.
    If the user's query itself is simple and has only one meaning, 
    output the original query as the value of key 'questions'.
    
    In any case, your output should correspond to the structure that definited in Function 'SubQueries'
    
    NOTE
    The number of sub-queries do not go longer than 3. 
    The combined set of all subqueries should be able to fully encompass all the meanings of this complex query.    
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")])
    # 构建处理链
    planner = prompt | llm.with_structured_output(SubQueries)
    result = planner.invoke({"query": query})

    try:
        return result.questions
    except:
        return result

if __name__ == '__main__':
    # 使用示例
    # query="is there any difference in tiger and cat?"
    # query="what is your name"
    # query="who are you"
    query="who am I"
    sub_qs = plan_query(query)
    # print(len(sub_qs))
    print(sub_qs)
