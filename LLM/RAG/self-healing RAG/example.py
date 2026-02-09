# # 导入基本消息类、用户消息类和工具消息类
# from langchain_core.messages import (HumanMessage, #用户提示词
#     ToolMessage, #工具消息
#     BaseMessage,)
# # 导入聊天提示模板和消息占位符
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # 导入状态图相关的常量和类
# from langgraph.graph import END, StateGraph, START
# # 导入Tavily搜索工具
# from langchain_community.tools.tavily_search import TavilySearchResults
# # 导入工具装饰器
# from langchain_core.tools import tool
# # 导入Python REPL工具
# from langchain_experimental.utilities import PythonREPL
# # 导入操作符和类型注解
# import operator
# # 导入注解类型
# from typing import Annotated, Sequence, TypedDict
# # 导入functools模块
# import functools
# # 导入AI消息类
# from langchain_core.messages import AIMessage
# # 导入预构建的工具节点
# from langgraph.prebuilt import ToolNode
# # 任一代理都可以决定结束
# from typing import Literal
# import os
# from langchain_community.chat_models import ChatTongyi
#
# os.environ["DASHSCOPE_API_KEY"] = "sk-271135c629c74e7d944f822b6130c6e4" # Tongyi API key
# os.environ["TAVILY_API_KEY"]="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" # 搜索工具TAVILY的key
#
# llm=ChatTongyi(model="qwen-plus")
#
# # 定义一个函数，用于创建代理
# def create_agent(llm, tools, system_message: str):
#     """创建一个代理。"""
#     # 创建一个聊天提示模板
#     prompt = ChatPromptTemplate.from_messages([(
#             "system",
#             "你是一个有帮助的AI助手，与其他助手合作。使用提供的工具来推进问题的回答。"
#             " 如果你不能完全回答，没关系，另一个拥有不同工具的助手"
#             " 会接着你的位置继续帮助。执行你能做的以取得进展。如果你或其他助手有最终答案或交付物，"
#             " 在你的回答前加上FINAL ANSWER，以便团队知道停止。"
#             " 你可以使用以下工具: {tool_names}。\n{system_message}", ),
#             MessagesPlaceholder(variable_name="messages"),]) # 消息占位符
#     # 传递系统消息参数。partial方法分次初始化模板
#     prompt = prompt.partial(system_message=system_message)
#     # 传递工具名称参数
#     prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
#     # 绑定工具并返回提示模板
#     return prompt | llm.bind_tools(tools)
#
# # 创建Tavily搜索工具实例，设置最大结果数为5
# tavily_tool = TavilySearchResults(max_results=5)
#
# # 警告：这会在本地执行代码，未沙箱化时可能不安全
# # 创建Python REPL实例
# repl = PythonREPL()
#
# # 定义一个工具函数，用于执行Python代码
# @tool
# def python_repl(code: Annotated[str, "要执行以生成图表的Python代码。"],):
#     """使用这个工具来执行Python代码。如果你想查看某个值的输出，应该使用print(...)。这个输出对用户可见。"""
#     try:
#         # 尝试执行代码matplotlib
#         result = repl.run(code)
#     except BaseException as e:
#         # 捕捉异常并返回错误信息
#         return f"执行失败。错误: {repr(e)}"
#     result_str = f"成功执行:\n```python\n{code}\n```\nStdout: {result}" # 返回执行结果
#     return ( result_str + "\n\n如果你已完成所有任务，请回复FINAL ANSWER。")
#
#
# # 定义一个对象，用于在图的每个节点之间传递
# # 我们将为每个代理和工具创建不同的节点
# class AgentState(TypedDict):
#     # messages字段用于存储消息的序列，并且通过 Annotated 和 operator.add 提供了额外的信息，解释如何处理这些消息。
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     # sender 用于存储当前消息的发送者。通过这个字段，系统可以知道当前消息是由哪个代理生成的。
#     sender: str
#
# # 辅助函数，用于为给定的代理创建节点
# def agent_node(state, agent, name):
#     result = agent.invoke(state) # 调用代理
#     # 检查 result 是否是 ToolMessage 类型的实例
#     if isinstance(result, ToolMessage):
#         pass
#     else:
#         # 将 tavily result 转换为 AIMessage 类型，
#         # 并将 name 作为发送者的名称附加到消息中，便以跟踪发送者，从而了解下一个传递给谁。
#         result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
#     return { "messages": [result],"sender": name,}
#
#
# # 创建两个辅助agent
# # 1 研究员agent
# research_agent = create_agent(llm,[tavily_tool],system_message="你应该提供准确的数据供chart_generator使用。",)
# # 创建一个检索节点，部分应用函数（partial function）
# # 其中第一个参数表示该节点要执行的桉树；agent参数表示用哪个agent；name 参数为对应agent的名字。
# research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
# # 2 图表生成的agent
# chart_agent = create_agent(llm,[python_repl],system_message="你展示的任何图表都将对用户可见。",)
# # 创建图表生成节点
# chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")
#
#
# def router(state) -> Literal["call_tool", "__end__", "continue"]:
#     '''
#     路由器函数
#     '''
#     messages = state["messages"]
#     last_message = messages[-1]
#     # 检查 last_message 是否包含工具调用（tool calls）
#     if last_message.tool_calls:
#         return "call_tool"
#     #如果已经获取到最终答案，则返回结束节点
#     if "FINAL ANSWER" in last_message.content:
#         # 任何代理决定工作完成
#         return "__end__"
#     return "continue"
#
#
# tools = [tavily_tool, python_repl] # 定义工具列表
# tool_node = ToolNode(tools) # 创建工具节点
#
# # 创建状态图实例 - 建立node和edge
# workflow = StateGraph(AgentState)
# workflow.add_node("Researcher", research_node) # 添加研究员节点
# workflow.add_node("chart_generator", chart_node) # 添加图表生成器节点
# workflow.add_node("call_tool", tool_node) # 添加工具调用节点
# # 添加起始边
# workflow.add_edge(START, "Researcher")
# # 添加条件边 - 即Researcher节点会连接path_map参数中指定的所有节点
# workflow.add_conditional_edges("Researcher",router,
#     {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},)
# workflow.add_conditional_edges("chart_generator",router,
#     {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},)
# # 添加条件边
# workflow.add_conditional_edges("call_tool",
#     # 这个 lambda 函数的作用是从状态中获取sender名称，以便在条件边的映射中使用。
#     # 即如果工具的调用方sender是"Researcher"，工作流将转移到 "Researcher" 节点。
#     # 如果 sender 是 "chart_generator"，工作流将转移到 "chart_generator" 节点。
#     lambda x: x["sender"],
#     {"Researcher": "Researcher", "chart_generator": "chart_generator",},)
#
# # 编译工作流图
# graph = workflow.compile()
#
# # 将生成的图片保存到文件
# graph_png = graph.get_graph().draw_mermaid_png()
# with open("collaboration.png", "wb") as f:
#     f.write(graph_png)
#
# # 事件流
# events = graph.stream({"messages": [HumanMessage(
#     content="获取过去5年AI软件市场规模，然后绘制一条折线图。一旦你编写好代码，完成任务。")],},
#     # 图中最多执行的步骤数
#     {"recursion_limit": 150},)
# # 打印事件流中的每个状态
# for s in events:
#     print(s)
#     print("----")


