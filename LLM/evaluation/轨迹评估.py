# 1 首先创建数据集
# 每个示例不仅包括参考答案，还包括 expected_steps——期望按顺序调用的工具名称列表：
# 问题列表，每个都有参考答案和预期工具轨迹
agent_questions = [
    (
        "Why was a $10 calculator app a top-rated Nintendo Switch game?",
        {
            "reference": "It became an internet meme due to its high price point.",
            "expected_steps": ["duck_duck_go"],  # 期望网络搜索
        },
    ),
    (
        "hi",
        {
            "reference": "Hello, how can I assist you?",
            "expected_steps": [],  # 期望无工具调用的直接响应
        },
    ),
    (
        "What's my first meeting on Friday?",
        {
            "reference": 'Your first meeting is 8:30 AM for "Team Standup"',
            "expected_steps": ["check_calendar"],  # 期望日历工具
        },
    ),
]


# 在 LangSmith 中创建数据集 - 数据集现在包含正确最终答案和到达路径的蓝图
dataset_name = "Agent Trajectory Eval"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Dataset for evaluating agent tool use and trajectory.",)

# 使用输入和多部分输出填充数据集
client.create_examples(
    inputs=[{"question": q[0]} for q in agent_questions],
    outputs=[q[1] for q in agent_questions],
    dataset_id=dataset.id,)

# 2 定义 Agent
# Agent将访问两个工具：duck_duck_go 网络搜索工具和模拟的 check_calendar 工具。
# 必须配置 Agent 返回其 intermediate_steps，以便评估器可以访问其轨迹：
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool


# 演示用的模拟工具
@tool
def check_calendar(date: str) -> list:
    """检查用户在指定日期的日历会议"""
    if "friday" in date.lower():
        return 'Your first meeting is 8:30 AM for "Team Standup"'
    return "You have no meetings."


# 此工厂函数创建 Agent 执行器
def create_agent_executor(inputs: dict):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [DuckDuckGoSearchResults(name="duck_duck_go"), check_calendar]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent_runnable = create_openai_tools_agent(llm, tools, prompt)

    # 关键步骤：`return_intermediate_steps=True` 使轨迹在输出中可用
    executor = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        return_intermediate_steps=True,
    )
    return executor.invoke(inputs)

# 3 自定义评估器来比较 Agent 的工具使用轨迹与标准答案。
@run_evaluator
def trajectory_evaluator(run: Run, example: Optional[Example] = None) -> dict:
    '''
    # 自定义评估器函数 - 可提供关于 Agent 是否按预期行为的清晰信号。
    此函数将从 Agent 运行对象解析 intermediate_steps，
    并将工具名称列表与数据集示例中的 expected_steps 比较
    '''
    # 1. 从运行输出获取 Agent 的实际工具调用
    # 'intermediate_steps' 是 (action, observation) 元组列表
    intermediate_steps = run.outputs.get("intermediate_steps", [])
    actual_trajectory = [action.tool for action, observation in intermediate_steps]

    # 2. 从数据集示例获取预期工具调用
    expected_trajectory = example.outputs.get("expected_steps", [])

    # 3. 比较并分配二进制分数
    score = int(actual_trajectory == expected_trajectory)

    # 4. 返回结果
    return {"key": "trajectory_correctness", "score": score}

# 4 运行评估
# 使用自定义 trajectory_evaluator 和内置 qa 评估器运行评估。
# qa 评估器对最终答案正确性评分，自定义评估器对过程评分，提供 Agent 性能的完整图片
# 'qa' 评估器需要知道用于输入、预测和参考的字段
qa_evaluator = LangChainStringEvaluator(
    "qa",
    prepare_data=lambda run, example: {
        "input": example.inputs["question"],
        "prediction": run.outputs["output"],
        "reference": example.outputs["reference"],
    },
)

# 使用两个评估器运行评估
# 运行完成后，可以转到 LangSmith 项目并按 trajectory_correctness 分数筛选。
# 可以找到 Agent 产生正确答案但采用错误路径的案例（或反之），为调试和改进 Agent 逻辑提供深入洞察。
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=create_agent_executor,
    evaluation=RunEvalConfig(
        # 包括自定义轨迹评估器和内置 QA 评估器
        evaluators=[qa_evaluator],
        custom_evaluators=[trajectory_evaluator],),
    max_concurrency=1,)