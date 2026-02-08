# 1 本节使用源自 ToolBench 基准的数据集，包含查询和一套物流相关 API 的预期工具
# 工具选择数据集的公共 URL
dev_dataset_url = "https://smith.langchain.com/public/bdf7611c-3420-4c71-a492-42715a32d61e/d"
dataset_name = "Tool Selection (Logistics) Dev"

# 将数据集克隆到 LangSmith 账户
client.clone_public_dataset(dev_dataset_url, dataset_name=dataset_name)

# 2 定义 tool_selection_precision 评估器。
from langsmith.evaluation import run_evaluator

@run_evaluator
def selected_tools_precision(run: Run, example: Example) -> dict:
    '''
    此函数比较预测工具集与预期工具集并计算精度分数
    此评估器为 Agent 工具选择准确性提供清晰指标
    '''
    # 数据集中的'expected'字段包含正确工具名称
    expected_tools = set(example.outputs["expected"][0])

    # Agent 输出是预测工具调用列表
    predicted_calls = run.outputs.get("output", [])
    predicted_tools = {tool["type"] for tool in predicted_calls}

    # 计算精度：（正确预测的工具）/（所有预测的工具）
    if not predicted_tools:
        score = 1 if not expected_tools else 0
    else:
        true_positives = predicted_tools.intersection(expected_tools)
        score = len(true_positives) / len(predicted_tools)

    return {"key": "tool_selection_precision", "score": score}

# 3 工具绑定LLM
# Agent 将是简单的函数调用链，从 JSON 文件加载大量真实世界工具定义并将它们绑定到 LLM
import json
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 从本地文件加载工具规范
with open("./data/tools.json") as f:
    tools = json.load(f)

# 定义提示并将工具绑定到 LLM - Agent 现配置为基于提供的工具列表根据其描述进行选择
assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond to the user's query using the provided tools."),
    ("user", "{query}"),])
llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)
chain = assistant_prompt | llm | JsonOutputToolsParser()


# 4 运行评估 - 查看 Agent 在原始工具描述下的表现
# 使用自定义精度评估器配置评估
eval_config = RunEvalConfig(custom_evaluators=[selected_tools_precision])

# 运行评估
test_results = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    verbose=True,
    project_metadata={"model": "gpt-3.5-turbo", "tool_variant": "original"},
)


# 5 构建"提示改进器"链
# 运行完成后，结果显示平均精度分数约为 0.63，表明 Agent 经常困惑。
# 通过检查 LangSmith 中的失败案例，可以看到它选择看似合理但不正确的工具，因为描述过于通用或重叠。
# 相比手动重写描述，可以构建"提示改进器"链。
# 该链采用映射-归约-提炼的方法：
    # 对于每个失败，LLM 查看查询、错误工具选择和正确工具选择，
    # 然后为涉及的工具建议更好的描述；按工具名称对所有建议的描述更改分组；
    # 对于每个工具，另一个 LLM 接受所有建议更改并将它们提炼成单一的新改进描述。
# 改进提示以纠正 Agent 工具调用
improver_prompt = ChatPromptTemplate.from_messages([
    # 此处为改进提示的具体实现
])

# 6 工具改进后的评估
# 现在运行完全相同的评估，但这次将带有改进描述的 new_tools 绑定到 LLM：
# 使用更新工具描述创建新链
llm_v2 = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(new_tools)
updated_chain = assistant_prompt | llm_v2 | JsonOutputToolsParser()

# 重新运行评估
# 通过比较两次运行的 tool_selection_precision 分数，可定量测量自动描述改进的有效性。
updated_test_results = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=updated_chain,
    evaluation=eval_config,
    verbose=True,
    project_metadata={"model": "gpt-3.5-turbo", "tool_variant": "improved"},
)