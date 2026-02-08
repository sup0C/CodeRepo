# 为了在 LangSmith 中有效实施这种方法，首先需要构建评估数据集。
# 在 LangSmith 框架中，数据集是由示例集合构成，每个示例包含输入和相应的预期输出（参考或标签）。
# 这些数据集构成了模型测试和评估的基础架构。

# 1 以下示例创建了包含两个问题的数据集，并为每个问题提供了期望的精确输出
# 创建数据集作为问答示例的容器
ds = client.create_dataset(dataset_name=dataset_name,
    description="A dataset for simple exact match questions.")

# 每个示例由输入字典和相应的输出字典组成
# 输入和输出在独立列表中提供，维持相同的顺序
client.create_examples(
    # 输入列表，每个输入都是字典格式
    inputs=[
        {"prompt_template": "State the year of the declaration of independence. Respond with just the year in digits, nothing else"},
        {"prompt_template": "What's the average speed of an unladen swallow?"},
    ],
    # 对应的输出列表
    outputs=[{"output": "1776"},  # 第一个提示的预期输出
        {"output": "5"}  # 第二个提示的预期输出（陷阱问题）
    ],
    # 示例将被添加到的数据集 ID
    dataset_id=ds.id,)

# 2 数据准备完成后，需要定义评估组件。
# 首要组件是待评估的模型或链。
# 定义待测试模型
model = "gpt-3.5-turbo"

# 被测试系统：接收输入字典，调用指定的 ChatOpenAI 模型，返回字典格式的输出
def predict_result(input_: dict) -> dict:
    '''
    该函数接收提示，将其发送至 OpenAI 的 gpt-3.5-turbo 模型，并返回模型响应
    '''
    # 输入字典包含 "prompt_template" 键，与数据集输入中定义的键一致
    prompt = input_["prompt_template"]
    # 初始化并调用模型
    response = ChatOpenAI(model=model, temperature=0).invoke(prompt)
    # 输出键 "output" 与数据集输出中的键匹配，用于比较
    return {"output": response.content}


# 3 评估器
from langsmith.evaluation import EvaluationResult, run_evaluator

# @run_evaluator 装饰器将函数注册为自定义评估器
@run_evaluator
def compare_label(run, example) -> EvaluationResult:
    """
    用于检查精确匹配的自定义评估器
    Args:
        run: LangSmith 运行对象，包含模型输出
        example: LangSmith 示例对象，包含参考数据
    Returns:
        包含键和分数的 EvaluationResult 对象
    """
    # 从运行输出字典获取模型预测
    # 键 'output' 必须与 `predict_result` 函数返回内容匹配
    prediction = run.outputs.get("output") or ""

    # 从示例输出字典获取参考答案
    # 键 'output' 必须与数据集中定义内容匹配
    target = example.outputs.get("output") or ""

    # 执行比较操作
    match = prediction == target

    # 返回结果，键值为结果中分数的命名方式
    # 精确匹配分数通常为二进制（匹配为 1，不匹配为 0）
    return EvaluationResult(key="matches_label", score=int(match))

# 4 评估
from langchain.smith import RunEvalConfig

# 定义评估运行的配置
eval_config = RunEvalConfig(
    # 通过字符串名称指定内置评估器
    evaluators=["exact_match"],
    # 在列表中直接传递自定义评估器函数
    custom_evaluators=[compare_label],)

# 5 触发评估执行
# 对数据集中每个示例运行 `predict_result` 函数
# 然后使用 `eval_config` 中的评估器对结果评分
# 评估执行将在样本数据上启动基于精确匹配方法的评估并显示进度，结果展示了多种统计信息
# 结果有评估数据中实体数量的 count、正确预测实体比例的 mean（0.5 表示一半实体被正确识别），及其他统计信息。
client.run_on_dataset(dataset_name=dataset_name,
    llm_or_chain_factory=predict_result,
    evaluation=eval_config,
    verbose=True,  # 打印进度条和链接
    project_metadata={"version": "1.0.1", "model": model},  # 项目可选元数据
)
