# 1 创建数据集
# 以著名的泰坦尼克号数据集构建问答系统为例。
# 不存储如"891 名乘客"这样的答案，而是存储计算答案的 pandas 代码片段：
# 问题列表和相应的 pandas 代码以找到答案
questions_with_references = [
    ("How many passengers were on the Titanic?", "len(df)"),
    ("How many passengers survived?", "df['Survived'].sum()"),
    ("What was the average age of the passengers?", "df['Age'].mean()"),
    ("How many male and female passengers were there?", "df['Sex'].value_counts()"),
    ("What was the average fare paid for the tickets?", "df['Fare'].mean()"),]

# 创建唯一数据集名称
dataset_name = "Dynamic Titanic QA"

# 在 LangSmith 中创建数据集
dataset = client.create_dataset(dataset_name=dataset_name,
    description="QA over the Titanic dataset with dynamic references.",)

# 填充数据集，输入为问题，输出为代码
client.create_examples(
    inputs=[{"question": q} for q, r in questions_with_references],
    outputs=[{"reference_code": r} for q, r in questions_with_references],
    dataset_id=dataset.id,)


# 被测试系统将是 pandas_dataframe_agent，设计用于通过在 pandas DataFrame 上生成和执行代码来回答问题。
# 首先加载初始数据：
import pandas as pd

# 从 URL 加载泰坦尼克号数据集
titanic_url = "https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv"
df = pd.read_csv(titanic_url) # 此 DataFrame 代表实时数据源。


# 2 定义创建和运行 Agent 的函数。该 Agent 在被调用时将访问当前的 df：
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 此函数在 `df` 当前状态下创建和调用 Agent
def predict_pandas_agent(inputs: dict):
    # Agent 使用当前 `df` 创建
    # 这种设置确保Agent始终查询数据源的最新版本。
    agent = create_pandas_dataframe_agent(agent_type="openai-tools", llm=llm, df=df)
    return agent.invoke({"input": inputs["question"]})

# 3 自定义评估器
# 能够获取 reference_code 字符串，执行它以获得当前答案，然后使用该结果进行评分。
# 通过继承 LabeledCriteriaEvalChain 并重写其输入处理方法来实现：
from typing import Optional
from langchain.evaluation.criteria.eval_chain import LabeledCriteriaEvalChain

class DynamicReferenceEvaluator(LabeledCriteriaEvalChain):
    '''
    这个自定义类在交给 LLM 评判者进行正确性检查之前获取实时标准答案
    '''
    def _get_eval_input(self,prediction: str,
            reference: Optional[str],input: Optional[str],) -> dict:
        # 从父类获取标准输入字典
        eval_input = super()._get_eval_input(prediction, reference, input)

        # 这里的'reference'是代码片段，例如"len(df)"
        # 执行它以获得实时标准答案值
        # 警告：使用 `eval` 可能有风险，仅运行可信代码
        live_ground_truth = eval(eval_input["reference"])

        # 用实际的实时答案替换代码片段
        eval_input["reference"] = str(live_ground_truth)

        return eval_input


# 4 现在配置并首次运行评估
# 创建自定义评估器链的实例
base_evaluator = DynamicReferenceEvaluator.from_llm(
    criteria="correctness", llm=ChatOpenAI(model="gpt-4", temperature=0))

# 将其包装在 LangChainStringEvaluator 中以正确映射运行/示例字段
dynamic_evaluator = LangChainStringEvaluator(
    base_evaluator,
    # 此函数将数据集字段映射到评估器期望的内容
    prepare_data=lambda run, example: {
    "prediction": run.outputs["output"],
    "reference": example.outputs["reference_code"],
    "input": example.inputs["question"],},)

# 在时间"T1"运行评估
client.run_on_dataset(dataset_name=dataset_name,
    llm_or_chain_factory=predict_pandas_agent,
    evaluation=RunEvalConfig(custom_evaluators=[dynamic_evaluator],),
    project_metadata={"time": "T1"},
    max_concurrency=1,  # Pandas Agent 不是线程安全的
)

# 5 模拟数据库更新 - 第一次测试运行现已完成，Agent 性能根据数据初始状态进行测量
# 通过复制行来修改 DataFrame，有效地改变所有问题的答案
df_doubled = pd.concat([df, df], ignore_index=True) # 通过数据加倍来模拟数据更新
df = df_doubled # df 对象现已改变。
# 由于 Agent 和评估器都引用这个全局 df，它们将在下次运行时自动使用新数据。
# 在时间"T2"对更新数据重新运行评估 - 重新运行完全相同的评估，无需更改数据集或评估器
client.run_on_dataset(dataset_name=dataset_name,
    llm_or_chain_factory=predict_pandas_agent,
    evaluation=RunEvalConfig(custom_evaluators=[dynamic_evaluator],),
    project_metadata={"time": "T2"},
    max_concurrency=1,)

# 6 现在可以在"数据集"页面查看测试结果。
# 前往"示例"选项卡探索每次测试运行的预测。点击任何数据集行以更新示例或查看所有运行的预测。
# 此案例中选择了问题："有多少男性和女性乘客？"的示例。
# 页面底部的链接行显示通过 run_on_dataset 自动链接的每次测试运行的预测。
# 有趣的是，运行之间的预测存在差异：
    # 第一次运行为 577 名男性、314 名女性；
    # 第二次运行为 1154 名男性、628 名女性。
    # 然而两次都被标记为"正确"，因为尽管底层数据发生变化，但每次检索过程都保持一致和准确。

# 7 抽查
# 为确保"正确"评级实际可靠，现在应当抽查自定义评估器的运行跟踪。
# 具体方法为：
    # 如果在表中看到"正确性"标签上的箭头，点击这些箭头直接查看评估跟踪；
    # 如果没有，点击进入运行，转到反馈选项卡，从那里找到该特定示例上自定义评估器的跟踪。

# 在截图中，"reference"键保存来自数据源的解引用值，与预测匹配：
    # 第一次运行为 577 名男性、314 名女性；
    # 第二次运行为 1154 名男性、628 名女性。
    # 这证实了评估器正确地将预测与来自变化数据源的当前标准答案进行比较。

# df更新后，评估器正确检索了新的参考值 1154 名男性和 628 名女性，与第二次测试运行的预测匹配。
# 这证实了问答系统即使在其知识库演变时也能可靠工作。