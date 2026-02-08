# 本节将评估一个从法律合同中提取关键详细信息的链。

# 1 首先克隆公共数据集到 LangSmith 账户中进行评估
# LangSmith 上公共数据集的 URL
dataset_url = "https://smith.langchain.com/public/08ab7912-006e-4c00-a973-0f833e74907b/d"
dataset_name = "Contract Extraction Eval Dataset"

# 将公共数据集克隆到账户中
client.clone_public_dataset(dataset_url, dataset_name=dataset_name)

# 2 为了指导 LLM 生成正确的结构化输出，需要使用 Pydantic 模型定义目标数据结构。
# 该模式充当待提取信息的结构
from typing import List, Optional
from pydantic import BaseModel

# 定义当事方地址的模式
class Address(BaseModel):
    street: str
    city: str
    state: str

# 定义合同中当事方的模式
class Party(BaseModel):
    name: str
    address: Address

# 整个合同的顶级模式
class Contract(BaseModel):
    document_title: str
    effective_date: str
    parties: List[Party]

# 3 接下来构建提取链。
# 使用专门为此任务设计的 create_extraction_chain，它接受 Pydantic 模式和能力强的 LLM（如 Anthropic 的 Claude 或具有函数调用功能的 OpenAI 模型）来执行提取
from langchain.chains import create_extraction_chain
from langchain_anthropic import ChatAnthropic

# 使用能够遵循复杂指令的强大模型
# 注：可以替换为等效的 OpenAI 模型
llm = ChatAnthropic(model="claude-2.1", temperature=0, max_tokens=4000)

# 创建提取链，提供模式和 LLM
extraction_chain = create_extraction_chain(Contract.schema(), llm)

# 4 现在配置为接受文本并返回包含提取 JSON 的字典。
from langsmith.evaluation import LangChainStringEvaluator

# 评估配置指定 JSON 感知评估器
eval_config = RunEvalConfig(evaluators=[
    # 使用 json_edit_distance 字符串评估器。将此评估器包装在 RunEvalConfig 中
    # 该评估器通过计算预测和参考 JSON 对象间的相似性，忽略键顺序等表面差异，
    # 比较两个JSON对象的结构和内容
        LangChainStringEvaluator("json_edit_distance")])

# 运行评估
# 使用 client.run_on_dataset 执行测试运行
# LangSmith 将在数据集中的每个合同上运行提取链，评估器将为每个结果评分。
# 输出中的链接直接导向项目仪表板以监控结果。
client.run_on_dataset(dataset_name=dataset_name,
    llm_or_chain_factory=extraction_chain,
    evaluation=eval_config,
    # 数据集中的输入键为'context'，映射到链的'input'键
    input_mapper=lambda x: {"input": x["context"]},
    # 链的输出为字典 {'text': [...]}，关注'text'值
    output_mapper=lambda x: x['text'],
    verbose=True,
    project_metadata={"version": "1.0.0", "model": "claude-2.1"},)