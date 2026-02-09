# 1 安装 RAGAS 库并设置必要组件
# 安装 RAGAS（如果尚未安装）
# pip install ragas
from ragas.langchain.evaluation import RagasEvaluatorChain
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
    context_precision,
)

# 2 设置 RAG 链和评估数据集
# 创建 RAGAS 评估数据集
dataset_name = "RAG Evaluation with RAGAS"

# 创建数据集示例，每个都包含问题、上下文和参考答案
ragas_examples = [
    {
        "inputs": {
            "question": "What is LangSmith?",
            "contexts": [
                "LangSmith is a platform for building production-grade LLM applications.",
                "It provides debugging, testing, and monitoring capabilities."
            ]
        },
        "outputs": {
            "answer": "LangSmith is a platform for building production-grade LLM applications with debugging and monitoring features.",
            "ground_truth": "LangSmith is a comprehensive platform for developing and deploying LLM applications."
        }
    }
]

# 在 LangSmith 中创建数据集
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[e["inputs"] for e in ragas_examples],
    outputs=[e["outputs"] for e in ragas_examples],
    dataset_id=dataset.id,
)

# 3 设置 RAGAS 评估器链
# 这些评估器将自动计算提到的四个核心指标
# 创建 RAGAS 评估器
evaluator_chains = []

for metric in [answer_relevancy, faithfulness, context_relevancy, context_precision]:
    evaluator_chain = RagasEvaluatorChain(metric=metric)
    evaluator_chains.append(evaluator_chain)

# 4 运行评估。
# RAGAS 将自动分析 RAG 系统在所有四个维度上的表现：
# 运行 RAGAS 评估 - 评估完成后，将在 LangSmith 仪表板中看到详细的指标分解
eval_config = RunEvalConfig(
    custom_evaluators=evaluator_chains)

client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=rag_chain,
    evaluation=eval_config,
    verbose=True,
    project_metadata={"evaluation_type": "ragas", "version": "1.0.0"},
)
