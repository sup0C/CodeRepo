# 1 创建数据集
# 数据集每个示例应都有问题和该被 LLM 应用为真实检索来源的特定文档名称
# 即包含可直接测试响应生成组件的自包含示例。
examples = [
    # 具体示例数据将在此处定义
]

dataset_name = "RAG Faithfulness Eval"
dataset = client.create_dataset(dataset_name=dataset_name)

# 在 LangSmith 中创建示例，传递复杂输入/输出对象
client.create_examples(
    inputs=[e["inputs"] for e in examples],
    dataset_id=dataset.id,)

# 2 单独评估response_synthesizer组件
# 对于此评估，"系统"不是完整 RAG 链，而仅是 response_synthesizer 部分。
# 此可运行组件接受问题和文档并将它们传递给 LLM：
# 通过单独测试此组件，可以确信任何失败都是由于提示或模型，而非检索器。

# response_synthesizer 的具体实现将在此处定义


# 3 忠实性评估
# 虽然"正确性"重要，但"忠实性"是可靠 RAG 系统的基石。
# 答案在现实世界中可能在事实上正确，但对提供上下文不忠实，这表明 RAG 系统未按预期工作。
# 创建自定义评估器，使用 LLM 检查生成答案是否忠实于提供文档。
# 该评估器专门测量 LLM 是否"坚持提供上下文的脚本"。

# 可以使用标准 qa 评估器进行正确性评估和自定义 FaithfulnessEvaluator 运行评估。

# 在 LangSmith 仪表板中，每次测试运行现在将有两个分数：正确性和忠实性。
# 这允许诊断细微失败。例如，在"LeBron"问题中，模型可能回答"LeBron 是著名篮球运动员"。
# 此答案在正确性上得高分但在忠实性上得低分，立即告知模型忽略了提供的上下文。