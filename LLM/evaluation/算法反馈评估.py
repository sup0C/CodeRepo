import re
import json
import time
from typing import Dict, Any
from datetime import datetime
from evaluate import client

# 1 实施各种算法反馈评估器
@run_evaluator
def response_time_evaluator(run, example) -> dict:
    """测量响应生成时间"""

    # 计算运行持续时间
    if hasattr(run, 'start_time') and hasattr(run, 'end_time'):
        duration = (run.end_time - run.start_time).total_seconds()
    else:
        duration = 0  # 备用

    # 分数基于响应时间（更快 = 更好）
    # 5秒以下 = 1.0，10秒以上 = 0.0
    score = max(0, min(1, (10 - duration) / 5))

    return {"key": "response_time", "score": score}


@run_evaluator
def json_validity_evaluator(run, example) -> dict:
    """检查输出是否为有效的 JSON"""

    output = run.outputs.get("output", "")

    try:
        json.loads(output)
        score = 1.0  # 有效 JSON
    except json.JSONDecodeError:
        score = 0.0  # 无效 JSON

    return {"key": "json_validity", "score": score}


@run_evaluator
def length_compliance_evaluator(run, example) -> dict:
    """检查响应长度是否在期望范围内"""

    output = run.outputs.get("output", "")
    expected_min = example.outputs.get("min_length", 50)
    expected_max = example.outputs.get("max_length", 500)

    actual_length = len(output)

    if expected_min <= actual_length <= expected_max:
        score = 1.0
    else:
        # 基于与目标范围距离的部分分数
        if actual_length < expected_min:
            score = actual_length / expected_min
        else:  # actual_length > expected_max
            score = max(0, 1 - (actual_length - expected_max) / expected_max)

    return {"key": "length_compliance", "score": score}


@run_evaluator
def keyword_presence_evaluator(run, example) -> dict:
    """检查响应中是否存在所需关键词"""

    output = run.outputs.get("output", "").lower()
    required_keywords = example.outputs.get("required_keywords", [])

    if not required_keywords:
        return {"key": "keyword_presence", "score": 1.0}

    present_keywords = sum(1 for keyword in required_keywords if keyword.lower() in output)
    score = present_keywords / len(required_keywords)

    return {"key": "keyword_presence", "score": score}


@run_evaluator
def format_compliance_evaluator(run, example) -> dict:
    """检查输出是否遵循指定格式"""

    output = run.outputs.get("output", "")
    expected_format = example.outputs.get("expected_format", "text")

    score = 0.0

    if expected_format == "email":
        # 检查电子邮件格式
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', output):
            score = 1.0
    elif expected_format == "phone":
        # 检查电话号码格式
        if re.search(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b', output):
            score = 1.0
    elif expected_format == "url":
        # 检查 URL 格式
        if re.search(r'https?://[^\s]+', output):
            score = 1.0
    elif expected_format == "markdown":
        # 检查基本 markdown 元素
        markdown_elements = ['#', '*', '**', '`', '[', ']', '(', ')']
        if any(element in output for element in markdown_elements):
            score = 1.0
    else:
        score = 1.0  # 默认为文本格式

    return {"key": "format_compliance", "score": score}


# 2 创建具有算法反馈的综合评估套件
def create_algorithmic_evaluation_dataset():
    """创建带有算法反馈标准的数据集"""
    dataset_name = "Algorithmic Feedback Evaluation"
    examples = [
        {"inputs": {"task": "Generate a JSON response with user data",
                "prompt": "创建一个包含姓名、电子邮件和电话号码的用户配置文件 JSON"},
            "outputs": {
                "expected_format": "json",
                "required_keywords": ["name", "email", "phone"],
                "min_length": 50,
                "max_length": 200}
        },
        {
            "inputs": {"task": "Write a professional email response",
                "prompt": "写一封专业的电子邮件回复客户询问"},
            "outputs": {
                "expected_format": "email",
                "required_keywords": ["dear", "thank", "regards"],
                "min_length": 100,
                "max_length": 300}
        }
    ]

    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_id=dataset.id,)

    return dataset_name


# 3 运行算法评估
def run_algorithmic_evaluation():
    """运行带有多个算法反馈评估器的评估"""

    dataset_name = create_algorithmic_evaluation_dataset()

    # 定义被测试系统
    def test_system(inputs: dict) -> dict:
        prompt = inputs["prompt"]
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        response = llm.invoke(prompt)
        return {"output": response.content}

    # 配置所有算法评估器
    algorithmic_evaluators = [
        response_time_evaluator,
        json_validity_evaluator,
        length_compliance_evaluator,
        keyword_presence_evaluator,
        format_compliance_evaluator
    ]

    eval_config = RunEvalConfig(custom_evaluators=algorithmic_evaluators)

    # 运行评估
    results = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=test_system,
        evaluation=eval_config,
        verbose=True,
        project_metadata={"evaluation_type": "algorithmic_feedback"},
    )

    return results


# 4 创建算法反馈仪表板
def generate_algorithmic_feedback_report():
    """生成算法反馈指标的详细报告"""

    # 获取最近的评估运行
    runs = client.list_runs(project_name="your-project-name", limit=100)

    metrics = {
        "response_time": [],
        "json_validity": [],
        "length_compliance": [],
        "keyword_presence": [],
        "format_compliance": []
    }

    for run in runs:
        feedback = client.list_feedback(run_ids=[run.id])
        for f in feedback:
            if f.key in metrics:
                metrics[f.key].append(f.score)

    # 计算汇总统计
    report = {}
    for metric, scores in metrics.items():
        if scores:
            report[metric] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
                "pass_rate": sum(1 for s in scores if s >= 0.8) / len(scores)
            }

    print("算法反馈报告")
    print("=" * 40)
    for metric, stats in report.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  平均分数: {stats['average']:.3f}")
        print(f"  通过率 (≥0.8): {stats['pass_rate']:.1%}")
        print(f"  范围: {stats['min']:.3f} - {stats['max']:.3f}")

    return report



# 5 实施性能基准测试
def benchmark_system_performance():
    """对系统性能进行基准测试"""

    test_queries = [
                       "生成一个简单的 JSON 对象",
                       "写一个长度为 100 字的段落",
                       "创建一个格式化的电子邮件"
                   ] * 10  # 重复以获得更好的统计

    response_times = []

    for query in test_queries:
        start_time = time.time()

        # 运行系统
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        response = llm.invoke(query)

        end_time = time.time()
        response_times.append(end_time - start_time)

    # 计算性能指标
    avg_time = sum(response_times) / len(response_times)
    p95_time = sorted(response_times)[int(0.95 * len(response_times))]