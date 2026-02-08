# 1 创建客户支持场景的仿真评估

from typing import List, Dict
import json

class SimulatedCustomer:
    """模拟具有特定问题和个性的客户"""
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.issue = scenario["issue"]
        self.personality = scenario["personality"]
        self.conversation_history = []
        self.issue_resolved = False

    def generate_response(self, agent_message: str) -> str:
        """基于 Agent 消息生成客户响应"""

        # 使用 LLM 生成现实的客户响应
        prompt = f"""您是一个{self.personality}的客户，有以下问题：{self.issue}

        对话历史：
        {self._format_history()}

        支持代理刚刚说："{agent_message}"

        作为客户，您会如何回应？保持您的个性和问题一致。
        如果您的问题得到解决，请明确表示满意。
        """

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = llm.invoke(prompt).content

        # 检查问题是否已解决
        if "谢谢" in response.lower() and "解决" in response.lower():
            self.issue_resolved = True

        return response

    def _format_history(self) -> str:
        """格式化对话历史"""
        return "\n".join([
            f"{'客户' if msg['role'] == 'customer' else '代理'}：{msg['content']}"
            for msg in self.conversation_history
        ])


# 2 定义客户支持 Agent
class CustomerSupportAgent:
    """客户支持 AI Agent"""

    def __init__(self):
        self.conversation_history = []

    def respond(self, customer_message: str, context: dict = None) -> str:
        """生成对客户消息的响应"""

        prompt = f"""
         您是一个有用且专业的客户支持代理。

         对话历史：
         {self._format_history()}

         客户刚刚说："{customer_message}"

         提供有用、准确和专业的响应。尝试解决客户的问题。
         """

        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        response = llm.invoke(prompt).content

        return response

    def _format_history(self) -> str:
        """格式化对话历史"""
        return "\n".join([
            f"{'客户' if msg['role'] == 'customer' else '代理'}：{msg['content']}"
            for msg in self.conversation_history
        ])

# 3 创建仿真评估框架
def run_conversation_simulation(scenario: dict, max_turns: int = 10) -> dict:
    """
    运行客户支持对话的仿真
    Args:
        scenario: 包含客户问题和个性的字典
        max_turns: 最大对话轮数
    Returns:
        包含对话历史和指标的字典
    """
    customer = SimulatedCustomer(scenario)
    agent = CustomerSupportAgent()
    conversation = []

    # 客户开始对话
    initial_message = f"你好，我有一个问题：{scenario['issue']}"
    conversation.append({"role": "customer", "content": initial_message})

    for turn in range(max_turns):
        # Agent 响应
        agent_response = agent.respond(
            conversation[-1]["content"] if conversation[-1]["role"] == "customer" else ""
        )
        conversation.append({"role": "agent", "content": agent_response})

        # 检查问题是否已解决
        if customer.issue_resolved:
            break

        # 客户响应
        customer_response = customer.generate_response(agent_response)
        conversation.append({"role": "customer", "content": customer_response})

        # 更新历史
        customer.conversation_history = conversation
        agent.conversation_history = conversation

    return {"conversation": conversation,
        "issue_resolved": customer.issue_resolved,
        "turns_to_resolution": len(conversation) // 2,
        "scenario": scenario}

# 4 定义仿真场景
simulation_scenarios = [
     {
         "issue": "我的订单迟到了，我需要退款",
         "personality": "不耐烦和愤怒"
     },
     {
         "issue": "我忘记了我的密码，无法登录",
         "personality": "困惑但礼貌"
     },
     {
         "issue": "您的产品坏了，我想换货",
         "personality": "挫败但合理"
     },
     {
         "issue": "我想取消我的订阅",
         "personality": "坚决但友好"
     }
 ]


# 5 运行仿真并收集指标
def evaluate_conversation_simulations():
    """运行多个仿真并分析结果"""

    results = []

    for scenario in simulation_scenarios:
        # 运行仿真
        result = run_conversation_simulation(scenario)
        results.append(result)

        # 将结果记录到 LangSmith
        client.create_run(
            name="conversation_simulation",
            run_type="chain",
            inputs={"scenario": scenario},
            outputs={
                "conversation": result["conversation"],
                "metrics": {
                    "issue_resolved": result["issue_resolved"],
                    "turns_to_resolution": result["turns_to_resolution"]
                }
            }
        )

    # 分析聚合指标
    total_simulations = len(results)
    resolved_count = sum(1 for r in results if r["issue_resolved"])
    avg_turns = sum(r["turns_to_resolution"] for r in results) / total_simulations

    print(f"仿真运行：{total_simulations}")
    print(f"问题解决率：{resolved_count / total_simulations:.2%}")
    print(f"平均解决轮数：{avg_turns:.1f}")

    return results

# 6 创建对话质量评估器
@run_evaluator
def conversation_quality_evaluator(run, example) -> dict:
    """评估整个对话的质量"""

    conversation = run.outputs.get("conversation", [])

    # 使用 LLM 评估对话质量
    prompt = f"""
     评估这个客户支持对话的质量：

     {json.dumps(conversation, indent=2, ensure_ascii=False)}

     请评分（1-10）：
     1. 专业性：代理是否保持专业语调？
     2. 有效性：代理是否有效解决了问题？
     3. 效率：对话是否简洁而重点明确？

     返回 JSON 格式：{{"professionalism": X, "effectiveness": X, "efficiency": X}}
     """

    judge = ChatOpenAI(model="gpt-4", temperature=0)
    evaluation = judge.invoke(prompt).content

    try:
        scores = json.loads(evaluation)
        overall_score = sum(scores.values()) / len(scores)
        return {"key": "conversation_quality", "score": overall_score}
    except:
        return {"key": "conversation_quality", "score": 5.0}  # 默认分数


