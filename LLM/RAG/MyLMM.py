from langchain.llms.base import LLM
# from zhipuai import ZhipuAI
from langchain_core.messages.ai import AIMessage
import os
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
os.environ["DASHSCOPE_API_KEY"] = "sk-271135c629c74e7d944f822b6130c6e4" # Tongyi API key

class MyLLM(LLM):
    client: object = None
    def __init__(self):
        super().__init__()
        # 智谱
        # zhipuai_api_key = os.getenv('ZHUPU_API_KEY')
        # self.client = ZhipuAI(api_key=zhipuai_api_key)
        # tongyi
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def _llm_type(self):
        return "tongyi"

    def invoke(self,prompt="你是谁？",stop=None,config={},history=[{"role": "system", "content": "You are a helpful assistant."}], *args, **kwargs):
        '''
        prompt：也就是输入
        '''
        if history is None:
            history=[]
        if not isinstance(prompt, str):
            prompt = prompt.to_string()

        # print(config, history)

        history.append({"role":"user","content":prompt })
        response = self.client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model = "qwen-turbo", # "qwen-plus"、"qwen-turbo"、"glm-4"
            messages =history,
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            extra_body={"enable_thinking": False},
        )

        result = response.choices[0].message.content
        return AIMessage(content=result)

    def _call(self, prompt,history=[],stop=None, *args, **kwargs):
        '''
        该方法接受一个字符串、一些可选的停用词，然后返回一个字符串。
        '''
        return self.invoke(prompt,history)

    def stream(self, prompt,config={},history=[], stop=None, *args, **kwargs):
        if history is None:
            history = []
        if not isinstance(prompt, str):
            prompt = prompt.to_string()

        # print(config,history)
        history.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model = "qwen-turbo", # "qwen-plus"、"qwen-turbo"、"glm-4"
            messages =history,
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            stream=True,
            extra_body={"enable_thinking": True},
        )
        for chunk in response:
            yield chunk.choices[0].delta.content



if __name__ == '__main__':
    # 1 不使用chain推理
    llm = MyLLM()
    # results=llm.invoke("请讲1个减肥的笑话")
    # print("2",results)
    # for deltaStr in llm.stream("如何鼓励自己减肥！"):
    #     print(deltaStr,end="")

    # 2 使用chain推理
    prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    results=chain.invoke({"topic": "康师傅绿茶"})
    print(results)

    for chunk in chain.stream({"topic": "康师傅绿茶"}):
        print(chunk,end="")