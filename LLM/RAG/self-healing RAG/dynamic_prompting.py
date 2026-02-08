# from llama_index.core import VectorStoreIndex, Document
# from llama_index.core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class LearningManager:
    def __init__(self):
        self.good_examples = []
        self.index = None

    def add_positive_example(self, query, answer):
        """
        用户遇见正面例子反馈时调用。将好的问答对存进向量检索器中
        input:
            query: str。用户查询
            answer: str。正例回答
        """
        # doc = Document(text=f"Q: {query}\nA: {answer}")
        doc = Document(page_content=f"Q: {query}\nA: {answer}")
        self.good_examples.append(doc)
        # 重建索引（生产环境建议用支持增量更新的向量库）
        self.index = Chroma.from_documents(self.good_examples, embedding=embeddings)
        # self.index = VectorStoreIndex.from_documents(self.good_examples)
        # print(self.index)

    def get_dynamic_prompt(self, current_query):
        '''
        根据用户当前的查询从积极案例库中进行检索匹配
        current_query：str。用户查询
        return：str。匹配的积极回答对，或空字符串
        '''
        if not self.index:
            return ""

        # 检索相似的历史成功案例
        # 这儿应该设置个相似性阈值，高于阈值则返回，没有合适的则返回空集。
        retriever = self.index.as_retriever(search_kwargs={"k": 2, # 最大检索数，
                  },) # 相似性距离阈值,
        # nodes = retriever.retrieve(current_query)
        nodes = retriever.invoke(current_query)
        # 这儿应该有个判断，是否检索出来了相似的积极问答对，如果有则正常运行，否则返回空字符串。
        print("nodes",nodes)
        # examples_text = "\n\n".join([n.text for n in nodes])
        examples_text = "\n\n".join([n.page_content for n in nodes])
        return f"Here are examples of how to answer correctly:\n{examples_text}"


if __name__ == '__main__':
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="znbang/bge:large-en-v1.5-q8_0")  # Qwen3-4b
    # 在管道中使用
    user_query='123'
    good_example=['123','132']
    manager = LearningManager()
    a=manager.add_positive_example(good_example[0],good_example[1])
    print('123',a)
    few_shot_context = manager.get_dynamic_prompt(user_query)
    print(few_shot_context)
    final_prompt_final_prompt= f"{few_shot_context}\n\nQuestion: {user_query}..."