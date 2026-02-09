from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder

# 1. 配置用于生成假设文档的LLM
model="qwen3-1.7b"
llm = ChatOllama(model=model)  # 使用通义千问 7b 模型

def build_hyde_engine(index,embeddings):
    '''
    index:Document.知识库
    '''
    embeddings = HypotheticalDocumentEmbedder.from_llm(llm, embeddings,
#                                                     # 加载预置的 Web Search Prompt
                                                    prompt_key="web_search")
    return embeddings

if __name__ == '__main__':
    # 使用示例
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="znbang/bge:large-en-v1.5-q8_0")  # Qwen3-4b
    documents = [
        Document(
            page_content="那是 個快樂的人",
            metadata={"source": "1"},
        ),]

    index = Chroma.from_documents(documents,embedding=embeddings)
    engine = build_hyde_engine(index,embeddings)
    result = engine.embed_query("Explain the self-correction mechanism in CRAG")
