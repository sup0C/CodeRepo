# import os
# import json
# import asyncio
from langchain_ollama import ChatOllama
from typing import List, Dict, Any, Optional
from datetime import datetime
# 导入各组件
# from hyde import build_hyde_engine, Settings
from query_decomposition import plan_query
from corrective_rag import create_grag
# from reranker import Reranker
from dynamic_prompting import LearningManager
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# 核心依赖
# from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
# from llama_index.llms.openai import OpenAI
from langchain_core.prompts import PromptTemplate
# from sentence_transformers import CrossEncoder

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="znbang/bge:large-en-v1.5-q8_0")  # Qwen3-4b


class SelfHealingRAGSystem:
    """
    完整自愈RAG系统，整合全部组件
    """
    def __init__(self, model:str='qwen3-1.7b',openai_api_key: str = None):
        """初始化RAG系统"""
            # 组件初始化
        print("🚀 Initializing Self-Healing RAG System...")
        # 核心LLM - 用户整合所有资料进行最后的回答
        self.llm = ChatOllama(model=model, temperature=0.001)

        # 初始化各组件
        # self.reranker = Reranker()
        self.learning_manager = LearningManager()
        self.vector_index = None #
        self.hyde_engine = None #
        self.web_seaerch=0 # 网络查询的次数
        # 演示数据
        self.sample_documents = self._create_sample_documents()
        self._setup_vector_index()
        # self.sample_texts=''

        # 统计
        self.query_stats = {
            "total_queries": 0,
            "hyde_used": 0,
            "decomposed_queries": 0,
            "crag_activated": 0,
            "reranked": 0,
            "learning_applied": 0
        }
        print("✅ System initialized successfully!")

    def _create_sample_documents(self) -> List[Document]:
        """创建演示用的示例文档"""
        sample_texts = [
            """Retrieval-Augmented Generation (RAG) is a technique that combines   
            pre-trained language models with external knowledge retrieval. RAG systems   
            retrieve relevant documents from a knowledge base and use them to generate   
            more accurate and factual responses.""",

            """Corrective RAG (CRAG) introduces a self-correction mechanism that grades   
            retrieved documents for relevance. If documents are deemed irrelevant, the   
            system triggers alternative retrieval strategies like web search.""",

            """HyDE (Hypothetical Document Embeddings) improves retrieval by generating   
            hypothetical documents that answer the query, then searching for real documents   
            similar to these hypothetical ones.""",

            """Cross-encoder reranking provides more accurate document scoring compared   
            to bi-encoder similarity search. It processes query-document pairs together   
            to produce refined relevance scores.""",

            """DSPy enables automatic prompt optimization by treating prompts as programs   
            that can be compiled and optimized against specific metrics like accuracy   
            or semantic similarity.""",

            """Self-healing RAG systems implement feedback loops that learn from successful   
            query-answer pairs, storing them as examples for future similar queries to   
            improve performance over time.""",

            """Query decomposition breaks complex multi-part questions into atomic   
            sub-queries that can be individually processed and then combined for   
            comprehensive answers.""",

            """Vector databases enable semantic search by converting documents into   
            high-dimensional embeddings that capture semantic meaning rather than   
            just keyword matches.""" ]

        return [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(sample_texts)]

    def _setup_vector_index(self):
        """用示例文档构建向量索引"""
        print("📚 Setting up vector index...")
        self.vector_index = Chroma.from_documents(self.sample_documents, embedding=embeddings)
        # self.vector_index = VectorStoreIndex.from_documents(self.sample_documents)
        # self.hyde_engine = build_hyde_engine(self.vector_index)
        print("✅ Vector index ready!")

    def enhanced_retrieve(self, query: str, use_hyde: bool = False, top_k: int = 2) -> List[Document]:
        """支持HyDE的增强检索
        query：用户的查询
        use_hyde：是否使用hyde方法
        top_k:留最相似的前N个答案
        """
        print(f"🔍 Retrieving documents for: '{query}'")
        if use_hyde:
            print(" 🧠 Using HyDE for enhanced retrieval...")
            response = self.hyde_engine.query(query)
            # 从HyDE响应提取文档
            documents = response.source_nodes
            self.query_stats["hyde_used"] += 1
        else:
            print("  📖 Using standard retrieval...")
            retriever = self.vector_index.as_retriever(search_kwargs={"k": top_k, # 最大检索数，
                  },)
            nodes = retriever.invoke(query)
            # retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
            # nodes = retriever.retrieve(query)
            documents = nodes
        # 转换为Document对象
        docs = []
        for node in documents:
            doc = Document(
                page_content=node.page_content if hasattr(node, 'text') else str(node),
                metadata=node.metadata if hasattr(node, 'metadata') else {})
            docs.append(doc)
        print(f"  ✅ Retrieved {len(docs)} documents")
        return docs

    def decompose_and_retrieve(self, query: str,top_k:int=3) -> tuple[List[str], List[Document]]:
        """分解复杂查询并分别检索
        在搜索回答时使用了hyde增强检索。
        input: query：用户的原始问题
        top_k:留最相似的前N个答案
        return：[query]：List[str]。复杂原始问题分解出的N个子问题，或者无需分解的原始问题
                docs：List[Document]。所有子问题的回答集合，或原始问题的回答.
                                一个问题可能会有多个回答。
        """
        print(f"🔧 Decomposing query: '{query}'")
        try:
            sub_queries = plan_query(query) # 将复杂问题分解出的N个子问题
            if len(sub_queries) > 1:
                print(f" 📝 Decomposed into {len(sub_queries)} sub-queries:")
                # 打印查看每个分解的子问题
                # for i, sq in enumerate(sub_queries):
                #     print(f"{i}. {sq}")

                # 对每个子查询检索
                all_docs = []
                for sq in sub_queries:
                    docs = self.enhanced_retrieve(sq, use_hyde=False, top_k=top_k)
                    all_docs.extend(docs)
                self.query_stats["decomposed_queries"] += 1
                return sub_queries, all_docs
            else:
                print("  ➡️ Query doesn't need decomposition")
                docs = self.enhanced_retrieve(query)
                return [query], docs
        except Exception as e:
            print(f"  ⚠️ Error in decomposition: {e}")
            docs = self.enhanced_retrieve(query)
            return [query], docs

    def apply_crag(self, query: str, documents: List[Document]) -> List[Document]: # tuple[List[Document], str]:
        """应用CRAG过滤文档
        input: query：用户的原始问题或分解出的子问题
            documents:针对每个子问题检索出的相关性文档。一个问题可能会有多个回答。
        """
        print("🔍 Applying CRAG (Corrective RAG)...")
        try:
            # 准备CRAG状态
            initial_state = {'question': query,
                             'generation': "",
                             'web_search': "No",
                             'documents': documents}
            app = create_grag()
            result = app.invoke(initial_state)

            if result['web_search']=='Yes':
                self.web_seaerch+=1 # 进行web_seaerch的次数
                # 对web数据分割
                # 计算相似度阈值
                # 使用LLM基于过滤后的文档进行回答
                pass
            else:
                # 使用LLM基于原始文档进行回答
                result['documents'] # N个相关性文档
                pass

            # # 正常情况下会跑完整CRAG流程
            # filtered_docs = [] # 相关性文档存储器
            # for doc in documents[:3]:  # 演示限制
            #     # 简单相关性检查（实际应该用LLM）
            #     if any(keyword in doc.page_content.lower() for keyword in query.lower().split()):
            #         filtered_docs.append(doc)

            # if len(filtered_docs) < len(documents):
            #     self.query_stats["crag_activated"] += 1
            #     print(f"  🚨 CRAG filtered {len(documents) - len(filtered_docs)} irrelevant documents")

            return "Documents filtered by CRAG"

        except Exception as e:
            print(f"  ⚠️ Error in CRAG: {e}")
            return documents, "CRAG not applied due to error"

    def apply_reranking(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """交叉编码器重排序"""
        print("🎯 Applying cross-encoder reranking...")
        try:
            # 提取文本用于重排序
            doc_texts = [doc.page_content for doc in documents]

            if len(doc_texts) > 1:
                reranked_texts = self.reranker.rerank(query, doc_texts, top_k)

                # 映射回Document对象
                reranked_docs = []
                for text in reranked_texts:
                    for doc in documents:
                        if doc.page_content == text:
                            reranked_docs.append(doc)
                            break

                self.query_stats["reranked"] += 1
                print(f"  ✅ Reranked to top {len(reranked_docs)} documents")
                return reranked_docs
            else:
                print("  ➡️ Not enough documents for reranking")
                return documents

        except Exception as e:
            print(f"  ⚠️ Error in reranking: {e}")
            return documents

    def apply_dynamic_prompting(self, query: str) -> str:
        """
        动态少样本学习。
        query：用户的查询
        return：筛选出的N个优秀问答对或空字符串。
        """
        print("🧠 Applying dynamic prompting...")
        try:
            # 添加积极问答对例子
            # few_shot_context = self.learning_manager.add_positive_example(query)
            few_shot_context = self.learning_manager.get_dynamic_prompt(query)
            if few_shot_context:
                self.query_stats["learning_applied"] += 1
                print("  ✅ Applied learned examples from previous successes")
            else:
                print("  ➡️ No relevant past examples found")
            return few_shot_context
        except Exception as e:
            print(f"  ⚠️ Error in dynamic prompting: {e}")
            return ""

    def generate_answer(self, query: str, documents: List[Document], few_shot_context: str = "") -> str:
        """基于检索文档生成答案"""
        print("✍️ Generating final answer...")
        # 合并文档内容
        context = "\n\n".join([doc.page_content for doc in documents])
        # 构建prompt，可选包含少样本示例
        # 提示词模板这里还需要再改进
        # prompt_parts = []
        # if few_shot_context:
        #     prompt_parts.append(few_shot_context)
        #
        # prompt_parts.extend(["Context:",context,
        #     f"\nQuestion: {query}",
        #     "\nAnswer based on the provided context:"])
        # prompt = "\n".join(prompt_parts)
        #
        # try:
        #     response = self.llm.complete(prompt)
        #     answer = response.text.strip()
            print("  ✅ Answer generated successfully")
            return answer
        except Exception as e:
            print(f"  ⚠️ Error generating answer: {e}")
            return f"I apologize, but I encountered an error generating an answer: {e}"

    def _get_components_used(self) -> List[str]:
        """获取本次查询用到的组件"""
        components = ["Vector Retrieval"]

        if self.query_stats["hyde_used"] > 0:
            components.append("HyDE")
        if self.query_stats["decomposed_queries"] > 0:
            components.append("Query Decomposition")
        if self.query_stats["crag_activated"] > 0:
            components.append("CRAG")
        if self.query_stats["reranked"] > 0:
            components.append("Cross-Encoder Reranking")
        if self.query_stats["learning_applied"] > 0:
            components.append("Dynamic Prompting")
        return components

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {"total_queries": self.query_stats["total_queries"],
            "hyde_usage_rate": f"{(self.query_stats['hyde_used'] / max(1, self.query_stats['total_queries']) * 100):.1f}%",
            "decomposition_rate": f"{(self.query_stats['decomposed_queries'] / max(1, self.query_stats['total_queries']) * 100):.1f}%",
            "crag_activation_rate": f"{(self.query_stats['crag_activated'] / max(1, self.query_stats['total_queries']) * 100):.1f}%",
            "reranking_rate": f"{(self.query_stats['reranked'] / max(1, self.query_stats['total_queries']) * 100):.1f}%",
            "learning_rate": f"{(self.query_stats['learning_applied'] / max(1, self.query_stats['total_queries']) * 100):.1f}%",
            "learned_examples": len(self.learning_manager.good_examples)}

    def full_pipeline(self, query: str, user_feedback: bool = None, previous_answer: str = None) -> Dict[str, Any]:
        """
        运行完整自愈RAG管道
        input
            query:用户的单条查询
            user_feedback：user_feedback -
            previous_answer：
        """
        start_time = datetime.now()
        print(f"\n🔄 Starting Self-Healing RAG Pipeline")
        print("=" * 60)
        self.query_stats["total_queries"] += 1

        # 步骤1：查询增强
        # 先问题分解，再进行知识检索。检索时，若能hyde则hyde，否则正常向量检索
        sub_queries, documents = self.decompose_and_retrieve(query)
        # 步骤2：文档校验（CRAG）
        filtered_docs = self.apply_crag(query, documents)
        # 步骤3：文档重排序
        # reranked_docs = self.apply_reranking(query, filtered_docs)
        # 步骤4：动态提示 - 自进化、自演化。
        # 将每个问答对都存入积极案例库中，逐渐增强系统的鲁棒性和稳定性
        # few_shot_context = self.apply_dynamic_prompting(query)
        # 步骤5：答案生成 -
        answer = self.generate_answer(query, filtered_docs)
        # 步骤6：学习（如有反馈） - 自更新动态提示
        # 可以考虑将这个积极案例库独立出来，然后根据准确率指标判断，如果大于每个阈值则认为为好的问答对。
        # if user_feedback is True:
        #     try:
        #         self.learning_manager.add_good_example(query, answer)
        #         print("📚 Added successful example to learning system")
        #     except Exception as e:
        #         print(f"⚠️ Error adding to learning system: {e}")
        end_time = datetime.now()
        # 单条问题的运行时间
        processing_time = (end_time - start_time).total_seconds()
        result = {
            "query": query,
            "sub_queries": sub_queries,
            "documents_found": len(documents),
            "documents_filtered": len(filtered_docs),
            # "final_documents": len(reranked_docs),
            "answer": answer,
            # "crag_status": crag_status,
            "processing_time": processing_time,
            "components_used": self._get_components_used()}
        print(f"✅ Pipeline completed in {processing_time:.2f} seconds")
        # print(f"📊 Documents: {len(documents)} → {len(filtered_docs)} → {len(reranked_docs)}")
        return result

def demo_interactive_session(demo_queries:List[str]=['']):
    """交互式演示
    demo_queries：用户的提问。
    """
    print("""🎯 Self-Healing RAG System Demo  
    ================================  
    This system demonstrates:
    • HyDE: Hypothetical Document Embeddings  
    • Query Decomposition: Breaking complex queries  
    • CRAG: Corrective RAG with document grading  
    • Cross-Encoder Reranking: Precision ranking  
    • Dynamic Learning: Few-shot from success examples""")
    # 初始化系统
    system = SelfHealingRAGSystem()
    # 演示用查询
    print("🔥 Running Demo Queries...\n","=" * 50)

    results = []
    # 对每个问题依次循环处理
    for i, query in enumerate(demo_queries):
        print(f"📋 Demo Query {i}/{len(demo_queries)}\n",
              f"Query: '{query}'")
        # 模拟正反馈用于学习
        if i > 1:  # 第二个查询开始加反馈
            result=system.full_pipeline(query, user_feedback=True)
        else:
            result = system.full_pipeline(query) # 开始运行自愈RAG系统
        results.append(result) # 将当前问题的结果保存
        print(f"💡 Answer:",f"{result['answer']}")
        print(f"\n📊 Components Used: {', '.join(result['components_used'])}")

    # 最终统计
    print("=" * 60,"\n📈 SYSTEM PERFORMANCE STATISTICS\n","=" * 60)
    stats = system.get_system_stats() # 获取系统执行的统计数据
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    return system, results


if __name__ == "__main__":
    demo_queries=["What is RAG and how does it work?",
        "Compare HyDE and standard retrieval methods",
        "How does CRAG improve retrieval quality and what are the benefits of cross-encoder reranking?",
        "Explain the self-correction mechanisms in modern RAG systems",
        "What are the advantages of DSPy optimization for prompts?"]
    # 设置OpenAI API密钥
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    demo_interactive_session(demo_queries)