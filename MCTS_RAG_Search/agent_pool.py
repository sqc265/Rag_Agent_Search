import re
from openai import OpenAI

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.retrieve_server.wiki18 import query_serve

client = OpenAI(
    api_key = "sk-C1VC67B3BGc0UN5JfwBMg0mnI9HiKHvbtibjC81wvpz0NPX7",
    base_url = "https://api.chatanywhere.tech/v1"
)

MAX_RETRIEVAL_TURNS = 5
TOP_K = 3  # 每次检索返回的top k个结果

class RAGagent:
    def __init__(self, system_prompt: str, model_name: str = "gpt-5-nano", client = client):
        self.model_name = model_name
        self.client = client
        self.system_prompt = system_prompt
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    
# 路由器，基于原始查询，决定查询策略
class Router(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert router.
            Your task is to analyze the original query and determine the most suitable retrieval strategy 
            based on its complexity.

            Retrieval strategies:
            - NO_RETRIEVAL: the query is simple and can be answered directly without retrieval
            - SINGLE_RETRIEVAL: the query can be solved with one retrieval pass
            - MULTI_TURN_RETRIEVAL: the query is complex and requires multiple retrieval passes

            Return only one of the following keywords exactly:
            NO_RETRIEVAL | SINGLE_RETRIEVAL | MULTI_TURN_RETRIEVAL
            Do not include any explanation or extra text.''', 
            model_name=model_name, client=client)
        
    def run(self, query: str, prev_info: None) -> str:
        prompt = f"Original Query: {query}\nDecide the retrieval strategy based on the above."
        return self.generate(prompt).strip()

# 检索器，基于当前信息，判断是否需要进行检索
# input: 原始查询 + 已检索信息
# output: 检索信息
class Retriever(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert retriever. 
            Based on the original query and the information already retrieved, decide whether further retrieval is needed.
            - If further retrieval is needed, return the next query or keywords in the format: <search>query</search>
            - If no further retrieval is needed, return: [EOR]
            Do not include any other text or explanation. Only return in the specified format.''', 
            model_name=model_name, client=client)
        
    def run(self, query: str, prev_info: str) -> str:
        prompt = f"Original Query: {query}\nPreviously Retrieved Information: {prev_info}\nDecide whether further retrieval is needed based on the above."
        text = self.generate(prompt)
        cnt = 0
        retrieve_info = []
        while '<search>' in text and cnt < MAX_RETRIEVAL_TURNS:
            # extract the query inside <search>...</search>
            search_pattern = r"<search>(.*?)</search>"
            search_match = re.search(search_pattern, text)
            query = search_match.group(1).strip()
            
            # retrieve the information for the new query
            print(f"第{cnt}次检索...")
            retrieve_info = query_serve.retrieve([query], TOP_K)
            cnt += 1
            for doc in retrieve_info:
                prev_info += "\n" + doc['text']
            prompt = f"Original Query: {query}\nPreviously Retrieved Information: {prev_info}\nDecide whether further retrieval is needed based on the above."
            text = self.generate(prompt)
            # print("检索结果：", retrieve_info)
        return retrieve_info

# 重排序器，基于当前信息，对检索结果进行重排序
class Reranker(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert reranker.
            Based on the original query and the retrieved chunk/document, rank the chunk/document by relevance to the query.
            Return the ranked chunk/document in descending order of relevance.
            Do not include any other text or explanation. Only return the ranked chunk/document in list:[d1, d2, ..., dn].''',
            model_name=model_name, client=client)
    
    def run(self, query: str, prev_info: str) -> str:
        prompt = f"Original Query: {query}\nRetrieved Document: {prev_info}\nRank the document based on relevance to the query."
        text = self.generate(prompt)
        return text.strip()

# 提取器，基于当前信息，从检索结果中提取相关信息
class Extractor(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert extractor. 
            Based on the original query and the retrieved chunk/document, extract the relevant information that can be used to answer the user's query.
            Return only the extracted information without any additional text or explanation in the format: <extract>information</extract>.''',
            model_name=model_name, client=client)
        
    def run(self, query: str, prev_info: str) -> str:
        prompt = f"Original Query: {query}\nRetrieved Document: {prev_info}\nExtract the relevant information based on the above."
        text = self.generate(prompt)
        pattern = r"<extract>(.*?)</extract>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text.strip()

# 生成器，基于当前信息，生成最终答案
class Generator(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert generator. 
            Based on the original query and the retrieved information, generate a comprehensive and accurate answer.
            Ensure the answer is clear, concise, and directly addresses the user's query.
            Do not include any other text or explanation. Only return the answer in the form of <answer>answer</answer>.
            Example:
            input: who plays malfoy's dad in harry potter?
            output: <answer>Jason Isaacs</answer>''', 
            model_name=model_name, client=client)
        
    def run(self, query: str, prev_info: str) -> str:
        prompt = f"Original Query: {query}\nRetrieved Information: {prev_info}\nGenerate the final answer based on the above."
        text = self.generate(prompt)
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text.strip()

# 调用LLM，扩展multi_agent_rag_system
def expand_system(multi_agent_rag_system: dict, dataset_name: str) -> str:
    system_prompt = f'''You are a professional Multi-Agent RAG System Optimizer Expert, skilled in Agentic RAG architectures, agent collaboration, and reinforcement-based optimization.'''
    prompt = f'''
    [Task Background]
    The current problem being solved is: {dataset_name}

    [Current Multi-Agent RAG System]
    The existing system consists of the following agents:
    {multi_agent_rag_system["agents"]}

    [System Origin]
    This system is adapted from another agent system, with the following modifications:
    {multi_agent_rag_system["modification"]}

    [Performance Feedback]
    After modification, the system performance compared to its parent system was: {multi_agent_rag_system["improvement"]}
    (A positive value indicates success, meaning the score improved compared to the parent system; 
    a negative value indicates failure, meaning the score decreased compared to the parent system.)


    [Available Agent Pool]
    Candidate agents available for modification:
    - Router: Determines the retrieval strategy based on the original query.
    - Retriever: Decides whether further retrieval is needed based on the original query and previously retrieved information.
    - Reranker: Ranks retrieved documents based on their relevance to the original query.
    - Extractor: Extracts relevant information from retrieved documents to answer the original query.
    - Generator: Generates the final answer based on the original query and retrieved information.

    [Your Task]
    Based on the above information, please modify or extend the current multi-agent RAG system to achieve better performance on the task: {dataset_name}.

    [Output Requirements]
    Return format: ['agent1', 'agent2', ..., 'agentn'].
    <modification>Describe the changes made to the system, including any new agents added or existing agents refined.</modification>
    
    Notes:
    - Prioritize system interpretability and cooperative efficiency.
    - You may introduce new agents or refine existing ones.
    - Keep the system size reasonable (3 - 8 agents preferred).
    - Only return the modified multi-agent RAG system and the modification in the specified format without any additional text or explanation.'''
    
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    
    router = Router()
    retriever = Retriever()
    reranker = Reranker()
    extractor = Extractor()
    generator = Generator()

    query = "what is the year of Swarg Narak film director year?"
    prev_info = ""
    
    router_result = router.run(query, prev_info)
    print("路由结果:\n", router_result)  # SINGLE_RETRIEVAL
    
    retriever_result = retriever.run(query, prev_info)
    print("检索结果:\n", retriever_result)  # <search>Swarg Narak director</search>
    
    reranker_result = reranker.run(query, str(retriever_result))
    print("重排序结果:\n", reranker_result)  # [d1, d2, ..., dn]
    
    extractor_result = extractor.run(query, str(reranker_result))
    print("提取结果:\n", extractor_result)  # <extract>...</extract>
    
    generator_result = generator.run(query, extractor_result)
    print("生成结果:\n", generator_result)  # <answer>...</answer>