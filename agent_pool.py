import re
from openai import OpenAI

client = OpenAI(
    api_key = "sk-C1VC67B3BGc0UN5JfwBMg0mnI9HiKHvbtibjC81wvpz0NPX7",
    base_url = "https://api.chatanywhere.tech/v1"
)

MAX_RETRIEVAL_TURNS = 5

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
# output: 需要检索，返回下一个查询或关键词 <search>query</search>
#         不需要检索，返回最终答案 <answer>answer</answer>
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
        cnt = 1
        while '[EOR]' not in text and cnt < MAX_RETRIEVAL_TURNS:
            cnt += 1
            prev_info += "\n" + text
            prompt = f"Original Query: {query}\nPreviously Retrieved Information: {prev_info}\nDecide whether further retrieval is needed based on the above."
            text = self.generate(prompt)
        search_pattern = r"<search>(.*?)</search>"
        answer_pattern = r"<answer>(.*?)</answer>"
        search_match = re.search(search_pattern, text)
        answer_match = re.search(answer_pattern, text)
        if search_match:
            return f"<search>{search_match.group(1).strip()}</search>"
        elif answer_match:
            return f"<answer>{answer_match.group(1).strip()}</answer>"
        return text.strip()

# 重排序器，基于当前信息，对检索结果进行重排序
class Reranker(RAGagent):
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert reranker.
            Based on the original query and the retrieved chunk/document, rank the chunk/document by relevance to the query.
            Return the ranked chunk/document in descending order of relevance.
            Do not include any other text or explanation. Only return the ranked chunk/document in list:[d1, d2, ..., dn].''',
            model_name=model_name, client=client)

# 提取器，基于当前信息，从检索结果中提取相关信息
class Extractor:
    def __init__(self, model_name: str = "gpt-5-nano", client = client):
        super().__init__(system_prompt='''You are an expert extractor. 
            Based on the original query and the retrieved chunk/document, extract the relevant information that directly answers the query.
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
        