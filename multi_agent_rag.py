# 定义multi_agent_rag_system的类

class MultiAgentRAGSystem:
    def __init__(
        self, 
        agents: list,  
        score: float,
        father: int,
        id: int,
    ) -> None:
        self.agents = agents  # List of agents in the system
        self.score = score  # Performance score of the system
        self.father = father  # ID of the parent system
        self.id = id  # Unique ID based on agents