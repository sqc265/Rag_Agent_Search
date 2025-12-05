# 实现蒙特卡洛树搜索的四个基本步骤：选择、扩展、模拟和反向传播
# 5层蒙特卡洛树搜索实现

import random
from evaluate import evaluate_rag
from hyperparameters import candidate_hyperparameters

class MCTSNode:
    def __init__(self, id: int, 
                 parent_id: int = -1, 
                 children: list[int] = None, 
                 visit_times: int = 0, 
                 score: float = 0.0,
                 is_fully_expanded: bool = False,
                 is_terminal: bool = True,
                 level: int = 0,
                 config: dict = None):
        self.id = id
        self.parent_id = parent_id # 父节点
        self.children = children # 子节点列表
        self.visit_times = visit_times # 节点访问次数
        self.score = score # 节点的累计得分
        self.is_fully_expanded = is_fully_expanded # 是否已完全扩展
        self.is_terminal = is_terminal # 是否为终止(叶子)节点
        self.level = level # 节点在树中的层级,用于确定所处RAG pipeline位置
        self.config = config

    
class MCTS:
    def __init__(self, root: MCTSNode, iterations: int = 20):
        self.root = root
        self.iterations = iterations
        self.all_nodes = [root]
        self.cnt = 1

    def select(self, node: MCTSNode) -> MCTSNode:
        # 选择阶段
        while not node.is_terminal:
            if node.is_fully_expanded and node.level < 5:
                node = self.get_best_child(node)
            else:
                return self.expand(node)
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        # 扩展阶段
        configs = candidate_hyperparameters[min(4, node.level)]
        
        for config in configs:
            child_node = MCTSNode(id=self.cnt, parent_id=node.id, config=config) # id与列表的index一致
            self.cnt += 1
            
            child_node.level = node.level + 1
            if node.children is None:
                node.children = []
            node.children.append(child_node.id)
            self.all_nodes.append(child_node)
            
            # 抵达RAG pipeline最后一步
            if child_node.level == 5:
                child_node.is_fully_expanded = True
                
        # 更新节点信息
        node.is_fully_expanded = True
        node.is_terminal = False
        self.all_nodes[node.id] = node 
        
        return  self.all_nodes[node.children[0]]
    
    def simulate(self, node: MCTSNode) -> float:
        # 模拟阶段，执行RAG pipeline并返回奖励
        rag_pipeline = []
        
        # 找到当前节点之前的配置信息
        current_node = node
        while current_node is not None and current_node.level > 0:
            rag_pipeline.append(current_node.config)
            current_node = self.all_nodes[current_node.parent_id]
        rag_pipeline = rag_pipeline[::-1]
        
        # 随机选择后续阶段的配置信息
        current_level = node.level
        while current_level < 5:
            action = random.choice(candidate_hyperparameters[current_level])
            rag_pipeline.append(action)
            current_level += 1
        
        score = evaluate_rag(rag_pipeline)
        
        if node.level < 5:
            node.is_terminal = False
            self.all_nodes[node.id] = node

        return score
    
    def backpropagate(self, node: MCTSNode, reward: float):
        # 反向传播模拟结果，更新节点的访问次数和价值
        while node is not None:
            print(node.id, end=" ")
            node.visit_times += 1
            node.score += reward
            self.all_nodes[node.id] = node
            node = self.all_nodes[node.parent_id] if node.parent_id != -1 else None
        return
    
    def get_best_child(self, node: MCTSNode, c_param: float = 1.4):
        # 使用UCB公式选择最佳子节点
        best_score = float('-inf')
        best_node = None
        for child_id in node.children:
            child = self.all_nodes[child_id]
            if child.visit_times == 0:
                ucb_score = float('inf')
            else:
                ucb_score = (child.score / child.visit_times) + c_param * ( (2 * (node.visit_times).bit_length()) / child.visit_times )**0.5
            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child
        return best_node
    
    def get_highest_scrore_child(self, node: MCTSNode):
        # 选择得分最高的子节点
        best_score = -1
        best_node = None
        
        for child_id in node.children:
            child = self.all_nodes[child_id]
            if child.score > best_score:
                best_score = child.score
                best_node = child
                
        return best_node
    
    def iterate(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            print("选择节点为：", node.id)
            score = self.simulate(node)
            print("模拟节点为：", node.id)
            print("反向传播经过的节点：", end="")
            self.backpropagate(node, score)
            print()
        # for node in search.all_nodes:
        #     print(f'''========================\nNode ID: {node.id}\nParent ID: {node.parent_id}\nChildren: {node.children}\nLevel: {node.level}\nVisits: {node.visit_times}\nScore: {node.score}\nFully Expanded: {node.is_fully_expanded}\nTerminal: {node.is_terminal}\nConfig: {node.config}''')
        
        final_rag_config = []
        current_node = self.root
        while current_node.level < 5 and not current_node.is_terminal and current_node.children is not None:
            node = self.get_highest_scrore_child(current_node)
            final_rag_config.append(node.config)
            current_node = node
            
        return final_rag_config
    
if __name__ == "__main__":
    # 示例用法
    root = MCTSNode(id=0, is_terminal=False)
    search = MCTS(root, iterations=200)
    rag_config = search.iterate()
    print("Optimized RAG Configuration:\n", rag_config)
    
    node0 = search.all_nodes[0]
    node1 = search.all_nodes[1]
    node2 = search.all_nodes[2]
    node3 = search.all_nodes[3]
    ucb1 = (node1.score / node1.visit_times) + 1.4 * ( (2 * (node0.visit_times).bit_length()) / node1.visit_times )**0.5
    ucb2 = (node2.score / node2.visit_times) + 1.4 * ( (2 * (node0.visit_times).bit_length()) / node2.visit_times )**0.5
    ucb3 = (node3.score / node3.visit_times) + 1.4 * ( (2 * (node0.visit_times).bit_length()) / node3.visit_times )**0.5
    print(ucb1, ucb2, ucb3)
    
#     for node in search.all_nodes:
#         print(f'''========================
# Node ID: {node.id}
# Parent ID: {node.parent_id}
# Level: {node.level}
# Visits: {node.visit_times}
# Score: {node.score}
# Fully Expanded: {node.is_fully_expanded}
# Terminal: {node.is_terminal}
# Config: {node.config}''')