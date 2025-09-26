import argparse
import json
from agent_pool import Router, Retriever, Reranker, Extractor, Generator
from multi_agent_rag import MultiAgentRAGSystem

THRESHOLD = 0.01  # 定义阈值
MAX_ROUNDS = 5  # 定义最大搜索轮数

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Agent System Search")
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Path to the dataset for evaluation."
    )
    return parser.parse_args()
    
# 在指定数据集上运行multi_agent_rag_system，返回得分（EM/f1/Accuracy）
def run_multi_agent_rag_system(multi_agent_rag_system: MultiAgentRAGSystem, dataset_name: str) -> float:
    agents = get_agents(multi_agent_rag_system.agents)
    dataset = f'dataset/{dataset_name}/dev.jsonl'
    total_em = 0.0
    total_data = 0
    with open(dataset, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data['question']
            golden_answers = data['golden_answers']
            output = ""
            for agent in agents:
                output = agent.run(question, output)
                print(output)
            total_em += get_em(output, golden_answers[0])
            total_data += 1
            break  # 仅测试一个样本，后续可删除
    return total_em / total_data if total_data > 0 else 0.0

def get_em(a: str, b: str) -> float:
    return 1.0 if a.strip() == b.strip() else 0.0

def get_agents(agent_names: list) -> list:
    agents = []
    for name in agent_names:
        if name == "Retriever":
            agents.append(Retriever())
        elif name == "Reranker":
            agents.append(Reranker())
        elif name == "Extractor":
            agents.append(Extractor())
        elif name == "Generator":
            agents.append(Generator())
        elif name == "Router":
            agents.append(Router())
    return agents

# 保存当前multi_agent_rag_system的相关信息及结果
def save_results(multi_agent_rag_system):
    pass

# 从已有multi_agent_rag_system中，选择一个进行扩展
# input: None(存储的文件)
# output: 扩展后新的multi_agent_rag_system
def get_multi_agent_rag_system():
    pass

def main():
    args = parse_args()
    dataset_name = args.dataset_name
    cnt = 0 # multi_agent_rag_system的id计数器
    
    # 初始化多智能体RAG系统，仅包含生成器
    multi_agent_rag_system = MultiAgentRAGSystem(
        agents = ["Generator"], 
        score = 0.0,
        father = -1,
        id = cnt
    )
    cnt += 1
    best_score = 0.0
    # 搜索次数
    round = 0
    # 针对特定任务，迭代搜索最优多智能体RAG系统
    while True:
        # 在测试集/验证集上评估当前多智能体RAG系统
        # 计算当前系统的得分/性能指标
        # run_multi_agent_rag_system的第二个参数，后续完善时可以替换为数据集名字
        multi_agent_rag_system.score = run_multi_agent_rag_system(multi_agent_rag_system, dataset_name)
        print(multi_agent_rag_system.score)
        break
        
        save_results(multi_agent_rag_system)
        # 迭代搜索MAX_ROUNDS次，或达到终止条件（改进/增幅小于某个阈值）
        if 0 <= multi_agent_rag_system.score - best_score <= THRESHOLD or round >= MAX_ROUNDS:
            break
        best_score = max(best_score, multi_agent_rag_system.score)
        # 基于已有系统，获得新的multi_agent_rag_system进行测试
        multi_agent_rag_system = get_multi_agent_rag_system()
        round += 1

if __name__ == "__main__":
    main()