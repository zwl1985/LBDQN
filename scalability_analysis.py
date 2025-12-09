import networkx as nx
import random
import pickle
import os
import time
from typing import Tuple, List, Dict, Any
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback

def sample_subgraph_with_attributes(G: nx.DiGraph, sample_ratio: float) -> nx.DiGraph:
    """
    采样子图并保留所有节点和边的属性
    """
    n_nodes_target = int(len(G) * sample_ratio)
    if n_nodes_target <= 0:
        n_nodes_target = 1
    
    all_nodes = list(G.nodes())
    
    # 尝试多次随机采样以获得连通子图
    for attempt in range(20):
        sampled_nodes = random.sample(all_nodes, min(n_nodes_target, len(all_nodes)))
        # 使用subgraph().copy()会保留所有属性
        subgraph = G.subgraph(sampled_nodes).copy()
        
        if nx.is_weakly_connected(subgraph) or len(sampled_nodes) <= 1:
            return subgraph
    
    # 如果无法获得连通子图，使用最大弱连通分量
    print(f"警告: 无法采样到连通子图，使用最大弱连通分量")
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    wcc_nodes = list(largest_wcc)
    
    if len(wcc_nodes) >= n_nodes_target:
        sampled_nodes = random.sample(wcc_nodes, n_nodes_target)
    else:
        sampled_nodes = wcc_nodes
    
    return G.subgraph(sampled_nodes).copy()


def generate_scalability_samples(G: nx.Graph, 
                                 dataset_name: str,
                                 deepwalk_embeddings: np.ndarray = None,
                                 node_id_mapping: Dict = None,
                                 ratios: List[float] = None,
                                 output_dir: str = "scalability_samples") -> Dict[float, Dict[str, Any]]:
    """
    生成不同规模的子图用于可扩展性测试，保留所有节点和边属性
    同时提取对应节点的DeepWalk向量
    
    Args:
        G: 原始图
        dataset_name: 数据集名称
        deepwalk_embeddings: DeepWalk节点向量矩阵 (n_nodes, embedding_dim)
        node_id_mapping: 节点ID到矩阵索引的映射 {node_id: index}
        ratios: 采样比例列表
        output_dir: 输出目录
    """
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    os.makedirs(output_dir, exist_ok=True)
    samples = {}
    
    print(f"开始为数据集 {dataset_name} 生成可扩展性样本...")
    print(f"原始图: {len(G)} 个节点, {G.number_of_edges()} 条边")
    
    if deepwalk_embeddings is not None:
        print(f"DeepWalk向量维度: {deepwalk_embeddings.shape}")
    
    # 检查原始图的边属性
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0]
        print(f"边属性示例: {sample_edge[2]}")
    
    print("-" * 50)
    
    original_nodes = list(G.nodes())
    
    for ratio in ratios:
        print(f"\n采样比例: {ratio*100:.0f}%")
        
        if ratio == 1.0:
            # 完整图 - 创建深拷贝以保留所有属性
            subgraph = G.copy()
        else:
            # 采样子图并保留属性
            if isinstance(G, nx.DiGraph):
                subgraph = sample_subgraph_with_attributes(G, ratio)
            else:
                # 对于无向图
                n_nodes = int(len(G) * ratio)
                if n_nodes <= 0:
                    n_nodes = 1
                
                # 使用BFS确保连通性
                start_node = random.choice(original_nodes)
                visited = set([start_node])
                queue = [start_node]
                
                while queue and len(visited) < n_nodes:
                    node = queue.pop(0)
                    neighbors = list(G.neighbors(node))
                    random.shuffle(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in visited and len(visited) < n_nodes:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                sampled_nodes = list(visited)
                # 使用subgraph().copy()保留所有属性
                subgraph = G.subgraph(sampled_nodes).copy()
        
        sampled_ratio = len(subgraph) / len(G)
        n_nodes = len(subgraph)
        n_edges = subgraph.number_of_edges()
        
        print(f"  实际比例: {sampled_ratio*100:.2f}%")
        print(f"  子图: {n_nodes} 个节点, {n_edges} 条边")
        
        # 验证边属性是否保留
        if n_edges > 0:
            sample_edge = list(subgraph.edges(data=True))[0]
            print(f"  子图边属性示例: {sample_edge[2]}")
        
        # 重新编号：将节点ID从0开始连续编号
        old_nodes = list(subgraph.nodes())
        
        # 创建旧节点ID到新节点ID的映射
        old_to_new = {old_node: new_id for new_id, old_node in enumerate(old_nodes)}
        
        # 创建重新编号的子图
        renumbered_subgraph = nx.DiGraph() if isinstance(subgraph, nx.DiGraph) else nx.Graph()
        
        # 添加节点（带属性）
        for old_node, new_id in old_to_new.items():
            if old_node in subgraph.nodes:
                # 复制节点属性
                node_attrs = subgraph.nodes[old_node]
                renumbered_subgraph.add_node(new_id, **node_attrs)
            else:
                renumbered_subgraph.add_node(new_id)
        
        # 添加边（带属性）
        for u, v, edge_data in subgraph.edges(data=True):
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            renumbered_subgraph.add_edge(new_u, new_v, **edge_data)
        
        print(f"  节点重新编号: {min(old_nodes)} ~ {max(old_nodes)} -> 0 ~ {n_nodes-1}")
        
        # 验证重新编号后的图
        if renumbered_subgraph.number_of_edges() > 0:
            sample_edge = list(renumbered_subgraph.edges(data=True))[0]
            print(f"  重新编号后边属性示例: {sample_edge[2]}")
        
        # 提取子图节点对应的DeepWalk向量
        subgraph_embeddings = None
        
        if deepwalk_embeddings is not None and node_id_mapping is not None:
            # 按新编号顺序提取向量
            subgraph_embeddings = np.zeros((n_nodes, deepwalk_embeddings.shape[1]))
            for new_id, old_node in enumerate(old_nodes):
                if old_node in node_id_mapping:
                    old_idx = node_id_mapping[old_node]
                    subgraph_embeddings[new_id] = deepwalk_embeddings[old_idx]
                else:
                    print(f"  警告: 节点 {old_node} 在DeepWalk映射中未找到")
            
            print(f"  提取DeepWalk向量: {subgraph_embeddings.shape}")
        
        # 保存重新编号的子图
        filename = f"{dataset_name}_{int(ratio*100):03d}pct.pkl"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(renumbered_subgraph, f)
        
        # 保存原始节点ID映射（用于追溯）
        old_to_new_filename = f"{dataset_name}_{int(ratio*100):03d}pct_old_to_new.pkl"
        old_to_new_filepath = os.path.join(output_dir, old_to_new_filename)
        with open(old_to_new_filepath, 'wb') as f:
            pickle.dump(old_to_new, f)
        print(f"  保存节点映射: {old_to_new_filepath}")
        
        # 保存子图的DeepWalk向量
        if subgraph_embeddings is not None:
            embedding_filename = f"{dataset_name}_{int(ratio*100):03d}pct_init_X.npy"
            embedding_filepath = os.path.join(output_dir, embedding_filename)
            np.save(embedding_filepath, subgraph_embeddings)
            
            print(f"  保存DeepWalk向量: {embedding_filepath}")
        
        samples[ratio] = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'filepath': filepath,
            'embedding_filepath': embedding_filepath if subgraph_embeddings is not None else None,
            'old_to_new_filepath': old_to_new_filepath,
            'target_ratio': ratio,
            'actual_ratio': sampled_ratio
        }
    
    return samples


def analyze_scalability_trend(samples: Dict[float, Dict[str, Any]], 
                             model_inference_func, 
                             agent,
                             num_features,
                             k: int = 10, 
                             n_trials: int = 3) -> Dict[float, Dict[str, Any]]:
    """
    分析模型在不同规模图上的推理时间
    """
    results = {}
    
    print("\n" + "="*60)
    print("开始可扩展性推理时间测试")
    print("="*60)
    
    for ratio in sorted(samples.keys()):
        info = samples[ratio]
        
        # 从pickle文件加载图（包含所有属性）
        with open(info['filepath'], 'rb') as f:
            graph = pickle.load(f)
        
        agent.graph_init_embedding_dict[id(graph)] = torch.randn(graph.number_of_nodes(), num_features)

        print(f"\n测试比例: {ratio*100:.0f}%")
        print(f"图大小: {info['n_nodes']} 节点, {info['n_edges']} 边")
        
        # 验证加载的图有正确的属性
        if graph.number_of_edges() > 0:
            sample_edge = list(graph.edges(data=True))[0]
            print(f"边属性: {sample_edge[2]}")
        
        total_time = 0
        all_times = []
        
        for trial in range(n_trials):
            print(f"  第 {trial+1}/{n_trials} 次推理...", end='')
            
            start_time = time.perf_counter()
            try:
                seeds = model_inference_func(graph, k)
                elapsed = time.perf_counter() - start_time
                
                total_time += elapsed
                all_times.append(elapsed)
                print(f" 耗时: {elapsed:.3f}秒")
            except Exception as e:
                print(f" 失败: {e}")
                traceback.print_exc()
                # 如果失败，使用一个较大的时间值作为惩罚
                elapsed = 999.0
                all_times.append(elapsed)
        
        if all_times:
            avg_time = sum(all_times) / len(all_times)
            if len(all_times) > 1:
                std_time = np.std(all_times)
            else:
                std_time = 0
        else:
            avg_time = 0
            std_time = 0
        
        results[ratio] = {
            'n_nodes': info['n_nodes'],
            'n_edges': info['n_edges'],
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'trials': all_times,
            'actual_ratio': info['actual_ratio']
        }
        
        print(f"  平均推理时间: {avg_time:.3f} ± {std_time:.3f} 秒")
        if info['n_nodes'] > 0:
            print(f"  单位节点时间: {avg_time/info['n_nodes']:.6f} 秒/节点")
    
    return results


def create_scalability_plot(results: Dict[float, Dict[str, Any]], 
                           dataset_name: str, 
                           output_dir: str = "scalability_plots") -> None:
    """
    创建可扩展性分析图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ratios = sorted(results.keys())
    nodes = [results[r]['n_nodes'] for r in ratios]
    times = [results[r]['avg_inference_time'] for r in ratios]
    stds = [results[r]['std_inference_time'] for r in ratios]
    
    plt.figure(figsize=(12, 8))
    
    # 子图1：推理时间 vs 节点数
    plt.subplot(2, 2, 1)
    plt.errorbar(nodes, times, yerr=stds, fmt='o-', linewidth=2, 
                 markersize=8, capsize=5, capthick=2, 
                 color='blue', label='Measured Data')
    
    if len(nodes) >= 2 and all(t > 0 for t in times):
        log_nodes = np.log(nodes)
        log_times = np.log(times)
        z = np.polyfit(log_nodes, log_times, 1)
        p_log = np.poly1d(z)
        
        x_fit = np.linspace(min(nodes), max(nodes)*2, 100)
        y_fit = np.exp(p_log(np.log(x_fit)))
        
        plt.plot(x_fit, y_fit, 'r--', alpha=0.7, 
                label=f'Power-law fit: t ∝ n^{z[0]:.2f}')
        
        million_nodes = 1e6
        extrapolated_time = np.exp(p_log(np.log(million_nodes)))
        
        plt.axvline(x=million_nodes, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=extrapolated_time, color='gray', linestyle=':', alpha=0.5)
        plt.plot(million_nodes, extrapolated_time, 's', markersize=10, 
                color='red', label=f'Extrapolated to 1M nodes: {extrapolated_time/60:.1f}min')
    
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Inference Time (s)', fontsize=12)
    plt.title(f'{dataset_name} - Inference Time vs. Scale', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=9)
    
    # 子图2：单位节点时间
    plt.subplot(2, 2, 2)
    time_per_node = [t/n if n > 0 else 0 for t, n in zip(times, nodes)]
    plt.plot(nodes, time_per_node, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time per Node (s/node)', fontsize=12)
    plt.title('Unit Computational Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # 子图3：实际采样比例
    plt.subplot(2, 2, 3)
    target_ratios = [r*100 for r in ratios]
    actual_ratios = [results[r]['actual_ratio']*100 for r in ratios]
    plt.plot(target_ratios, actual_ratios, 'o-', linewidth=2, markersize=8, color='orange')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Ideal line')
    plt.xlabel('Target Sampling Ratio (%)', fontsize=12)
    plt.ylabel('Actual Sampling Ratio (%)', fontsize=12)
    plt.title('Sampling Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    # 子图4：数据表格
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = []
    for ratio in ratios:
        info = results[ratio]
        table_data.append([
            f"{ratio*100:.0f}%",
            f"{info['n_nodes']}",
            f"{info['n_edges']}",
            f"{info['avg_inference_time']:.3f}",
            f"{info['avg_inference_time']/info['n_nodes']:.6f}" if info['n_nodes'] > 0 else "N/A"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Ratio', 'Nodes', 'Edges', 'Time(s)', 'Time/Node'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.15, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title('Detailed Summary', fontsize=12, y=0.9)
    
    plt.suptitle(f'{dataset_name} - Scalability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'scalability_{dataset_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'scalability_{dataset_name}.pdf'), bbox_inches='tight')
    
    print(f"\nPlots saved to: {plot_path}")


def main():
    """主函数"""
    print("="*60)
    print("LBDQN-IM Scalability Analysis")
    print("="*60)
    
    # 1. 加载图数据
    print("\nStep 1: Loading graph data...")
    graph_name = 'Gowalla'
    location_counts = 1
    
    try:
        with open(f'{graph_name}/g_{location_counts}.pkl', 'rb') as f:
            G = pickle.load(f)
        print(f"Successfully loaded graph: {len(G)} nodes, {G.number_of_edges()} edges")
        print(f"Graph type: {type(G)}")
        
        # 检查边属性
        if G.number_of_edges() > 0:
            sample_edge = list(G.edges(data=True))[0]
            print(f"Edge attributes: {sample_edge[2]}")
            
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return
    
    # 1.5 加载DeepWalk向量
    print("\nStep 1.5: Loading DeepWalk embeddings...")
    deepwalk_embeddings = None
    node_id_mapping = None
    
    try:
        # 加载DeepWalk向量矩阵
        embedding_path = f'{graph_name}/init_X.npy'
        deepwalk_embeddings = np.load(embedding_path)
        print(f"Loaded DeepWalk embeddings: {deepwalk_embeddings.shape}")
        
        # 创建节点ID到索引的映射
        # 假设节点ID是按顺序排列的，如果不是，需要根据实际情况调整
        all_nodes = list(G.nodes())
        node_id_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        print(f"Created node mapping for {len(node_id_mapping)} nodes")
        
        # 验证向量数量与节点数量是否匹配
        if deepwalk_embeddings.shape[0] != len(all_nodes):
            print(f"Warning: Embedding count ({deepwalk_embeddings.shape[0]}) != Node count ({len(all_nodes)})")
            print("Please verify the node-to-index mapping!")
        
    except Exception as e:
        print(f"Failed to load DeepWalk embeddings: {e}")
        print("Will continue without embeddings...")
    
    # 2. 生成可扩展性样本
    print("\nStep 2: Generating scalability samples...")
    
    sample_ratios = [0.1, 0.3, 0.5, 0.8, 1.0]
    
    samples = generate_scalability_samples(
        G=G,
        dataset_name=graph_name,
        deepwalk_embeddings=deepwalk_embeddings,
        node_id_mapping=node_id_mapping,
        ratios=sample_ratios,
        output_dir="scalability_samples"
    )
    
    # 3. 准备模型推理
    print("\nStep 3: Preparing model inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 导入必要的模块
    try:
        from agent import Agent
        
        aggr_direction = 'all'
        use_deepwalk = True
        num_features = 64
        n_steps = 1
        
        print("Loading pretrained model...")
        model_path = f'qnet_ddqn/q_net_{30 * 10}.model'
        
        agent = Agent(
            num_features // 2,
            0, 0, 0, device, 0, n_steps, 
            aggr_direction=aggr_direction, 
            graph_name=graph_name, 
            use_deepwalk=use_deepwalk
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        agent.q_net.load_state_dict(checkpoint.state_dict())
        
        agent.q_net.to(device)
        agent.q_net.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Using random inference as fallback...")
        agent = None
    
    # 4. 定义模型推理函数
    def my_model_inference(graph, k):
        """模型推理函数"""
        if agent is None:
            print("  Warning: Using random selection")
            return random.sample(list(graph.nodes()), min(k, len(graph)))
        
        try:
            # agent.graph_init_embedding_dict[id(graph)] = torch.randn(graph.number_of_nodes(), num_features).to(device)
            test_graphs = [graph]
            data = agent.get_q_net_input(test_graphs)
            states = [[0] * graph.number_of_nodes()]
            
            with torch.no_grad():
                res = agent.q_net(
                    data.x.to(device), 
                    data.edge_index.to(device), 
                    data.edge_weights.to(device),
                    data.batch.to(device),
                    states
                )
            
            seeds = torch.topk(res, k=k)[1].tolist()
            return seeds
            
        except Exception as e:
            traceback.print_exc()
            print(f"  Inference error: {e}")
            return random.sample(list(graph.nodes()), min(k, len(graph)))
    
    # 5. 运行可扩展性测试
    print("\nStep 4: Running scalability test...")
    
    k = 200
    n_trials = 3

    results = analyze_scalability_trend(
        samples=samples,
        model_inference_func=my_model_inference,
        agent=agent,
        num_features=num_features,
        k=k,
        n_trials=n_trials
    )
    
    # 6. 创建图表
    print("\nStep 5: Creating plots...")
    create_scalability_plot(results, graph_name, "scalability_plots")
    
    # 7. 保存结果
    print("\nStep 6: Saving results...")
    
    with open(f'{graph_name}_scalability_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    import csv
    with open(f'{graph_name}_scalability_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ratio(%)', 'Nodes', 'Edges', 'Avg Time(s)', 'Std(s)', 'Time/Node(s/node)'])
        
        for ratio in sorted(results.keys()):
            info = results[ratio]
            time_per_node = info['avg_inference_time']/info['n_nodes'] if info['n_nodes'] > 0 else 0
            writer.writerow([
                f"{ratio*100:.1f}",
                info['n_nodes'],
                info['n_edges'],
                f"{info['avg_inference_time']:.4f}",
                f"{info['std_inference_time']:.4f}",
                f"{time_per_node:.8f}"
            ])
    
    # 8. 打印汇总报告
    print("\n" + "="*60)
    print("Scalability Analysis Complete!")
    print("="*60)
    
    print(f"\nDataset: {graph_name}")
    print(f"Test parameters: k={k}, trials={n_trials}")
    print()
    
    print(f"{'Ratio(%)':<8} {'Nodes':<10} {'Edges':<12} {'Time(s)':<15} {'Time/Node':<15}")
    print("-"*70)
    
    for ratio in sorted(results.keys()):
        info = results[ratio]
        time_per_node = info['avg_inference_time'] / info['n_nodes'] if info['n_nodes'] > 0 else 0
        
        print(f"{ratio*100:>6.1f}%  {info['n_nodes']:<10} {info['n_edges']:<12} "
              f"{info['avg_inference_time']:8.4f} ± {info['std_inference_time']:6.4f}  "
              f"{time_per_node:12.8f}")
    
    print("\nGenerated files:")
    print("1. scalability_samples/ - Graph samples at different scales")
    print("   - *_pct.pkl: Subgraph files (nodes renumbered from 0)")
    print("   - *_pct_init_X.npy: DeepWalk embeddings for subgraphs")
    print("   - *_pct_old_to_new.pkl: Original node ID to new ID mapping")
    print(f"2. {graph_name}_scalability_results.pkl - Complete results (binary)")
    print(f"3. {graph_name}_scalability_results.csv - Results (CSV format)")
    print(f"4. {graph_name}_scalability_plots/ - Visualization plots")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    main()