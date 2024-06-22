
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from collections import defaultdict, Counter
import seaborn as sns
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP


def analyze_llama_model(model_name="meta-llama/Llama-2-7b-hf"):
    # 加载模型配置
    config = LlamaConfig.from_pretrained(model_name)
    
    # 从配置创建模型
    model = LlamaForCausalLM(config)
    
    # 打印模型整体结构
    print("模型结构概览:")
    print(model)
    print("\n" + "= " *50 + "\n")

    # 统计参数总量和各层参数量
    total_params = 0
    param_counts = defaultdict(int)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # 根据参数名称分类
        if 'embed' in name:
            param_counts['Embedding'] += num_params
        elif 'attention' in name:
            param_counts['Attention'] += num_params
        elif 'mlp' in name:
            param_counts['MLP (Feed Forward)'] += num_params
        elif 'norm' in name:
            param_counts['Layer Norm'] += num_params
        elif 'lm_head' in name:
            param_counts['LM Head'] += num_params
        else:
            param_counts['Other'] += num_params

    # 打印统计结果
    print(f"总参数量: {total_params:,}")
    print("\n各组件参数量:")
    for component, count in param_counts.items():
        print(f"{component}: {count:,} ({coun t /total_param s *100:.2f}%)")

    # 打印模型配置信息
    print("\n模型配置信息:")
    print(f"层数: {config.num_hidden_layers}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"隐藏层维度: {config.hidden_size}")
    print(f"中间层维度 (MLP): {config.intermediate_size}")
    print(f"最大序列长度: {config.max_position_embeddings}")
    print(f"词汇表大小: {config.vocab_size}")


def analyze_word_embeddings(model_name="meta-llama/Llama-2-7b-hf", n_sample=10000):
    # 创建保存图片的文件夹
    if not os.path.exists('embedding_analysis'):
        os.makedirs('embedding_analysis')

    # 加载分词器和模型
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(model_name)

    # 获取词嵌入矩阵
    embedding_matrix = model.get_input_embeddings().weight.detach().numpy()

    # 分析权重分布
    print("词嵌入权重分布:")
    print(f"Mean: {np.mean(embedding_matrix):.4f}")
    print(f"Std: {np.std(embedding_matrix):.4f}")
    print(f"Min: {np.min(embedding_matrix):.4f}")
    print(f"Max: {np.max(embedding_matrix):.4f}")

    # 绘制权重分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(embedding_matrix.flatten(), bins=50)
    plt.title("Word Embedding Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.savefig('embedding_analysis/weight_distribution.png')
    plt.close()

    # 随机采样一部分词进行可视化
    vocab_size = embedding_matrix.shape[0]
    sample_indices = np.random.choice(vocab_size, n_sample, replace=False)
    sample_embeddings = embedding_matrix[sample_indices]

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(sample_embeddings)

    # 使用UMAP进行降维
    umap = UMAP(n_components=2, random_state=42)
    umap_results = umap.fit_transform(sample_embeddings)

    # 可视化t-SNE结果
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6)
    plt.title("t-SNE Visualization of Word Embeddings")
    plt.savefig('embedding_analysis/tsne_visualization.png')
    plt.close()

    # 可视化UMAP结果
    plt.figure(figsize=(12, 10))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], alpha=0.6)
    plt.title("UMAP Visualization of Word Embeddings")
    plt.savefig('embedding_analysis/umap_visualization.png')
    plt.close()

    # 分析常用词和罕见词
    word_freq = Counter()
    for token, id in tokenizer.get_vocab().items():
        if id in sample_indices:
            word_freq[token] = sample_indices.tolist().index(id)

    common_words = [word for word, _ in word_freq.most_common(10)]
    rare_words = [word for word, _ in word_freq.most_common()[-10:]]

    print("\n常用词:")
    for word in common_words:
        print(word)

    print("\n罕见词:")
    for word in rare_words:
        print(word)

    # 可视化常用词和罕见词
    def plot_words(words, embeddings, title, filename):
        plt.figure(figsize=(12, 10))
        for word in words:
            idx = word_freq[word]
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1], alpha=0.6)
            plt.annotate(word, (embeddings[idx, 0], embeddings[idx, 1]))
        plt.title(title)
        plt.savefig(f'embedding_analysis/{filename}.png')
        plt.close()

    plot_words(common_words, umap_results, "Common Words in UMAP Space", "common_words_umap")
    plot_words(rare_words, umap_results, "Rare Words in UMAP Space", "rare_words_umap")



def analyze_attention_weights(model_name="meta-llama/Llama-2-7b-hf", text="Hello, how are you?"):
    # 加载模型和分词器
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt")

    # 获取注意力输出
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 获取所有层的注意力权重
    attention_weights = outputs.attentions

    # 创建保存图片的文件夹
    if not os.path.exists('attention_analysis'):
        os.makedirs('attention_analysis')

    # 可视化每一层的注意力模式
    for layer, attn in enumerate(attention_weights):
        attn = attn.squeeze(0).mean(dim=0).numpy()  # 平均所有头的注意力
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, cmap='viridis')
        plt.title(f"Layer {laye r +1} Attention Pattern")
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.savefig(f'attention_analysis/layer_{laye r +1}_attention.png')
        plt.close()

    # 分析自注意力矩阵（以最后一层为例）
    last_layer_attn = attention_weights[-1].squeeze(0).mean(dim=0).numpy()
    plt.figure(figsize=(12, 10))
    sns.heatmap(last_layer_attn, cmap='viridis')
    plt.title("Last Layer Self-Attention Matrix")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.savefig('attention_analysis/last_layer_self_attention.png')
    plt.close()

    # 比较不同头的注意力分布（以最后一层为例）
    last_layer_heads = attention_weights[-1].squeeze(0).numpy()
    num_heads = last_layer_heads.shape[0]
    fig, axes = plt.subplots(num_head s/ /2, 2, figsize=(20, 5* num_heads))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(last_layer_heads[i], ax=ax, cmap='viridis')
        ax.set_title(f"Head {i + 1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
    plt.tight_layout()
    plt.savefig('attention_analysis/last_layer_heads_comparison.png')
    plt.close()

    # 分析对角线注意力
    diagonal_attention = np.diagonal(last_layer_attn)
    plt.figure(figsize=(10, 6))
    plt.plot(diagonal_attention)
    plt.title("Diagonal Attention in Last Layer")
    plt.xlabel("Token Position")
    plt.ylabel("Attention Weight")
    plt.savefig('attention_analysis/diagonal_attention.png')
    plt.close()


def analyze_position_encoding(model_name="meta-llama/Llama-2-7b-hf", seq_length=128):
    # 加载模型配置
    config = LlamaConfig.from_pretrained(model_name)

    # 创建一个小型的LLaMA模型用于分析
    small_config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256
    )
    model = LlamaForCausalLM(small_config)

    # 获取位置编码函数
    rotary_emb = model.model.layers[0].self_attn.rotary_emb

    # 生成位置编码
    position_ids = torch.arange(seq_length).unsqueeze(0)

    # 尝试不同的方法来获取 cos 和 sin
    if hasattr(rotary_emb, 'get_cos_sin_cache'):
        cos, sin = rotary_emb.get_cos_sin_cache(seq_length, device='cpu')
    elif hasattr(rotary_emb, 'cos_cached') and hasattr(rotary_emb, 'sin_cached'):
        cos = rotary_emb.cos_cached[:seq_length, :]
        sin = rotary_emb.sin_cached[:seq_length, :]
    else:
        # 如果上述方法都不可用，我们自己计算
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_emb.dim, 2).float() / rotary_emb.dim))
        t = torch.arange(seq_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # 可视化cos和sin
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cos.detach().numpy(), cmap='viridis')
    plt.title('Cosine of Position Encodings')
    plt.xlabel('Dimension')
    plt.ylabel('Position')

    plt.subplot(1, 2, 2)
    sns.heatmap(sin.detach().numpy(), cmap='viridis')
    plt.title('Sine of Position Encodings')
    plt.xlabel('Dimension')
    plt.ylabel('Position')

    plt.tight_layout()
    plt.savefig('position_encoding_analysis.png')
    plt.close()

    # 分析不同位置的编码
    positions_to_analyze = [0, 10, 50, 100]
    plt.figure(figsize=(15, 10))
    for i, pos in enumerate(positions_to_analyze):
        encoding = torch.cat([cos[pos], sin[pos]])
        plt.subplot(2, 2, i + 1)
        plt.plot(encoding.detach().numpy())
        plt.title(f'Position Encoding at position {pos}')
        plt.xlabel('Dimension')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig('position_encoding_comparison.png')
    plt.close()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def analyze_layer_weights(model_name="meta-llama/Llama-2-7b-hf", output_dir="res_ans"):
    ensure_dir(output_dir)

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    print("Collecting weights...")
    layer_weights = []
    residual_weights = []

    for layer in model.model.layers:
        attn_weights = layer.self_attn.q_proj.weight.data.flatten().cpu().numpy()
        layer_weights.append(attn_weights)

        residual = layer.input_layernorm.weight.data.cpu().numpy()
        residual_weights.append(residual)

    print("Plotting weight distributions...")
    plt.figure(figsize=(15, 10))
    for i, weights in enumerate(layer_weights):
        if i % 8 == 0:
            sns.kdeplot(weights, label=f'Layer {i + 1}')
    plt.title('Weight Distribution Across Layers')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'layer_weight_distribution.png'))
    plt.close()

    print("Plotting residual weights...")
    plt.figure(figsize=(15, 10))
    sns.heatmap(residual_weights, cmap='viridis')
    plt.title('Residual Connection Weights Across Layers')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Layer')
    plt.savefig(os.path.join(output_dir, 'residual_weights.png'))
    plt.close()

    print("Calculating weight correlations...")
    sampled_weights = layer_weights[::8]
    weight_correlations = np.corrcoef(np.array(sampled_weights))
    plt.figure(figsize=(12, 10))
    sns.heatmap(weight_correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Layer Weight Correlations (Every 8th Layer)')
    plt.savefig(os.path.join(output_dir, 'layer_weight_correlations.png'))
    plt.close()

    print("Analyzing deep vs shallow layers...")
    shallow_layers = layer_weights[:5]
    deep_layers = layer_weights[-5:]

    plt.figure(figsize=(15, 10))
    for i, weights in enumerate(shallow_layers):
        sns.kdeplot(weights, label=f'Shallow Layer {i + 1}')
    for i, weights in enumerate(deep_layers):
        sns.kdeplot(weights, label=f'Deep Layer {len(layer_weights) - 4 + i}')
    plt.title('Weight Distribution: Shallow vs Deep Layers')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'shallow_vs_deep_layers.png'))
    plt.close()


def analyze_activations(model_name="meta-llama/Llama-2-7b-hf",
                        sample_text="The quick brown fox jumps over the lazy dog."):
    print("Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    print("Tokenizing sample text...")
    inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

    print("Analyzing activations...")
    activation_stats = []
    dead_neurons = []

    # 注册钩子来捕获激活值
    def hook_fn(module, input, output):
        activation_stats.append(output.detach().cpu().numpy())

    hooks = []
    for name, module in model.named_modules():
        if "mlp" in name and "up_proj" in name:  # 我们关注MLP层的上投影
            hooks.append(module.register_forward_hook(hook_fn))

    # 运行前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    print("Processing activation statistics...")
    for i, act in enumerate(activation_stats):
        act = act.reshape(-1)  # 展平激活值

        # 计算基本统计量
        mean = np.mean(act)
        std = np.std(act)
        max_val = np.max(act)
        min_val = np.min(act)

        # 检查死亡神经元
        dead_count = np.sum(act == 0)
        dead_percentage = dead_count / len(act) * 100
        dead_neurons.append(dead_percentage)

        print(f"Layer {i + 1}:")
        print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"  Max: {max_val:.4f}, Min: {min_val:.4f}")
        print(f"  Dead neurons: {dead_percentage:.2f}%")

        # 绘制激活值分布
        plt.figure(figsize=(10, 6))
        sns.histplot(act, kde=True, bins=50)
        plt.title(f"Activation Distribution for Layer {i + 1}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.savefig(f"activation_dist_layer_{i + 1}.png")
        plt.close()

    # 绘制死亡神经元百分比
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(dead_neurons) + 1), dead_neurons, marker='o')
    plt.title("Percentage of Dead Neurons Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Percentage of Dead Neurons")
    plt.savefig("dead_neurons.png")
    plt.close()


def analyze_gradients(model_name="meta-llama/Llama-2-7b-hf",
                      prompt="Translate the following English text to French: 'Hello, how are you?'",
                      target="Bonjour, comment allez-vous ?"):
    print("Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.train()  # Set model to training mode to enable gradient computation

    print("Tokenizing input...")
    # Tokenize input and target
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    target_ids = tokenizer(target, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    # Ensure input and target have the same sequence length
    max_length = max(inputs.input_ids.shape[1], target_ids.shape[1])
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    target_ids = tokenizer(target, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)[
        "input_ids"]

    # Move tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    target_ids = target_ids.to(model.device)

    print("Running forward pass and computing loss...")
    outputs = model(**inputs, labels=target_ids)
    loss = outputs.loss

    print("Computing gradients...")
    loss.backward()

    print("Analyzing gradients...")
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

    # Sort parameters by gradient norm
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 parameters with highest gradient norms:")
    for name, norm in sorted_grads[:10]:
        print(f"{name}: {norm}")

    # Visualize gradient norms
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[name.split('.')[-2] for name, _ in sorted_grads[:20]], y=[norm for _, norm in sorted_grads[:20]])
    plt.title("Top 20 Parameter Gradient Norms")
    plt.xlabel("Parameter")
    plt.ylabel("Gradient Norm")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("gradient_norms.png")
    plt.close()

    # Analyze gradients across layers
    layer_grads = {}
    for name, norm in grad_norms.items():
        if 'layers' in name:
            layer = int(name.split('.')[2])
            if layer not in layer_grads:
                layer_grads[layer] = []
            layer_grads[layer].append(norm)

    avg_layer_grads = {layer: np.mean(grads) for layer, grads in layer_grads.items()}

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=list(avg_layer_grads.keys()), y=list(avg_layer_grads.values()))
    plt.title("Average Gradient Norm Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Average Gradient Norm")
    plt.savefig("layer_gradients.png")
    plt.close()


if __name__ == "__main__":
    analyze_gradients()
