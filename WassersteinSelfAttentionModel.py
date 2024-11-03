import torch
import torch.nn as nn

class WassersteinSelfAttentionModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(WassersteinSelfAttentionModel, self).__init__()
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

    def wasserstein_distance(self, a, b):
        a_numpy = a.detach().cpu().numpy()
        b_numpy = b.detach().cpu().numpy()
        n_a = a_numpy.shape[0]
        n_b = b_numpy.shape[0]
        dim = a_numpy.shape[1]

        # 计算源域和目标域数据的概率分布
        source_distribution = np.ones(n_a) / n_a
        target_distribution = np.ones(n_b) / n_b

        # 定义线性规划问题的系数矩阵和约束条件
        c = np.ones(n_a * n_b)
        A_eq = np.zeros((n_a + n_b, n_a * n_b))
        b_eq = np.concatenate((source_distribution, target_distribution))
        for i in range(n_a):
            for j in range(n_b):
                A_eq[i, i * n_b + j] = 1
                A_eq[n_a + j, i * n_b + j] = 1

        # 定义线性规划问题的边界条件
        bounds = [(0, None)] * n_a * n_b

        # 计算每个特征维度上的距离
        distances = []
        for d in range(dim):
            source_dim_data = a_numpy[:, d]
            target_dim_data = b_numpy[:, d]
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            distances.append(res.fun)

        # 这里简单取平均
        return torch.tensor(np.mean(distances), device=a.device)

    def forward(self, x):
        batch_size, seq_len, in_dim = x.size()
        queries = self.query_proj(x).view(batch_size, seq_len, -1)
        keys = self.key_proj(x).view(batch_size, seq_len, -1)
        values = self.value_proj(x).view(batch_size, seq_len, -1)

        attention_weights = torch.zeros(batch_size, seq_len, seq_len).to(x.device)
        for i in range(seq_len):
            for j in range(seq_len):
                attention_weights[:, i, j] = self.wasserstein_distance(queries[:, i], keys[:, j])

        attention_weights = torch.nn.functional.softmax(-attention_weights, dim=-1)
        output = torch.bmm(attention_weights, values).view(batch_size, -1)
        return output
