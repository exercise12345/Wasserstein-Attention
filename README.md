# Wasserstein Self - Attention Mechanism

The Wasserstein Self - Attention Mechanism is an enhanced version of the traditional self - attention mechanism.

## Input Representation
Suppose there is an input feature sequence `X = [x_1, x_2,..., x_n]`, where `x_i` is the `i`-th feature vector with dimension `d`.

## Generation of Query, Key and Value Vectors
Similar to the traditional self - attention, query vectors `Q`, key vectors `K` and value vectors `V` are generated through linear transformations. Let `W_Q`, `W_K` and `W_V` be learnable weight matrices. Then `Q = XW_Q`, `K = XW_K` and `V = XW_V`.

## Calculation of Attention Weights
In traditional self - attention, the attention weight `A_ij` is calculated by the softmax function: `A_ij = exp(q_i * k_j) / sum(exp(q_i * k_k)) (k = 1 to n)`, where `q_i` is the `i`-th query vector in `Q` and `k_j` is the `j`-th key vector in `K`.

In the Wasserstein self - attention mechanism, the attention weight calculation is based on the Wasserstein distance. First, calculate the Wasserstein distance `W_ij` between two distributions. This distance can be obtained by methods like solving the optimal transport problem. Then, use a function (e.g., sigmoid) to convert `W_ij` to the attention weight `A_ij`. Thus, feature pairs with a small distance have a larger weight, and more attention is paid to features of similar distributions.

## Output Calculation
The final output `Y` is the weighted sum of the attention weights and the value vectors: `Y = sum(A_ij * v_j) (j = 1 to n)`, where `v_j` is the `j`-th value vector in `V`.

This mechanism can better capture feature relationships and is especially useful in feature extraction and sequence - based data processing tasks.


graph TD;
    A[输入特征序列] --> B[生成查询向量 Q];
    A --> C[生成键向量 K];
    A --> D[生成值向量 V];
    B --> E[对每个查询向量 q_i];
    C --> E;
    E --> F[对每个键向量 k_j];
    F --> G[计算 Wasserstein 距离 W(q_i, k_j)];
    G --> H[通过函数转换为注意力权重 A_{ij}];
    D --> I[对每个值向量 v_j];
    H --> J[计算加权和得到输出 Y];
    I --> J;
    J --> K[最终输出];
