# Wasserstein-Attention
The code for Wasserstein Attention.
# Wasserstein Self - Attention Mechanism

The Wasserstein Self - Attention Mechanism is an enhanced version of the traditional self - attention mechanism.

## Input Representation
Suppose there is an input feature sequence \(X = [x_1, x_2,\cdots,x_n]\), where \(x_i\) is the \(i\)-th feature vector with dimension \(d\).

## Generation of Query, Key and Value Vectors
Similar to the traditional self - attention, query vectors \(Q\), key vectors \(K\) and value vectors \(V\) are generated through linear transformations. Let \(W_Q\), \(W_K\) and \(W_V\) be learnable weight matrices. Then \(Q = XW_Q\), \(K = XW_K\) and \(V = XW_V\).

## Calculation of Attention Weights
In traditional self - attention, the attention weight \(A_{ij}\) is calculated by the softmax function: \(A_{ij}=\frac{\exp(q_i\cdot k_j)}{\sum_{k = 1}^{n}\exp(q_i\cdot k_k)}\), where \(q_i\) is the \(i\)-th query vector in \(Q\) and \(k_j\) is the \(j\)-th key vector in \(K\).

In the Wasserstein self - attention mechanism, the attention weight calculation is based on the Wasserstein distance. First, calculate the Wasserstein distance \(W_{ij}\) between two distributions. This distance can be obtained by methods like solving the optimal transport problem. Then, use a function (e.g., sigmoid) to convert \(W_{ij}\) to the attention weight \(A_{ij}\). Thus, feature pairs with a small distance have a larger weight, and more attention is paid to features of similar distributions.

## Output Calculation
The final output \(Y\) is the weighted sum of the attention weights and the value vectors: \(Y=\sum_{j = 1}^{n}A_{ij}v_j\), where \(v_j\) is the \(j\)-th value vector in \(V\).

This mechanism can better capture feature relationships and is especially useful in feature extraction and sequence - based data processing tasks.
