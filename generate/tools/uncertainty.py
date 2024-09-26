import torch
import torch.nn.functional as F

# A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
def MCP(logits):
    prob = F.softmax(logits, dim=-1)
    return torch.max(prob, dim=-1)[0]

# Predictive Uncertainty Estimation via Prior Networks
def entropy(logits):
    prob = F.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    return -torch.sum(prob * log_prob, dim=-1)

# Evidential Deep Learning to Quantify Classification Uncertainty
def evidence(logits):
    alpha = F.relu(logits) + 1
    S = torch.sum(alpha, dim=-1)
    return logits.shape[-1] / S

if __name__ == "__main__":
    logits = torch.randn(4, 3, 5)
    print(F.softmax(logits, dim=-1).shape)
    print(MCP(logits).shape)
    print(entropy(logits).shape)
    print(evidence(logits).shape)