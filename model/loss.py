import torch
import torch.nn.functional as F

def prototypical_loss(emb_sup, sup_y, emb_qry, qry_y):
    n_way = len(torch.unique(sup_y))  # Number of classes in episode

    # Compute prototypes by averaging support embeddings per class
    prototypes = []
    for c in torch.unique(sup_y):
        class_mask = sup_y == c
        class_emb = emb_sup[class_mask]
        prototype = class_emb.mean(dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)  # Shape: (n_way, D)

    # Compute squared Euclidean distances from queries to prototypes
    dists = torch.cdist(emb_qry, prototypes)  # Shape: (N_query, n_way)

    # Compute log probabilities for query samples over classes (softmax over -distances)
    log_p_y = F.log_softmax(-dists, dim=1)

    # Compute negative log-likelihood loss on true labels
    loss = -log_p_y[range(len(qry_y)), qry_y].mean()

    # Predict class with max log probability
    _, y_hat = log_p_y.max(1)
    acc = (y_hat == qry_y).float().mean()

    return loss, acc.item()