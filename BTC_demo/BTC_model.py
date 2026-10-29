"""BTC inference architecture. Source: database_label_finetune/finetune_core.py."""
import torch
from torch import nn
from torch.nn import functional as F

class HeteroRelationLayer(nn.Module):
    def __init__(self, node_types, relations, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.relations = relations
        self.self_linear = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim) for name in node_types}
        )
        self.relation_linear = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, hidden_dim, bias=False)
                for name in relations
            }
        )
        self.norm = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in node_types}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_index, edge_weight):
        output = {
            name: self.self_linear[name](value) for name, value in hidden.items()
        }
        for relation in self.relations:
            source_type, _, target_type = relation.split("__")
            edge = edge_index[relation]
            if edge.shape[1] == 0:
                continue
            weight = edge_weight[relation].float()
            message = (
                self.relation_linear[relation](hidden[source_type][edge[0]])
                * weight[:, None]
            )
            aggregate = torch.zeros_like(hidden[target_type])
            aggregate.index_add_(0, edge[1], message)
            degree = torch.zeros(len(hidden[target_type]), device=message.device)
            degree.index_add_(0, edge[1], weight.abs())
            output[target_type] += aggregate / degree.clamp_min(1e-6)[:, None]
        return {
            name: self.norm[name](
                hidden[name] + self.dropout(F.gelu(output[name]))
            )
            for name in hidden
        }

class HeteroFinetuner(nn.Module):
    def __init__(self, input_dims, relations, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
                )
                for name, dim in input_dims.items()
            }
        )
        self.layers = nn.ModuleList(
            [
                HeteroRelationLayer(list(input_dims), relations, hidden_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.membership_relation = "metacell__contains__metatcr"
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_weight):
        hidden = {
            name: self.input_projection[name](value.float())
            for name, value in node_features.items()
        }
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_weight)
        edge = edge_index[self.membership_relation]
        weight = edge_weight[self.membership_relation].float()
        context = hidden["metatcr"].new_zeros(hidden["metatcr"].shape)
        denominator = weight.new_zeros(len(hidden["metatcr"]))
        context.index_add_(
            0, edge[1], hidden["metacell"][edge[0]] * weight[:, None]
        )
        denominator.index_add_(0, edge[1], weight.abs())
        if (denominator <= 0).any():
            raise AssertionError("Every metatcr must have a metacell membership edge")
        context = context / denominator[:, None]
        return self.classifier(
            torch.cat([hidden["metatcr"], context], dim=1)
        ).squeeze(1)
