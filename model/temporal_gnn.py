"""Temporal GNN with S4 but without GSL. aka GraphS4mer w/o GSL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import (
    GINEConv,
    GINConv,
    TransformerConv,
    GATv2Conv,
    SAGEConv,
    GCNConv,
)
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch_geometric
import scipy
from model.graphs4mer import get_knn_graph
from model.graph_learner import *
from model.s4 import S4Model
from model.decoders import SequenceDecoder


class TemporalGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_nodes,
        dropout,
        g_conv,
        num_gnn_layers,
        hidden_dim,
        max_seq_len,
        num_temporal_layers,
        state_dim=64,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        temporal_pool="mean",
        prenorm=False,
        postact=None,
        gin_mlp=False,
        train_eps=False,
        graph_pool="sum",
        activation_fn="relu",
        num_classes=1,
        undirected_graph=True,
        use_prior=False,
        K=3,
        **kwargs
    ):
        """Baseline Temporal GNN model"""
        super().__init__()

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.graph_pool = graph_pool
        self.hidden_dim = hidden_dim
        self.undirected_graph = undirected_graph
        self.use_prior = use_prior
        self.K = K
        self.temporal_model = temporal_model
        self.temporal_pool = temporal_pool

        # temporal layer
        if temporal_model == "gru":
            self.t_model = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif temporal_model == "s4":
            self.t_model = S4Model(
                d_input=input_dim,
                d_model=hidden_dim,
                d_state=state_dim,
                channels=channels,
                n_layers=num_temporal_layers,
                dropout=dropout,
                prenorm=prenorm,
                l_max=max_seq_len,
                bidirectional=bidirectional,
                postact=postact,  # none or 'glu'
                add_decoder=False,
                pool=False,  # hard-coded
                temporal_pool=None,
            )

        else:
            raise NotImplementedError

        # temporal pool, no trainable param
        if temporal_pool == "mean":
            pass
        else:
            self.temporal_pool_layer = SequenceDecoder(
                d_model=hidden_dim,
                d_output=None,
                l_output=1,  # pool to 1
                use_lengths=False,
                mode=temporal_pool,
            )

        # gnn layers
        self.gnn_layers = nn.ModuleList()
        if g_conv == "graphsage":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, **kwargs))
        elif g_conv == "gine":
            for _ in range(num_gnn_layers):
                if gin_mlp:
                    gin_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                else:
                    gin_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
                self.gnn_layers.append(
                    GINEConv(
                        nn=gin_nn, eps=0.0, train_eps=train_eps, edge_dim=1, **kwargs
                    )
                )
        elif g_conv == "gat":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        negative_slope=negative_slope,
                        dropout=dropout,
                        edge_dim=1,
                        fill_value=1.0,
                    )
                )
        else:
            raise NotImplementedError

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_fn == "elu":
            self.activation = nn.ELU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, lengths=None):
        """
        Args:
            data: torch geometric data object
        """
        x = data.x  # (batch * num_nodes, seq_len, 1)
        batch = x.shape[0] // self.num_nodes
        _, seq_len, _ = x.shape
        num_nodes = self.num_nodes
        batch_idx = data.batch

        if lengths is not None:
            lengths = torch.repeat_interleave(lengths, num_nodes, dim=0)

        if self.use_prior:
            edge_index, edge_weight = data.edge_index, data.edge_attr
        else:
            # initialize with a knn graph
            edge_index, edge_weight, adj_mat = get_knn_graph(
                x.reshape(batch, num_nodes, -1),
                self.K,
                dist_measure="cosine",
                undirected=self.undirected_graph,
            )
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
            adj_mat = adj_mat.to(x.device)
            
        # add self-loop
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index, edge_attr=edge_weight
        )
        edge_index, edge_weight = torch_geometric.utils.add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1,
        )

        # temporal layer
        if self.temporal_model == "s4":
            x = self.t_model(x, lengths=lengths)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            x, _ = self.t_model(x)

        # temporal pool
        if self.temporal_pool == "mean":
            if lengths is None:
                x = torch.mean(x, dim=1)  # (batch * num_nodes, hidden_dim)
            else:
                x = torch.stack(
                    [
                        torch.mean(out[:length, :], dim=0)
                        for out, length in zip(torch.unbind(x, dim=0), lengths)
                    ],
                    dim=0,
                ) # (batch * num_nodes, hidden_dim)
        else:
            x = self.temporal_pool_layer(x).squeeze(
                1
            )  # (batch * num_nodes, hidden_dim)

        for i in range(len(self.gnn_layers)):

            # gnn layer
            x = self.gnn_layers[i](
                x, edge_index=edge_index, edge_attr=edge_weight.reshape(-1, 1)
            )
            x = self.dropout(self.activation(x))  # (batch * num_nodes, hidden_dim)

        x = x.reshape(batch, num_nodes, -1)

        # graph pool
        if self.graph_pool == "sum":
            x = torch.sum(x, dim=-2)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x = torch.mean(x, dim=-2)
        elif self.graph_pool == "max":
            x, _ = torch.max(x, dim=-2)
        else:
            raise NotImplementedError

        # classifier
        x = self.classifier(x)

        return x

class TemporalGNN_Regression(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_nodes,
        dropout,
        g_conv,
        num_gnn_layers,
        hidden_dim,
        max_seq_len,
        output_seq_len,
        num_temporal_layers,
        state_dim=64,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        prenorm=False,
        postact=None,
        gin_mlp=False,
        train_eps=False,
        activation_fn="relu",
        undirected_graph=True,
        use_prior=False,
        K=3,
        **kwargs
    ):
        """Baseline Temporal GNN model"""
        super().__init__()

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.undirected_graph = undirected_graph
        self.use_prior = use_prior
        self.K = K
        self.temporal_model = temporal_model

        # temporal layer
        if temporal_model == "gru":
            self.t_model = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif temporal_model == "s4":
            self.t_model = S4Model(
                d_input=input_dim,
                d_model=hidden_dim,
                d_state=state_dim,
                channels=channels,
                n_layers=num_temporal_layers,
                dropout=dropout,
                prenorm=prenorm,
                l_max=max_seq_len,
                bidirectional=bidirectional,
                postact=postact,  # none or 'glu'
                add_decoder=False,
                pool=False,  # hard-coded
                temporal_pool=None,
            )

        else:
            raise NotImplementedError

        # gnn layers
        self.gnn_layers = nn.ModuleList()
        if g_conv == "graphsage":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, **kwargs))
        elif g_conv == "gine":
            for _ in range(num_gnn_layers):
                if gin_mlp:
                    gin_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                else:
                    gin_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
                self.gnn_layers.append(
                    GINEConv(
                        nn=gin_nn, eps=0.0, train_eps=train_eps, edge_dim=1, **kwargs
                    )
                )

        elif g_conv == "gat":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=num_heads,
                        negative_slope=negative_slope,
                        dropout=dropout,
                        edge_dim=1,
                        fill_value=1.0,
                    )
                )
        else:
            raise NotImplementedError

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_fn == "elu":
            self.activation = nn.ELU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)
        # self.decoder = nn.Linear(hidden_dim, output_dim)

        self.decoder = SequenceDecoder(
            d_model=hidden_dim,
            d_output=output_dim,
            l_output=output_seq_len,
            use_lengths=False,
            mode="last",  # equivalent to no pooling
        )

    def forward(self, data):
        """
        Args:
            data: torch geometric data object
        """
        x = data.x  # (batch * num_nodes, seq_len, 1)
        batch = x.shape[0] // self.num_nodes
        _, seq_len, _ = x.shape
        num_nodes = self.num_nodes
        batch_idx = data.batch

        if self.use_prior:
            edge_index, edge_weight, adj_mat = data.edge_index, data.edge_attr, data.adj_mat
        else:
            # initialize with a knn graph
            edge_index, edge_weight, adj_mat = get_knn_graph(
                x.reshape(batch, num_nodes, -1),
                self.K,
                dist_measure="cosine",
                undirected=self.undirected_graph,
            )
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
            adj_mat = adj_mat.to(x.device)

        # temporal layer
        # print("x before s4:", x.shape)
        if self.temporal_model == "s4":
            x = self.t_model(x)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            x, _ = self.t_model(x)

        ## batched implementation
        adj_mat = torch.cat([adj_mat] * seq_len)
        edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

        # add self-loop
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index, edge_attr=edge_weight
        )
        edge_index, edge_weight = torch_geometric.utils.add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1,
        )
        
        # treating seq_len as batch
        x = x.reshape(batch, num_nodes, seq_len, -1).transpose(1, 2) # (batch, seq_len, num_nodes, hidden_dim)
        x = x.reshape(batch * seq_len, num_nodes, -1).reshape(batch * seq_len * num_nodes, -1)
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index=edge_index, edge_attr=edge_weight.reshape(-1,1))
            x = self.dropout(self.activation(x))
        x = x.reshape(batch * seq_len, num_nodes, -1).reshape(batch, seq_len, num_nodes, -1).transpose(1, 2)
        x = x.reshape(batch * num_nodes, seq_len, -1)

        # decoder
        x = self.decoder(x)

        return x
