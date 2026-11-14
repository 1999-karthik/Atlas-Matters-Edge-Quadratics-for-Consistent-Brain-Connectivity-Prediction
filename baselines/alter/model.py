import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from .dec import DEC
from typing import List
from .transformer_encoder import InterpretableTransformerEncoder
from .rrwp import add_full_rrwp
from omegaconf import DictConfig
from .base import BaseModel, pick_num_heads

ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "elu": F.elu, "leaky_relu": F.leaky_relu}
import torch.nn.functional as F

class TransPoolingEncoder(nn.Module):

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, dropout=0.1, activation=F.relu, encoder_hidden=32):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=pick_num_heads(input_feature_size, 4),
                                                           dim_feedforward=hidden_size, dropout=dropout, activation=activation,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = encoder_hidden  # upstream: 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()
        
        # Store config as instance variable
        self.config = config

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == 'rrwp':
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim



        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=config.model.get('ffn_dim', 1024),  # upstream: 1024
                                    dropout=config.model.get('dropout', 0.1),  # upstream: nn.TransformerEncoderLayer default 0.1
                                    activation=ACTIVATIONS[config.model.get('activation', 'relu')],  # upstream: relu
                                    encoder_hidden=config.model.get('encoder_hidden', 32),  # upstream: 32
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        # Get number of classes from config, default to 2 for backward compatibility
        num_classes = getattr(config.dataset, 'num_classes', 2)
        print(f"Model initialization: num_classes = {num_classes}")
        
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        if self.pos_encoding == 'rrwp':
            # Use pos_embed_dim for RRWP to match the expected dimension
            rrwp_config = getattr(self.config.model, 'rrwp', {})
            threshold = rrwp_config.get('threshold', 0.0)  # upstream ALTER keeps every positive edge (data > 0)
            
            pos_emb = add_full_rrwp(node_feature, self.pos_embed_dim, threshold=threshold)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
