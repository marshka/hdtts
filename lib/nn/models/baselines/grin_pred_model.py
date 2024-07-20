from typing import Optional, Tuple, List

import torch
from torch import Tensor
from tsl.nn.blocks import MLPDecoder, MLP
from tsl.nn.layers import NodeEmbedding, GRINCell
from tsl.nn.models import BaseModel


class GRINPredictionModel(BaseModel):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 n_layers: int,
                 horizon: int,
                 exog_size: int = 0,
                 embedding_size: Optional[int] = None,
                 kernel_size: int = 2,
                 decoder_order: int = 1,
                 layer_norm: bool = False,
                 activation: str = "relu",
                 dropout: float = 0.):
        super().__init__()
        self.fwd_gril = GRINCell(input_size=input_size,
                                 hidden_size=hidden_size,
                                 exog_size=exog_size,
                                 n_layers=n_layers,
                                 dropout=dropout,
                                 kernel_size=kernel_size,
                                 decoder_order=decoder_order,
                                 n_nodes=n_nodes,
                                 layer_norm=layer_norm)

        if embedding_size is not None and embedding_size > 0:
            assert n_nodes is not None
            self.emb = NodeEmbedding(n_nodes, embedding_size)
        else:
            self.register_parameter('emb', None)
            embedding_size = 0

        #  READOUT MODULES  ###################################################
        ro_input_size = 2 * hidden_size + input_size + embedding_size
        self.imputation_readout = MLP(input_size=ro_input_size,
                                      hidden_size=hidden_size * 2,
                                      output_size=input_size,
                                      activation=activation,
                                      dropout=dropout)
        self.prediction_readout = MLPDecoder(input_size=hidden_size,
                                             hidden_size=hidden_size * 2,
                                             output_size=input_size,
                                             horizon=horizon,
                                             activation=activation,
                                             dropout=dropout)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                input_mask: Optional[Tensor] = None,
                u: Optional[Tensor] = None) -> Tuple[Tensor, List, None, None]:
        """"""
        # x: [batch, time, node, features]
        # u: [batch, time, *, features]
        if input_mask is not None:
            x = x * input_mask
        if u is not None and u.ndim == 3:
            u = u.unsqueeze(-2).expand(-1, -1, x.size(-2), -1)
        # u: [batch, time, node, features]
        fwd_out, fwd_pred, fwd_repr, fwd_h = self.fwd_gril(x,
                                                           edge_index,
                                                           edge_weight,
                                                           mask=input_mask,
                                                           u=u)

        inputs = [fwd_repr, input_mask]
        if self.emb is not None:
            b, s, *_ = fwd_repr.size()  # fwd_repr: [b t n f]
            inputs += [self.emb(expand=(b, s, -1, -1))]
        h = torch.cat(inputs, dim=-1)

        #  IMPUTATION READOUT  ################################################
        x_hat = self.imputation_readout(h)
        imputations = [x_hat, fwd_out, fwd_pred]

        #  PREDICTION READOUT  ################################################
        # fwd_h: [layers b time n f]
        h = fwd_h[-1, :, -1]  # select last state of last layer
        y_hat = self.prediction_readout(h)

        return y_hat, imputations, None, None

    def predict(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                input_mask: Optional[Tensor] = None,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        y_hat = self.forward(x=x,
                             edge_index=edge_index,
                             edge_weight=edge_weight,
                             input_mask=input_mask,
                             u=u)[0]
        return y_hat
