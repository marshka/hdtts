from typing import List, Optional, Tuple

import torch
from einops import rearrange
from einops import repeat
from torch import Tensor, nn
from tsl.nn.blocks import MLPDecoder
from tsl.nn.blocks.encoders.recurrent import RNNIBase
from tsl.nn.layers.recurrent import GRUCell, LSTMCell, StateType
from tsl.nn.models import BaseModel

from lib.nn.layers import GRUD


class RNNI(RNNIBase):
    """RNN encoder for sequences with missing data.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        exog_size (int): Size of the optional exogenous variables.
            (default: ``0.``)
        cell (str): Type of recurrent cell to be used, one of [:obj:`gru`,
            :obj:`lstm`].
            (default: :obj:`gru`)
        concat_mask (bool): If :obj:`True`, then the input tensor is
            concatenated to the mask when fed to the RNN cell.
            (default: :obj:`True`)
        unitary_mask (bool): If :obj:`True`, then the mask is a single value
            and applies to all features.
            (default: :obj:`False`)
        flip_time (bool): If :obj:`True`, then the time is folded in the
            backward direction.
            (default: :obj:`False`)
        n_layers (int, optional): Number of hidden layers.
            (default: :obj:`1`)
        detach_input (bool): If :obj:`True`, call :meth:`~torch.Tensor.detach`
            on predictions before they are used to fill the gaps, breaking the
            error backpropagation.
            (default: :obj:`False`)
        cat_states_layers (bool): If :obj:`True`, then the states of the RNN are
            concatenated together.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 unitary_mask: bool = False,
                 flip_time: bool = False,
                 n_layers: int = 1,
                 detach_input: bool = False,
                 cat_states_layers: bool = False):

        if cell == 'gru':
            cell = GRUCell
        elif cell == 'lstm':
            cell = LSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.mask_size = 1 if unitary_mask else input_size

        cells = [cell(hidden_size, hidden_size) for _ in range(n_layers)]

        super(RNNI, self).__init__(cells, detach_input, concat_mask, flip_time,
                                   cat_states_layers)

        if concat_mask:
            input_size = input_size + self.mask_size
        input_size = input_size + exog_size

        self.encoder = nn.Linear(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, self.input_size)

    def state_readout(self, h: List[StateType]):
        return self.readout(h[-1])

    def preprocess_input(self,
                         x: Tensor,
                         x_hat: Tensor,
                         input_mask: Tensor,
                         step: int,
                         u: Optional[Tensor] = None,
                         h: Optional[List[StateType]] = None):
        x_t = super().preprocess_input(x, x_hat, input_mask, step)
        if u is not None:
            x_t = torch.cat([x_t, u[:, step]], -1)
        return self.encoder(x_t)

    def single_pass(self, x: Tensor, h: List[StateType], *args,
                    **kwargs) -> List[StateType]:
        return super().single_pass(x, h)

    def forward(self, x: Tensor, input_mask: Tensor, u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None) \
            -> Tuple[Tensor, Tensor, List[StateType]]:
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        return super().forward(x, input_mask, u=u, h=h)


class RNNIPredictionModel(BaseModel):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 n_layers: int = 1,
                 cell: str = 'gru',
                 detach_input: bool = False,
                 cat_states_layers: bool = False,
                 concat_mask: bool = False):
        super().__init__()
        self.cat_states_layers = cat_states_layers

        self.rnn = RNNI(input_size=input_size,
                        hidden_size=hidden_size,
                        exog_size=exog_size,
                        cell=cell,
                        concat_mask=concat_mask,
                        unitary_mask=False,
                        n_layers=n_layers,
                        detach_input=detach_input,
                        cat_states_layers=False)

        #  READOUT MODULES  ###################################################
        ro_input_size = hidden_size * (n_layers if cat_states_layers else 1)
        self.readout = MLPDecoder(input_size=ro_input_size,
                                  hidden_size=hidden_size * 2,
                                  output_size=input_size,
                                  horizon=horizon)

    def forward(self, x: Tensor, input_mask: Optional[Tensor],
                u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None):
        # x: [batch, time, node, features]
        b, n = None, x.size(-2)
        if x.ndim == 4:
            b = x.size(0)
            x = rearrange(x, 'b t n f -> (b n) t f')
            input_mask = rearrange(input_mask, 'b t n f -> (b n) t f')
            if u is not None:
                if u.ndim == 3:
                    u = repeat(u, 'b t f -> (b n) t f', n=n)
                else:  # u.ndim == 4
                    u = rearrange(u, 'b t n f -> (b n) t f')

        # x_hat: [batch, time, features]
        # h: [batch, time, features]
        x_hat, _, h = self.rnn(x, input_mask, u, h)

        if self.cat_states_layers:
            h = torch.cat(h, dim=-1)
        else:
            h = h[-1]  # select last RNN layer

        if b is not None:
            h = rearrange(h, '(b n) f -> b n f', b=b)
            x_hat = rearrange(x_hat, '(b n) t f -> b t n f', b=b)

        # readout
        out = self.readout(h)  # -> [batch time features]

        return out, x_hat, None, None

    def predict(self, x: Tensor, input_mask: Optional[Tensor],
                u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None):
        out, _, _ = self(x, input_mask, u, h)
        return out


class GRUDModel(BaseModel):
    """GRUD model with MLP readout"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 n_nodes: int = None,
                 n_layers: int = 1,
                 x_mean: Tensor = None,
                 ff_size: int = 64,
                 ff_layers: int = 1,
                 ff_dropout: float = 0.,
                 activation: str = 'relu',
                 cat_states_layers: bool = False,
                 imputation_loss: bool = False,
                 fc: bool = False,
                 global_exog: bool = True):
        super().__init__()
        self.cat_states_layers = cat_states_layers
        self.imputation_loss = imputation_loss
        self.fc = fc
        self.global_exog = global_exog

        if fc:
            assert n_nodes is not None
            input_size = input_size * n_nodes
            exog_size = exog_size * (1 if global_exog else n_nodes)
            if x_mean is not None:
                x_mean = rearrange(x_mean, 'n f -> 1 (n f)')

        self.rnn = GRUD(input_size=input_size,
                        hidden_size=hidden_size,
                        exog_size=exog_size,
                        n_layers=n_layers,
                        x_mean=x_mean,
                        cat_states_layers=cat_states_layers,
                        return_only_last_state=False)

        #  READOUT MODULES  ###################################################
        ro_input_size = hidden_size * (n_layers if cat_states_layers else 1)
        self.readout = MLPDecoder(input_size=ro_input_size,
                                  hidden_size=ff_size,
                                  output_size=input_size,
                                  horizon=horizon,
                                  n_layers=ff_layers,
                                  activation=activation,
                                  dropout=ff_dropout)
        if self.imputation_loss:
            self.imputation_readout = nn.Sequential(
                nn.ReLU(),
                nn.Linear(ro_input_size, input_size)
            )

    def forward(self, x: Tensor, input_mask: Tensor,
                x_mean: Optional[Tensor] = None,
                u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None):

        if self.fc:
            _, _, n, _ = x.size()
            x = rearrange(x, 'b t n f -> b t 1 (n f)')
            input_mask = rearrange(input_mask, 'b t n f -> b t 1 (n f)')
            if u is not None and not self.global_exog:
                u = rearrange(u, 'b t n f -> b t 1 (n f)')

        if x_mean is not None:
            out_shape = '1 1 (n f)' if self.fc else '1 n f'
            x_mean = rearrange(x_mean, 'n f -> ' + out_shape)

        x_hat, h = self.rnn(x, mask=input_mask, x_mean=x_mean, u=u, h=h)

        if self.cat_states_layers:
            h = torch.cat(h, dim=-1)
        else:
            h = h[-1]  # select last RNN layer

        # readout
        out = self.readout(h)  # -> [batch time features]
        if self.fc:
            out = rearrange(out, 'b t 1 (n f) -> b t n f', n=n)

        # imputation readout
        if self.imputation_loss:
            x_hat = self.imputation_readout(x_hat)
            if self.fc:
                x_hat = rearrange(x_hat, 'b t 1 (n f) -> b t n f', n=n)
        else:
            x_hat = None

        return out, x_hat, None, None
