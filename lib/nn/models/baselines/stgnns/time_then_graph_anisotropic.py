from typing import Union, List

from tsl.nn.blocks.encoders import RNN
from tsl.utils import ensure_list

from lib.nn.layers import GraphAnisoConv
from .prototypes import TimeThenSpace


class TimeThenGraphAnisoModel(TimeThenSpace):

    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 emb_size: int = 0,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 time_layers: int = 1,
                 graph_layers: int = 1,
                 activation: str = 'elu'):
        rnn = RNN(input_size=hidden_size,
                  hidden_size=hidden_size,
                  n_layers=time_layers,
                  return_only_last_state=True,
                  cell='gru')
        self.temporal_layers = time_layers

        add_embedding_before = ensure_list(add_embedding_before)
        mp_in_size = hidden_size
        if 'message_passing' in add_embedding_before:
            mp_in_size += emb_size
        mp_layers = [
            GraphAnisoConv(mp_in_size, hidden_size, activation=activation)
            for _ in range(graph_layers)
        ]
        super(TimeThenGraphAnisoModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=rnn,
            spatial_encoder=mp_layers,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )
