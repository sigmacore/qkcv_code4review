__all__ = ['C_Generator']

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm
import pandas as pd
import torch.nn.functional as F
from typing import Tuple, Optional
from neuralforecast.models.tft import GRN, VariableSelectionNetwork

class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)

 
class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class _GRN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        context_hidden_size=None,
        dropout=0,
    ):
        super().__init__()

        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x


class _VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, dropout):
        super().__init__()
        self.joint_grn = GRN(
            input_size=hidden_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            context_hidden_size=hidden_size,
        )
        self.var_grns = nn.ModuleList(
            [
                GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
                for _ in range(num_inputs)
            ]
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        # the line below performs batched matrix vector multiplication
        # for temporal features it's bthf,btf->bth
        # for static features it's bhf,bf->bh
        variable_ctx = torch.matmul(
            transformed_embed, sparse_weights.unsqueeze(-1)
        ).squeeze(-1)

        return variable_ctx, sparse_weights


class TFTEmbedding_qkcv(nn.Module):
    def __init__(
        self, hidden_size, stat_input_size
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.stat_input_size = stat_input_size

        # Instantiate Continuous Embeddings if size is not None
        for attr, size in [
            ("stat_exog_embedding", stat_input_size),
        ]:
            if size:
                vectors = nn.Parameter(torch.Tensor(size, hidden_size))
                bias = nn.Parameter(torch.zeros(size, hidden_size))
                torch.nn.init.xavier_normal_(vectors)
                setattr(self, attr + "_vectors", vectors)
                setattr(self, attr + "_bias", bias)
            else:
                setattr(self, attr + "_vectors", None)
                setattr(self, attr + "_bias", None)

    def _apply_embedding(
        self,
        cont: Optional[Tensor],
        cont_emb: Tensor,
        cont_bias: Tensor,
    ):
        if cont is not None:
            # the line below is equivalent to following einsums
            # e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            # e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
            e_cont = e_cont + cont_bias
            return e_cont

        return None

    def forward(self, stat_exog=None):
        # temporal/static categorical/continuous known/observed input
        # tries to get input, if fails returns None

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        stat_exog = stat_exog[:, :] if stat_exog is not None else None

        s_inp = self._apply_embedding(
            cont=stat_exog,
            cont_emb=self.stat_exog_embedding_vectors,
            cont_bias=self.stat_exog_embedding_bias,
        )

        return s_inp
    

class StaticCovariateEncoder(nn.Module):
    def __init__(self, hidden_size, num_static_vars, dropout, lstm_layer=1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            hidden_size=hidden_size, num_inputs=num_static_vars, dropout=dropout
        )
        self.context_grn = GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        # cs, ce, ch, cc, ch1, cc1 
        x = self.context_grn(variable_ctx)

        return x # ca

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = output_size if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # print(f'[MLP] x.shape: {x.dtype}, {x.shape}')

        x = x.to(dtype=torch.float)
        return self.mlp(x)
    

class C_Generator(nn.Module):
    def __init__(self, hidden_size, stat_exog_size, embedding_type=0, dropout=0.1):
        super().__init__()
        self.embedding_type = embedding_type
        if self.embedding_type==0:
            print(f'[C_Generator] Applying TFT embedding: {self.embedding_type}')
            self.embedding = TFTEmbedding_qkcv(
                hidden_size=hidden_size,
                stat_input_size=stat_exog_size,
            )

            self.static_encoder = StaticCovariateEncoder(
                hidden_size=hidden_size,
                num_static_vars=stat_exog_size,
                dropout=dropout, 
            )

        elif self.embedding_type==1:
            print(f'[C_Generator] Applying MLP embedding: {self.embedding_type}')
            self.embedding = MLP(
                input_size=stat_exog_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                num_layers=2,
                dropout=dropout,
            )


    def forward(self, stat_exog, dropout=0.1):
        # stat_exog = windows_batch["stat_exog"]

        if self.embedding_type==0:

            s_inp = self.embedding(
                stat_exog=stat_exog,
            )

            # print(f'[C_Generator] stat_exog.shape: {stat_exog.shape}, s_inp.shape: {s_inp.shape}')
            # [C_Generator] stat_exog.shape: torch.Size([256, 1, 5]), s_inp.shape: torch.Size([256, 1, 5, 1280])

            # -------------------------------- Inputs ------------------------------#
            # Static context
            if s_inp is not None:
                x = self.static_encoder(s_inp)

        elif self.embedding_type==1:

            # Use multi-layer perceptron (mlp) to embed stat_exog into vector x
            # x = nn.Sequential(
            #     nn.Linear(x.size(-1), self.embedding.hidden_size),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(self.embedding.hidden_size, self.embedding.hidden_size),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            # )(x)
            x = self.embedding(stat_exog)

        return x


