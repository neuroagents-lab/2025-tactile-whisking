import torch.nn as nn

try:
    from enc_att_dec.s4_base import S4Block as S4
except ImportError:
    print("Failed to import S4 blocks from s4_base")

try:
    from mamba_ssm.models.mixer_seq_simple import create_block
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("Failed to import Triton LayerNorm / RMSNorm kernels")


__all__ = ["S4Att", "MambaAtt"]  # currently only support the non-embedding & non-decoder version


class S4Att(nn.Module):
    def __init__(
            self,
            # model_dim,
            # dropout,
            # nlayers,
            # prenorm,
            cfg
    ):
        super().__init__()

        attender_cfg = cfg.model.attender

        model_dim = cfg.model.repr_dim  # output dim of the encoder

        dropout = attender_cfg.dropout
        nlayers = attender_cfg.nlayers
        prenorm = attender_cfg.prenorm

        self.prenorm = prenorm
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.s4_layers.append(S4(model_dim, dropout=dropout, transposed=False))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, lengths=None, return_activations=False):
        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        activations = dict()
        for idx, (layer, norm, dropout) in enumerate(zip(self.s4_layers, self.norms, self.dropouts)):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            activations[f'att_s4_l{idx}'] = x

        x = x.transpose(-1, -2)
        if return_activations:
            return x, activations
        else:
            return x


class MambaAtt(nn.Module):
    def __init__(
            self,
            # model_dim,
            # d_intermediate,  # if 0, no mlp applied to the hidden states of each Mamba block
            # # see:
            # # https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/models/mixer_seq_simple.py#L67
            # # https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/models/mixer_seq_simple.py#L76
            # # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py#L69
            # nlayers,
            # ssm_cfg=None,
            # norm_epsilon=1e-5,
            # rms_norm=False,
            # fused_add_norm=False,
            # residual_in_fp32=False,
            cfg
    ):
        super().__init__()

        attender_cfg = cfg.model.attender
        model_dim = cfg.model.repr_dim  # output dim of the encoder

        d_intermediate = attender_cfg.d_intermediate
        nlayers = attender_cfg.nlayers
        ssm_cfg = attender_cfg.ssm_cfg
        norm_epsilon = attender_cfg.norm_epsilon
        rms_norm = attender_cfg.rms_norm
        fused_add_norm = attender_cfg.fused_add_norm
        residual_in_fp32 = attender_cfg.residual_in_fp32

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    model_dim,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(nlayers)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            model_dim, eps=norm_epsilon
        )

    def forward(self, hidden_states, lengths=None, inference_params=None, return_activations=False):
        residual = None
        activations = dict()
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            activations[f'att_mamba_l{idx}'] = hidden_states

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        activations[f'att_mamba_norm'] = hidden_states

        if return_activations:
            return hidden_states, activations
        else:
            return hidden_states
