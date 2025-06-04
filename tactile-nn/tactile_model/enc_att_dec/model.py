from . import encoder as enc
from . import attender as att
from . import decoder as dec
from torch import nn


class EncAttDec(nn.Module):
    def __init__(self, cfg):
        super(EncAttDec, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data

        self.encoder_cfg = cfg.model.encoder
        self.attender_cfg = cfg.model.get('attender', None)  # if attender is null in the yaml, then set it to None
        self.decoder_cfg = cfg.model.get('decoder', None)  # if decoder is null in the yaml, then set it to None

        assert self.encoder_cfg is not None, 'the encoder needs to be specified'

        self.encoder = self._init_module(module_cfg=self.encoder_cfg, lib=enc)
        # maps (bs, t, C, H, W) ---> (bs, t, d) or (bs, C, H, W) ---> (bs, d)
        self.attender = self._init_module(module_cfg=self.attender_cfg, lib=att)
        # maps (bs, t, d) ---> (bs, t, d) or identity [e.g., (bs, d) ---> (bs, d)]
        self.decoder = self._init_module(module_cfg=self.decoder_cfg, lib=dec)
        # maps (bs, t, d) ---> (bs, num_classes) or identity [e.g., (bs, d) ---> (bs, d)]

    def _init_module(self, module_cfg, lib):
        if module_cfg is None:
            module = nn.Identity()
        else:
            assert "name" in module_cfg  # the module_cfg is just to specify the class name
            module = getattr(lib, module_cfg.name)(cfg=self.cfg)
        return module

    def forward(self, x, return_activations=False):
        if return_activations:
            # encode, (bs, t, C, H, W) ---> (bs, t, d) or (bs, C, H, W) ---> (bs, d)
            x, enc_activations = self.encoder(x, return_activations=True)

            # attend, (bs, t, d) ---> (bs, t, d) or identity [e.g., (bs, d) ---> (bs, d)]
            if isinstance(self.attender, nn.Identity):
                x = self.attender(x)
                att_activations = {}
            else:
                x, att_activations = self.attender(x, return_activations=True)

            # decode/predict, (bs, t, d) ---> (bs, num_classes) or identity [e.g., (bs, d) ---> (bs, d)]
            if isinstance(self.decoder, nn.Identity):
                pred = self.decoder(x)
                dec_activations = {}
            else:
                pred, dec_activations = self.decoder(x, return_activations=True)
            return pred, {'enc': enc_activations, 'att': att_activations, 'dec': dec_activations}
        else:
            # encode, (bs, t, C, H, W) ---> (bs, t, d) or (bs, C, H, W) ---> (bs, d)
            x = self.encoder(x)
            # attend, (bs, t, d) ---> (bs, t, d) or identity [e.g., (bs, d) ---> (bs, d)]
            x = self.attender(x)
            # decode/predict, (bs, t, d) ---> (bs, num_classes) or identity [e.g., (bs, d) ---> (bs, d)]
            pred = self.decoder(x)

            return pred
