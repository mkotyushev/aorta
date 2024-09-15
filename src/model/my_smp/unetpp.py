import copy
import types
import torch
import segmentation_models_pytorch_3d as smp


class Unetpp(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model = smp.UnetPlusPlus(**kwargs)
        
        del model.decoder.blocks
        
        blocks = {}
        skips = {}
        
        nblocks = 5
        
        in_channels = model.encoder.out_channels[1: ][::-1]
        skip_channels = (*in_channels[1:], 0)
        out_channels = in_channels
        
        for idx in range(nblocks):
            skips[f's_{idx+1}_{idx}'] = in_channels[-idx-1]
        
        for idx in range(nblocks):
            for jdx in range(nblocks - idx):
                depth = jdx
                layer = idx+jdx
        
                in_ = in_channels[-depth-1]
                skip_ = skip_channels[-depth-1]
                out_ = out_channels[-depth-1]
        
                if depth > 0:
                    for sdx in range(layer-depth):
                        skip_ += skips[f's_{depth}_{layer-sdx-2}']
        
                skips[f's_{depth}_{layer}'] = out_
        
                block = smp.decoders.unetplusplus.decoder.DecoderBlock(in_, skip_, out_)
                blocks[f'b_{depth}_{layer}'] = block
        
            if idx == 0:
                in_channels = (0, *in_channels[:-1])
                skip_channels = (0, *skip_channels[:-2], 0)
        
        model.decoder.blocks = torch.nn.ModuleDict(blocks)
        model.decoder.nblocks = nblocks

        
        model.heads = torch.nn.ModuleDict()
        
        for idx in range(model.decoder.depth + 1):
            model.heads[f'{idx}'] = torch.nn.Conv3d(
                24,
                24,
                3,
                1,
                1
        )
        
        def decoder_forward(self, *feats):
            xs = dict()
        
            for idx, x in enumerate(feats):
                xs[f'x_{idx}_{idx-1}'] = x
        
            for idx in range(self.nblocks):
                for jdx in range(self.nblocks - idx):
                    depth = jdx
                    layer = idx+jdx
        
                    block = self.blocks[f'b_{depth}_{layer}']
        
                    if depth == 0:
                        skip = None
                        shape = xs[f'x_{0}_{-1}'].shape
                    else:
                        skip = torch.concat([ xs[f'x_{depth}_{layer-sdx-1}'] for sdx in range(layer-depth+1) ], axis=1)
                        shape = xs[f'x_{depth}_{layer-1}'].shape
        
                    x = xs[f'x_{depth+1}_{layer}']
                    x = block(x, skip)
                    xs[f'x_{depth}_{layer}'] = x
        
            return xs
        
        def model_forward(self, x):
            f = self.encoder(x)
            xs = self.decoder(*f)
        
            out = tuple()

            for idx in range(self.decoder.nblocks):
                x = xs[f'x_{0}_{idx}']
                x = self.heads[f'{idx}'](x)
        
                out = (*out, x)
        
            if self.training:
                return out
            else:
                return x
        
        # model.decoder.nblocks = len(model.decoder.blocks)
        
        model.decoder.forward = types.MethodType(decoder_forward, model.decoder)
        model.forward = types.MethodType(model_forward, model)

        self.model = model

    def forward(self, x):
        return self.model(x)
