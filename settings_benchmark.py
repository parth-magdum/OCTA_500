from models.frnet import *
from models.csnet import *
from models.attentionunet import *
from models.unetppp import *

class ObjectCreator:
    def __init__(self, args, cls) -> None:
        self.args = args
        self.cls_net = cls
    def __call__(self):
        return self.cls_net(**self.args)


models = {
    "FRNet-base": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock
    )),
    "FRNet": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock
    )),
    "AttUNet": ObjectCreator(cls=AttUNet, args=dict(
        in_channels=1, n_classes=1, channels=32, is_deconv=True, is_batchnorm=True
    )),
    "UNetppp": ObjectCreator(cls=UNetppp, args=dict(
        in_channels=1, n_classes=1, channels=32, is_deconv=True, is_batchnorm=True
    )),
    "CSNet": ObjectCreator(cls=CSNet, args=dict(
        in_channels=1, n_classes=1
    )),
    # More models can be added here......
}
