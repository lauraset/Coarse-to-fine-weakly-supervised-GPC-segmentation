""" Bring-Your-Own-Attention Network

A flexible network w/ dataclass based config for stacking NN blocks including
self-attention (or similar) layers.

Currently used to implement experimential variants of:
  * Bottleneck Transformers
  * Lambda ResNets
  * HaloNets

Consider all of the models definitions here as experimental WIP and likely to change.

Hacked together by / copyright Ross Wightman, 2021.
"""
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .byobnet import ByoBlockCfg, ByoModelCfg, ByobNet, interleave_blocks
from .helpers import build_model_with_cfg
from .registry import register_model

__all__ = []


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1.conv', 'classifier': 'head.fc',
        'fixed_input_size': False, 'min_input_size': (3, 224, 224),
        **kwargs
    }


default_cfgs = {
    # GPU-Efficient (ResNet) weights
    'botnet26t_256': _cfg(
        url='',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'botnet50ts_256': _cfg(
        url='',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),
    'eca_botnext26ts_256': _cfg(
        url='',
        fixed_input_size=True, input_size=(3, 256, 256), pool_size=(8, 8)),

    'halonet_h1': _cfg(url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'halonet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/halonet26t_256-9b4bf0b3.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'sehalonet33ts': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/sehalonet33ts_256-87e053f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256), crop_pct=0.94),
    'halonet50ts': _cfg(
        url='', input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),
    'eca_halonext26ts': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/eca_halonext26ts_256-1e55880b.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), min_input_size=(3, 256, 256)),

    'lambda_resnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/lambda_resnet26t_256-b040fce6.pth',
        min_input_size=(3, 128, 128), input_size=(3, 256, 256), pool_size=(8, 8)),
}


model_cfgs = dict(

    botnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        fixed_input_size=True,
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    botnet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=4, d=4, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        fixed_input_size=True,
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),
    eca_botnext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        fixed_input_size=True,
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='bottleneck',
        self_attn_kwargs=dict()
    ),

    halonet_h1=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='self_attn', d=3, c=64, s=1, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=3, c=128, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=10, c=256, s=2, gs=0, br=1.0),
            ByoBlockCfg(type='self_attn', d=3, c=512, s=2, gs=0, br=1.0),
        ),
        stem_chs=64,
        stem_type='7x7',
        stem_pool='maxpool',

        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3),
    ),
    halonet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2, dim_head=16)
    ),
    sehalonet33ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), every=[2], d=3, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg('self_attn', d=2, c=1536, s=2, gs=0, br=0.333),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='',
        act_layer='silu',
        num_features=1280,
        attn_layer='se',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3)
    ),
    halonet50ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=3, c=256, s=1, gs=0, br=0.25),
            interleave_blocks(
                types=('bottle', 'self_attn'), every=4, d=4, c=512, s=2, gs=0, br=0.25,
                self_attn_layer='halo', self_attn_kwargs=dict(block_size=8, halo_size=3, num_heads=4)),
            interleave_blocks(types=('bottle', 'self_attn'), d=6, c=1024, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=3, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=3)
    ),
    eca_halonext26ts=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=16, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=16, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=16, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=16, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        act_layer='silu',
        attn_layer='eca',
        self_attn_layer='halo',
        self_attn_kwargs=dict(block_size=8, halo_size=2, dim_head=16)
    ),

    lambda_resnet26t=ByoModelCfg(
        blocks=(
            ByoBlockCfg(type='bottle', d=2, c=256, s=1, gs=0, br=0.25),
            ByoBlockCfg(type='bottle', d=2, c=512, s=2, gs=0, br=0.25),
            interleave_blocks(types=('bottle', 'self_attn'), d=2, c=1024, s=2, gs=0, br=0.25),
            ByoBlockCfg(type='self_attn', d=2, c=2048, s=2, gs=0, br=0.25),
        ),
        stem_chs=64,
        stem_type='tiered',
        stem_pool='maxpool',
        self_attn_layer='lambda',
        self_attn_kwargs=dict(r=9)
    ),
)


def _create_byoanet(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)


@register_model
def botnet26t_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet26-T backbone.
    NOTE: this isn't performing well, may remove
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet26t_256', 'botnet26t', pretrained=pretrained, **kwargs)


@register_model
def botnet50ts_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet50-T backbone, silu act.
    NOTE: this isn't performing well, may remove
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('botnet50ts_256', 'botnet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_botnext26ts_256(pretrained=False, **kwargs):
    """ Bottleneck Transformer w/ ResNet26-T backbone, silu act.
    NOTE: this isn't performing well, may remove
    """
    kwargs.setdefault('img_size', 256)
    return _create_byoanet('eca_botnext26ts_256', 'eca_botnext26ts', pretrained=pretrained, **kwargs)


@register_model
def halonet_h1(pretrained=False, **kwargs):
    """ HaloNet-H1. Halo attention in all stages as per the paper.
    NOTE: This runs very slowly!
    """
    return _create_byoanet('halonet_h1', pretrained=pretrained, **kwargs)


@register_model
def halonet26t(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet26-t backbone. Halo attention in final two stages
    """
    return _create_byoanet('halonet26t', pretrained=pretrained, **kwargs)


@register_model
def sehalonet33ts(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet33-t backbone, SE attn for non Halo blocks, SiLU, 1-2 Halo in stage 2,3,4.
    """
    return _create_byoanet('sehalonet33ts', pretrained=pretrained, **kwargs)


@register_model
def halonet50ts(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet50-t backbone, silu act. Halo attention in final two stages
    """
    return _create_byoanet('halonet50ts', pretrained=pretrained, **kwargs)


@register_model
def eca_halonext26ts(pretrained=False, **kwargs):
    """ HaloNet w/ a ResNet26-t backbone, silu act. Halo attention in final two stages
    """
    return _create_byoanet('eca_halonext26ts', pretrained=pretrained, **kwargs)


@register_model
def lambda_resnet26t(pretrained=False, **kwargs):
    """ Lambda-ResNet-26T. Lambda layers in last two stages.
    """
    return _create_byoanet('lambda_resnet26t', pretrained=pretrained, **kwargs)
