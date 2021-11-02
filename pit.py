# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import paddle
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from einops import rearrange
from paddle import nn
import math

from functools import partial
# from vision_transformer import trunc_normal_
from vision_transformer import Block as transformer_block
# from timm.models.registry import register_model

trunc_normal_ = TruncatedNormal(std=.02)

class Transformer(nn.Layer):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.LayerList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.LayerList([
            transformer_block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, epsilon=1e-6)
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        b, c, h, w= x.shape[0:4]
        # print('before : ', x.detach())
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape([0, h * w, -1])
        # print('after : ', x.detach())

        token_length = cls_tokens.shape[1]
        x = paddle.concat([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x)
        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        # print('before : ', x.detach())
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = x.transpose([0, 2, 1])
        x = x.reshape([0, c, h, w])
        # print('after : ', x.detach())
        return x, cls_tokens


class conv_head_pooling(nn.Layer):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2D(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Layer):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias_attr=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Layer):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        # self.pos_embed = nn.Parameter(
        #     paddle.randn([1, base_dims[0] * heads[0], width, width]),
        #     requires_grad=True
        # )
        tmp = paddle.randn([1, base_dims[0] * heads[0], width, width])
        self.pos_embed = paddle.create_parameter(shape=tmp.shape,
                        dtype=str(tmp.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(tmp))
        self.pos_embed.stop_gradient = False

        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        # self.cls_token = nn.Parameter(
        #     paddle.randn([1, 1, base_dims[0] * heads[0]]),
        #     requires_grad=True
        # )
        tmp1 = paddle.randn([1, 1, base_dims[0] * heads[0]])
        self.cls_token = paddle.create_parameter(
            shape=tmp1.shape,
            dtype=str(tmp1.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(tmp1))
        self.cls_token.stop_gradient = False

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.LayerList([])
        self.pools = nn.LayerList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], epsilon=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            m.bias.set_value(paddle.zeros_like(m.bias))
            m.weight.set_value(paddle.ones_like(m.weight))

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1])
        # print('x.shape = ', x.shape, ', x = ', x.detach())
        # print('cls_tokens.shape = ', cls_tokens.shape, ', cls_tokens = ', cls_tokens.detach())
        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            # print('x = ', x.detach())
            # print('cls_tokens = ', cls_tokens.detach())
            x, cls_tokens = self.pools[stage](x, cls_tokens)
            # print('x = ', x.detach())
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
        cls_tokens = self.norm(cls_tokens)
        # print(cls_tokens.detach())
        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(
            paddle.randn([1, 2, self.base_dims[0] * self.heads[0]]),
            requires_grad=True)
        if self.num_classes > 0:
            self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1],
                                       self.num_classes)
        else:
            self.head_dist = nn.Identity()

        trunc_normal_(self.cls_token)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2

# @register_model
def pit_b(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_b_820.pth')
        model.load_state_dict(state_dict)
    return model

# @register_model
def pit_s(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_s_809.pth')
        model.load_state_dict(state_dict)
    return model


# @register_model
def pit_xs(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_xs_781.pth')
        model.load_state_dict(state_dict)
    return model

# @register_model
def pit_ti(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_ti_730.pth')
        model.load_state_dict(state_dict)
    return model


# @register_model
def pit_b_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_b_distill_840.pth')
        model.load_state_dict(state_dict)
    return model


# @register_model
def pit_s_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_s_distill_819.pth')
        model.load_state_dict(state_dict)
    return model


# @register_model
def pit_xs_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_xs_distill_791.pth')
        model.load_state_dict(state_dict)
    return model


# @register_model
def pit_ti_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        paddle.load('weights/pit_ti_distill_746.pth')
        model.load_state_dict(state_dict)
    return model
