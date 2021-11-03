import torch
import paddle
import numpy as np


def transfer():
    input_fp = "pit_ti_730.pth"
    output_fp = "pit_ti_730.pdparams"
    torch_dict = torch.load(input_fp)
    paddle_dict = {}
    fc_names = [
        "transformers.0.blocks.0.attn.qkv.weight", "transformers.0.blocks.0.mlp.fc1.weight", "transformers.0.blocks.0.mlp.fc2.weight",\
        "transformers.0.blocks.1.attn.qkv.weight", "transformers.0.blocks.1.mlp.fc1.weight", "transformers.0.blocks.1.mlp.fc2.weight",\
        "transformers.1.blocks.0.attn.qkv.weight", "transformers.1.blocks.0.mlp.fc1.weight", "transformers.1.blocks.0.mlp.fc2.weight",\
        "transformers.1.blocks.1.attn.qkv.weight", "transformers.1.blocks.1.mlp.fc1.weight", "transformers.1.blocks.1.mlp.fc2.weight",\
        "transformers.1.blocks.2.attn.qkv.weight", "transformers.1.blocks.2.mlp.fc1.weight", "transformers.1.blocks.2.mlp.fc2.weight",\
        "transformers.1.blocks.3.attn.qkv.weight", "transformers.1.blocks.3.mlp.fc1.weight", "transformers.1.blocks.3.mlp.fc2.weight",\
        "transformers.1.blocks.4.attn.qkv.weight", "transformers.1.blocks.4.mlp.fc1.weight", "transformers.1.blocks.4.mlp.fc2.weight",\
        "transformers.1.blocks.5.attn.qkv.weight", "transformers.1.blocks.5.mlp.fc1.weight", "transformers.1.blocks.5.mlp.fc2.weight",\
        "transformers.2.blocks.0.attn.qkv.weight", "transformers.2.blocks.0.mlp.fc1.weight", "transformers.2.blocks.0.mlp.fc2.weight",\
        "transformers.2.blocks.1.attn.qkv.weight", "transformers.2.blocks.1.mlp.fc1.weight", "transformers.2.blocks.1.mlp.fc2.weight",\
        "transformers.2.blocks.2.attn.qkv.weight", "transformers.2.blocks.2.mlp.fc1.weight", "transformers.2.blocks.2.mlp.fc2.weight",\
        "transformers.2.blocks.3.attn.qkv.weight", "transformers.2.blocks.3.mlp.fc1.weight", "transformers.2.blocks.3.mlp.fc2.weight",\
        "pools.0.fc.weight", "pools.1.fc.weight", "head.weight",\
        "transformers.0.blocks.0.attn.proj.weight", "transformers.0.blocks.1.attn.proj.weight", "transformers.1.blocks.0.attn.proj.weight",\
        "transformers.1.blocks.1.attn.proj.weight", "transformers.1.blocks.2.attn.proj.weight", "transformers.1.blocks.1.attn.proj.weight", \
        "transformers.1.blocks.3.attn.proj.weight", "transformers.1.blocks.4.attn.proj.weight", "transformers.1.blocks.5.attn.proj.weight", \
        "transformers.2.blocks.0.attn.proj.weight", "transformers.2.blocks.1.attn.proj.weight", "transformers.2.blocks.2.attn.proj.weight",\
        "transformers.2.blocks.3.attn.proj.weight",

    ]
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        # print(key)
        # key_ends = key.split('.')[-1]
        # if key_ends == 'running_mean':
        #     key_starts = key.split('running_mean')[0]
        #     key = key_starts + '_mean'
        # if key_ends == 'running_var':
        #     key_starts = key.split('running_var')[0]
        #     key = key_starts + '_variance'
        flag = [i in key for i in fc_names]
        if any(flag):
            print("{} need to be trans".format(key))
            weight = weight.transpose()
        print(key)
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)


transfer()
