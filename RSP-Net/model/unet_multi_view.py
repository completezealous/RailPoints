import functools
import warnings
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.core import ConvAlgo
from collections import OrderedDict
from torch_scatter import scatter_mean
from model.Bi_GRU_attention import SphereFormer

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, dilation=1,padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, dilation=1,padding=1, bias=False, indice_key=indice_key)
        )


    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output

class LearnableWeightFeatureFusionModule(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cnn_fc = nn.Linear(output_dim, output_dim)
        self.transformer_fc = nn.Linear(output_dim, output_dim)
        self.weight_fc = nn.Linear(output_dim, 1)
        self.layer_norm=nn.LayerNorm(output_dim)

    def forward(self, cnn_features, transformer_features):
        cnn_output = self.cnn_fc(cnn_features)#(N,C)

        transformer_output = self.transformer_fc(transformer_features)
        add_features=torch.add(cnn_output,transformer_output)


        attention_weights = torch.softmax(self.weight_fc(add_features),1)


        fused_features =  (1.0+attention_weights) *(cnn_features+transformer_features)

        fused_features=self.layer_norm(fused_features)
        return fused_features

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features#
        self.fc1 = nn.Linear(in_features, hidden_features) #
        self.act = act_layer()#
        self.fc2 = nn.Linear(hidden_features, hidden_features)#
        self.drop = nn.Dropout(drop, inplace=True)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)#[N,C]
        x = self.drop(x)
        x=self.fc2(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc3(x)
        x=self.drop(x)


        return x


class SparseConv3dToFormer(nn.Module):

    def __init__(self, fusion_dim):
        super().__init__()
        self.fusion_dim = fusion_dim


        self.cnn_projection = nn.Linear(fusion_dim, fusion_dim)
        self.transformer_projection = nn.Linear(fusion_dim, fusion_dim)

        self.att_dropout = nn.Dropout(0.5,inplace=True)

        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=2)


        self.layer_norm1=nn.LayerNorm(fusion_dim)
        self.layer_norm2=nn.LayerNorm(fusion_dim)

    def forward(self, cnn_features, transformer_features):
        cnn_proj = self.cnn_projection(cnn_features)
        cnn_proj=self.layer_norm1(cnn_proj)
        transformer_proj = self.transformer_projection(transformer_features)
        transformer_proj=self.layer_norm1(transformer_proj)

        key = cnn_proj.unsqueeze(0)
        value = cnn_proj.unsqueeze(0)
        query = transformer_proj.unsqueeze(0)

        attention_output, _ = self.cross_attention(query, key, value)

        fused_features = transformer_features+self.att_dropout(attention_output.squeeze(0))
        fused_features=fused_features+self.att_dropout(self.layer_norm2(fused_features))
        fused_features=fused_features+self.att_dropout((self.layer_norm2(fused_features)))

        return fused_features

class FormerToSparseConv3d(nn.Module):
    def __init__(self, fusion_dim):
        super().__init__()
        self.fusion_dim = fusion_dim


        self.cnn_projection = nn.Linear(fusion_dim, fusion_dim)
        self.transformer_projection = nn.Linear(fusion_dim, fusion_dim)

        self.att_dropout = nn.Dropout(0.5,inplace=True)

        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=2)

        self.layer_norm1=nn.LayerNorm(fusion_dim)
        self.layer_norm2=nn.LayerNorm(fusion_dim)

    def forward(self, cnn_features, transformer_features):

        cnn_proj = self.cnn_projection(cnn_features)
        cnn_proj=self.layer_norm1(cnn_proj)
        transformer_proj = self.transformer_projection(transformer_features)
        transformer_proj=self.layer_norm2(transformer_proj)


        key = transformer_proj.unsqueeze(0)
        value = transformer_proj.unsqueeze(0)
        query = cnn_proj.unsqueeze(0)

        attention_output, _ = self.cross_attention(query, key, value)


        fused_features = cnn_features+self.att_dropout(attention_output.squeeze(0))
        fused_features=fused_features+self.att_dropout(self.layer_norm2(fused_features))
        fused_features = fused_features + self.att_dropout((self.layer_norm2(fused_features)))

        return fused_features
def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next


class UBlock(nn.Module):  # 构建具有多层块和 Transformer 结构的神经网络
    def __init__(self, nPlanes,  # 一个列表，表示每个块的输入和输出通道数 [32, 64, 128, 256, 256]
                 norm_fn,
                 block_reps,  # 表示每个块要重复的次数。
                 block,  # 表示块的类型。
                 window_size,
                 window_size_sphere,
                 quant_size,  # 表示每个块的输入和输出的网格尺寸。
                 quant_size_sphere,
                 head_dim=16,  # 表示 Transformer 中每个头的维度。
                 window_size_scale=[2.0, 2.0],
                 rel_query=True,
                 rel_key=True,
                 rel_value=True,
                 drop_path=0.0,
                 indice_key_id=1,
                 grad_checkpoint_layers=[],  # 表示要应用梯度检查点的层。
                 sphere_layers=[1, 2, 3, 4],  # 表示应用 SphereFormer 模块的层。
                 a=0.05 * 0.25,
                 ):

        super().__init__()
        # 如果 nPlanes 的长度大于 1，表示有下一级的块，构建了一个下采样块 self.u 和上采样块 self.deconv，然后通过 blocks_tail 构建了一系列尾部块。
        self.nPlanes = nPlanes
        print('nPlanes', nPlanes)
        self.indice_key_id = indice_key_id
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.sphere_layers = sphere_layers

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(
            blocks)  # 用于构建多层块。它接收了一些参数，包括输入通道数 nPlanes[0]、输出通道数 nPlanes[0]、块的重复次数 block_reps、块的类型 block、以及块的索引键 indice_key。

        if indice_key_id in sphere_layers:
            self.window_size = window_size
            self.window_size_sphere = window_size_sphere
            num_heads = nPlanes[0] // head_dim
            self.transformer_block = SphereFormer(
                nPlanes[0],  # 表示输入和输出的通道数。[32, 64, 128, 256, 256]
                num_heads,
                window_size,
                window_size_sphere,
                quant_size,
                quant_size_sphere,
                indice_key='sphereformer{}'.format(indice_key_id),
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[0],
                a=a,
            )
        self.fusion_model=LearnableWeightFeatureFusionModule(nPlanes[0])
        self.sparseconv3dtoformer=SparseConv3dToFormer(nPlanes[0])
        self.formertosparseconv3d=FormerToSparseConv3d(nPlanes[0])

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2,dilation=1, bias=False,
                                   indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array([
                window_size[0] * window_size_scale_cubic,
                window_size[1] * window_size_scale_cubic,
                window_size[2] * window_size_scale_cubic
            ])
            quant_size_next = np.array([
                quant_size[0] * window_size_scale_cubic,
                quant_size[1] * window_size_scale_cubic,
                quant_size[2] * window_size_scale_cubic
            ])
            window_size_sphere_next = np.array([
                window_size_sphere[0] * window_size_scale_sphere,
                window_size_sphere[1] * window_size_scale_sphere,
                window_size_sphere[2]
            ])
            quant_size_sphere_next = np.array([
                quant_size_sphere[0] * window_size_scale_sphere,
                quant_size_sphere[1] * window_size_scale_sphere,
                quant_size_sphere[2]
            ])
            self.u = UBlock(nPlanes[1:],  # 构建下一级块
                            norm_fn,
                            block_reps,
                            block,
                            window_size_next,
                            window_size_sphere_next,
                            quant_size_next,
                            quant_size_sphere_next,
                            window_size_scale=window_size_scale,
                            rel_query=rel_query,
                            rel_key=rel_key,
                            rel_value=rel_value,
                            drop_path=drop_path[1:],
                            indice_key_id=indice_key_id + 1,
                            grad_checkpoint_layers=grad_checkpoint_layers,
                            sphere_layers=sphere_layers,
                            a=a
                            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2,bias=False,
                                           indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn,
                                                         indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)

            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, inp_0,inp_1,xyz, batch):

        assert (inp_0.indices[:, 0] == batch).all()
        assert (inp_1.indices[:, 0] == batch).all()

        output = self.blocks(inp_0)


        # transformer

        if self.indice_key_id in self.sphere_layers:
            if self.indice_key_id in self.grad_checkpoint_layers:
                def run(feats_, xyz_, batch_):
                    return self.transformer_block(feats_, xyz_, batch_)


                transformer_features = torch.utils.checkpoint.checkpoint(run, inp_1.features+output.features, xyz, batch)

            else:

                transformer_features = self.transformer_block(inp_1.features+output.features, xyz, batch)


            globalTolocal_features = self.formertosparseconv3d(output.features, transformer_features)
            localToglobal_features = self.sparseconv3dtoformer(output.features, transformer_features)



            fusion_features = self.fusion_model(globalTolocal_features+output.features,localToglobal_features+transformer_features)

            output = output.replace_feature(fusion_features)



        # downsample
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)#残差块中的跳跃连接

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)#下采样
            globalTolocal = output.replace_feature(globalTolocal_features)
            output_decoder_3dcnn = self.conv(globalTolocal)
            localToglobal = output.replace_feature(localToglobal_features)
            output_decoder_former = self.conv(localToglobal)


            # downsample
            indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)

            output_decoder= self.u(output_decoder_3dcnn,output_decoder_former,xyz_next, batch_next.long())

            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        return output


class Semantic(nn.Module):
    def __init__(self,
                 input_c,  # 表示输入的通道数。4
                 m,  # 表示 U 形块的输出通道数。32
                 classes,
                 block_reps,
                 block_residual,
                 layers,
                 window_size,
                 window_size_sphere,
                 quant_size,
                 quant_size_sphere,
                 rel_query=True,
                 rel_key=True,
                 rel_value=True,
                 drop_path_rate=0.0,
                 window_size_scale=2.0,
                 grad_checkpoint_layers=[],
                 sphere_layers=[1, 2, 3, 4, 5],
                 a=0.05 * 0.25,

                 ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                7)]

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, dilation=1,padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock(layers,
                           norm_fn,  # 表示归一化函数。
                           block_reps,  # 2
                           block,  # 表示块的类型。
                           window_size,
                           window_size_sphere,
                           quant_size,
                           quant_size_sphere,
                           window_size_scale=window_size_scale,
                           rel_query=rel_query,
                           rel_key=rel_key,
                           rel_value=rel_value,
                           drop_path=dpr,
                           indice_key_id=1,
                           grad_checkpoint_layers=grad_checkpoint_layers,
                           sphere_layers=sphere_layers,
                           a=a,
                           )

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )


        self.linear = nn.Linear(m, classes)

        self.apply(self.set_bn_init)


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, xyz, batch):

        output = self.input_conv(input)

        output = self.unet(output,output,xyz, batch)

        output = self.output_layer(output)

        semantic_scores = self.linear(output.features)  # (N, nClass), float
        return semantic_scores

