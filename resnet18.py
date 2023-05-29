import sys
sys.path.append("~/scalehls/build/tools/scalehls/python_packages/scalehls_core/scalehls")

from scalehls.ir import BlockArgument
from scalehls.ir import BlockArgumentList
from scalehls.ir import RankedTensorType
from scalehls.dialects.linalg.opdsl.lang import *
from scalehls.dialects import _structured_transform_ops_gen
from scalehls.dialects import transform
from scalehls.dialects import linalg
from scalehls.ir import AffineMapAttr
from scalehls.ir import AffineMap
from scalehls.ir import AffineExpr, AffineExprList
from scalehls.ir import AffineCeilDivExpr, AffineModExpr, AffineDimExpr, AffineSymbolExpr
from scalehls.ir import IntegerAttr
from scalehls.ir import ArrayAttr
from scalehls.ir import F32Type, IndexType
from scalehls.ir import TypeAttr
from scalehls.ir import Location
from scalehls.ir import InsertionPoint
from scalehls.ir import Module, Context
from scalehls.dialects import hls
import scalehls
import torch
import torch.nn as nn
import torch_mlir
from scalehls.passmanager import PassManager
import io
import shutil


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


model = resnet18(pretrained=True)
model.train(False)

torch_mlir_module = torch_mlir.compile(model, torch.ones(1, 3, 224, 224),
                                       output_type="linalg-on-tensors")


ctx = Context()
scalehls.register_everything(ctx)

# Parse module from torch_mlir_module and create new insert point and location.
with ctx:
    module = Module.parse(str(torch_mlir_module))
    insert = InsertionPoint.at_block_begin(module.body)
    loc = Location.unknown()

# # Create a new "vitis" library.
# with ctx, loc, insert:
#     lib = hls.LibraryOp("vitis")
#     lib_body = lib.body.blocks.append()

# insert = InsertionPoint.at_block_begin(lib_body)
# with ctx, loc, insert:
#     gemm_ip = hls.DeclareOp("gemm")
#     gemm_ip_meta = gemm_ip.meta.blocks.append()


# insert = InsertionPoint.at_block_begin(gemm_ip_meta)
# with ctx, loc, insert:
#     include = hls.IncludeOp(
#         "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp")

#     tparam_type = hls.TypeParamType.get()
#     dtype = hls.TypeParamOp(tparam_type, ArrayAttr.get(
#         [TypeAttr.get(F32Type.get())]), "t_DataType")
#     itype = hls.TypeParamOp(tparam_type, ArrayAttr.get(
#         [TypeAttr.get(IndexType.get())]), "t_IndexType")

#     vparam_type = hls.ValueParamType.get()
#     static_kind = hls.ValueParamKindAttr.get(hls.ValueParamKind.static)
#     buffer_dim = hls.ValueParamOp(
#         vparam_type, itype, [], static_kind, "k_KBufferDim")
#     par_entries = hls.ValueParamOp(vparam_type, itype, [
#     ], static_kind, "t_ParEntries", candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), 8)]))
#     max_size_c = hls.ValueParamOp(
#         vparam_type, itype, [], static_kind, "t_MaxSizeC")

#     dynamic_kind = hls.ValueParamKindAttr.get(hls.ValueParamKind.dynamic)
#     p_m = hls.ValueParamOp(vparam_type, itype, [], dynamic_kind, "p_m")
#     p_n = hls.ValueParamOp(vparam_type, itype, [], dynamic_kind, "p_n")
#     p_k = hls.ValueParamOp(vparam_type, itype, [], dynamic_kind, "p_k")

#     p_alpha = hls.ValueParamOp(vparam_type, dtype, [], dynamic_kind, "p_alpha")
#     p_beta = hls.ValueParamOp(vparam_type, dtype, [], dynamic_kind, "p_beta")

#     port_type = hls.PortType.get()
#     layout = AffineMapAttr.get(AffineMap.get_identity(2))

#     input_direction = hls.PortDirectionAttr.get(hls.PortDirection.input)
#     p_a = hls.PortOp(port_type, dtype, [
#                      p_m, p_k], layout, input_direction, "p_a")
#     p_b = hls.PortOp(port_type, dtype, [
#                      p_k, p_n], layout, input_direction, "p_b")
#     p_c = hls.PortOp(port_type, dtype, [
#                      p_m, p_n], layout, input_direction, "p_c")

#     output_direction = hls.PortDirectionAttr.get(hls.PortDirection.output)
#     p_r = hls.PortOp(port_type, dtype, [
#                      p_m, p_n], layout, output_direction, "p_r")

#     semantics = hls.SemanticsOp([p_a, p_b, p_c], [p_r], [])
#     hls.semantics_init_args(semantics)
#     semantics_blk = semantics.body.blocks[0]
#     semantics_args = semantics_blk.arguments


# @linalg_structured_op
# def matmul_mono(
#         A=TensorDef(T, S.M, S.K),
#         B=TensorDef(T, S.K, S.N),
#         C=TensorDef(T, S.M, S.N, output=True)):
#     domain(D.m, D.n, D.k)
#     C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


# insert = InsertionPoint.at_block_begin(semantics_blk)
# with ctx, loc, insert:
#     matmul_mono(semantics_args[0], semantics_args[1], outs=[semantics_args[2]])
#     result = semantics_blk.operations[0]
#     hls.OutputOp([result], [semantics_args[3]])

with ctx:
    pm = PassManager.parse('''builtin.module(scalehls-pytorch-pipeline)''')
    pm.run(module.operation)  # type: ignore
    PassManager.parse("builtin.module(canonicalize)").run(
        module.operation)  # type: ignore

# print(module)

buf = io.StringIO()
scalehls.emit_hlscpp(module, buf)
print(buf.getvalue())
