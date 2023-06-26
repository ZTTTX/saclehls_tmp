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
from scalehls.ir import IntegerAttr, StringAttr, FloatAttr
from scalehls.ir import ArrayAttr
from scalehls.ir import F32Type, IndexType
from scalehls.ir import TypeAttr
from scalehls.ir import Location
from scalehls.ir import InsertionPoint
from scalehls.ir import Module, Context
from scalehls.ir import Operation
from scalehls.dialects import hls
import scalehls
import torch
import torch.nn as nn
import torch_mlir
from scalehls.passmanager import PassManager
import io
import shutil

import random
random.seed(123)


class ConvMLP(nn.Module):
    def __init__(self):
        super(ConvMLP, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = ConvMLP()
model.train(False)

torch_mlir_module = torch_mlir.compile(model, torch.ones(1, 1, 32, 32),
                                       output_type="linalg-on-tensors")


# from ip_register_ver5_struct import IPRegistration

# obj = IPRegistration(torch_mlir_module)
# obj.Add_Lib('vitis')
# obj.Add_IP('gemm', "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp")
# obj.Add_Template('t_DataType', 'type', 'float')
# obj.Add_Template('t_IndexType', 'type', 'int')
# obj.Add_Template('k_KBufferDim', 'para', 'int', [], [32])
# obj.Add_Template('t_ParEntries', 'para', 'int', [], [2, 4])
# obj.Add_Template('t_MaxSizeC', 'para', 'int', [], [1024])
# obj.Add_Port('input', 'p_m', 'para', 't_IndexType')
# obj.Add_Port('input', 'p_n', 'para', 't_IndexType')
# obj.Add_Port('input', 'p_k', 'para', 't_IndexType')
# obj.Add_Port('input', 'alpha', 'para', 't_DataType',[] ,1)
# obj.Add_Port('input', 'beta', 'para', 't_DataType',[] ,0)
# obj.Add_Port('input', 'p_a', 'data', 't_DataType', ['p_m', 'p_k'])
# obj.Add_Port('input', 'p_b', 'data', 't_DataType', ['p_k', 'p_n'])
# obj.Add_Port('input', 'p_c', 'data', 't_DataType', ['p_m', 'p_n'])
# obj.Add_Port('output', 'p_r', 'data', 't_DataType', ['p_m', 'p_n'])

# obj.Add_Template('dummy_type', 'type', 'float', [], None, False)
# obj.Add_Template('dummy_int', 'para', 'int', [], [15], False)
# obj.Add_Struct('my_struct', ['dummy_type', 'dummy_int', 't_IndexType', 't_ParEntries'], True)
# obj.IO_Warpper()


# @linalg_structured_op
# def matmul_mono(
#         A=TensorDef(T, S.M, S.K), 
#         B=TensorDef(T, S.K, S.N),
#         C=TensorDef(T, S.M, S.N, output=True)):
#     domain(D.m, D.n, D.k)
#     C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

# module, ctx = obj.IP_Wrapper(matmul_mono, [['input','p_a'], ['input', 'p_b'], ['output', 'p_c']], [['ip_output', 'p_r']])


#===================================================================================================
from ip_register_ver5_struct import IPRegistration

obj = IPRegistration(torch_mlir_module)

obj.Add_Lib('hls4ml')
obj.Add_IP('conv_2d_cl', "src/nnet_conv2d.h")
#add templates that will be in the strcut
obj.Add_Template('bias_t', 'type', 'float', [], None, False)
obj.Add_Template('weight_t', 'type', 'float', [], None, False)
obj.Add_Template('accum_t', 'type', 'float', [], None, False)
obj.Add_Template('pad_top', 'para', 'int', [], [0], False)
obj.Add_Template('pad_bottom', 'para', 'int', [], [0], False)
obj.Add_Template('pad_left', 'para', 'int', [], [0], False)
obj.Add_Template('pad_right', 'para', 'int', [], [0], False)
obj.Add_Template('in_height', 'para', 'int', [], [32], False)
obj.Add_Template('in_width', 'para', 'int', [], [32], False)
obj.Add_Template('n_chan', 'para', 'int', [], [1], False)
obj.Add_Template('filt_height', 'para', 'int', [], [3], False)
obj.Add_Template('filt_width', 'para', 'int', [], [3], False)
obj.Add_Template('n_filt', 'para', 'int', [], [16], False)
obj.Add_Template('stride_height', 'para', 'int', [], [1], False)
obj.Add_Template('stride_width', 'para', 'int', [], [1], False)
obj.Add_Template('out_height', 'para', 'int', [], [30], False)
obj.Add_Template('out_width', 'para', 'int', [], [30], False)
obj.Add_Template('dilation_height', 'para', 'int', [], [1], False)
obj.Add_Template('dilation_width', 'para', 'int', [], [1], False)
obj.Add_Template('reuse_factor', 'para', 'int', [], [1], False)
# obj.Add_Template('store_weights_in_bram', 'para', 'bool', [], [False], False)
obj.Add_Template('n_zeros', 'para', 'int', [], [0], False)

#from strcut
obj.Add_Struct('conv2d_config', ['bias_t', 'weight_t', 'accum_t', 'pad_top', 'pad_bottom', 'pad_left', 'pad_right', 'in_height', 'in_width', 'n_chan', 'filt_height', 'filt_width', 'n_filt', 'stride_height', 'stride_width', 'out_height', 'out_width', 'dilation_height', 'dilation_width', 'reuse_factor', 'n_zeros'], True)

#Need dummy size for the port size
obj.Add_Template('data_size', 'para', 'int', [], [32*32*1], False)
obj.Add_Template('result_size', 'para', 'int', [], [30*30*16], False)
obj.Add_Template('weight_size', 'para', 'int', [], [3*3*1*16], False)

#add templates and ports needed in the actuall call
obj.Add_Template('data_T', 'type', 'float')
obj.Add_Template('res_T', 'type', 'float')
obj.Add_Port('input', 'data', 'data', 'data_T', ['data_size'])
obj.Add_Port('input', 'result', 'data', 'res_T', ['result_size'])
obj.Add_Port('input', 'weights', 'data', 'weight_t', ['weight_size'])
obj.Add_Port('input', 'biases', 'data', 'bias_t', ['n_filt'])
obj.IO_Warpper()

@linalg_structured_op
def conv_2d(
        input=TensorDef(T, S.H_in, S.W_in, S.C),
        weights=TensorDef(T, S.K_h, S.K_w, S.C, S.K_out),
        output=TensorDef(T, S.H_out, S.W_out, S.K_out, output=True)):
    domain(D.in_h, D.wt_h, D.in_w, D.wt_w, D.in_c, D.out_k)
    # domain(D.in_h[D.wt_h, D.wt_c], D.in_w[D.wt_w, D.wt_c], D.in_c, D.out_k)
    output[D.in_h, D.in_w, D.out_k] += input[D.in_h + D.wt_h, D.in_w + D.wt_w, D.in_c] * weights[D.wt_h, D.wt_w, D.in_c, D.out_k]
    

module, ctx = obj.IP_Wrapper(conv_2d, [['input','data'], ['input', 'weights']], [['ip_output', 'result']])


#===================================================================================================

# ctx = Context()
# scalehls.register_everything(ctx)

# with ctx:
#     module = Module.parse(str(torch_mlir_module))
#     insert = InsertionPoint.at_block_begin(module.body)
#     loc = Location.unknown()

#===================================================================================================
with ctx:
    pm = PassManager()
    scalehls.add_linalg_transform_passes(pm)
    scalehls.add_convert_linalg_to_dataflow_passes(pm)
    scalehls.add_generate_design_space_passes(pm)
    pm.run(module.operation)  # type: ignore

for space in module.body.operations:
    if (isinstance(space, hls.SpaceOp)):
        params = []

        def get_params(op: Operation):
            param = op.opview
            if (isinstance(param, hls.ParamOp)):
                params.append(param)

        scalehls.walk_operation(space.operation, get_params)
        with ctx:
            for param in params:
                if param.kind == hls.ParamKind.tile:
                    # For now, we always set tile size to 0.
                    param.value = IntegerAttr.get(IndexType.get(), 0)
                elif param.kind == hls.ParamKind.parallel:
                    param.value = IntegerAttr.get(IndexType.get(), 0)
                elif param.kind == hls.ParamKind.template:
                    param.value = IntegerAttr.get(IndexType.get(), random.randint(0, 10))
                elif param.kind == hls.ParamKind.impl:
                    *_, param.value = param.candidates

with ctx:
    pm = PassManager.parse(
        "builtin.module(scalehls-implement-task-design-space)")
    pm.run(module.operation)  # type: ignore

with ctx:
    pm = PassManager()
    scalehls.add_comprehensive_bufferize_passes(pm)
    scalehls.add_lower_dataflow_passes(pm)
    scalehls.add_convert_dataflow_to_func_passes(pm)
    pm.run(module.operation)  # type: ignore

# print(module)

buf = io.StringIO()
scalehls.emit_hlscpp(module, buf)
print(buf.getvalue())
