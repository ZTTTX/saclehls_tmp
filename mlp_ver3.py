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


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


model = MLP()
model.train(False)

torch_mlir_module = torch_mlir.compile(model, torch.ones(1, 3, 32, 32),
                                       output_type="linalg-on-tensors")


from ip_register_ver3 import IPRegistration

obj = IPRegistration(torch_mlir_module)
obj.Add_Lib('vitis')
obj.Add_IP('gemm', "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp")
obj.Add_Template('t_DataType', 'type', 'float')
obj.Add_Template('t_IndexType', 'type', 'int')
obj.Add_Template('k_KBufferDim', 'para', 'int', [], 32)
obj.Add_Template('t_ParEntries', 'para', 'int', [], 2)
obj.Add_Template('t_MaxSizeC', 'para', 'int', [], 1024)
obj.Add_Port('input', 'p_m', 'para', 't_IndexType')
obj.Add_Port('input', 'p_n', 'para', 't_IndexType')
obj.Add_Port('input', 'p_k', 'para', 't_IndexType')
obj.Add_Port('input', 'alpha', 'para', 't_IndexType')
obj.Add_Port('input', 'beta', 'para', 't_IndexType')
obj.Add_Port('input', 'p_b', 'data', 't_IndexType', ['p_k', 'p_n'])
obj.Add_Port('input', 'p_a', 'data', 't_IndexType', ['p_m', 'p_k'])
obj.Add_Port('input', 'p_c', 'data', 't_IndexType', ['p_m', 'p_n'])
obj.Add_Port('output', 'p_r', 'data', 't_IndexType', ['p_m', 'p_n'])
obj.IO_Warpper()

@linalg_structured_op
def matmul_mono(
        B=TensorDef(T, S.K, S.N),
        A=TensorDef(T, S.M, S.K), 
        C=TensorDef(T, S.M, S.N, output=True)):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

module, ctx = obj.IP_Wrapper(matmul_mono, [['input','p_b'], ['input', 'p_a'], ['output', 'p_c']], [['ip_output', 'p_r']])



# insert = InsertionPoint.at_block_begin(semantics_blk)
# with ctx, loc, insert:
#     matmul_mono(semantics_args[1], semantics_args[0], outs=[semantics_args[2]])
#     result = semantics_blk.operations[0]
#     hls.SemanticsOutputOp([result], [semantics_args[3]])

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
                    param.value = IntegerAttr.get(IndexType.get(), 2)
                elif param.kind == hls.ParamKind.template:
                    param.value = IntegerAttr.get(IndexType.get(), 4)
                elif param.kind == hls.ParamKind.impl:
                    *_, param.value = param.candidates

with ctx:
    pm = PassManager.parse(
        "builtin.module(scalehls-implement-task-design-space)")
    pm.run(module.operation)  # type: ignore

# with ctx:
#     pm = PassManager()
#     scalehls.add_comprehensive_bufferize_passes(pm)
#     scalehls.add_lower_dataflow_passes(pm)
#     scalehls.add_convert_dataflow_to_func_passes(pm)
#     pm.run(module.operation)  # type: ignore

print(module)

# buf = io.StringIO()
# scalehls.emit_hlscpp(module, buf)
# print(buf.getvalue())
