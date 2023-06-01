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
obj.Add_Port('input', 'alpha', 'para', 't_DataType',[] ,1)
obj.Add_Port('input', 'beta', 'para', 't_DataType',[] ,0)
obj.Add_Port('input', 'p_b', 'data', 't_DataType', ['p_k', 'p_n'])
obj.Add_Port('input', 'p_a', 'data', 't_DataType', ['p_m', 'p_k'])
obj.Add_Port('input', 'p_c', 'data', 't_DataType', ['p_m', 'p_n'])
obj.Add_Port('output', 'p_r', 'data', 't_DataType', ['p_m', 'p_n'])
obj.IO_Warpper()

@linalg_structured_op
def matmul_mono(
        A=TensorDef(T, S.M, S.K), 
        B=TensorDef(T, S.K, S.N),
        C=TensorDef(T, S.M, S.N, output=True)):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

module, ctx = obj.IP_Wrapper(matmul_mono, [['input','p_a'], ['input', 'p_b'], ['output', 'p_c']], [['ip_output', 'p_r']])


# #Dummy lib and ip test
# obj.Add_Lib('vitis_DUMMY')
# obj.Add_IP('gemm_DUMMY', "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm_DUMMY.hpp")
# obj.Add_Template('t_DataType_1', 'type', 'float')
# obj.Add_Template('t_IndexType_1', 'type', 'int')
# obj.Add_Template('k_KBufferDim_1', 'para', 'int', [], 32)
# obj.Add_Template('t_ParEntries_1', 'para', 'int', [], 2)
# obj.Add_Template('t_MaxSizeC_1', 'para', 'int', [], 1024)
# obj.Add_Port('input', 'p_m_1', 'para', 't_IndexType_1')
# obj.Add_Port('input', 'p_n_1', 'para', 't_IndexType_1')
# obj.Add_Port('input', 'p_k_1', 'para', 't_IndexType_1')
# obj.Add_Port('input', 'alpha_1', 'para', 't_DataType_1')
# obj.Add_Port('input', 'beta_1', 'para', 't_DataType_1')
# obj.Add_Port('input', 'p_b_1', 'data', 't_DataType_1', ['p_k_1', 'p_n_1'])
# obj.Add_Port('input', 'p_a_1', 'data', 't_DataType_1', ['p_m_1', 'p_k_1'])
# obj.Add_Port('input', 'p_c_1', 'data', 't_DataType_1', ['p_m_1', 'p_n_1'])
# obj.Add_Port('output', 'p_r_1', 'data', 't_DataType_1', ['p_m_1', 'p_n_1'])
# obj.IO_Warpper()

# @linalg_structured_op
# def matmul_mono(
#         A=TensorDef(T, S.M, S.K), 
#         B=TensorDef(T, S.K, S.N),
#         C=TensorDef(T, S.M, S.N, output=True)):
#     domain(D.m, D.n, D.k)
#     C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

# module, ctx = obj.IP_Wrapper(matmul_mono, [['input','p_a_1'], ['input', 'p_b_1'], ['output', 'p_c_1']], [['ip_output', 'p_r_1']])


# obj.Add_IP('scal', 'Vitis_Libraries/blas/L1/include/hw/xf_blas/scal.hpp')
# obj.Add_Template('t_DataType', 'type', 'float')
# obj.Add_Template('t_IndexType', 'type', 'int')
# obj.Add_Template('t_ParEntries', 'para', 'int', [], 2)
# obj.Add_Port('input', 'p_n', 'para', 't_IndexType')
# obj.Add_Port('input', 'p_alpha', 'data', 't_DataType', ['0'])
# obj.Add_Port('input', 'p_x', 'data', 't_DataType', ['p_n'])
# obj.Add_Port('output', 'p_res', 'data', 't_DataType', ['p_n'])
# obj.IO_Warpper()

# @linalg_structured_op
# def copy_and_scale(p_alpha=ScalarDef(T),
#                    I=TensorDef(T, S.N),
#                    O=TensorDef(T, S.N, output=True)):
#   O[D.n] = I[D.n] * p_alpha

# module, ctx = obj.IP_Wrapper(copy_and_scale, [['input', 'p_alpha'], ['input', 'p_x'], ['output', 'p_res']], [['ip_output', 'p_res']])



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

with ctx:
    pm = PassManager()
    scalehls.add_comprehensive_bufferize_passes(pm)
    scalehls.add_lower_dataflow_passes(pm)
    scalehls.add_convert_dataflow_to_func_passes(pm)
    pm.run(module.operation)  # type: ignore

print(module)

# buf = io.StringIO()
# scalehls.emit_hlscpp(module, buf)
# print(buf.getvalue())
