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

# ctx = Context()
# scalehls.register_everything(ctx)

# # Parse module from torch_mlir_module and create new insert point and location.
# with ctx:
#     module = Module.parse(str(torch_mlir_module))
#     insert = InsertionPoint.at_block_begin(module.body)
#     loc = Location.unknown()

# from ip_register_test import IPRegistration
# obj = IPRegistration(ctx, loc, insert)
# obj.Add_Lib('vitis')
# obj.Add_IP('gemm')
# gemm_path = "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp"
# gemm_para = [['template', 'type', [], ['float']], #t_DataType 0
#         ['template', 'type', [], ['int']], #t_IndexType 1
#         ['template', 'para', [], [32]], #k_bufferDim 2
#         ['template', 'para', [], [2]], #t_par 3
#         ['template', 'para', [], [1024]], #Max c size 4
#         ['input', 'para', [], ['var_1']], #m 5
#         ['input', 'para', [], ['var_1']], #n 6
#         ['input', 'para', [], ['var_1']], #k 7
#         ['input', 'para', [], ['var_1']], #alpha 8
#         ['input', 'para', [], ['var_1']], #beta 9
#         ['input', 'data', ['var_5','var_7'], ['var_0']], #A: m*k
#         ['input', 'data', ['var_7','var_6'], ['var_0']], #B: k*n
#         ['input', 'data', ['var_5','var_6'], ['var_0']], #C: m*n
#         ['output', 'data', ['var_5','var_6'], ['var_0']]] #R: m*n


# semantics = obj.Define_IP_Basic(gemm_path, gemm_para)

# with ctx, loc, insert:   
#     hls.semantics_init_args(semantics)
#     semantics_blk = semantics.body.blocks[0]
#     semantics_args = semantics_blk.arguments

# Create a new "vitis" library.
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

#     template_kind = hls.ParamKindAttr.get(hls.ParamKind.template)
#     type_type = hls.TypeType.get()
#     dtype = hls.ParamOp(type_type, [], template_kind, "t_DataType",
#                         candidates=ArrayAttr.get([TypeAttr.get(F32Type.get())]))
#     itype = hls.ParamOp(type_type, [], template_kind, "t_IndexType",
#                         candidates=ArrayAttr.get([TypeAttr.get(IndexType.get())]))

#     index_type = IndexType.get()
#     buffer_dim = hls.ParamOp(index_type, [], template_kind, "k_KBufferDim",
#                              candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), 32)]))
#     par_entries = hls.ParamOp(index_type, [], template_kind, "t_ParEntries", candidates=ArrayAttr.get(
#         [IntegerAttr.get(IndexType.get(), 2), IntegerAttr.get(IndexType.get(), 4), IntegerAttr.get(IndexType.get(), 8)]))
#     max_size_c = hls.ParamOp(index_type, [], template_kind, "t_MaxSizeC",
#                              candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), 1024)]))

#     port_type = hls.PortType.get()
#     param_layout = AffineMapAttr.get(AffineMap.get_empty())
#     param_kind = hls.PortKindAttr.get(hls.PortKind.param)
#     p_m = hls.PortOp(port_type, itype, [], param_layout, param_kind, "p_m")
#     p_n = hls.PortOp(port_type, itype, [], param_layout, param_kind, "p_n")
#     p_k = hls.PortOp(port_type, itype, [], param_layout, param_kind, "p_k")

#     p_alpha = hls.PortOp(port_type, dtype, [],
#                          param_layout, param_kind, "p_alpha")
#     p_beta = hls.PortOp(port_type, dtype, [],
#                         param_layout, param_kind, "p_beta")

#     input_layout = AffineMapAttr.get(AffineMap.get_identity(2))
#     input_kind = hls.PortKindAttr.get(hls.PortKind.input)
#     p_a = hls.PortOp(port_type, dtype, [
#                      p_m, p_k], input_layout, input_kind, "p_a")
#     p_b = hls.PortOp(port_type, dtype, [
#                      p_k, p_n], input_layout, input_kind, "p_b")
#     p_c = hls.PortOp(port_type, dtype, [
#                      p_m, p_n], input_layout, input_kind, "p_c")

#     output_layout = AffineMapAttr.get(AffineMap.get_identity(2))
#     output_kind = hls.PortKindAttr.get(hls.PortKind.output)
#     p_r = hls.PortOp(port_type, dtype, [
#                      p_m, p_n], output_layout, output_kind, "p_r")

    # semantics = hls.SemanticsOp([p_a, p_b, p_c], [p_r], [])


    # hls.semantics_init_args(semantics)
    # semantics_blk = semantics.body.blocks[0]
    # semantics_args = semantics_blk.arguments

from ip_register_ver2 import IPRegistration

obj = IPRegistration(torch_mlir_module)
obj.Add_Lib('vitis')
obj.Add_IP('gemm', "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp")
obj.Add_Template('t_DataType', 'type', 'float')
obj.Add_Template('t_IndexType', 'type', 'int')
obj.Add_Template('k_KBufferDim', 'para', 'int', [], 32)
obj.Add_Template('t_ParEntries', 'para', 'int', [], 2)
obj.Add_Template('t_MaxSizeC', 'para', 'int', [], 1024)
obj.Add_Input('p_m', 'para', 't_IndexType')
obj.Add_Input('p_n', 'para', 't_IndexType')
obj.Add_Input('p_k', 'para', 't_IndexType')
obj.Add_Input('alpha', 'para', 't_IndexType')
obj.Add_Input('beta', 'para', 't_IndexType')
obj.Add_Input('p_a', 'data', 't_IndexType', ['p_m', 'p_k'])
obj.Add_Input('p_b', 'data', 't_IndexType', ['p_k', 'p_n'])
obj.Add_Input('p_c', 'data', 't_IndexType', ['p_m', 'p_n'])
obj.Add_Output('p_r', 'data', 't_IndexType', ['p_m', 'p_n'])
obj.IO_Warpper()

@linalg_structured_op
def matmul_mono(
        A=TensorDef(T, S.M, S.K),
        B=TensorDef(T, S.K, S.N),
        C=TensorDef(T, S.M, S.N, output=True)):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

module, ctx = obj.IP_Wrapper(matmul_mono)

# insert = InsertionPoint.at_block_begin(semantics_blk)
# with ctx, loc, insert:
#     matmul_mono(semantics_args[0], semantics_args[1], outs=[semantics_args[2]])
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
                    param.value = IntegerAttr.get(IndexType.get(), 2)
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
