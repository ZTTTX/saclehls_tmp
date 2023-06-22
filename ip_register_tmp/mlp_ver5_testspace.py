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
from scalehls.ir import AffineCeilDivExpr, AffineModExpr, AffineDimExpr, AffineSymbolExpr, FlatSymbolRefAttr
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
            nn.Linear(32 * 32 * 1, 10),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


model = MLP()
model.train(False)

torch_mlir_module = torch_mlir.compile(model, torch.ones(1, 1, 32, 32),
                                       output_type="linalg-on-tensors")

ctx = Context()
scalehls.register_everything(ctx)

# Parse module from torch_mlir_module and create new insert point and location.
with ctx:
    module = Module.parse(str(torch_mlir_module))
    insert = InsertionPoint.at_block_begin(module.body)
    loc = Location.unknown()

# Create a new "vitis" library.
with ctx, loc, insert:
    lib = hls.LibraryOp("vitis")
    lib_body = lib.body.blocks.append()

insert = InsertionPoint.at_block_begin(lib_body)
with ctx, loc, insert:
    gemm_ip = hls.DeclareOp("gemm")
    gemm_ip_meta = gemm_ip.meta.blocks.append()


insert = InsertionPoint.at_block_begin(gemm_ip_meta)
with ctx, loc, insert:
    include = hls.IncludeOp(ArrayAttr.get(
        [StringAttr.get("Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp")]))

    template_kind = hls.ParamKindAttr.get(hls.ParamKind.template)
    type_type = hls.TypeType.get()
    dtype = hls.ParamOp(type_type, [], template_kind, "t_DataType",
                        candidates=ArrayAttr.get([TypeAttr.get(F32Type.get())]))
    itype = hls.ParamOp(type_type, [], template_kind, "t_IndexType",
                        candidates=ArrayAttr.get([TypeAttr.get(IndexType.get())]))

    index_type = IndexType.get()
    buffer_dim = hls.ParamOp(index_type, [], template_kind, "k_KBufferDim",
                             candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), 32)]))
    par_entries = hls.ParamOp(index_type, [], template_kind, "t_ParEntries", candidates=ArrayAttr.get(
        [IntegerAttr.get(IndexType.get(), 2), IntegerAttr.get(IndexType.get(), 4), IntegerAttr.get(IndexType.get(), 8)]))
    max_size_c = hls.ParamOp(index_type, [], template_kind, "t_MaxSizeC",
                             candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), 1024)]))

    port_type = hls.PortType.get()
    param_layout = AffineMapAttr.get(AffineMap.get_empty())
    param_kind = hls.PortKindAttr.get(hls.PortKind.param)
    p_m = hls.PortOp(port_type, itype, [], param_layout, param_kind,  "p_m")
    p_n = hls.PortOp(port_type, itype, [], param_layout, param_kind,  "p_n")
    p_k = hls.PortOp(port_type, itype, [], param_layout, param_kind,  "p_k")

    p_alpha = hls.PortOp(port_type, dtype, [], param_layout, param_kind, 
                         "p_alpha", value=FloatAttr.get(F32Type.get(), 1.0))
    p_beta = hls.PortOp(port_type, dtype, [], param_layout, param_kind, 
                        "p_beta", value=FloatAttr.get(F32Type.get(), 0.0))

    input_layout = AffineMapAttr.get(AffineMap.get_identity(2))
    input_kind = hls.PortKindAttr.get(hls.PortKind.input)

    sturct_name = FlatSymbolRefAttr.get("my_test_struct")
    # sturct_payload = ArrayAttr.get([StringAttr.get("p_a"), StringAttr.get("p_b"), StringAttr.get("p_c")])
    test_struct = hls.StructOp(sturct_name, [buffer_dim, par_entries, max_size_c], ArrayAttr.get([]))
    


    # p_aMoveRule = lambda d0, d1, s0: (d0 / s0, d1 / s0, d0 % s0, d1 % s0)
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    s0 = AffineSymbolExpr.get(0)
    div0 = AffineCeilDivExpr.get(d0, s0)
    div1 = AffineCeilDivExpr.get(d1, s0)
    input_layout_a = AffineMapAttr.get(AffineMap.get(2, 1, [div0, div1]))

    p_a = hls.PortOp(port_type, dtype, [
                     p_m, p_k], input_layout_a, input_kind, "p_a")
    p_b = hls.PortOp(port_type, dtype, [
                     p_k, p_n], input_layout, input_kind, "p_b")
    p_c = hls.PortOp(port_type, dtype, [
                     p_m, p_n], input_layout, input_kind, "p_c")

    output_layout = AffineMapAttr.get(AffineMap.get_identity(2))
    output_kind = hls.PortKindAttr.get(hls.PortKind.output)
    p_r = hls.PortOp(port_type, dtype, [
                     p_m, p_n], output_layout, output_kind, "p_r")

    semantics = hls.SemanticsOp([p_a, p_b, p_c, p_r, p_m, p_n, p_k, p_alpha, p_beta], [
                                dtype, itype, buffer_dim, par_entries, max_size_c], ArrayAttr.get([]))
    semantics.init_args([p_a.result, p_b.result, p_c.result, p_r.result])
    semantics_blk = semantics.body.blocks[0]
    semantics_args = semantics_blk.arguments


@linalg_structured_op
def matmul_mono(
        B=TensorDef(T, S.K, S.N),
        A=TensorDef(T, S.M, S.K),
        C=TensorDef(T, S.M, S.N, output=True)):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += B[D.k, D.n] * A[D.m, D.k]


insert = InsertionPoint.at_block_begin(semantics_blk)
with ctx, loc, insert:
    matmul_mono(semantics_args[1], semantics_args[0], outs=[semantics_args[2]])
    result = semantics_blk.operations[0]
    hls.SemanticsOutputOp([result], [semantics_args[3]])

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
