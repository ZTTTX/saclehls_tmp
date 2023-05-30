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

class IPRegistration:

    def __init__(self, torch_mlir_module):
        self.context = Context()
        scalehls.register_everything(self.context)
        with self.context:
            self.module = Module.parse(str(torch_mlir_module))
            self.insertion = InsertionPoint.at_block_begin(self.module.body)
            self.location = Location.unknown()

    def Add_Lib(self, lib_name):
        with self.context, self.location, self.insertion:
            self.lib = hls.LibraryOp(lib_name)
            self.lib_body = self.lib.body.blocks.append()
        self.insertion = InsertionPoint.at_block_begin(self.lib_body)

    def Add_IP(self, ip_name, ip_path):       
        self.var_dict = {}
        self.input_list = []
        self.output_list = []
        self.para_list = []
        with self.context, self.location, self.insertion:
            self.new_ip = hls.DeclareOp(ip_name)
            self.new_ip_meta = self.new_ip.meta.blocks.append()
        self.insertion = InsertionPoint.at_block_begin(self.new_ip_meta)
        with self.context, self.location, self.insertion:
            self.include = hls.IncludeOp(ip_path)
        
    
    def IO_Warpper(self):
        with self.context, self.location, self.insertion:
            self.semantics = hls.SemanticsOp(self.input_list, self.output_list, self.para_list)
            hls.semantics_init_args(self.semantics)
            self.semantics_blk = self.semantics.body.blocks[0]
            self.semantics_args = self.semantics_blk.arguments
            # return self.context, self.location, self.insertion, self.semantics, self.semantics_blk, self.semantics_args


    def IP_Wrapper(self, func):
        self.insertion = InsertionPoint.at_block_begin(self.semantics_blk)
        with self.context, self.location, self.insertion:
            func(self.semantics_args[0], self.semantics_args[1], outs=[self.semantics_args[2]])
            self.result = self.semantics_blk.operations[0]
            hls.SemanticsOutputOp([self.result], [self.semantics_args[3]])
            return self.module, self.context



    def Add_Template(self, name, type, datatype, size=[], default_value=None): #IF it is only number, size will be []
        with self.context, self.location, self.insertion:
            self.var_name = name
            self.template_type = type
            self.template_datatype = datatype
            self.template_size = size
            self.template_default_value = default_value

            self.ip_template_kind = hls.ParamKindAttr.get(hls.ParamKind.template)

            if self.template_type == 'type': #Indicate this template para is a TYPE
                self.ip_template_type = hls.TypeType.get()
                if self.template_datatype == 'float':
                    self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, self.template_size, 
                                            self.ip_template_kind, self.var_name, 
                                            candidates=ArrayAttr.get([TypeAttr.get(F32Type.get())]))
                    
                if self.template_datatype == 'int':
                    self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, self.template_size, 
                                            self.ip_template_kind, self.var_name, 
                                            candidates=ArrayAttr.get([TypeAttr.get(IndexType.get())]))
                            
            if self.template_type == 'para': #Indicate this template para is an integer
                self.ip_template_type = IndexType.get()
                self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, self.template_size, self.ip_template_kind, self.var_name, 
                                            candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), self.template_default_value)]))
    
    def Add_Input(self, name, type, datatype, size=[], default_value=None): #To make reference to templates, input the variable name in str fashion in "datatype"
        with self.context, self.location, self.insertion:
            self.port_type = hls.PortType.get()
            self.var_name = name
            self.input_type = type
            self.input_datatype = datatype
            self.input_size = size
            self.input_default_value = default_value


            if self.input_type == 'para': #Indicate this is input value, which is a number
                self.input_layout = AffineMapAttr.get(AffineMap.get_empty())
                self.input_kind = hls.PortKindAttr.get(hls.PortKind.param)
                if self.input_datatype in self.var_dict:
                    self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.var_dict[self.input_datatype], self.input_size, 
                                                                  self.input_layout, self.input_kind, self.var_name)


                    
            if self.input_type == 'data': #Indicate this is a data access,
                self.input_layout = AffineMapAttr.get(AffineMap.get_identity(len(self.input_size)))
                self.input_kind = hls.PortKindAttr.get(hls.PortKind.input)
                self.size_item = []
                for item in self.input_size: #Check for pointing indexes for size
                    self.size_item.append(self.var_dict[item])
                self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.var_dict[self.input_datatype], self.size_item, 
                                                                  self.input_layout, self.input_kind, self.var_name)
                self.input_list.append(self.var_dict[self.var_name])

    def Add_Output(self, name, type, datatype, size=[], default_value=None):
        with self.context, self.location, self.insertion:
            self.port_type = hls.PortType.get()
            self.var_name = name
            self.output_type = type
            self.output_datatype = datatype
            self.output_size = size
            self.output_default_value = default_value
            if self.output_type == 'data': #Indicate this is a data access,
                self.output_layout = AffineMapAttr.get(AffineMap.get_identity(len(self.output_size)))
                self.output_kind = hls.PortKindAttr.get(hls.PortKind.output)
                self.size_item = []
                for item in self.output_size: #Check for pointing indexes for size
                    self.size_item.append(self.var_dict[item])
                self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.var_dict[self.input_datatype], self.size_item, 
                                                                  self.output_layout, self.output_kind, self.var_name)
                self.output_list.append(self.var_dict[self.var_name])

         


                

    
