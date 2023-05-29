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

    def __init__(self, ctx, loc, insert):
        self.context = ctx
        self.location = loc
        self.insertion = insert

    def Add_Lib(self, lib_name):
        with self.context, self.location, self.insertion:
            self.lib = hls.LibraryOp(lib_name)
            self.lib_body = self.lib.body.blocks.append()
        self.insertion = InsertionPoint.at_block_begin(self.lib_body)

    def Add_IP(self, ip_name):
        with self.context, self.location, self.insertion:
            self.new_ip = hls.DeclareOp(ip_name)
            self.new_ip_meta = self.new_ip.meta.blocks.append()
        self.insertion = InsertionPoint.at_block_begin(self.new_ip_meta)
    


    def Define_IP_Basic(self, ip_path, para):
        """
        ip_path: path to your ip
        para: [[entry_function, entry_type, entry_size, entry_default_value], ... ]
                entry_function: can be one of "template"/"input"/"output"
                entry_type: For entry_function "template", can be one of "type"/"para"
                                "type": Support "float" and "int", which should be filled to entry_default_value 
                                "para": Support "int" only, default value should be filled to entry_default_value
                            For entry_function "input", can be one of "para"/"data"
                                "para": Use entry_default_value to refer to an entity previously created, with key word "var_num", num being the location of the entry
                                "data": Use entry_size to refer a previously created entity. Use entry_default_value for datatype.
                            For entry_function "output", same as "input"-"data"
                                
        
        """ 
        with self.context, self.location, self.insertion:
            self.include = hls.IncludeOp(ip_path)
            
            self.var_dict = {}
            self.ip_para = para
            self.num_of_para = len(self.ip_para)
            self.input_list = []
            self.output_list = []
            self.para_list = []
            
            # print(self.num_of_para)
            for i in range(self.num_of_para): #Loop each para 
                # print('1')
                # print(self.var_dict)
                self.var_name = 'var_' + str(i)
                
                if self.ip_para[i][0] == 'template':
                    self.ip_template_kind = hls.ParamKindAttr.get(hls.ParamKind.template)

                    if self.ip_para[i][1] == 'type': #Indicate this template para is a TYPE
                        self.ip_template_type = hls.TypeType.get()
                        if self.ip_para[i][3][0] == 'float':
                            self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, [], 
                                                 self.ip_template_kind, self.var_name, 
                                                 candidates=ArrayAttr.get([TypeAttr.get(F32Type.get())]))
                        if self.ip_para[i][3][0] == 'int':
                            self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, [], 
                                                 self.ip_template_kind, self.var_name, 
                                                 candidates=ArrayAttr.get([TypeAttr.get(IndexType.get())]))
                            
                    if self.ip_para[i][1] == 'para': #Indicate this template para is an integer
                        self.ip_template_type = IndexType.get()
                        self.var_dict[self.var_name] = hls.ParamOp(self.ip_template_type, [], self.ip_template_kind, self.var_name, 
                                                     candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), self.ip_para[i][3][0])]))
                
                if self.ip_para[i][0] == 'input':
                    self.port_type = hls.PortType.get()
                    self.type_index = []
                    self.size_index = []
                    if self.ip_para[i][1] == 'para': #Indicate this is input value, which is a number
                        self.input_layout = AffineMapAttr.get(AffineMap.get_empty())
                        self.input_kind = hls.PortKindAttr.get(hls.PortKind.param)
                        for item in self.ip_para[i][3]: #Check for pointing indexes, for para it can only work with type
                            if 'var' in item:
                                self.type_index.append(self.var_dict[item])
                        self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.type_index[0], [], 
                                                                  self.input_layout, self.input_kind, self.var_name)
                    
                    if self.ip_para[i][1] == 'data': #Indicate this is a data access,
                        self.input_layout = AffineMapAttr.get(AffineMap.get_identity(len(self.ip_para[i][2])))
                        self.input_kind = hls.PortKindAttr.get(hls.PortKind.input)
                        for item in self.ip_para[i][3]: #Check for pointing indexes for type
                            if 'var' in item:
                                self.type_index.append(self.var_dict[item])
                        for item in self.ip_para[i][2]: #Check for pointing indexes for size
                            if 'var' in item:
                                self.size_index.append(self.var_dict[item])
                        self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.type_index[0], self.size_index, 
                                                                  self.input_layout, self.input_kind, self.var_name)
                        self.input_list.append(self.var_dict[self.var_name])
         
                if self.ip_para[i][0] == 'output':
                    self.port_type = hls.PortType.get()
                    self.type_index = []
                    self.size_index = []
                    self.output_layout = AffineMapAttr.get(AffineMap.get_identity(len(self.ip_para[i][2])))
                    self.output_kind = hls.PortKindAttr.get(hls.PortKind.input)
                    for item in self.ip_para[i][3]: #Check for pointing indexes for type
                        if 'var' in item:
                            self.type_index.append(self.var_dict[item])
                    for item in self.ip_para[i][2]: #Check for pointing indexes for size
                        if 'var' in item:
                            self.size_index.append(self.var_dict[item])
                    self.var_dict[self.var_name] = hls.PortOp(self.port_type, self.type_index[0], self.size_index, 
                                                              self.input_layout, self.input_kind, self.var_name)
                    self.output_list.append(self.var_dict[self.var_name])
                
            # print(self.input_list)
            self.semantics = hls.SemanticsOp(self.input_list, self.output_list, self.para_list)
            return self.semantics
            # hls.semantics_init_args(self.semantics)
            # self.semantics_blk = self.semantics.body.blocks[0]
            # self.semantics_args = self.semantics_blk.arguments
    
    # def RegistorOp_to_Matcher(self, target_class, )
        






    # def Define_IP_Basic(self, ip_path, para): 
    #     with self.context, self.location, self.insertion:
    #         self.include = hls.IncludeOp(ip_path)

    #         self.ip_para = para
    #         self.num_of_para = len(self.ip_para)
    #         for i in range(self.num_of_para): #Loop each para 
    #             self.var_name = f"var{i}"
    #             if self.ip_para[i][0] == 'template':
    #                 self.ip_template_kind = hls.ParamKindAttr.get(hls.ParamKind.template)

    #                 if self.ip_para[i][1] == 'type': #Indicate this template para is a TYPE
    #                     self.ip_template_type = hls.TypeType.get()
    #                     if self.ip_para[i][3] == 'float':
    #                         self.dummy_var = hls.ParamOp(self.ip_template_type, [], 
    #                                              self.ip_template_kind, self.var_name, 
    #                                              candidates=ArrayAttr.get([TypeAttr.get(F32Type.get())]))
    #                     if self.ip_para[i][3] == 'int':
    #                         self.dummy_var = hls.ParamOp(self.ip_template_type, [], 
    #                                              self.ip_template_kind, self.var_name, 
    #                                              candidates=ArrayAttr.get([TypeAttr.get(IndexType.get())]))
                            
    #                 if self.ip_para[i][1] == 'para': #Indicate this template para is an integer
    #                     self.ip_template_type = IndexType.get()
    #                     self.dummy_var = hls.ParamOp(self.ip_template_type, [], self.ip_template_kind, self.var_name, 
    #                                                  candidates=ArrayAttr.get([IntegerAttr.get(IndexType.get(), self.ip_para[i][3][0])]))
                    
    #             if self.ip_para[i][0] == 'input':
    #                 self.port_type = hls.PortType.get()

    #                 if self.ip_para[i][1] == 'para': #Indicate this is input value, which is a number
    #                     self.pinput_layout = AffineMapAttr.get(AffineMap.get_empty())
    #                     self.input_kind = hls.PortKindAttr.get(hls.PortKind.param)
                        
                    
    #                 if self.ip_para[i][1] == 'data': #Indicate this is a data access, for now we assume it is an 2D array
    #                     self.input_layout = AffineMapAttr.get(AffineMap.get_identity(2))
    #                     self.input_kind = hls.PortKindAttr.get(hls.PortKind.input)








                    



   