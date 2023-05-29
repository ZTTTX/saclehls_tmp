# How to run:

0. Check for build status, go to /build

    > ninja check-scalehls

1. go to ~/scalehls run:

    > export PATH=$PATH:$PWD/build/bin
    > export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core

2. go to /tmp
3. run:

    > python3 resnet18.py > tmp1.cpp

or

    > python3 mlp.py > tmp2.cpp



# IP_register

To register an IP, we need all the essential info for this ip

para: This entry contains all information needed to setup the ip. It should include each entry as a sub-list
[[entry_function, entry_type, entry_size, entry_default_value]]

[[str, str, [num, num,...], [num, num,  ]]]
## entry_function: can be one of template, input, output
    
### If entry function is template, it will look for entry type. 

If entry type is "type", it will register this para with a TypeAttribute, with the type found in the default_value. Currently, we only support float and int as datatype. Size of the entry is assumed to be [].

If entry type is "para", it will register the parameter assuming it is integer (FOR NOW), size of the entry is assumed to be []. It will take in the first in the default value list as its default value. It can only take one default value (FOR NOW)

### If entry is function is input

If entry type is "para". We generate it as a portOP. You can refer to any previous para by using key word var_i, where i is the ith para in the para list.

If you have multiple points of reference, for example you have: 

p_a = hls.PortOp(port_type, dtype, [p_m, p_k], input_layout, input_kind, "p_a") 

Where dtype, [p_m, p_k] are all reference. we put the reference indicator in series inside the default_value list, for example this can be 

entry_default_value = ['Var_1', 'Var_4', 'Var_5']

In the code we take the first one as datatype, and all subsequent ones are size.