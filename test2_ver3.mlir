#map = affine_map<() -> (0, 10)>
#map1 = affine_map<()[s0] -> (0, s0)>
#map2 = affine_map<() -> (0, 32)>
#map3 = affine_map<() -> (0, 64)>
#map4 = affine_map<() -> (0, 3072)>
#map5 = affine_map<() -> ()>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> (0, d1)>
#map11 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "MLP"} {
  %0 = hls.dse.space @global() : () -> !hls.space {
    %1 = hls.dse.space @task0() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%8, %9) : (index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index):
        %12 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %13 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %12, %13 : index, index
      }
      %11 = hls.dse.space_select %7 [#hls.impl<"" _ "">] %10 : !hls.impl, (!hls.space) -> !hls.space
      hls.dse.space_pack %8, %9, %11 : index, index, !hls.space
    }
    %2 = hls.dse.space @task1() : () -> !hls.space {
      %7 = hls.dse.const_param <tile> {value = 0 : index} : index
      %8 = hls.dse.param @tile1 <tile> range() #map step 1 {value = 0 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map2 step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 32 : index} : index
        %15 = hls.dse.const_param <template> {value = 2 : index} : index
        %16 = hls.dse.const_param <template> {value = 1024 : index} : index
        hls.dse.space_pack %14, %15, %16 ["k_KBufferDim", "t_ParEntries", "t_MaxSizeC"] : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
    %3 = hls.dse.space @task2() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map2 step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%8, %9) : (index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index):
        %12 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %13 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %12, %13 : index, index
      }
      %11 = hls.dse.space_select %7 [#hls.impl<"" _ "">] %10 : !hls.impl, (!hls.space) -> !hls.space
      hls.dse.space_pack %8, %9, %11 : index, index, !hls.space
    }
    %4 = hls.dse.space @task3() : () -> !hls.space {
      %7 = hls.dse.const_param <tile> {value = 0 : index} : index
      %8 = hls.dse.param @tile1 <tile> range() #map2 step 1 {value = 0 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map3 step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 32 : index} : index
        %15 = hls.dse.const_param <template> {value = 2 : index} : index
        %16 = hls.dse.const_param <template> {value = 1024 : index} : index
        hls.dse.space_pack %14, %15, %16 ["k_KBufferDim", "t_ParEntries", "t_MaxSizeC"] : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
    %5 = hls.dse.space @task4() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map3 step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%8, %9) : (index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index):
        %12 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %13 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %12, %13 : index, index
      }
      %11 = hls.dse.space_select %7 [#hls.impl<"" _ "">] %10 : !hls.impl, (!hls.space) -> !hls.space
      hls.dse.space_pack %8, %9, %11 : index, index, !hls.space
    }
    %6 = hls.dse.space @task5() : () -> !hls.space {
      %7 = hls.dse.const_param <tile> {value = 0 : index} : index
      %8 = hls.dse.param @tile1 <tile> range() #map3 step 1 {value = 0 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map4 step 1 {value = 0 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 32 : index} : index
        %15 = hls.dse.const_param <template> {value = 2 : index} : index
        %16 = hls.dse.const_param <template> {value = 1024 : index} : index
        hls.dse.space_pack %14, %15, %16 ["k_KBufferDim", "t_ParEntries", "t_MaxSizeC"] : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
  }
  hls.uip.library @vitis {
    hls.uip.declare @gemm {
      hls.uip.include ["Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp"]
      %1 = hls.dse.param @t_DataType <template> candidates [f32] : !hls.type
      %2 = hls.dse.param @t_IndexType <template> candidates [index] : !hls.type
      %3 = hls.dse.param @k_KBufferDim <template> candidates [32 : index] : index
      %4 = hls.dse.param @t_ParEntries <template> candidates [2 : index] : index
      %5 = hls.dse.param @t_MaxSizeC <template> candidates [1024 : index] : index
      %6 = hls.uip.port @p_m <param> type %2 sizes() #map5 : () -> !hls.port
      %7 = hls.uip.port @p_n <param> type %2 sizes() #map5 : () -> !hls.port
      %8 = hls.uip.port @p_k <param> type %2 sizes() #map5 : () -> !hls.port
      %9 = hls.uip.port @alpha <param> type %2 sizes() #map5 : () -> !hls.port
      %10 = hls.uip.port @beta <param> type %2 sizes() #map5 : () -> !hls.port
      %11 = hls.uip.port @p_b <input> type %2 sizes(%8, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %12 = hls.uip.port @p_a <input> type %2 sizes(%6, %8) #map6 : (!hls.port, !hls.port) -> !hls.port
      %13 = hls.uip.port @p_c <input> type %2 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %14 = hls.uip.port @p_r <output> type %2 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      hls.uip.semantics<%1, %2, %3, %4, %5> (%6, %7, %8, %9, %10, %11, %12, %13, %14) [5 : index, 6 : index, 7 : index, 8 : index] : <!hls.type, !hls.type, index, index, index> (!hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port) {
      ^bb0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>):
        %15 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %16 = arith.mulf %in_0, %in : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        } -> tensor<?x?xf32>
        hls.uip.semantics.output %15 -> %arg3 : (tensor<?x?xf32>) -> tensor<?x?xf32>
      }
    }
  }
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<[3.20490304E-4, 0.00297085987, -0.0151902791, 0.00816963613, -0.0157840326, 0.00674360571, 0.0132276742, 0.0177070647, 0.0102539668, 0.00145303132, -0.00431816652, 0.00499606784, -0.00662339525, -0.00718710199, 0.0147221778, 0.0082229413, 0.0115343947, 0.00389332394, -0.0102071548, -0.00121453591, -0.0177267343, 0.0173472334, -0.00845747068, 0.00120154722, -0.0101301288, -0.00835017394, -0.0147694862, 0.0173417646, -0.00211789017, -0.012208919, -0.0178934578, -0.0177751649, -0.00420461968, 8.38374369E-4, -0.0121586462, 0.0142530734, -0.015777953, 3.26555542E-4, 0.00774956774, -0.0111678988, -0.0111586247, 0.00474465452, 0.015173791, -3.802470e-03, -0.0130240843, -0.0129222702, -0.00240610773, -0.010571924, 0.0150261754, 0.0174863767, -0.0114875939, -0.00362562062, 0.0147703318, 0.0138352141, -0.0026183743, 0.00902822148, 7.482170e-03, -0.00906530768, 0.00612409879, 0.0081237806, -0.0149353342, -0.0135029554, -0.0173319224, -0.00790041219]> : tensor<64xf32>
    %cst_1 = arith.constant dense<[0.0330218971, 0.0954179615, -0.12046732, 0.103317112, -0.0563988537, 0.0578674227, 0.0163800716, -0.0475305021, 0.0735184252, 0.0022983849, 0.114757806, -0.0289237946, 0.120515108, 0.113760903, -0.0608102977, 0.0229682177, -0.101213753, 0.0613418221, 0.017042473, -0.0470626056, 0.0995501726, 0.101796955, 0.0491928458, -0.0265373141, 0.0718662143, 8.049630e-02, -0.0614857971, -0.09457466, 0.0736731738, -0.111784384, 0.0959455222, 0.0563677102]> : tensor<32xf32>
    %cst_2 = arith.constant dense<[-0.0599529222, 0.140413851, 0.0184263065, -0.11831554, -0.139851138, 0.11523629, -0.0818133726, 0.154019222, -0.0684074312, 0.0846099629]> : tensor<10xf32>
    %cst_4 = arith.constant dense<"0x6A4FD13DE8E1F4BD1CDAE93DC889003D000A083CD49616BD9A0F90BD78F1983D768E9BBD84316A3DD088B03D44A7AA3D40A1CD3BC62EFFBD38CB8F3D5415503DAECBB1BDDAA587BDE0F8A43D2845DCBDE8CD54BDB0B9693CF4CC263D660DDDBD54B8883DD43208BDE064F13C86C0CE3DD08D8FBC70D78A3C2C2D963DD6FAAD3D4015E2BBE413EB3DF4D46F3D406F793C18EBA23DC81F24BDAA42BFBD448DF8BD389064BD5854D3BD063FAD3D001FA63AC4B537BD7880BA3DEC5946BD9009333CFA159FBD00EE773AF0B4DABDFEBBB83DE07B40BCCE46BDBDA8D2823D4011A3BBC48EAC3DE2429D3D8005E83D2E97C53D8C854C3DC2A486BDAC9140BDB0B58F3CE25EFABD10078ABC0015373B0897B4BD40AD343D8A5BF73D88FBF0BCD0F6763DC091DCBBA0B60C3DA061CEBB680CE3BD18F866BDB0D067BCA01162BDA4CEBB3D4CEAEABD94A0183DC8A999BC9000983C00C47CB90CE1BC3D1837013DF0A175BDDE5DD9BD889AA13C4492233DACD6E8BD64DB25BD4A74F7BDB40C8BBDC84BC23C08700FBDEC0DFCBD24FE473D3CF5D6BDC8ABC1BD729EDEBD2E23B9BD7078153C08C2BBBD50B08DBC8AEBFCBD0044133CA0E937BDE0942D3C3C78D6BDBA4CBA3D5429463D00624BBAB81126BD80AE083BE4D4E33D62E5CEBD4C7E60BD98E2CBBC8AEC8DBDD8B6A73C500340BC107BAB3CE007713CE076B03B48B0353DEA5A923D94AB19BDA0BBAEBD80FC913D4E1F9FBDC04478BD662D81BDC414B9BD14F6E1BDF0731DBD7AA2AE3D2050BBBD36C9FA3D20E5D73B00F927BD40393C3D2AD6B53D2224C33D1A80883DE0A9EA3D8AA3F6BD0087BCBD304EF73C18E22ABD107285BD509D0BBC508FD03DA0F2173CE0B9A73C4A6ED2BDBA0A91BD38D3AA3CF038CDBD7CF68F3D90E23FBD4862CE3C40EC38BB6A5CD03D70CA163DA008F53D280928BD6088EDBDB649D73D046BF1BD2C178CBD8A17973D3680B53DC0A7C23C20633F3C14F56DBDF624DA3D4C2C473D1468AB3DF01210BC3C81C3BD84CBC73D206FA7BB1013503CD47EE6BD8C51DF3D468DF63DC2EB983DEE3292BD1C6AE2BD40D9113B148BE63D1E45AFBDAE62FDBDA885543D08EEB73D24C1AEBD48D69B3C48D7ADBC22C1CEBD70D3253C20895B3D50E3603D186CD93D242E583D1ED6A4BD7AA3FC3D245DE53DEA87B23D4E31ECBD64A8C43D6C402D3DEA32FC3D80DB523B6AD9B4BDA003133C08B047BDA004F43D1841E1BDAE5782BD1EEBCC3D48D8FC3DD077AABC8AA5E1BDD46007BD60667E3DB8B3673D0AB38EBDA24ADBBD7EA6FCBD005BA13C5E8BFEBD5050C5BC40C176BDB0C508BC4898A73DFC3E3E3D6082103D5A68A03D44C76E3DEC24C1BDBA14833D56B5E53DDEA088BD4E0CEDBD7E49B53D443A143D80D6C33C9AEEE7BD4EBBCD3DD6CCBF3D98D8AC3C1ADFCB3D6058A03C40DDF0BBCC0F033DF8D3E3BC243344BD4C1F883DFA44A6BD56E8C0BDF881D93C385180BCE4A9433D68A983BC4EE399BDC88DFABC8E01D23D644AAEBD56988D3D2A8EC13DE05EF0BDB4760BBDC861F73C908B103D3E70A13D346EFB3DC0D192BCA4AFC9BDDC8FA5BD0E52ECBD80F6E23DB06D6B3C6801ACBDD297CFBDA0C6B2BCB060313DD8C1AD3D8096A83A8A21B8BD80BAE7BC78B8FB3C408D99BCF6E8F4BD20F0E33B7C5A75BD2E38B5BDE0D2B6BC68A99FBC8ADD8CBDAAA3F5BD7ADFC33DC274C6BDDC9E643D1858CBBDC88479BDC2509EBDD664EC3D72B5B6BDD091B0BDC66DEA3D9EB69B3DEAD7973DC4D1053D7496C4BDC80D8DBD16ABCF3D406D063D108634BCEE11873D005FE3BBD21184BD7CE2483DEA41843D4C1802BDBA85EB3DB082A4BCEC0F4DBD0C66E1BD4A92DCBDF0D030BCDCA7DA3D80E9F23C36ABD93D9A53EB3D04DA96BD3A19B0BDE4DDB4BD704FA8BDCAB68F3D10C64A3C7E8F96BD08E0F5BCD8FAACBC3697CD3D008E913B9C23F23D549869BD546C38BDA056C83B88CBA7BCA0AFC23D70B52B3CC657F8BD8A5EB73D54B6E7BD18A4B0BD42C5AC3DB00CCEBD30F8273D1C9EC4BDEE4CB83D24FF9C3DF01C66BC240F243D503AAE3D12F2DABD12EBDABD005BE13A18B5F2BD20D4E0BD98F7BCBC707F73BD7CC3623D506E94BD40BCE03D2C8E11BD8C521BBD3089C33D003A79BC706EDE3D4C03803DECD3A8BD0A62CF3D00CBC63B6406AB3D60A4EABCB8C8953C90E6883D868E91BD6802B4BD922FA6BD50165DBCC084B2BBB024D83C58AAE53DB0D7AABD409994BD6CE73EBD5AA0A7BD3AAFAE3D0C528BBD7CE2F8BD6C29EB3D6017D83C2A7E9D3D1EE6F13DC84C1D3D908CC9BD484957BD4ED38CBD3EDDA9BD38D626BDA4CBDF3D8CC51C3D346CA8BDD8F758BD0864C03DB0134CBC20B580BBC4C2A9BD24C973BDA644D23D38A4863D44897BBD2A89A9BD78190DBD88C72D3D0CB2C53D646DF8BD689DBE3DC218F83DDCD1A43D80FBCE3CEE5FD73D7870693DD01248BD50393ABD0CE5A0BD283A9B3C1EFB9CBD6C0F9D3D622E8A3D0833E3BCD8996E3DF6C1C5BDD865F23DDAF2C83D720AE7BD3691BC3D24B88C3D52B4F23D54B37DBD6C92673DF09ABE3D50C5D63D8C5CB33DB4A563BDA49C773D043C7B3DAA5A953DB0907D3D30C0673D967BAF3DE2A6D0BD7000563C5619BC3D549ECC3D0C49C83D74F2B13DC423B6BDDC52B53DD8E56ABD6AE4C4BDD63CE93D506BCBBCAED2A4BD6CCDAC3D7888EE3CC0DD79BB50D84C3DF8959FBCC680823DFC12FD3D24446C3DAA5685BDB0123FBD406513BC00418EBBBEA9A9BD046956BD5E44C93D78CE413D9AE9E4BDFA1281BDF093843D6A24943D24F72F3D9853183DC82B983C82388EBDC42BC4BDAEA3B03D04FBDEBDBA3A933D9CE6B0BD6E25B9BD746D6E3D6628D63D641B1B3D4E5CD9BD58BBC03D806CAA3B8029463D80FA8ABC2880903D00A1733CF6EE8EBD842D75BD5ADD8CBD008724BD9C8BB9BDB0B8FABC780DCD3C704EDDBC10EA633C786DBFBD4C40D33DD80AB4BCAC9D1FBD80577F3D1AFAEFBDD8D58FBD286B7C3D48D40C3DCCFE343D583F903DB81AE33C48CAEE3C203B13BC4809253D0CC8A1BD5AF2F33D505D853D1046753D8872BC3C0015B13BF44F863D806173BD0096AB3B78F66FBD5EE0E13D5224E7BD705697BD001F23BDF03577BD841A82BDAA1D823D00E0B8BB78FC64BD38D38ABD50FEAE3DDC970CBDA0366C3D22EFD93D20F7913C98B5263D48E7B3BCC0C26EBDBA2DEDBD08D899BDC0AB103B6AD0BEBD006AAD3C88A4D9BCC41352BDEE0A84BD60F1A4BDB84A2DBD4C9B583D44774A3D9AF8FCBDB0CE5E3C201754BCCC738C3D8CE1F33D40D8383BE0AAC43B88BE74BD18A984BD501797BDF2DFD73DC0065EBC3E2FDA3D805418BB20554FBC1878973D465492BD50EB693DDCF0DD3D24F66D3D448BBB3DAAAE98BD0010D43B5C9B59BD9CFD4C3D48FDF1BD76B790BD607F71BDD60B82BDE09BCBBDB898BF3DD8F5163D00F3DDBD4841E5BD4439003D0624D7BDEAD0DF3D98B09EBD802E0B3CEE76893D20F1FEBBE4D92F3D8CB995BD38243A3D003982BD64A0203DB0EDB5BDD690D13DAC1074BD7860D2BC0AC8CD3D96F7D93D800B9F3D1C605E3D8AF0CA3D5C90CC3DB03762BD781587BCC0B3233B8850FFBD6000EFBC78E8373D266E80BD48AFA9BD7001703C08C2DE3C7AAAAC3DB02BEB3D1EB9FD3DF8A5F03C846FB6BDB8AD603D2C211B3D54AB1CBDF238A63D20DDEE3DA6EFF6BDD87B093DC00C1FBBC01DFBBC2A8EE83D88114CBD02A2DD3DFC1174BD1041453D3C07FD3DEABFD6BD1855E7BCF064313D8069E83B389F3FBDA03DD5BCA069963D20FF45BD623EA9BDD6ABD1BDD0ED3EBD8CB9E6BD844420BD84DCF4BD505919BCFA87E0BD0079A7BD14F54B3D60A4CEBBACAE7A3D401B5FBB00DECBBCBCA3A9BDEAA3D03DDC6C61BD383D963D70153C3D002E903DF89BE3BC08C7EEBD407C96BC846A14BD7A3484BDB6AF9EBDF62AC9BD787EEABCEA22CA3D70A16D3C9A38F4BDB091C63D80397CBB807C5FBC8841F93CB2E1F33D10AC0BBDB0F52FBCB85C1FBD9292E73DB25CE9BDBC15593D8C0E3EBD6240983D6CF8133D2A13EC3DA0DDA0BB443EB3BD602338BC60DCC8BD60D3603D386583BC009CEFBD1C9E49BD58A088BC44669E3D7C2B093D44A9E0BD8683C6BDBCDCAD3DCCA2093DF0A36C3D60C4113D04519BBD1872A53D6244D63D54E0E3BDAC34433D9A7F8E3D803BC13DACD05ABD006DC23B0090CB3A94D4C93D466CAF3D10E87ABD34ABDFBD3016D0BDA888FDBC5CD2563D8EC0C2BD9C1CB43D0636E0BD8046073DBCC3C4BDC00754BC001281BD107027BC18CD023D2083323C288DCDBC403A63BB60F3E83BA0D48CBCE6AED0BD72B1C1BD241D003D00D0AC397CB15CBDC4F7213D4E98EA3DB09F81BC328EE1BD646B263D4831FBBD800ADBBAE80A853DE0BEADBB3E3CA33D1EB6C73DE001F33BBC00B03D20364BBD64EF0E3DBE75B4BD5418B03DBA72B1BDA0BF22BD729F8EBD44673CBDE824C63C589721BD107A053C5C4CFF3DB8DDE93C70147B3C4633933D2E51943DA02F9D3DE854CE3C00B6D3BCDC982B3DF44725BD56749F3D0021543C2201BCBDC813AB3C34027F3D227DCFBD224CE2BDF2D3E4BD5462F0BD3E4AE6BD083BF83C9282D6BD0C9E6D3D8C35DF3D504EA53C9C4A563D2C00B03DE2F2FFBD402A643C2CFD71BDC0275C3D18BD8D3CB4F6AEBDD8CEC4BC90FA6C3D80D9EBBBA424463DDCCB2BBD2814BE3CBA21EB3D80C5B13A907DE93D80F6F43D3CFDF43D20F26C3C1C7AD0BD94B11A3D744A49BD28A6DFBDD44072BDCC823E3D5A5D8C3D70AE8CBD9EF8B33D42FEE6BD0A32F53D5878B93DE8E39E3D6229CD3D1094AE3C88902B3D8245EF3D586FD63D86FE96BD86D0E3BDB812B0BD006E75BD3681ED3DC0E7A9BDAA48C6BD0004453C228FD63D10BB06BD78FB453DACEA76BD6CF59CBD34B849BD8470DC3D6C4CEABDB8F2CCBC9877DB3D2817D63DF0F78A3C2C1077BDC4AA23BD58CBD03C60E7443D1254ABBDB8D820BD4E9AFB3D70AAC23DEC01263DA4C6ABBD383BDEBC1EE6A9BD30E3B3BC3AE6F13D48EBD73DAA8AC53DC0A9BC3C28A396BD323CE83DCC97BDBD1A1DCFBD2EAED23D5A9D97BD489BA9BC4084133B668AAC3DF6EACB3D00F09F39A095D2BDA091853BB2E7CD3DBA7C8DBD1085313C6E6ECE3D82BEB33DE0D0BC3D6E8D93BD1064F5BCC0275ABB9CBCD3BD98DEAE3C3CFD11BD00040D3A94264E3DE07209BDEA27BCBD203DC23C9872383D7A75CA3D68E664BD80BD543BF0B5463C90132B3C7C56E73D0EF4B53D1215853D84E706BDAC3DACBD105E38BD70687ABC1091B1BC601BF93C80EBF83DA065A4BD9009D23CB0E17A3C189C773D54B79BBD00668CBA2AB394BD02DB9ABD485D82BC7CBECB3D0C6E14BD807A15BD84BCDD3DA4634EBD50988F3DCE1A943D22F2BC3D98879C3D8CAC66BD8CDEB2BD36BFFF3D4ADBEC3DE8BA5BBDE8344A3D2E00BCBDBAEEA03D2027D7BC8AE59D3D90A61CBC000CD1B924CE3FBD64ADD2BD2066F3BB1C888E3D6AA795BDB6CECDBD383F46BDD85511BDA283983DA0B9F5BC2EDEB8BD06A8EBBD1439053DBC05C23DAC145F3D70C79ABDE0E3CABC0804C7BCF0EDDFBCE8C0E03CE0C5E5BB2EA9903D0A24ADBD38B7D0BCFCFA4F3DC08B383B68A18CBD9C9B913DB01DDB3D0C16B6BD308FC1BC00B9AE3B9C825D3DD8528ABC2C7068BD2C27F7BD126ABC3D2046503C00D0103972DC853DCC6509BD1AD181BDE4B897BD60711C3DD89082BCC894893CD465D53DB636D13D389B01BD687D433D1C7C4BBD10BA47BD54E75ABDE07EAB3D7CC23ABD00C2A13C4EB7D7BD40939F3DE81E7BBD4838923D8084083DBA25B93DE82F38BDEC33FEBD98C059BD10C3C1BDA229E83D6A1ED5BD00D4783A08B6763DB2948BBDF8EFF53C584F5FBD00383A3CD233C7BDCCD937BDA83CDA3D4C74A3BD8C7073BD9E9F9C3D909BCB3CDE5AC83D46C4D53D063AE5BD2A8CF83D407EB03DD0F7263D38F989BCBC94CA3D209AD13B08EB3B3DD6E99CBD209E56BC2023D1BCFE12B73DFA42E9BD70E06EBCF0FCEFBC664CD2BD343299BD8079B2BC566CECBDD6DFD3BDAA8DD7BDA0ABC93BF8E2C2BDACF8363D2C1ADEBDD4E4CFBD3009F4BC900221BC000034B820F3A3BC883CBA3D2486083D4067F5BC540E64BD2662E8BD46368ABDC871A23D44D0AC3DBA37DABD4CD8643D1A3381BD3C92B7BD109C30BD9A55F13DB2B3D13DF6868D3DF068B33DE0B3E3BB6673DA3D704527BDC2C7B5BD800FEC3BA86F59BDB048213CD8828B3D2C964BBDC8CEBC3D607EA9BCF818F4BDC00E10BB6C3347BD6C243EBD707ECF3D00399B3DAEACD83DA4739D3D209C223D70E37A3D681497BDDC22503D2220B5BD9ECA923D00102EBAC02379BB60AEEE3DA0C768BD086F54BDDCC7063D7C6D44BD00877D3A182939BD805DFE3A1C45BBBDD82860BD74F53E3DA0AC1E3DDEF7CA3D26D8C53D18E2E13C30674E3D62ADD43D34A5993DFE31C0BDDA59993DECA000BD142B203D3C78A0BDD011FFBD58D7BEBD2A3CC03DCC7E903DC091E5BD58ED00BDF839203D40B2CFBCDA528BBD7035B83DE6C8DEBD283797BCCA5F9D3DA45FAEBDB0C8A8BDB02AB63DE26A91BD00E9A43DD83A6EBD303FD73D722DBD3D003CA23BBCECE8BDC81DC53D0E7EFBBD3414F03DF826333DDA81C33DC8E83F3D90EBB1BD54A982BDF82963BD8447BBBDC43AF5BD2EBBE1BDC4DF50BDC043D8BD16D1B7BD7C900DBDA682883D58E5AF3DC60AB2BD68BBEA3C445343BD301391BCF0501EBDD8A49E3C4242A0BD50A9F2BD94370D3D0012813CD2B8F23D544369BD90FED0BC90E7353C0034D23CDEC1BA3D40F2E13C800AAE3C64065C3D6EA4D83D400EA83B400B03BBB4E0DE3D009D3E3CC203F0BD666CE33D3A39EF3DF08066BCB08E5DBCBC687BBD3068D73DFA1DAA3D8C2B30BDFE2881BDEE0BD9BDB4A5E1BD388C0C3DE415E73DF007D93DB08F733C7694FE3D86ED873DD8ADE7BC68C4CC3DEC6D7DBD541C563D802B5BBBA02CF1BBC0EF4B3B0002AAB912DF9DBD349FD23DD816CFBDE844ACBCC602ABBD7AB389BD009863BA0080BDBB84E6043D2C2D933D04AE74BD081C27BD4447E83D223CC83D3816A2BDDEC8CE3DD091983C6019B13CD4C4F7BD84D9D4BD5C2576BD802D8EBB60FABABC503522BD2098683DF05577BD2AADA43DDAD1BCBD5096D43D100C883C60A07FBD9E15FEBDC4206FBD0040D9B8BCBD293D52D3B5BD1AD59B3D72989ABDF0213CBD46B697BDAC4AC93D800F773B40E7AABD064E89BD307FC7BD90353DBC3ECAF7BD8CC231BD9811B33D1E7297BD4087E93CE40C7ABDC8F0D0BDC269C33D6AEDD73D0292AABD1013593C1693EC3DFE0CD73D0CF076BD34740EBD684380BDD600AF3D40C2C6BDDCFB0B3DC2FEA9BD40D262BC348A09BDF631973D0C0918BD3298D53D2EA1AA3D10FE79BD9E10853D40E2333D405F69BCB846053D0497643D00A7343BF449993D20FDE63D0EA5863D7E40CE3D8A4FF4BD607783BB3C4659BD86F195BDD88CC9BD787E99BC28FFF63D0C652DBD9801CBBD165CF2BDEAB0903DE0F3F23C2839473D36F8DB3D807ED83A6477D83D20EAF03B200905BDBCC0C73D8EAABBBD98F9073D68B4F7BC1043F9BC4A46D6BDD07334BD84E9E63D206FC53BB0FE883C409EC8BB0CE48FBDD8E1E6BC384F8F3D108FA83D34607BBDDE43963D589AC4BD887358BD3EE3F4BDEABCA53D005937BAC44A36BD1E38E33DF647873DD04217BCC270F4BD18B993BCF20B82BD48A0DA3D7CA108BD80CBD73B90F4313C9E47C5BD0081263B080DDFBD6867F1BCE01BD83C4062073BB8ACFEBC88BECDBDCA46953D1481773D78918ABD1295E5BD24D0AEBD5094E9BC80EAB43C967DA13D40EC913CFC7E12BD800648BD621DE13DC2AE99BD484A7E3D7E72EABD3069E9BC20206B3DC0A4B43D86B8A9BD303829BD2084003D8A4C81BD7E4CA83DF888E0BCC40E613DDC514CBDCC67273D1A42EDBD6E40F1BD2C4991BD900166BCDC6895BDF80B523D30DE8A3D6007D33C28A6833DE062623D4C2F66BDF4C3E7BDAC65243DEC7042BD004333BBEA378A3DD00F22BC00A027BC543DD3BDB0CEF23DBC81A3BD020AD23D50B222BD3CBF8EBDEAF599BD905275BD22ACC6BDBC2CF63DF86B9B3CB016A73C4898A5BDD452C3BDF0BD313D105AF03D1CF0CF3D6039B93CF66383BD684F653D80813CBBB0FB843C9242DA3D54E5EDBDF4616DBD7C3BD2BD80BF95BAF8F77D3DE85C8C3D6AFF893DFA9FB2BD586D843C507E68BD4E68A23D4C16A23DEC4C64BD80F177BB3E35EA3D8C77E6BD48BB79BD6629863D3C093EBD6883B1BC280691BC227D853D9EA396BDE0801E3CA0A5FABCB0D5EC3D00EC2A3996D5A2BD4C99003D90624D3DFE9A85BD7642D1BDD0BA473D40B9303C58B1393D8673F3BD42C8EB3DE6C2F53D48F9DB3C502003BC6866FD3D84CDD0BDE0FA7EBCE079933C24FA133D30644FBC58A634BD3C7DCF3D80D4BB3C5455B8BDB41A0B3DC888CA3DF857033D705903BCD8E7683DF0B5C13C309A3ABD006E913930CAECBDAE489ABDBCCD59BDB8F106BD2089C6BDECDED7BDEC3DCFBD28192D3DC83DE1BCF890153DE241DA3D7811E43CA0299A3B40FC5D3DD6C8A13DC273BA3D0E71C73D4461DE3D00632F3DA0E1AEBC68E8F9BCB0884A3D762CBBBD806BAA3C38E483BCE0236ABD3800893C487F913DC00C803B4014CFBDD0A7753CBCB9EE3D60A8ABBC7655F13DC0DAC3BB503DAB3D88A697BD4E37F23DC8F597BD42ADBA3D223D9F3DA42A57BD48698F3DF093DBBC1C256ABD60BBFBBB9CCC50BDC05F013BD03F353C14F53B3DC87BA03DACF1FFBDFCC6B73DA0ABB23B1038633D6081753D1861B93C2C5108BDD472EEBD000E68BB2CE4B93DD2E4AABD705845BDB000F2BC94621BBD3CCD623DA42A31BD90B9FFBCA0A8DF3CA2EFA23D88563FBD163DB1BD34158F3D40DA22BC7246E83DF08A2EBDBA53AD3DF09D49BD2C74793D40CC71BDD0A63BBD406EF83DA42DDC3DF0A21F3C5012263C306F5FBC5077113D18D660BDE0DD623D002452BD0CAC70BD1A41CBBD9A69913D7839A4BDC0647CBBE00DF9BD22E5E1BD3C3EF5BD4ECC9EBD6C55CE3DE4ADEBBD9CC21DBD7EA2B03D70DF8C3C1064A2BD8C4B86BD0ED38D3D7201A3BD063C9F3D3C28803D3488183D6A4C823D0084F5B95279CE3D84543BBDD66EB7BD109DEA3C343AEEBD0869823C360491BDC618E3BD78F9A33CCE20F13D68004F3D0C5B683D2A90E13D2CF5C63D189E353D44B99ABD00CF93BD887BCCBC587A473DA077E93CAEC2AA3DC4DE9D3D3401C73D7886A33D3876DABC80BD953B6E1CC6BD72C8A3BDC01298BD80CA743CF084ADBC3EEFD83DD642E23D0025E83ADE1AD2BDD8B2DA3D082DAB3CC0A676BDD0A1CD3CD0A28BBD0037753D90648D3CE0FCC03C8034433D10B3E2BC16DAABBDE03AF63B5843933D0C3E74BD8655F2BDB8C982BD7E678ABD204358BD788DF1BC30AB193DD03C5E3C007ABF3C4C54353D6EF38F3D809B0A3DA24780BDA4693C3DC6C1B43D72B9E53D5C2C51BDD0A1F2BCAC654D3DAA22B1BD44A2283D04E5FCBDC0F89E3CD88CF13CA27BE23D18BF28BDCE8BE33D9414093DA0483B3C1A1CAF3D661DD5BD0443BF3D20B601BD709B23BC285EAFBC70BD7F3DFA2CA23D1CE56CBD7A4F9CBD58FA983DECDD263D745C1FBD9037B83D20463E3DA60DA3BD5007B93D6839E9BDA0B4A6BDE4FF0E3D10657E3DB0CD073DBCB2533D1842B23C2C5AB93D2AD8F33D08DBF03D9CF212BD3C29D7BD88E592BC82BFB23D00E86CBB2812D9BCDCA4E1BD4C41A5BDA4299ABDC853AE3D807A83BA8C9C78BD18FC483D34CD1F3DF8E1A73D38461BBD50E99C3C16CD9DBDA200F93DF45B42BD4CBC33BD10C8D6BDDC38EB3DE8BF9D3D721BDEBD0EFBAE3DD619FABD6017B8BC0CCCEBBDD0FA05BC044628BD40BD87BB047F273D7CDC8CBDF45F64BD7A5FDC3DF49A75BDB4EF4FBD52C4C2BD8835103D8846F03DE09A75BC90BFA9BC0488D03DA00B0EBC386040BD38A7EC3DE8A31DBD2053F43D08881F3D905971BC88E1973CD679A53D80A1663BA08F6FBDF620A63DA0A8C9BC4E9FC3BD52BEADBDC2839DBDE453AC3DAC8551BDB474E13DD0680B3D522BA93D505C66BC6C34A23D90C5F0BCCE4DA53D740D11BD4642B5BD18E0D13D60F3F93B1803593D38D287BD6ED0B33D186B773DCEB0F23D807C52BD5208E3BD000D743C4A43C13D08478ABD78DB5C3DC03C6E3B107305BD003690BCB4267B3D4C25463D0C8AC63D603AC63D4020623C7C7680BD6005E6BBF45DD4BD5C7DF03D607F0FBD48AA6B3D5248F4BD4E5C8BBD001057BDE8F9003D94B9E0BD68FCFEBDF069FF3DE6F3DF3D307F203D301FFC3C5CB026BDF836AD3C308A893CFCA81EBD969DF93D88EFEABC6EA6C6BDC84FA23C72CBFBBD04A763BD608A8EBCD07E83BD2A4FA73D625A94BD4060AF3CC005F6BCB86BCA3C702E763CD451A53DF057743D0056A7B944CF72BDDA48F13D4228B8BDF6B1DFBD0C24AE3D406DD3BB9E55DA3D5078D33D20BCD7BBAE289B3D982F843CAAD4FB3D28ECF7BD2E18DF3DD02CD5BCC0C86EBB0212E13D3C1D31BD2898E73C9000C03D001A273B36E2AE3DF49798BD6C21503D4040563B70BE3BBCD829E93D48E9953D508135BCEC0B153D54C26DBD3826E2BC401E123BC481353D185CC6BDF83288BD8A2086BDA0544E3DA807D13D40C749BC60D2BEBC4CAF613DC06716BDA6ECB0BDE41E60BD50CB10BDE091753C70A7C43D5098583C28C1DE3DE0A5FABD8830C93D300ABFBD6E8190BDB8F8703DCAEAEEBD04F55B3D08D7683DE4C97DBD84211B3D741A16BD8076FBBA0CDA73BD005131BCB4519EBD540CD6BDC4DB8E3DBA58AFBD388F73BDD0AE49BC40BC1F3BB058513D1C409B3DE0F7A53BF00C363C606D93BCD895D23D64236ABD02BDDD3D4CD48FBD7856B2BD60378BBC4CE42C3D1C97973D2AECC6BDB8389ABC74E2DCBDFCD6E7BD34163B3D4E06823D2E1EFF3D808632BB80B0833B84E675BD3CC4173D84ABE4BDB8A4383D2C06B6BDAC5DE13D7CE94E3D08AF07BD30B87C3D7001F03C6806E83C1CAC1BBD72C0D83D083A8E3D80D661BC2CD51F3D185AA83D802638BBC87981BDDC84493D26EDC03D5C84C0BD0A60A33D2E5394BDAAD1883D4C0D5C3DF0D1563C2499723DB87EAABDC0B4C03CA0BDE1BD9A9BF7BDA60ECABD789D99BD1A9ECE3D8AA3CC3DDEB79CBDF8C9D93D6C0F443D14472CBDA0867C3C505D3F3C8885C8BCF6A9B93D5850A5BC466CC93D343F66BDC0CDBEBD40D5D7BB38E8B2BD844AD2BD3ACDCABD508E76BCA4A133BD12E1EABDF4367F3D02E4893D2086E3BCA42200BDA6FBE93D182BD53DCCED80BD80568BBA2E7FEB3D309D5ABCE03DC93B728EDCBDD0EF5EBDF0EF843D0212B73DCCF79BBD581DA33CC403EE3D6AFD95BD0A5FA03D9E2BB4BD1030743C0456EB3DC039603B647F69BDCC20F1BD705F8FBDC21DA2BD44F65CBD94B1EB3D9C98B1BD5A5AD93DB0FBCA3D3A0FBF3DDE3FF5BDC25FB1BD10F0BABD083CD33C986C463DC898E2BC"> : tensor<64x32xf32>
    %cst_5 = arith.constant dense<"0x6E5007BD7639BEBD418269BDC97C78BD9D850D3E6A0C2DBC12679A3D1BCCA13D781B803DE743AB3D7E023A3D5C12B13D2620023E718F85BB3AF5AEBD475498BD4F49BDBDFA9D20BE8F42BDBC3BC12EBE825DAC3D05BDB53D0016863D108F14BE8E2DAA3D32A39E3D52899CBD35B29D3D620C203E4E172BBEBD3CE93D9366D73D5BA719BD4643313E0395CE3D7D6C18BE7976AFBDB05D253CB21D68BD85838B3CDA0E933C8DE7BA3D9AB7C0BD427D213E3803713C350909BEB72C653D060D2BBDAAB7053EE4F2ECBCD49C623DD1132ABE3DFE29BE885408BE51DEA03DF3625B3D253213BE4950ADBD9E972F3E145B1C3E118421BBC78B15BE8AAC8E3B30019D3D3D640A3ECF48EC3D3527A3BDD583AC3D41114FBD379820BE6E5D5E3D4AC0313EC0A2F1BD2A92993C71EDFBBC7B94F63B2503EA3CA94A433DDA2EBC3DFF9FC53D85AD243E05E1C7BD89512E3E4DB8093DA27B8FBD31C7FFBDD822E8BB1522033EDC9998BD35FDA33D618F61BD88C0F83DB20C29BE27D0FFBDD81EF93CF7AA26BE8E90143D04A31E3ED98D40BDF993103ED4361D3E89ADD13DE9B101BDEA790E3D901E2A3E69CCA63D45E5FE3DDA7A163E24954D3D0AA8D83B6003DE3DD37485BDEB7EE5BD0E91C8BC2EF18B3D3594A6BDC3FE05BEA9F5D53D169C08BB5459613D8C0EABBCF17CAEBD186A003EE4D3523DECA81F3E0A854B3CB9B9B1BD246C94BD1B8E1C3E910D85BDF924B43C9BCBDCBDCD11D6BDF9A563BD20DDF03DF9E8C3BDAF5C943D3711673D4C2286BD2764E4BDF6C60B3DDCBE273C7A66FC3DCC977A3DDBF721BD74CAC4BD9F1109BDD4F909BE4B1885BCAF64D73D2A7F75BDE91554BD3E8BAFBD2AA08E3A56FE02BEA57AF5BD81DEDB3DDD26293E1D1D09BDE5E6E93D8EC7183DEF8E21BE64E4213D28E2863DB134243EB694B5BC9C8019BE398E90BD4109E03CA979D6BD0B219A3DB23FB33B6057EFBC99C75C3D4C98C93D968731BE31F7B93D3184E23CB71022BE3020323D7159C33D7E4215BE80A4DFBDEA9364BACD034E3DEC728C3C257FABBDAD02AA3D8118153E5E2CCBBD0454D2BCCA1D02BE7B42AABD52B5FE3CE59585BCF63D133D563384BDB9AA4ABCF878013EF1830DBED1FE2FBE79ECE33DA250523D100807BCBC3E293E785AD63DEF75153E72B39D3DCAB2223E7CE3F7BD372F843D4708053E54FF87BC1D3529BE543201BEF41A0D3E0A11A63D1C7CD0BCF0FB94BDCDCECBBD2FD07F3DECD6773C91FAA9BD8941BABDF2A316BD35808BBC56C4D63DC0F92F3EED9F9ABDE9511B3EFACD1C3EC9A3E9BD28138B3D6608E73DC5431B3E1ECA603B3DB0FEBD60340E3D2B9118BE240E343E2A7522BE21D1933DD37A2A3E2EFFD4BDEFF41B3ECCA6253E099211BE5567F1BDFE7E24BDBF0917BC5270183EEB633EBD52EFABBD15F8D3BB456097BDB28FF03DD3EAA2BDE35805BE283FF83D3B0306BD855AC03C8BDD64BC15780CBEE172E8BC1B83AA3D6B6E0F3DF7E7B5BDD434213DA9E8F23DCB15B8BC66ADD2BD55D2E5BD41353D3C095B0DBE518D453D786A45BD368C323EC61CF1BDEF4E073EB70CFF3D5F03A23DE88F2E3E736C07BD594955BD0060AD3D61C230BE6B56BABD739DD33D104C733D602D073EA1DB01BEEDCBD4BD9F57A5BD4F9249BD6E0175BB52F8F13B62A5B1BD44C7343ED06A71BC1B2CD03D760AEA3D155B2E3E24B5F2BDC0E85FBDC41128BC5306F7BDCE3CF13D2B09ECBDB59E233C09CCE73DA8D5913D983206BEB67C3A3D0573F33C3C3B803D4222F7BA035AF0BDE32318BEA65E90BD25620A3E"> : tensor<32x10xf32>
    %1 = hls.fdf.dispatch : tensor<1x10xf32> {
      %2 = hls.fdf.task : tensor<1x3072xf32> {
        %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<1x3x32x32xf32> into tensor<1x3072xf32>
        hls.fdf.yield %collapsed : tensor<1x3072xf32>
      }
      %3 = hls.fdf.alloc_tensor : () -> tensor<1x64xf32>
      %4 = hls.fdf.alloc_tensor %cst : (f32) -> tensor<1x64xf32>
      %5 = hls.fdf.task : tensor<1x64xf32> {
        %15 = linalg.matmul ins(%2, %cst_3 : tensor<1x3072xf32>, tensor<3072x64xf32>) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
        hls.fdf.yield %15 : tensor<1x64xf32>
      }
      %6 = hls.fdf.task : tensor<1x64xf32> {
        %15 = linalg.generic {indexing_maps = [#map10, #map11, #map6], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_0 : tensor<1x64xf32>, tensor<64xf32>) outs(%3 : tensor<1x64xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %16 = arith.addf %in, %in_6 : f32
          %17 = arith.cmpf ugt, %16, %cst : f32
          %18 = arith.select %17, %16, %cst : f32
          linalg.yield %18 : f32
        } -> tensor<1x64xf32>
        hls.fdf.yield %15 : tensor<1x64xf32>
      }
      %7 = hls.fdf.alloc_tensor : () -> tensor<1x32xf32>
      %8 = hls.fdf.alloc_tensor %cst : (f32) -> tensor<1x32xf32>
      %9 = hls.fdf.task : tensor<1x32xf32> {
        %15 = linalg.matmul ins(%6, %cst_4 : tensor<1x64xf32>, tensor<64x32xf32>) outs(%8 : tensor<1x32xf32>) -> tensor<1x32xf32>
        hls.fdf.yield %15 : tensor<1x32xf32>
      }
      %10 = hls.fdf.task : tensor<1x32xf32> {
        %15 = linalg.generic {indexing_maps = [#map10, #map11, #map6], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_1 : tensor<1x32xf32>, tensor<32xf32>) outs(%7 : tensor<1x32xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %16 = arith.addf %in, %in_6 : f32
          %17 = arith.cmpf ugt, %16, %cst : f32
          %18 = arith.select %17, %16, %cst : f32
          linalg.yield %18 : f32
        } -> tensor<1x32xf32>
        hls.fdf.yield %15 : tensor<1x32xf32>
      }
      %11 = hls.fdf.alloc_tensor : () -> tensor<1x10xf32>
      %12 = hls.fdf.alloc_tensor %cst : (f32) -> tensor<1x10xf32>
      %13 = hls.fdf.task : tensor<1x10xf32> {
        %15 = linalg.matmul ins(%10, %cst_5 : tensor<1x32xf32>, tensor<32x10xf32>) outs(%12 : tensor<1x10xf32>) -> tensor<1x10xf32>
        hls.fdf.yield %15 : tensor<1x10xf32>
      }
      %14 = hls.fdf.task : tensor<1x10xf32> {
        %15 = linalg.generic {indexing_maps = [#map10, #map11, #map6], iterator_types = ["parallel", "parallel"]} ins(%13, %cst_2 : tensor<1x10xf32>, tensor<10xf32>) outs(%11 : tensor<1x10xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %16 = arith.addf %in, %in_6 : f32
          linalg.yield %16 : f32
        } -> tensor<1x10xf32>
        hls.fdf.yield %15 : tensor<1x10xf32>
      }
      hls.fdf.yield %14 : tensor<1x10xf32>
    }
    return %1 : tensor<1x10xf32>
  }
}
