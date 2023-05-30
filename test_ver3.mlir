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
      %11 = hls.uip.port @p_a <input> type %2 sizes(%6, %8) #map6 : (!hls.port, !hls.port) -> !hls.port
      %12 = hls.uip.port @p_b <input> type %2 sizes(%8, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %13 = hls.uip.port @p_c <input> type %2 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %14 = hls.uip.port @p_r <output> type %2 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      hls.uip.semantics<%1, %2, %3, %4, %5> (%6, %7, %8, %9, %10, %11, %12, %13, %14) [5 : index, 6 : index, 7 : index, 8 : index] : <!hls.type, !hls.type, index, index, index> (!hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port, !hls.port) {
      ^bb0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>):
        %15 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
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
    %cst_0 = arith.constant dense<[-0.01643309, -0.0159807354, -0.0132200476, -0.00416501472, 0.0143718664, -0.0115279574, 0.0115099484, -0.00377486018, 0.01570048, -0.00858280621, -0.016224768, 0.00107262412, -0.00742499949, -0.0104035698, 0.00725634722, -0.0117612844, 0.0121261738, 0.0151887611, 0.0131708765, 0.0162942559, 0.00119492272, -0.01400996, 0.014263073, 0.00779373245, -0.0102744345, -0.0062809689, -0.00397067517, 0.00991225242, -0.0157686397, -0.0130805299, -0.00111844041, -0.0170939714, 0.0119812852, -1.9641727E-4, 0.0104117282, -0.00169158704, -0.00148221978, -0.00625276798, -0.0104510728, -0.00810334645, -0.00436564116, 0.0146538448, -0.0161085594, -0.00392595166, 0.0125637278, 0.0102614351, 0.0106048975, 0.0062730927, -0.00894793589, 0.0169414729, 0.0165913962, -0.00410439027, -0.00271076383, 0.0016998332, -2.23377501E-4, 0.0107971355, 8.133720e-03, -0.0168336425, 0.001235476, -0.011163597, 0.00330023724, 0.00679903151, -0.00557004474, 0.00868789665]> : tensor<64xf32>
    %cst_1 = arith.constant dense<[-0.0736412406, -0.0207702518, -0.0884304494, -0.113735914, -0.102729946, 0.0933341831, 0.0267892033, 0.114384517, -0.113161623, -0.0804256796, -0.102543667, 0.103653073, -0.115446091, -0.0591729432, 0.0101165771, -0.0860360116, -0.100620434, 0.0763669908, -0.0111385435, -0.100433037, -0.0118710995, 0.114551678, 0.0899527668, -0.0719346106, -0.0593000203, 0.0744338929, 0.117481932, 0.0847624689, 0.0765694827, -0.0457409024, 0.0198619664, 0.12010166]> : tensor<32xf32>
    %cst_2 = arith.constant dense<[-0.100163266, 0.140212715, -0.00315256324, -0.167290598, 5.654990e-03, 0.064721331, 0.15347825, -0.151908815, -1.337560e-01, -0.0715311467]> : tensor<10xf32>
    %cst_4 = arith.constant dense<"0xFC102B3D10C302BC3069B1BC983F343DF028B4BDB865843C206EA63C20BB16BC2CA0563D00F3C8BBCC46EEBD2EDEF5BD38C8753DBA04CD3DB0CDEB3DF6C7BCBD3AD5A0BD46C2FF3D00A253BDE826D4BD64764D3DE0F43ABCEC00533D6CD42BBD40B1943D4C6B6A3DC8857D3D0828A83D903A393CC8FD11BD5C170D3D80D2003C9C10093D8048643D10A9F53D004F7DBD783E7A3D18510F3D2408773DBC3A793D60A2313C38C0EABCCE62C23D04A1CD3DFC2F18BDDCD59E3D9E659A3DBCC4F1BDEC54703D1C37D23D2442F63D2E7D943DC22EADBD3045BB3DFA45B8BDB89CD13D0011963CAC4C633DD077B03D5A81C03D04B2F83D343191BDC8FA733D8270CF3D28FD61BD1AF2BC3DDCB5003D00AFFDBAD611B2BDF8F8E33CEAA09FBDA098BFBDC477BABD92F8CDBDC4AD27BD403100BB723BB2BD18CB793D8C7F613D00FB4FBBC2EE8CBD1E21E53D502E253DBC6B03BD8041EC3C9C7F483D40B594BC58A4063DCA7180BD90CCCA3D3CBBDE3DC0DE4DBDE0455F3C601B233DF232CCBD3CE1303DB455F23D10DA64BC602045BC9CC9C6BD2815E2BDC0B61F3BE889263D0A52B1BD988D42BD4C0AAD3DCC92813DF268F53DFC5713BD5AAA8C3D8CBCFF3D08F3E83CC2F2A43DA8C29F3C162CB33DD4039F3DE6EC87BD50F8B5BD680B8FBD54F152BD10D0E33DFC7A8C3D400CECBCA49E15BD3030F43D1C112ABDE8BDEF3CE6FBC43D38C1DABC78A5FE3DE8235BBD5810A33DB4CC0FBD6004813B7456BD3D9CBCEB3DA0FC0B3D6CA413BDCCCE4FBDCECA9F3D6E22F83DF01BD9BD983A98BC1067583D0EBC98BDB0B81D3D04DBA1BD202717BD04B8AABD42E7953D7012AABCC07342BDC6ACB5BDCC9BABBD1A668A3DC2F0D1BD7C14FCBD346480BD1C6953BDA69AD43D5ED6C7BD90D67DBC2C8857BD4A3C97BD90716CBCAE23B93D826FC0BDE894DD3D78FA9CBD8216EB3DEC92163D44ACE33DCC098DBD3C30C0BD58C65E3D743022BD5E80E13D105242BD0A6E9F3D0092823904F4C7BD54724C3D6872E2BDC629A7BD205D0ABDAC22113D806ED2BC0074933B2065823C6AB7BE3D80500EBB7850FFBC46E8D63D4038223B507F883DA422C83D7CE88E3D2E46C63D306859BC40517CBBE04FE4BCE036063C8CDF4BBD1E5E8C3DC447CBBDA4D9263DFCC244BDEA0BFABD3CED14BD9871C43D3E8AB33D184EA63D001878BD80B0B3BBF0D1CBBDD0468CBDF45C6EBDA0AE9E3CC0C169BD6C11FD3DF04D95BD38CAB03D944D943D5CEAFABD806840BC6AEFA4BD58E6E83C2893093DE42699BDCE42C93DB0B5743DC468EBBD8208AEBD8CC8C5BD94544CBD6474F0BD2070603DC0C15BBD9C1B1D3D9A04BF3DE21CA73D10568A3DB6B4E8BD04D67A3D7805DE3C486E05BDC6B3CABD2470CEBD60EC6A3D903687BC602ABC3BF802FCBCB24890BDE8A0123D28457FBDA01DE6BD1041613C147C04BD2E4D9D3D984B073DB08C3CBD681AFD3CE4FA0A3DC8E785BCDAFCA13D3444D83D249B043D0625A83DEEEB90BD6421BA3D227D8A3D8CBCF2BD16D2AF3D76ACD4BD3C8837BD0E2EF4BDD0C291BDDEF6F6BD6A88B8BD00D7273DBC192E3D505E92BD04A9B93D6EFED53DC00616BB184A383DC046283BA26986BD447851BD645FA3BD6C27043D64962BBD507D963DACAE163D587890BCFA81EFBD182C483D840A153DA049C1BCF8D2953C2C0F34BD5603B7BD7887E2BC3AFFBDBD9671DF3D00FB0F3D308880BD1A9196BD3A9E99BD7C7647BDFEF08A3DD415E83D906005BDFAAE893D240DDCBDB47686BDF012B0BCBA68C83D145C0B3DF03B303CE0DE7CBCE653E9BD6E2DB13D4E15BCBD2CA5DE3D40FAC03C44BCF8BDDCD7A6BD3030EA3C009DED3AB003DBBD789AD3BC909EC73C444E8F3D5026853CC0A2BBBDA8DFED3C585D9EBD206461BD46FFBD3D1EE98FBD70CCCB3CF624903D14FDD43DBA10F0BD9CDC41BD129EF93D58C0ABBC402CEC3B145F993D38D0F3BCA446C0BD50408B3D94C51A3DCABCA83DE2E8BABD4824EEBD36F4E63DA050D23B1006C63CBAA2953DBA32B43D6E26DA3DA81F9A3DCAC8853D7AC199BD8295EB3DCE19823D7215833D0038ECB8A091B4BB0EDFEA3D606E8D3D9469493DB835C03D109A9C3CF09F3EBDF818F9BCC05485BD4EBDD73D6645A6BD70B5C13C82A5DF3D40E3ACBCE0DB53BD30B3233CB0BEDBBD2C120DBDC4DA14BD0C8CB33D00C1F33C20C18EBDD09D35BC802AE7BBEA70A1BD8469053D7EF380BD241D643D128CDA3DBAF48BBDD465533D7076F7BCD6B49E3D963984BDA2EFF73DF44286BD309212BC0229893DF632B33D20ADE93C987EAF3D4647CFBD5261E83D8600823DFCCA92BD56F3B23D9089613C1087423DFCC390BD84CCA13D3201883D78C7AC3CEAE4F6BD546E983D08E4CDBC80CF8A3A34EF493D4031213B1C0D483DB870D63DD0BDBBBDF07533BDE001A8BD0C8F2B3D44A8143DC0AB523B7A57F9BD20F52B3DFC65DC3DFCA8B43DA04B58BC0653B7BD4CA3E13D4041A03B4C3A7F3D9E6694BD2895ED3CBAD4EF3D9C98B8BD7492C73DF01A2EBC2050E13DD4C861BDD67BF53D304E83BC8681C3BDCA3C8C3DE4922DBD845F273D7AEFF1BD4055CFBDA4CDC4BD6E9BD1BD423D883D3E1CEEBD9C64D9BDBC8497BD68DA303D04C03ABD963FF33D467EC13D0A6EB83D3C8820BD60A397BCE4E44A3DF0FBA8BD2853033D70E052BD8CFA0DBDA80C91BCB89DA2BCD84CC13D08E04FBDA0777B3DAC5A433DA028D2BBBCDFC53D0A2FDBBD6CFA6A3D2451BFBD90D4F9BDFC9B68BD1C02F03D80F04EBC3ABAC53D9CE85FBD807DBA3A10F2363C34F5DA3DE676CDBD9451773DB235C93DC06FE73CF0E3243DE8FDB43C10005A3CBA8FEA3D707C28BC10E7D6BC3864CE3DD828D93C0419A0BD7E6DA9BDF81895BD0846BEBD0836D13C6CB7B23D3081C0BDF22DE03DE08B313D00A991BBCAA4993D4E45B33DC04C1FBBBCD4B0BDC0868A3DBEA7FC3D10EFDABDA0C61A3C4CF874BD6418BB3DA0193A3D40EB9FBC0C7F4DBDCE1CFD3D8457B6BD20BBA93B88D653BDF064CCBCB0ED983DC07E33BBA2788BBDCCA6E2BDEAD181BDE285923D7A68AA3D38B198BC42EFFCBD3E89DC3D5CFE61BD188D983C0EA2A4BD4031DABB80D354BB7EF3B2BDA0FF89BB8AE9F83D18CDB3BCBC9B303D202F9BBBEC30333DC8210A3D0C2FA3BDE2FCC63D7050043CE2A8A6BDA0C4163C404531BD00F854BDF80212BD00CC21BD1C986ABD284B053D64AC813D38BC66BD8801D93C46EDB53D40360A3DE4D0DB3D0047303D38FBDF3D50A6A33D40E6483CA81C3B3D605821BCC821DCBDFEB7BDBD64E8ECBDA83D23BD060FB13D04B328BD4A51BD3D20810FBC583950BD8CC67B3DCAB0D2BD6E2FBDBD3CB0E73DB8B3EABD68062D3D3C9EDE3D9272ADBDFC11913D64F7883D8AAAA2BD9A4BCEBDD4C9123D8099F43C202D763C20587B3D90A243BC04D5F0BD2873DFBD188FB9BC2C4B583D5A81D1BD949B65BD5ADE93BD30C10BBC4801DE3D70A766BCA87B59BD40EC283C884498BCD071813D00E1DA3D0089193B545CBABD48C4823D9CBF263DC06AAA3BE80F273D6C4273BDA80D7E3D8022D2BA12B1A6BD54078C3DBE92A93DE0D001BC885F263D4A70EE3D90446B3D7406FA3D2A85D43D581B953DC06439BC4AE49EBD60C5AABD0AA181BD8A90EA3DE01643BD02C8C63D7C37AC3D7631D1BD6080D1BC38F0063D002BA4BC469FE6BD4470113DC01F763BB6BFD4BD94F04F3D7CAFECBD681D5CBD1808C73D88CF973D28D4B1BC68BFA13C2225C3BDE68587BDD2049F3DE8D9EA3D2091A6BCC0082FBD02E0B1BD80E460BB58C022BD2AB4D3BD10183A3D88F2413DBC90503D969AD33D5C9A8DBD646EF03D40E1FBBC889B2F3D0E4ABABDE0A16E3C34D15FBD9897A3BD00CED93BA8E13BBDA01F483C86ADB03D400F563B3096D63C407047BD840D63BD529EF9BD06A2E03D9A7EEEBD72FEF03D06CEEA3DE08518BD00D2A73AE6B59CBD6E49C53D229FFFBD185616BD6816C23CCA29B13DA0E758BC5E44BFBDD09D363CF041043D86909A3D506E2ABC48769FBD48FF82BDD8C908BDD4672F3DA4778B3D40A2DC3B4C4155BD8886AD3CC0A9503D76F784BDECB173BDD6188C3D0CE079BD7CB0CBBD4C940EBD68BF7B3D5493F83D7ECCD13DDC915ABDD885E13D1444863D80D2A43B30203CBCD84C573D10343B3D4029F13C04181EBDD8E491BDCCF1ADBD3254DFBD4868C13C00C52ABB20A3FD3BD4C043BD4043713DC4AF64BD68BF73BD6C5DB9BDE01E87BB00E2023AB6E2943D26B6DA3D0822CEBD28F9A03CA0A966BC4C389BBD9CB3963D6691FA3D5C6C1B3D9A42843D641728BDE0684C3CEC16543D24B98CBD9C680A3D70D118BD5419813D80A0A5BBE88F253DDE89B63D88E385BDBC62E3BDACDF113DB838FA3C367CD23D206530BD8CE6E1BD9ECEF1BD40013D3C407C543B74315D3D842AA9BDD0E0533DA876863C34D3EE3D9290A73D5EC0883DA0EFDABD52DFF0BD807FCCBD34AADABDF00EA5BDD8D5813C98A440BD9CDACE3D185C99BCF464FCBDEE0183BDB0D40A3CC0F5533D8C5242BDCE7DE63D8A14CC3D5812A4BD48338BBDCA3EBA3D462C9F3D6023D6BB84D8943D164889BD18640BBD609BAB3C865C8E3DA0ADF53C247CF8BD48D194BC78BCC9BDC02172BC40BBAABD2C71D63D3009E2BD9E70B0BDE0FDCCBDCC8F87BD7E7AE0BDB24180BD96C4FC3D9C96E13D9080FE3CA658A8BD309D743D6CBFC2BD8414ADBD28A9BA3C12A4B73D6E5A92BD60E021BC7E4BDA3DECFF2F3D20082B3DD85D20BDCC7E3DBDE269AE3D2004CEBDA0FEF7BC380625BD581D79BDFADD8C3D38EB7ABD9C642B3DF00D7ABC986BD7BCD29086BDECD1C7BDE01CBFBD500CC63D0406B53DD4AD16BD00A946BD447223BDF85DAA3D6EF9EF3DEEBCBE3DC84E28BDFAFF883D729DDA3D0E13DEBDF06A49BC5814F8BC48607EBD20F5913B3E1AE73D724AE63D80898DBB50B0123D700496BD6A2ED63DBC413ABDF209E2BDC0629E3B3A2AFABD386EDC3C40FC02BDCEA68E3D547AA43DE0DAB4BC308B01BDC62B8B3DBCFCA6BD4C88D3BD220EB8BD40BBA23DD2C7863D345B6EBDC86CF33C60DB3ABD508D713CF033D2BDBC8C213DA8C3DDBC342EF43D60B89BBB502F4F3CCC0F203DD2D3A03D40D668BDF85A6B3DC0885D3B58B2AC3CC8F1A6BDFCC471BD103E013CA474AA3D48E3DABCD0C4E6BD940B43BD20D0EFBB8C4C70BD3C9CD6BDA205F2BD14228ABD3C39C6BD649431BDE4F3173D9211E7BD506D893CCCD816BDC8D59CBD14B800BD70559ABC3AE1B9BD8879CF3C18529CBD3889CABDAC5FCEBDB2A1B43DD8AC943CF8FDEC3CFCCBB63DB8AF4C3D64EB2B3D16269ABD563296BDDCF0DE3D80A132BC4EE1E1BDD436ABBDF445F33D2826843CCAC78F3D081865BD9813373D1839FEBC2CCA4C3D80B1DCBBA0308D3D904BBEBC66C8B53D0032F03982B8E2BDE259F3BD840795BDAC4AF23D1CCD11BDB23AEE3DD81BA63CDE0D98BD643C753DA845ED3D8AF3E7BD2059F1BB7052ECBCC42AE8BD9ED2BF3DB097103DB02B80BD3645933D9A2FBFBDBCBAD63D8C9288BD60FDC83DF83F243DCCC95C3DD62198BD904498BC4689EABDF0B8A9BCE44C09BDC2D9F3BDD05ABBBC929C94BD9443613DD687A83D98066ABDE0F696BC1A51E0BD7CF2ACBD7453DCBD5C09713D5EB9A43DAAE2A2BDFAFCD0BD80FDAE3B3C8562BDE4B34F3DD016653C7034DDBC20D0D23BC45F6FBDD05EC0BC000014BCF816F73DBC4DAC3DA468743D782134BDCC7A733D900F66BD0C0700BDA0E0BC3BA8E3943C923D9FBD106A0B3D0E17B43D54D2753D385592BCD030043DAEFAE43D0E13C03D009B5C3A98B3973C700E6ABDF218BD3DE65FB9BD203ABCBC00E53CBAF2C2B6BDFC806BBD985C68BDC603FF3D800EEF3A0090F33C605BC83BA2FDDC3D6E73AABD762FA8BDB8D1AB3C40F54C3C605386BBC8EF3ABDE07DE93D0846433DC817A2BD908E00BD545B8F3DE86B47BDC239EEBD124BF93D4AC5F5BDDCE886BD9457413DC2BFB73D380ABD3CF65688BD54EA033D606A633DD08FEDBD804EF93B986455BD8CBC5D3D809CFB3BC04900BDC08DD3BBFC0EDA3D2600C93D309E57BC6ACD99BD46D8C2BDD809EFBCE0A1EFBDC4BA253D90DFB9BDD0940DBC9EB2B33DFAC7803D9AA1E23DC0E813BD4892723D1AB4A73D90D7F93D4C15133DF44F6DBDF0F01A3C6EB4ED3DF0A7FABC28DE013DF85D6FBD40FD2F3C8684833D346351BD32FDAA3DD02D903CEE3DC8BD4833AE3C3AB9953DDC1C663D58F1753D40E0553D4C57003DA80FEABD587CD83CEC85FF3D2C39153D64764C3D7637B93D5497983D80069B3B9896D4BD544A743DE489C53DECB6D2BD249C30BDE2CCD03DD245DCBDC4D57EBDC299A9BD22EE8D3D844711BD40AE2EBD505526BD6267C3BD98B5853D8202FA3DA4B331BDD6BBE03D08BCFB3C743BF7BDD8505F3D40195CBB3A9BEEBD82508B3D22CFBB3D18BBB03D14760ABDF859EABD4833143D9E27CDBD8E649B3DFC177C3DD2C0DBBD3079653C7A7EEABDD491BCBDE453ABBDD8A88B3C84C94DBD70F340BD44A16EBD4C086C3D44E8CFBDBCE549BDD054E1BC5C42A6BD5E78F63DE6CE823D4A7AE83D00D1F53B00559ABA2CCFFBBD18E76ABD80C0FABB10BA17BCD01A7CBD428DF13D60CA243D703E2DBC9C2EAD3DAC918FBDD4D977BD9257D2BDF89BD6BC806059BCFCE02C3DB010253CA40BEDBD00451CBAC82FC4BC268CA6BD04EA333DE025E6BBE0B60B3DA636A63D36D7D43DB451AC3DD61F8D3DC0D305BC2080323D408CBFBC782544BD40FD3FBDA41D033D4057913D8865D53C8C15C3BD1C4BFCBD60E91CBC42C487BD0A6EFFBD368FFD3D30C55ABD0CE90ABD10DBF13D505BECBC1C385B3DE00FA8BCB27CE13DBE6FBB3DBA2BE53D9A0AA93D80C397BACC961C3D4424363D48DC66BD609A283D109BDDBC8003073D2CA67B3D7021B0BD30BC153D78A5003D0CDBA83DC050173B247A7E3D200642BC2CBC0A3DD031F3BC000FF13DE68D9B3DB8385FBD900398BCDEE0EBBD4862D63CE03463BD9EC4953D5E4FD93D0086443B005E293C22479D3D886D1E3D409AF2BDE808A53DC01AE23BBA46B9BD689490BCD41932BD1C90CC3D6067C63DA09951BDCA18A1BD44EAB0BDC8C7C43C0C89C4BD0075653C34324A3DB6C3CDBDF0E6573D00C0873BE072BFBC10945DBCE249D8BDCAE588BD44CB4E3DC04883BD90838ABD70F94A3C189343BDE6C5EA3DA05C7D3CFA778FBD4C61E33D221A9DBDCE4790BDC047E43BB05CD93D10EBC03C60B28D3DE0A019BDA0A5FBBCBE7DA43DF8287F3D603D853C001EDBBA0C4D80BDAA02CABDD025013DC0E2B93B9023833C0EF3D1BD1061A83C1EA8C7BD00A2D23B2892EDBC84EBC33DD0202CBD48FAFD3C68288D3CFC1E62BD88C39FBDF4E6E63DA055CFBCDCFE513D9292F13D48D6173D400446BBE0B1B23DE4185E3D1CA7D93D28D7E93D48F6D4BD863FB43DA0F0CBBC00C8C8BAB8981EBDA0E3BA3BCEB0BCBD206506BDB873BDBC4CC3CDBDD0C195BDD83A29BDA050D3BC2CEFE43D7CE94E3D48831DBD7410DD3D44354D3DDC3D1ABDE0E09CBD3C755B3D381DCBBCE8F6ACBDC449DDBDCC9D19BD9CA74BBDE026F93C1E22B73D74ED64BD604FEEBCA828CEBCAE089ABDF864B3BD08DB74BD7E85BC3D94A58F3D74CF01BD88DEA43DE430893DBC8E05BD3E7AF33D9C237CBDA07C0BBD86D8A6BD3C00F63D86A9CEBDDA95983DBC815A3D3835C13C8C0104BD4670B1BD98646A3DE05D86BB663EC13D285CD2BC481490BC586AB3BDA80692BD884747BD4415793DB46DF5BD08C966BD46A483BD1824E83C083EF43C206F1B3C7A48F23D8CE040BD528BD53D5CD2FE3DA078E4BD9403313DB03E57BC0ECE8F3D0490A03DF8CAC43D98CECDBDCC2C773D26C4F13D885F603D005F4EBBBC6FA5BDBEF0EC3DCE1D92BD881F5D3D90244B3CA67CAEBD7035653DD02329BC9E2E9B3DEC86963D8C5F0E3DE4B3033D7068423D5AEFB5BDF443653D785AA7BDA4F321BD183DB83CD47FBFBD9A2DCEBDC69EBE3DD054EB3CFAC386BDD4CE223DB671D1BD923494BDE40DEC3DB2B88CBDA288AC3D009E6FBDC079AABCA02D5ABC1294813D168EE93D68CB28BDDCA1F93D40B9C83D84562BBDF298E6BD681FB1BCC0453F3D1AD095BDA43CB93DF26D843D545EA5BD703238BC907C953C3CB9DEBDE09A38BC24DD86BD3085013D408ACE3CF4DDFC3D7CCBBA3DCCA9E93D18C529BD0080EFB642F6FDBD2C0A223D44E4AB3D2828423D585DF73D325ABC3D50CADEBD945E403DFC5A273D3626953D626BA43D00A804B9D0F6823CE0E217BDE85C21BD9C3214BDE83670BD6637D6BD8A11C5BD60386B3D24DC74BDBE9584BD281CC23D84E8103D085C683D1CC56B3DF068EDBC1057413D10D475BDC232A73D54A196BD50D413BC6042F4BC80A76BBD1C14D53DF85CC63D801DED3B4E3AFA3DC00857BD6E5DCDBD14FDE5BD5252CB3DA0C0ECBDE4885F3DEEC2E73D9E2BC13DF2ECB1BD48B1D53D285A70BDECC1FC3DB2FFC4BD50FB9DBCE0A214BD9CD6803DD42B023D2CE1373DAC33343D12B0F03D3ED5883DF476963D9856D33DE4EC5B3DD4D6EC3D88C3263D78DED1BCCEA794BDF893A5BDE0F6CBBC5E8CCCBD6CFA8BBD5040903C0CFB1A3D3002A23DC812F6BC3A8AA73DF452013DA404ADBDB284D83D78CECABD5CB766BD70D2ADBD80CA4D3D5C86283D74EAE9BD1430EC3D4C92973DD812AEBD447A13BDCC8FBBBDF44F483D1869B03C6C540DBDB0ED37BD58A5263D703B793CC0C598BCB6BE9C3DCA3AC53D60A299BB80DE753D508B713D1822CA3CB8CC3FBD68B11ABDC0F4963C60E66FBD561A9C3DB278EABD9058A7BC9430733D30EA163DCAAF8ABDD8A7A0BDF6FA98BDE632A13DC4E5233DC08A0B3C16AF823DB46F1BBD28EC763D0A81DE3D849B45BDEC636E3D483DCBBC302E6BBCF6F6E03D18DDF43C30A265BCB640F23DFC8F87BD6C763F3DC26E943DE671C63D507606BDA4E0ECBD9C469DBDD016743C98F270BDEA3AB0BD6EE9BEBD4EEEAEBD8C3FA63D08149A3DB0E30C3D00A82DB964BC7BBD14B5343DFC6E3C3D700142BDE0AC21BC6071493CACCA813DA262E13D9054FEBD9070DA3C905F3DBDB2EA953D4A08A7BD3EDCF1BD7E41EFBD40C122BD5EFBCC3D2EFEED3DD80BD7BCB44E32BD24AF63BDAE5BC83D9CAF7A3DB8B3873DB802053D7EB69EBDE02F6A3DB0B60E3CE0A2F63B20A247BC309F29BCCC90333D3A24DEBD5860F03C40D319BB6031273DF090A13DA026FD3C00BA9D3D001748BD62C6AC3D2498D33DAC34343DFE87D63DFA8D9E3D1A25DBBDB220C8BD606C473C06CF9EBD5205DFBD303B1EBD08C48C3D78C5833C1C36B5BD40ACDFBCE2859A3DB4C7EF3D6CD9E43D74640EBD3CDF87BD98FDE33D4677CABD80E8DEBB5654923D789B493D863BB93D683A8B3D4C838B3D98AF81BD9072383DDCFD21BDFC3BCFBDB213B43DBCAE68BDB27A89BD7C9C99BD2072F83B48F9623D7011693DE83A37BD740B403DCAAA9B3D7074B1BDC4B1DABDEAD9E03DA0D10D3D18DC40BD102DD1BD704EA13D1A1B8C3D8884A13CA059713CC22AE63D647F3B3D7A9FE0BD94764E3DD8D3413DE048E0BDA00342BD286AE43D90EE3B3CF0FA1D3CBCCA9CBDF89D993CB09DCFBCB85BDB3DACF1923D1073D3BD086ABEBD7C27803D24351F3D024CFBBD307490BCB45A2A3DE84144BD68F5C5BCA85E8E3C8030F4BC180CDFBC9456DDBD5E9F8A3DB49FE9BD24E19F3DBCE75E3DB01062BC80EA7F3CA0BDB63CBA02D9BD903EE6BDEC12223DA07ADC3BF0A4273D907EE53C681BE5BDF077C83C48C909BD8E66D13DE094D73D80B3B03CA41435BD9299A93D4696F4BD9C7FD1BD003CFC3D74A470BDE0FC003D909980BCEE00DA3D50BC7B3C485188BDB0E5F7BD205CBB3BE062E53BB04E6C3D8223A93D80B6383DFAABAB3D72FCCDBD3640B13D38C0AFBC3CCF41BDD0AF94BDE4CFB3BD1815CB3C60B68BBCE06A8BBC00ED5CBD34451BBD3E8AA6BDE08DB83B326587BD5667A43D00A3263D08629F3CC6578DBDD8C68C3D420E8E3D545C12BD1413B1BD48B248BD2065B2BC6EA5EA3D70D5D7BC08A13E3D3464D03D220D93BDC8BDAD3D4428C7BDE0C6313D786C933C7AB3CCBD240FC73D5233C23DBC5C81BDE09DF7BD36A9B7BD90990B3CB0A6553C527885BD26A3BE3D48C1433DF240CBBD30EAD7BDE0AE6FBD80DCDABB18245A3DA88EDBBDB28AE33DE08D503CA0E8743CBE7E9E3DC863CD3D589B7D3D409C1F3B08BC0FBD009BCDBDD8DF123D5A6CD0BDE8F527BDA49456BDFCA8903D2CA7E63DE2D7D23DDC4B593DA4C10F3D7CFDD8BD4807793D9094D93C48FBA73DC0ECD23CF647843D364AAB3D221A83BD50ADDFBDFEA0AB3D089FA63DAE2AB3BDD8A5913DA87F1FBDF4CF2ABDB885523DBC2BEABD80B2333B40D6AFBBAAE1903D18E08F3CFC254CBD80D26A3DC4DA03BD1842D3BD581E5DBD2AB8F73D900CA63CE4D06D3D00289ABADA1D8C3DD8FDE03C84C8AFBD7EA7E53D9C01083D1A9EAD3DE89FDABC60FDF6BC8C15CABD90AE433D744B813D7ED1C93DB073A2BD00CC823A507CB5BC301450BC2059323C02DFEE3D780DAFBCE685F0BD40AC8DBBF026203DBC59813D1CA08D3DC88ADFBD889DDC3CD09668BC2C173EBD3C8565BDF228A3BDCA75C53D901103BCC01CE43C3408B8BD607E2B3D50564D3C70C128BD9C0412BDB87E73BDA8E2F1BD1676C63DB8BB983C10B0CFBDE022DA3CA67DD53D78496B3DA08F17BC4A989F3D0C91153D3EAC9CBDFAAF9ABDC024ED3D74A97F3D0A85E4BD306BA53CB8DC503DC882183D407281BD44D6F03D44EB21BD806F813B3821473DE856B4BCBCCE1ABD4822B2BCA675893D4A1CA3BD34D45E3D4A98BABD145FE3BD949225BD401AADBCF49F2B3D58EE0B3DA007FA3B7C81E6BDA0B209BD78F6BEBCD8B21D3DDCFF313D66A9D83D54D3663DA6C2E8BD7CE6A9BD5A06973DBE1CD7BD70E27C3C6ADFCABD8433DCBD14FDCC3D6CD8413DA4A684BDCCAC71BD0C03483D0EE0EF3D58E8E43C8039A33A409AB2BDAA58CBBD8CCEFD3D1E9A91BD8C334D3D76309ABDF6FEDD3D0090933CF86C1CBDA03735BD2C0D32BD40A5713D924EE73DE098173CB0B469BD6EE8BB3D302E8CBC0E03A4BD1616F0BD84F28BBD00DAD9BA14A0173D00BCB53BC0D035BB90BD5F3D98898DBDD010533CAA53F33D8073AEBAEC4A15BD30FF39BDB823A73DE48371BDB0EC16BD4EE7A3BD40EE86BD00B221BC4A769DBD389C903DD0C6C4BC6639F03DCED5853DCCC02E3D9828A0BDC8C6E2BC6C36603DD831C23D92B8EC3DD236B2BD84730BBDD829C0BD7E4BE5BD60C875BDF81883BDEC9911BDA849A4BDDAD0AB3D005AE63AE8EC243DD60AB0BDF0B1843DA04A5D3D8090A3BCBAB7BABDBCC3B63D745C243D94F186BD8E02EBBDEC214F3DC8B0CC3C683B033D20A7ADBDF633DBBDE8304C3D76B9C23D10D3F63D7CC8EEBD42B48B3DE0A10DBCF405E63DE0BC82BB983228BD38AED93D804887BAB0221A3D9C4EF3BD78B4DB3CD6F2AFBD2840943C"> : tensor<64x32xf32>
    %cst_5 = arith.constant dense<"0x8529B53D5D9F5FBCD0DEE4BDCBCFC13D299BDEBD2C506FBC81AA2BBE5E9FFE3DC78A05BE287994BCE181843D6CD0273EEE2F163E20BDA9BADF7F4F3D2A7E283E9A6206BDCA927DBC8AB6213DC5A01D3D58A03CBDE83E293D25A23A3D10F0353CB4032F3EF2DA9C3DECC30E3ECAA20E3EB4DB1ABEA65B37BDC9131A3D2390533BCD531A3E49706E3B0A6B10BE9DC01DBD3F0F26BEF08173BDCCC73A3DCF30BC3D606D1DBC6891083E33571EBEA938AABD5A9BC8BC5B31D63DA8667EBB6831073E3F6F31BE4E1896BD8E57E0BDE9CE2B3E95B12E3E199A16BEE40F1F3EC390973C42C0993C8D49003E9975083E3860B3BAC065843D572D14BEB3172A3E9E17BEBC80FD39BDEAE9CF3D827DADBD52FC0E3E16D70CBEF190343D67EACF3D1A9B133E7E42673D92AB983DAD21873D544E433D1B2007BC0B27B3BDC8F5533C9412333E57CE703D8A86293ECE0B9BBC1549BC3DBDF5E8BB7E832CBE603CEF3DF7330A3E3A648EBD7CDE8B3DD22CB83BE01C793DF61128BEFF3279BDC1F6123EEBA3E3BDE110A53B12243C3D337EC3BD84C763BD1559A93D170226BD8392183EF125D03D8E069BBD4ED8103E3A9233BE30268ABD519C6BBD78CF1FBEF8E32EBC7CF8033E6309143D156EC73DCBB513BDAAF9A6BBB429C93B37C02ABE9C1245BD441159BD88F5763C9E7E24BE6E36C5BDB232383DC4327BBD79E5DCBD781F953DC4BB8CBDD15E06BE0EF8EF3DCE85383DFA0032BEE3CCBBBD27653BBDBDCBD8BDB865BA3D0CB024BD4B4C7E3DDCCFAA3D57D9EDBD7CE82DBE5D0C133EE743E83A31B1963DC269BFBD0B25AB3C6C54313EA4A5213E901BBA3D4FBB05BEDE07D23D49C6C23DBDC6073DB6A90EBB05A317BEA232AFBC78F954BC5DFF81BD840CE83D26F3E53D48520BBEBFC6CDBC14E6AEBD8515793DA0F60BBE2BDFE1BCB6E19EBD8696293EB70126BD6BFAD8BCF46190BD005E233E8983273E7D51213ED606A73C4EB2563DD443DF3D56E77C3DABC6DCBD2EE9253E445BDE3D76AD19BE1B57B1BDF1BFE2BC1541F33C3F540F3E9C692EBE400D193DD7B9C13DCCD8023CE79F6EBDC080823D6C0DB13D9F3DD43D9DA8E7BDC6DC16BEECF3083E735926BE1C41CCBDEF69DF3DAAD82FBEAD3D843DD20905BE44DE653D746E483D64210FBCA442C23D183308BE46FBECBDCDA1B83CD301203E52F87EBB7B970B3E95A880BDA83FE63D5167173EDD884BBD2F952C3E0AF91B3E18BB0ABE52D2DF3D1CDBCDBDB02434BDAA7EC3BD979530BEED3F743DBA1F293D397F20BE1F60983DA4511D3E54F30C3D0E8CC5BC7C453B3DECBED4BD7F6C46BD2C5B19BEB39C513DE10F50BC9D501D3E404B44BA185596BDD84232BE78559E3D10CC7D3C88E10BBE08B3023E6C1B703DC559203ECADF8DBC5E17A0BDDC058A3DABDD26BD05EF943D4CC31D3E0CB50E3D8AE79FBDD301CE3D347F82BD86E475BDA1F789BB81D10DBE9062163D656F153EEDD29CBD1131D03D08677A3DFC805EBCCF3887BD28D4993D9D1D2CBD70A8053EC7D757BD674D18BD8DC512BE1CD1C13CEB78C5BC18F9213D2290A1BD726AD63D7F4A3CBDFC4347BD54B8B23DCB924EBD571A673DBAC616BE5EE0ED3D1D99D23DD45011BECBCD41BD144AB33DDA79873D46E90F3E0D494ABD63CEF83DB2B7AC3D90F42EBD2D7F853C2586C73D09FA033E8516EFBD1E92FABCDF20FE3C83E4CDBDD366183E45F2E8BDC00D513D92B323BEC569E9BD9E0CBB3DCFE1F43D3DCC013E5D08F63D2BBFE23D939B37BD3BA9593DB666233E499A2C3D057590BD162C2EBE0BE7FBBD"> : tensor<32x10xf32>
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
