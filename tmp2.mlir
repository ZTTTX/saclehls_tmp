#map = affine_map<() -> (0, 10)>
#map1 = affine_map<()[s0] -> (0, s0)>
#map2 = affine_map<() -> (0, 32)>
#map3 = affine_map<() -> (0, 64)>
#map4 = affine_map<() -> (0, 3072)>
#map5 = affine_map<() -> ()>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> (0, d1)>
#map11 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "MLP"} {
  %0 = hls.dse.space @global() : () -> !hls.space {
    %1 = hls.dse.space @task0() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map step 1 {value = 4 : index} : index
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
      %8 = hls.dse.param @tile1 <tile> range() #map step 1 {value = 4 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map2 step 1 {value = 4 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 1024 : index} : index
        %15 = hls.dse.const_param <template> {value = 32 : index} : index
        %16 = hls.dse.param @t_ParEntries <template> candidates [2 : index, 4 : index, 8 : index] {value = 4 : index} : index
        hls.dse.space_pack %15, %16, %14 : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
    %3 = hls.dse.space @task2() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map2 step 1 {value = 4 : index} : index
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
      %8 = hls.dse.param @tile1 <tile> range() #map2 step 1 {value = 4 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map3 step 1 {value = 4 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 1024 : index} : index
        %15 = hls.dse.const_param <template> {value = 32 : index} : index
        %16 = hls.dse.param @t_ParEntries <template> candidates [2 : index, 4 : index, 8 : index] {value = 4 : index} : index
        hls.dse.space_pack %15, %16, %14 : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
    %5 = hls.dse.space @task4() : () -> !hls.space {
      %7 = hls.dse.const_param <impl> {value = #hls.impl<"" _ "">} : !hls.impl
      %8 = hls.dse.const_param <tile> {value = 0 : index} : index
      %9 = hls.dse.param @tile1 <tile> range() #map3 step 1 {value = 4 : index} : index
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
      %8 = hls.dse.param @tile1 <tile> range() #map3 step 1 {value = 4 : index} : index
      %9 = hls.dse.param @tile2 <tile> range() #map4 step 1 {value = 4 : index} : index
      %10 = hls.dse.space @default(%7, %8, %9) : (index, index, index) -> !hls.space {
      ^bb0(%arg0: index, %arg1: index, %arg2: index):
        %14 = hls.dse.param @parallel0 <parallel> range(%arg0) #map1 step 1 {value = 2 : index} : index
        %15 = hls.dse.param @parallel1 <parallel> range(%arg1) #map1 step 1 {value = 2 : index} : index
        %16 = hls.dse.param @parallel2 <parallel> range(%arg2) #map1 step 1 {value = 2 : index} : index
        hls.dse.space_pack %14, %15, %16 : index, index, index
      }
      %11 = hls.dse.space @vitis_gemm() : () -> !hls.space {
        %14 = hls.dse.const_param <template> {value = 1024 : index} : index
        %15 = hls.dse.const_param <template> {value = 32 : index} : index
        %16 = hls.dse.param @t_ParEntries <template> candidates [2 : index, 4 : index, 8 : index] {value = 4 : index} : index
        hls.dse.space_pack %15, %16, %14 : index, index, index
      }
      %12 = hls.dse.param @candidates <impl> candidates [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] {value = #hls.impl<"vitis" _ "gemm">} : !hls.impl
      %13 = hls.dse.space_select %12 [#hls.impl<"" _ "">, #hls.impl<"vitis" _ "gemm">] %10, %11 : !hls.impl, (!hls.space, !hls.space) -> !hls.space
      hls.dse.space_pack %7, %8, %9, %13 : index, index, index, !hls.space
    }
  }
  hls.uip.library @vitis {
    hls.uip.declare @gemm {
      hls.uip.include "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp"
      %1 = hls.dse.param @t_DataType <template> candidates [f32] : !hls.type
      %2 = hls.dse.param @t_IndexType <template> candidates [index] : !hls.type
      %3 = hls.dse.param @k_KBufferDim <template> candidates [32 : index] : index
      %4 = hls.dse.param @t_ParEntries <template> candidates [2 : index, 4 : index, 8 : index] : index
      %5 = hls.dse.param @t_MaxSizeC <template> candidates [1024 : index] : index
      %6 = hls.uip.port @p_m <param> type %2 sizes() #map5 : () -> !hls.port
      %7 = hls.uip.port @p_n <param> type %2 sizes() #map5 : () -> !hls.port
      %8 = hls.uip.port @p_k <param> type %2 sizes() #map5 : () -> !hls.port
      %9 = hls.uip.port @p_alpha <param> type %1 sizes() #map5 : () -> !hls.port
      %10 = hls.uip.port @p_beta <param> type %1 sizes() #map5 : () -> !hls.port
      %11 = hls.uip.port @p_a <input> type %1 sizes(%6, %8) #map6 : (!hls.port, !hls.port) -> !hls.port
      %12 = hls.uip.port @p_b <input> type %1 sizes(%8, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %13 = hls.uip.port @p_c <input> type %1 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      %14 = hls.uip.port @p_r <output> type %1 sizes(%6, %7) #map6 : (!hls.port, !hls.port) -> !hls.port
      hls.uip.semantics(%11, %12, %13) -> (%14) : (!hls.port, !hls.port, !hls.port) -> !hls.port {
      ^bb0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>):
        %15 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %16 = arith.mulf %in, %in_0 : f32
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
    %cst_0 = arith.constant dense<[0.009907878, 0.00459480658, 0.00633573253, -7.179910e-03, -0.00368875079, 0.00217673369, -6.364190e-03, -0.0119520277, -0.0166412201, 4.934740e-03, 0.0128686223, 0.0128187453, 0.0048067891, -9.59986879E-4, 0.00861522927, -0.0153386351, -4.63625882E-4, 0.00929156504, 0.00853155088, -0.0121426722, 0.0022998068, 0.00773415761, -0.0052546151, -0.0168961883, 0.0159659591, -0.00210556388, -0.0074358033, -0.00787711889, 0.00336780446, -0.0168454461, -0.0138732083, -0.0148228798, 0.00275536277, -0.0168636125, -0.00547274901, 0.0101003293, 0.0149516352, 0.0160753801, -0.00456111832, 1.396810e-02, -0.00586819649, -0.00148204772, -0.0149874976, 0.00589617435, 0.0129158497, 4.587410e-03, 0.0172967967, -3.94286326E-4, 0.0149347559, -0.00917499419, -0.0168312378, 0.013043304, 0.00826549716, -0.0031133953, -0.0170203988, -0.00911680422, 0.0172713641, -0.00572223729, -0.00789839868, 0.00946812145, -0.00475389464, -0.0176625084, -0.00237407163, 0.0169635229]> : tensor<64xf32>
    %cst_1 = arith.constant dense<[0.0667207241, -0.0436323732, -0.0211737752, -0.0792340189, -0.0357274711, 0.086349681, 0.103366762, -0.0648124814, 0.0134359896, 0.013658464, -0.0218243748, 0.0140103549, 0.038062349, -0.122464672, -0.0464532673, -0.0907561928, 0.12265116, 0.0398997068, -0.0859652161, -0.0534368306, 0.101274341, -8.072470e-02, 0.0500278324, -0.112598598, 0.0173958242, 0.0782292783, 0.00800904631, 0.0653923154, -0.033252582, -0.0130667984, 0.0931779146, -0.0263978094]> : tensor<32xf32>
    %cst_2 = arith.constant dense<[-0.0893619582, 0.0512764454, 0.0217502173, -0.103750639, 0.121906452, -0.00669692317, 0.0754107609, -0.131430611, -0.133836761, 0.08827883]> : tensor<10xf32>
    %cst_4 = arith.constant dense<"0xA278F13DD654ECBD50FBF83CA09CD8BC5AB3E23D5AEF87BDEE5A92BD160A8F3DC0FCF5BDD440A63DC210F13D4622EABDA02D5FBD2869FA3DA6C5E2BD882941BDA835693D764AC63D006AA63A3CD79E3D5EF1FE3D52CAA63DA2F781BDCC086BBDD81278BDD0E983BCC49C26BD70C3CCBD185067BD6C7269BDBE148D3D508DD43D049D49BD0E74DBBDA0C7D43C58268A3DD06853BC86F6813D801C353D6827C13DC685D4BD3C686B3D7C683FBDD04602BCFAE8F2BD305B21BC18F38BBCF04ABABC1858CE3C4CE4DABDACD35ABD5460F23D080F60BD86B988BDF04CD0BDBC202FBD30AD6CBC66FAC9BD80A5C3BC2075CDBB40165C3B36CAF4BDF41C81BD0AC9993D40AF643BC850E53D76F2D3BDA6A2A0BD7C0D59BDFE2DBABD281E14BDF090223CD8E5813D8440C2BD5C6B9DBD4855BEBDD06D503D2667FDBD98961DBD740C703D88594A3D20FC0DBC98F3103DE059C43D68B99DBD229C853DDA37CEBD1AFCC23D4820F1BC5C033D3D1E05B2BD84EC1DBD7C9F7D3D608517BC78B7F13D70583D3C2469603D008D38BDE4BA1B3D4EC2E13D2C887D3DAABEC33D5860573D86ABD3BD987ABCBD2E34EFBD5EFDB8BD384E92BD905FA33C3417243D185DBCBD8857A23D201C58BC484DF63C14DE4CBD20CD4BBC9C55CE3DB8C387BC08E2CC3CC8184EBD5013743C58E3E03DB49911BDB4BA823D50877EBCC00C78BB98EC9EBDC2A8F63D4CA35CBDC013A03DCC80483D5CA33D3D922FC2BDF0FD0A3D80FE363D047BE7BD88EF5BBD4645D93D94A578BDA0E9A23BBE3FB2BDA230D33DE43E2FBD087C923C94C458BD7432CABD80F894BB1014B93CF098D73DC6B0ADBD72B0A6BDBE91B93DE45C163DF0A6B5BC20951C3C1A04B13D649A05BDF0811A3C70DFDFBDD898D1BD80208FBB14A08CBD1283DB3D0A2EC93DBC005DBD0C9EECBD1689F63D085CA6BC001C41B9C4BFEB3D08D90CBD905258BCCE5190BD74B4E03D588E75BD389209BDF4891F3D802FC23A209DDA3D5C91DABDC82290BD004036B95034B43D86D3F83DF6F5CCBD18C6153D6EBAE63D1C08C9BDB80E5F3DA4E4B43D0E88ECBD98BB9F3C90A4BA3CE454D73DBE73B0BD5840D2BD6820F53DB219ECBD64CBFCBD7691B43DC4DB7B3D2C93313DFC3BE0BDFC5F3E3D92C3933DD2558BBD80893BBB58F79C3D0831B03C0C7F5FBD98C5083DA864FB3C80A8843ADC072BBD441DF93DD87ED8BD9001D13D728DBA3D68B5913C0CDE57BD7289CDBD80148BBC0002EF397404433DD8B4B03C50E894BCDE4292BD406C173D722FC6BD44C4F1BD628EE5BD50DA2D3C28286CBD085904BDF8A1F93D4049543D403F153B082C373D7E0BF7BD107718BC42AF853D28D4FEBD600C19BD801F67BDEEF5EC3D12A7C63D90FD8E3C6008D6BBA0C4F03DD2C7ABBD902BD1BC1050B33C68DAB3BD401AE6BDD08682BDC85A98BD02ECD83DACE617BDD83836BD5064513D380D0E3DAE03F8BD8034D83D66B3B7BDE854F93CA06D993D8ED8A83D6884CD3C34F0163D2AC0933DD0E05E3D58B729BDAC8A8CBD4886DFBD402ECDBBC2A5C93D0082683CA09E283D50DA73BCF0DCD7BD083EC0BCD0A73D3D1810B53D18B2EA3D6CEC6BBD960BA7BD5885A43D360AE5BD5C7BBBBD20EE913B809DA53BC4C3993DA081F33C00645C3B60F46A3C30E6A4BD5E26ABBDE07E973D7805E23DD0C333BCC0B2A53D3023E1BCC84CEA3D6ABE83BD88D3B5BCB0B2AA3CC08E713D4E6BE1BD485968BD028ED4BD1EDBC6BD344B8BBD6E94A8BD7016DB3C5E549ABD40ACBF3B4E759B3DC293993DA802B2BCBAACC1BD4E0383BDD87B9DBDF82BEDBCFC914DBDF02937BC5071D6BCB036DA3DB4ABDE3D6E67C33DE8C9093D603CE73DD212D7BDFCD40E3D4CA46F3D4A83F3BD003F503A301D19BD1047F2BCE03B803D4611ABBD42BDC53D70BBC6BC9C39AB3DECC7163DC0261C3C14E8DF3D788F0E3D5CB801BD0E3BBF3DC0CDB1BB6006EB3C003E293A64B3663DDC04E33DF603A5BD64AA603DE402253DF887E93C4C4F343D60D6C6BB869CCC3D7657A03DC814583DC4B458BD38B9463DBA03EA3DE0D2AC3DC2DA963D9C8BB8BD50F41BBCC07DDCBC5E25FBBD90347ABDE82A42BD78D5963D865EDEBD18AFF1BCA073DFBB3A3E8BBD0C0C5F3DAAFFD53D284A81BDEEE9F63D6AB0F73DC687FDBD38ED01BD143978BD143B333D506CD5BC0046843DDE9B983D1C6639BD884474BD508D37BD04CC643DD84DC33C74400BBD3C4965BDFC75813D3CBEDDBDE074EEBD208FFB3B0CDA2ABDD02668BD26C3AD3DE2BCD3BD1A95D5BDCCC7AE3D80A65B3DCA1691BDF03C62BD2E15BE3D6046D8BD3A5FC03D041E10BD7A17BABD70D511BCE8DAFF3D94738ABDEE56BDBD58BFD1BC0494813D68AE4FBDF07AF2BD6822B13C849C743DE49850BD309AF43C54E3693DA4B67BBD5A07BA3D4800C53D205313BD7C6DF13D80C0FE3B0C7394BD9C9C8FBD300217BDD279BC3D60BC44BD00E874395C1885BD04CF54BDD0BC693D165ED33DF267F1BD6234A1BDF4CACABD9855953CE07B753DE06AE0BBCA93BABD3EC39F3D401644BB4223C6BD682AD63DE8B87F3D38E9F1BD8674A63D2C8BF93D722FEEBD34990A3D808E743DC84C323D949CF9BDD80CE5BCA446BABD801FD03BF05BA83D3C8D6A3D3E96FBBD80583EBC8457BE3D2857233DA62AF33DA83BC7BC223AB8BD48A8373D929280BD7289AFBD14C658BDF4C426BDE203C83D96DDC33D101F4BBD903A7C3CE074B43DD00D13BCF81394BC9048B1BC080AEE3D62A3E03D6866F9BDBA01DF3D005186BC14577ABD788DA73C3EE9D8BD608419BD604176BD10A0443C581CBFBC30D83ABC6C44083DBE8B8B3DDC469BBD9029FDBC70C8B3BC2E01E5BDD6368B3D00BE43BD50E3123D7652F23DAC2807BD30B5C53D762FE7BDC06BDDBC02F2843DC0D5A33BF0FE3BBDF81AACBCC08E243B787987BDEC09DBBD0AB18C3D78D6E23D049ADFBDC0B21BBD20427F3DCA0D91BD987E7EBD262EA63D30F4A33C3862113D003AFB3BCA2AB13DF007A03CF84915BD40ECF53CE860E73D40CB3DBC4EDAD23DF4A100BD04E6DABD6CEB293D509F41BD6C9299BD1AAA803DC0E427BDD8544B3D403FE6BB04ACBA3D561F91BD90660C3C10A0D83D909EB0BD98F6BF3D80B3C33B70117CBC9E18EDBD2A81D63D104728BCA40C433D101F5EBD605193BCDC9243BD00BBB03D002E2D3CF4117D3D24938A3D84FE5D3D7822BD3DD8EB83BC5E818C3D1CF61EBD7CC485BD84EAD53DD622F5BD12D2E73D00EAE2BC7441D83D9E85B83DA4119DBD827B82BD50ECDFBD94EEB9BDC4B67A3DA8775C3D7E43C33D4045FD3C8CE7E23DD010783D849C2F3DA0CDE33CAA1CFFBD1805E4BD8E1189BD4068033C2C88E9BDC0915ABD4A2BEDBD30B0073CACDC713D7040613CB0A42EBCCEF3F1BDC0CBC43BC072B03CD8A7CBBD901D4E3C60986FBDA82458BD3E79A8BDA0C3203DC054673DF0D6B4BCB8874E3DEA27B63DDA31DABDE27FD13D10D77BBD343A85BD220EDB3D72DFC6BDC06E913BB05AD13C1897A2BD3860A4BDF4456BBD7A4FF4BD80F0FDBDA89E8A3C6AF19DBDBE6AC6BD80F1C63B3628CEBDD651CCBD0C76B63D9CB382BD20C2643DAE7E933D48B8BBBCDA8AD53D285A0BBD18E4803CB8B22FBD80CB24BB769D91BD2A2E873DEA47A5BD8C9480BD8A65C4BDC2C0C53D084A073D6C62043DC0DE013CB01CCCBC5615E93D0C7A40BD3AAFCD3D9EF4A53D00AE9839800CE53C60DB84BB04348A3D8894D43CD8D7A23DAC5E423D4AD6923DFC9CCB3D8C04173DC0BC4B3DB4AF95BD7AFAF8BD9EE9B0BD02619E3D64E2F83D0083E3BDEE1BC4BD0038E6B918EB6B3DB00015BD6ACABABD246727BDB09D6ABD0ED2ABBD08755CBD1C16DDBD148C07BD4656BCBDAC13673DC4E5E1BDBCF5DA3DC06B063B0EA7C43DC0A740BCA001EE3D107B443D7E1A9C3D0E30FA3D50DBAEBCDC706FBDFC25203D4027C23B4444FFBDCA8AEF3D185049BDA06F2CBCE6F7F13D6895833C74301A3DD689E1BD7E2DC23D4071C6BDD45052BDCCBB79BD48E1FB3C1404A83DB0DA63BDC856153D800704BB64C02CBDC222F5BDE426203D101953BD20E6DBBDC4FDB03DCC3FCABD9078323DF074FD3D1EBE8F3D1031CEBDD0B0063C3C878B3DB461293D96B0F33DB23EEF3DD80EBDBDB4DA01BD24FB42BD34EFDC3D7459B13DE0A7ACBCD0D99CBD809D12BD789BDC3D08AACB3C902B583C68FBDD3DE00AE0BD30C7233C602C57BCB06853BD18FE15BD98AD32BDF0BA643C64815CBDB0D526BCCC31C3BDBCD759BD8074423B1665F23D1EEBE7BD9CB58CBD40D2723C10EBDB3D78901FBD749F303D3414BABDB4D705BD1C38C5BD80F4933DD0C18D3CBC30643D2A709EBDC8F16CBDB05736BCCC19D43DF03B5CBCB0EAEF3DB8E6BABD90F09BBDE0FE9CBD90298CBC928FCE3DAA8ABA3D782E8BBC528FA3BD9E8DD3BDE00143BC3062C83CD655BEBDB03ECB3D6A18A53D500C36BC70F4FF3DDA9A863D641936BD8058863B5CFE5EBDCE178D3D800D2DBC4686BABDA03774BC3861A83C26C7B4BD2E1A853D1EF6B23D6AB6923DA81E90BDDC71B2BDFC75603DAC8F5FBDA057973BA62CB8BD7C31CBBDE801DABC206081BC603C94BC9E0BD13D24AD843D34B5683DA858603D842FDE3D540A6F3D3C0B6ABD60A4A5BCF89E953C9CC0A03DD0FDA7BD442AB3BDC4D743BD76AAF83DEA0F81BD3601F3BDC81A80BD8CE965BD10DBB03C02CFFD3D8ECDF1BD3080983D30915FBD888BB13CA84C853D264EAC3D24B74BBD40F6873B809F66BD9CE0BD3D3C551E3D804909BB407C503C265F813D2003D0BBB08C52BC30DEFCBD348893BDA03E553D1823A33C28858E3D00AF0FBDA2D6BC3D103B3ABC789D343DD8ADB4BC34CB4B3D1C73EF3D0807E4BC66C996BDC8ED743D7EA7AE3D08C2EC3CD42B953DD4265DBD6E29B7BD8037CA3C8C667E3D7C33843D509FE0BC2C8BE3BD90B14D3D38A55EBDC858BEBD08B0AE3C701620BCCC65743DDECD83BD6810CEBC3076073DD0C11A3CFA85903DAEAFE2BD28C3BDBD103440BD60BE54BCF8A9FDBD0CD1BE3D0006A23C3A76963D0CA28FBDECB38CBDD43AA33DCC08933DF081E3BC108261BCE094A5BCACA1833DA259A5BD189DC23CE86DF1BD584316BD04FE753DEC2D34BDC087E1BC92CADE3D0A42943DD8D1ADBD0E29EA3D7093F93D12999C3DFACBD4BDFA0BEFBD4C7BFA3D48940C3DDC66363DA4BDA63D74E78F3D402254BB2890243DC0F8563D908BB4BC80FCE93BC0A5A33C6866183DD004473C34C29B3D3E73F5BD743F093DF0426F3C34F59A3DA08A213CD66CD5BDD2AB86BD7080B6BC30A5A9BC20403ABDE07ED9BDB4A2DDBD2CDD24BD34E8153DF2EEA13D60B62FBC9011543DBA49FEBDEE5BB2BD5EFAC3BD563B8A3DC6D98A3D6095C93D7890F23CC828F93D78A53C3DC870B5BCD866F2BDC884A6BDA634B6BDBEE3BDBD10D660BDF0CEBDBCE023EEBCB4B9B1BD6881BABDC440FD3D5082B33CA89718BD00E8363D0C3136BD447DC6BDAC8B473D7273D13D9CD420BD6896E0BDB445FF3DB885E33D78E5CD3D68DA70BDCA58BF3DF27AA83D906B24BD0282D4BDBC9D10BDE82DFB3CE042E43C04C9AD3DB089DBBDAECAB23D983D85BCB683BFBD6C2557BD20F0E13B2AA1C0BDF86DA6BD40F61C3DCE17E0BDEAF7E2BDF0C95FBDAAF6D4BD705A92BCB89475BD4863B7BC100C133CF8A4EB3D126AD13D8819FDBD60C7C53B0E16F7BDB095BE3DBC99A6BD2447D7BD6041D9BC1C86C53DA0BAD63BB834A2BC806EE4BB04E326BD7ABED13DE86B4BBDE4BD36BDC41BD9BD281C98BCAC3BA33D1E00C7BD465AB8BDF080373DBA549ABDA8C9C33CB62DFF3D361381BDFCC74F3DDEB6EB3D6061EB3DA0709DBB286FFD3D3EA0BD3DBA3098BD60B015BD1254E9BD40118CBBF628D2BD66F8D8BD38D7F0BDB204E83D9CF39FBD4C29543DA8A11EBD40CE98BB7860C03C8E5D9DBD12D6BF3D3E41E33DB2E0D5BD52B684BD004F903A008F12BC48E4B9BD847DCF3D60E6DDBD581DFABCD068403CD4F01ABD9ACFD3BD20F2B03DBA0BFEBD60305FBCE2C9F1BD182C9FBDB82BEEBD40A0DB3B08F0FE3CC0A4C5BC10FDE2BD528592BDD603AB3D9615E5BD10D60ABD00E34A3B0071433B0C76CFBD9C09FCBDE0EDA23B96B4B33DCE57E2BD2E75953D103D92BCB050C7BD9018293D3013DA3C6037F63CC8C8BD3C3865D0BD30FB4EBDD80705BDF27DC63DA0E23FBD90989CBDB0DA01BCC094E4BC7EC0AD3D9C3A7B3DFC63F8BDF03529BC00BB2EBCA87540BD58D1EDBC70C452BC40A1BC3C4E3AF6BDD8DEA13D2073AC3C704EA23D0087693CB0F7FD3D3063A43C46F8C3BD60F9DEBB3C33C53DB0D5D7BC84E2DFBDB01AA8BC6C4CBDBD70B3243CB856E53C70DAD33D2CFFD03D42FBEE3DCA27E2BDFAEDBDBDF02CADBCD0A81CBDD2C2803DD077ECBDF0E8023DC0B24DBBB0B3A3BC10F6E3BCA673B5BDCA44FDBDA8C6663D6CD98C3D0048E7BD68DF02BD205CB9BB92B7C13D30F9603D7A15E2BD005D763B1CB1B2BDF4D8C63DF06D133C98E8B53C48BBF03C9E12F5BDAEF887BD08DDC1BD70A5B33C90BC16BDC6A18B3D4411BCBD1026BEBC2224D3BD98C4703D8432DA3D306600BC9A0D90BD0C04493D40227C3D40754CBC60A4343CFA78C1BD10F8F9BCDAA8F23D40BC343BE09FBCBDD866F63DE44F5A3D20E30C3C805DFDBCB08EEEBD407AD8BBC0AAF33BC49D1FBD70B2583D10BFB33C3A16FBBD4EE2BEBDA83FCC3D98F09C3CC004E1BBD4C5833D22D1CE3D20E68FBC801BBEBC9CD9283D60849FBBA01BA9BBC4BD4D3DC086D13B708FA8BD7ED1EEBD6697F13D40B043BBD826BF3CBA9AFE3D9275B2BD348D3CBD6EBAD93D844DFDBD3258F1BDD8B0CBBC00404FBA8C4DA8BDF0978F3C1CFF4E3DC08BB4BB3041BA3D680DD13C84C6EA3D808DC7BDC00845BDB6CAAEBDBC43BFBDA2B0C03D4E74D3BDE676DB3D2019533D5E90EBBD281DC33CFC47713DFE0ABC3D4CF42A3DA07FBDBCA4F81ABD80C3DCBCC271C4BD3C32FDBD405360BCA04DAEBCB0E64BBCC6AFBBBD38E9DABD2EDAD5BDEA8FCCBD00C4E0BB88656C3D3C03413D283DC5BCB6A7E8BD90EBD8BC6CA4063D30934CBC0CA841BD085AC33D88C7653D6C191F3DE895B8BC3AE9A23D64485B3D8042F13BB608963D14040BBD58A869BD70DCB73D8C8FE53D8077E33C7038E73C60F565BD7010A3BDB29FF6BDF8E7403DD4E1743DE475E93D449A753DC06C7CBD50A4433C2AFFB6BDD03FDA3C3883DF3D2695EEBD607F563CDA3EBC3D3C2918BD9E65993DF0EF37BD6020C23B5A2AADBD285D63BDC882F8BCEA3BDE3D84C6593DCEADCFBDF4D567BD3229D63D38A1BABD4064FF3C8876963DB0A9703C4020CEBD641762BDC42E2C3D0A93D63D9C2ED63D70358EBC902872BDE0A2883C6607F73DB01E38BCE8A5243D9C29B1BDD82020BD20ED27BCD08110BC78EC9D3C68C4833CE03E6F3D10C9FC3CC0C192BB64C117BDA20AF3BDA01C773C783823BD583C68BD12C7DC3DC0CFF6BDFC28B3BDA096B6BBD02B69BC426DBEBD8097EF3BB07B71BC063DD23D526DBB3D724FA73D743A36BD0C5107BD38EC9FBD6038CB3CB883C0BD0A12F53DA4915C3DFC2262BD50B0833D200164BC7052393C6058D4BC204F603C242DA83D842B883DD8CD28BDF036FC3C0A62A7BD2E53E13D681AA03D0E9EB3BD3EA6D23D1035373CFCDEA8BD866CD73D8450F0BD683C37BDAE45C2BDA242DFBDC0A90EBD00C71BBC5EF4B13D9C047F3D707F85BC009ACA3A806DE83D489C55BD9E41AF3DAC5C9CBD10F685BDE0E5D63DDCA3E53D5E4DA1BD4CEE9C3D66E4BB3DAC28D4BD2606F63DA46F543D2016863B90E7DC3DB830CF3C382ACD3DA4CAEABDEE798EBD78B39A3CCC296F3D000C6A3AD0DDAE3DDC42033DB0B7A8BCE4AACFBD0456DBBD1C4A853D80C9DE3DBA8CD63D747BEABD1CDF943D1CEBCCBDE06B16BD622DCBBD9E38AEBD00A386BDE6CCBCBDACF354BD40C1FA3C0826FA3CD81C1D3D14EA8A3DF0FD673DC8A27CBD9AC5813D60128ABB68BA4EBD60BEB03DB0CF963C3E9BCB3DF214BB3DD801A9BD50D4B8BDE4B33DBD4C16413DB448F5BD6EDADBBDAC6E0DBD5C4A263DB014FE3CDC1D0CBDD80987BC04BE053D1080C5BD90B5A73D00DCA83C42A4FABD005FB53B0EFBC43D1201ABBD20366BBCC6B2973D3E36EA3DA080983D5872693DE42B153D1C137BBD1E52A93DBABDA8BDB0D9B5BD98151D3DC6DABD3D802216BB7CEF843D745A32BDB842A13CE006933B30F394BC1812873C90AC1E3C4041D2BBE853D9BD6C39B23DBA69A73DD05842BC8C90CD3D207B0D3C586F783D80AA6F3B146AD8BD70B0853CC25FB7BD1053F03C409434BB0008F03C441786BD5AA7A3BDD4B505BD78CFCA3D80D095BC4C0C333DE21E923DECC781BD2C1D81BD64E025BD40582EBDD002BA3C209C03BDAC84803DC0F4953D6C06F5BDFAFF893D1802C63CAC878A3D20C272BD4054DC3DF08A98BD0C58DC3DE0BDAC3D764BB7BD5AB9BE3DFEEA9CBD40E7793C1CD4C9BD38E3D73D76E5CD3D6C4CD4BD0C36303DE6E4C53D7416A03D46F58DBD527AB13D9E7F963D601C1EBDA0C25D3C9854E23C8087F03DF0BE133C40AD6CBD86619EBD300B1C3DC0BCA23D4CFE1ABD5AA6993D30B4883D00AAAE3A389E193D148B0F3D72D5BD3DB8E6FF3C12CF98BD5E08CE3D3489103D345BE5BD2039B4BC6C2D12BD1881E2BC26E7CC3DD29DFEBD70C7C63C40B8EABB7841C8BD0062EA3B0E0489BD360B94BDC8BE933DB6ECFFBDDADD833DEE998FBD346605BD9AD9D2BDE0F706BD00B4BFBB38BF303D48837C3D5C5FBF3D54B6A63D9076A0BDEE9BEC3D56E6E63D98489BBCDA76B23DC05881BB8071C53BE24D8BBD78F8CF3DF4152EBDCC6E76BD743C0BBD4449A03D6653C8BDD463ADBD601307BD0063F23C3CF6D03DC8CB613D78E864BD3809EDBCEC5F80BDD88C44BD602611BDE637823DA0D92B3D8CA412BD848685BDF862A0BD983AFD3CD6B0C83D1071B1BCB608F93D2A17B3BD3CDC3C3D00E6713BF07300BDECB5C73D46FFB7BDB832CC3D7C0D3F3DC6A2B13DA0D9F6BD8240D63D00C653BC80B748BBF0F5603D60C760BDC0CA6B3BF8069C3CE086F4BD20732CBC948E02BDFC6B023D4018E53C54C45C3DD814823D2881FF3DD80F273DD8D2043D006E2B3B881BA33C0089A2BCE455AB3D007B37BCB0E25E3C5C8F6B3D0050B1BCF6499CBDF03986BD6E08913D244466BD50B0C13C5C9FEE3D9CD034BDDA29A2BDDABAEC3DDC13583D98E1DFBC2A55CF3DA074E1BD3E65DC3D52468A3D902AD2BCCC597E3D2083E73CE251D53D34C644BDCC7CC63D16EE843D18FC76BD40C8A83D96A4F9BD68AE20BD301A0EBD6E48843DF895A9BD662DC0BD8E7DC93D94731B3D4020E03B60E4D33CC0C1763B804ACCBD38053F3DA868203D50DADD3C20EAF93D082E3D3DA4C1943DE8E590BD006C59BC00E7233C245DAEBDF0EDFEBD7638AABDB618ADBD0A158DBD805AEEBB948EB8BDA81A473DA4B94CBDD8A95C3DA0948A3D44FF5A3D9093D73CF641A3BD00F8BDBDF83CFABD0439EFBD40F900BCE0A14C3D7EE9C63D9877C5BC54A8393DA818FE3D54B1673D12B9913DB07F163C3CE559BD50D2573D00B7E93D2051193D0E7E87BD1A35F3BD8847873CE82C9BBDF238CC3D4C1B65BD2C8001BDACB7BCBD186DE63DD0EF8ABDF02B643D007E993AE2A580BD404889BCCCE187BD583D0C3D7843713DD8F81DBDB04D793D587A8D3C8068233CB026DA3DFCFE8E3D1EEEDB3D766EA1BD0ACCAA3DE46BB9BD6662F9BDD8A8913C7035C23C369BAD3DB2F1EFBD14A0B0BDF8B5133DA02CC03B2CEC13BD6014DA3C78B36C3DE045623CC8D3553D6A72DB3D16559E3DF4046CBD90EF75BC78AABBBC384BE2BC50E2253C4C15F53D504DCCBD04A01C3D8052EFBBF40F473D483DF9BC96FAE23D626B9BBD8037653B4C8F8EBD40871E3B0444A83D86FCC0BD9EA3B83D92BBCE3D9C8176BDB6E6CBBDA02ED9BCEA42DABD721583BDE884CE3DB087FEBD40F1C63C94BCF8BD2483C03DA29EDB3D9A6187BD44B56CBDD093CEBD7831D3BC843D523DD076EDBC8002F1BBF8E573BDC034873D3EB38FBD805FBE3BE03DF3BC90594C3C48524D3D18891A3D743DCEBDB298F73D0E37873DC0A9D1BD207D9B3D90A409BC4EFCA8BDF4F3A23D40B2AABD80642F3D50B0E8BD50B9353C102631BCACC9273D0458B6BD32EFA6BD60B6D9BB004DBEBB9ACDC63D9C95663D6CC288BDF46D10BD64A09A3D1CA459BDBC053DBDC214D7BD0C36713DD635A2BD68C0B1BD1A63C6BD7690B2BDF0BF41BD00289038BA91D1BD60D0243D8812F93DF0FA13BDD446E23D4C7CF8BD7C75E73D2A81923DA097F33D401AB53B46D9E8BDB2D1F7BD7820B6BC80AD7E3C48591C3D5273D1BD98FBCE3C40D123BBE84CCCBD400FEC3B68E5F9BDC2AFB93DF8F2D23D48E9B63C40BBE63BA8D8C8BC9079363C549E90BD60C4FBBD00D4FCBCE848E53D7E59A1BD14BB913D980FF33D1C29C23D80BDC3BC2A4BCC3DD0611C3D54E589BD5682BE3D807E8CBD50672EBC444F97BDFCD855BDEAB18ABD307464BCF0048E3D28F8B63DE4666EBD1AC1893D8629AABD0C08F73D68B4A33D94F3C8BD166785BD00401BBC920EABBD5C5696BD3C8AA7BD044A41BDC4AB843D2088623D588354BDA0EF7BBDA01D63BDA8C0603DC01585BCD4A6FDBDC046B1BD14E016BD80A1CD3B6C1FA13D348F98BD4EDA91BDB417C33D7843CCBD4617883D2ED8B93DFE178E3D6A88C3BDDE99B53D94681C3DCC87B1BD62ABEFBDBEF5B63D0C57BE3DDCBEDE3DE813913D900D9ABCA697C9BD3641A9BDF43F7EBD60A465BDD0C76BBD0C99D0BDC01AA33B10C5A53D04D0FDBD7876B2BC0654B63D00D733BD7C2E32BDF8BDBDBC1875633DD02CE93CBAA7A3BD4CDA7BBDC2BAC0BD928A8E3DE8C980BDA8D99CBC44B3753D7C714FBD904D883C2A59D53D7424BC3D9412C93D40C6533BA6FF9D3D90192B3C403159BCB6698F3DC85492BC821DCDBD48DA85BD821A90BD409EF63C00BFF53C88A8523D004FD9BA5818033D0818D93D64A85DBD180C8DBDD29CC5BD0066723A64D8CA3D40BFE0BBA658873DFE57A83DD491D23D2C11E23DF4A58B3D509D06BD7A7FD13D00F3AA3D14C61DBDFCDE2C3DDE10E7BDCE23CABD747B51BD32A8DD3DEE8AF53DF0DF793CA8E563BDF4BDED3D8808023D60AFB23D68BACFBC1AA9D6BDA01F5EBC7898DBBC2469A9BD3409C9BD10ADB73D583BF2BC5CDA8ABD7264CC3DF4DD30BD989F89BD508C453C9C14ACBD8C2F123D40FABF3BA08393BB18D5123D7806413D9613A2BD08BD9CBDF80EAB3D1455DCBD8230823D061A8B3D280282BD5C3F07BD52C0EC3D0C70C6BD1C956E3D703E433C4CB2773D4251A03D30796EBC54A2D5BD9456DE3DFC218EBD508CF43CC0B77A3B4033D23B4C831C3DE00197BD7A0ED43D90DE81BC60AEE03C2299ACBDF629DDBDA6F48FBDE4F3523DCE1BC7BD3C7BADBD46B0AEBD40A694BC206AE73B0889803D30AF083C282C28BD04219EBD6ECBFA3D36F9D4BDF0B2E43D"> : tensor<64x32xf32>
    %cst_5 = arith.constant dense<"0x367D2C3E4E451B3E0933043DAB47E6BB8167F6BDD26B3C3DD9F3C0BD89DC553D35DA813D913E2C3E185520BEE184173E5BFE2CBEBFAEE53DC7F982BD6B5AECBCDBF2A9BD323EB43D41249F3D66B317BE76AFFF39988F123E70D56B3DEE0EAE3D44959CBDA581973D9535ABBD1FB0163CC3841CBEF585E2BC72B6283E2171DCBC8AC969BD42DB223EE552C4BDA65B48BDD76B383C1ACBBCBD109D603DC89F1C3D6B868F3DD84511BE5B5044BD74D3013E5ADBC3BC41DCE9BB5E82C73C98EEE33D47CE56B906F1DF3D55B00B3D801F003E80DA1FBE67A400BE3F069FBD01D91EBC6088323E21B4E93C643F0A3E0E1A39BA4FEEBDBD24C4163E805DA9BD9FF1133E5AC8E63DE97BE13C58459BBC3BE1D33C7681E0BD6896033EE961A0BD7242F8BD05EA273EE5B4EABCCD28C83CC3501EBE8981B0BBD2E391BD4834DB3D363A193EC08327BD9DCDFA3D1E54C93DE179E53D9B46C7BD0ECB77BD25BB0B3EE694183E788793BD68130ABE5647FDBDDE8C333E48511E3EB49F11BEAD88333E6ADB953C5408E83CDF5816BE0C9795BD79C24B3C9B8D0EBCEF2306BD9EE8023EF10920BD19B0C7BDA6C42BBEF0E9153EB3BF8EBD3469EC3DDA7BAF3D9AE0A33CE439E83D7693053E242317BEE4759BBDA51767BCFD412B3D4B00903D62BF313D91DC9FBD7E16CFBD1C7ACABD234089BB5906D3BD7B830ABDC6B3AA3D5D14103E0545723D59EDA63DB14DFFBD8100CDBD6AA8D2BDD2CC1B3D2303AFBD14AB263EE3B78C3DE6700BBE735D35BD1843553C24FBF7BDE2009ABD60F368BD877ACCBD944B0B3E7C1C933DF4CEC1BDDADF053E3C3CEC3D5EB4BBBD1E25E43D261135BCDB61873DD445B7BD24BB123E5CED973D050D113EC16FC63DA34EE03CB1082D3E03756E3D7F63823D4BEF973B335BA33D1A36983DBCBB42BC33D37D3D01E64B3C922EE3BD867309BEFFECB4BDF349E33D7B93EF3D8DBBED3D2E7886BDC833173E052A2C3E4A46F0BCF2AA91BDC1F7D2BC36F7083E53D2C5BCD8CD1ABEF3A9F23D1BBCDE3D5FDC063EFADB133D89A312BE17B9DEBD0C05853DB37CF73DC46406BE025D5F3D2528143C9D07243E3EBEFFBDF9D0DBBDF51704BE2AE36F3C922106BEF5C393BC242B903DBCBDC2BD2F53EA3C9F8B343EC0CD09BE6AC3CC3D88172EBE5AD11B3E224A83BD8646173EA330413C581C6D3C714F18BE4E9FB83D34F51BBE5E90A33DBC4C50BDE3C3223E95220FBD0CEDF03DD89D34BE44CDF7BD13049A3DD3F1BA3CA8B5C2BDB4CDF83DB9E42D3D7158C23D6871EF3CBF5BEEBD763A303E9E5732BE898D17BDEAEB11BE136EB5BD6A13AA3DA0E4BE3DA878E83D369D8DBDF72D88BD5C6273BD23F908BE48CDFF3D4C0EB93D1091153EB569F13D0ADC3A3D20BDA73DD578F1BD617B303EADED1C3E743FF43DFB9C563DD4BD13BEF8E76D3D9986313E59E91ABEF34ED13D1563EA3D3717C33DD0700D3EDC138A3D2A941B3E7BD22ABEEE57B6BDFDB8953D7C60663D34C52A3ECA6C0E3D236FE1BDA76188BDFBC908BD80FFFABC0ED5753D85DF19BEF72098BCCB2E12BE6A7A4FBD2157B73DA29B01BE37BC9BBDA66F843D10B5B5BD45042F3EC6F7193EF70410BE0572CABC64C9343E2AC3D23D430CC8BD05DEDF3DF4ABB83D56B20EBE4D6F353D874E113E8E309C3DB985A4BD4F4B7C3DE8C085BD129FB23DC9AB9BBD773EEA3DCE2DD93D581A243EADE3333E9AFBB83DEB8C343D23A40BBE0E99A6BCCA3039BD3517F53DB8659D3D886F7BBD325F353DD6E053BCD1CD383DE8CEC63D6CA2DBBD7321A1BD04EB3FBD"> : tensor<32x10xf32>
    %1 = hls.fdf.dispatch : tensor<1x10xf32> {
      %2 = hls.fdf.task : tensor<1x3072xf32> {
        %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<1x3x32x32xf32> into tensor<1x3072xf32>
        hls.fdf.yield %collapsed : tensor<1x3072xf32>
      }
      %3 = hls.fdf.alloc_tensor : () -> tensor<1x64xf32>
      %4 = hls.fdf.alloc_tensor %cst : (f32) -> tensor<1x64xf32>
      %5 = hls.fdf.task space @global::@task5 : tensor<1x64xf32> {
        %15 = linalg.matmul ins(%2, %cst_3 : tensor<1x3072xf32>, tensor<3072x64xf32>) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
        hls.fdf.yield %15 : tensor<1x64xf32>
      }
      %6 = hls.fdf.task space @global::@task4 : tensor<1x64xf32> {
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
      %9 = hls.fdf.task space @global::@task3 : tensor<1x32xf32> {
        %15 = linalg.matmul ins(%6, %cst_4 : tensor<1x64xf32>, tensor<64x32xf32>) outs(%8 : tensor<1x32xf32>) -> tensor<1x32xf32>
        hls.fdf.yield %15 : tensor<1x32xf32>
      }
      %10 = hls.fdf.task space @global::@task2 : tensor<1x32xf32> {
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
      %13 = hls.fdf.task space @global::@task1 : tensor<1x10xf32> {
        %15 = linalg.matmul ins(%10, %cst_5 : tensor<1x32xf32>, tensor<32x10xf32>) outs(%12 : tensor<1x10xf32>) -> tensor<1x10xf32>
        hls.fdf.yield %15 : tensor<1x10xf32>
      }
      %14 = hls.fdf.task space @global::@task0 : tensor<1x10xf32> {
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
