
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "Vitis_Libraries/blas/L1/include/hw/xf_blas/gemm.hpp"

using namespace std
void forward_node0(
  float v0[1][10],
  float v1[10],
  float v2[1][10]
) {	//
  for (int v3 = 0; v3 < 1; v3 += 1) {	// L62
    for (int v4 = 0; v4 < 10; v4 += 1) {	// L62
      float v5 = v0[0][v4];	// L62
      float v6 = v1[v4];	// L62
      float v7 = v5 + v6;	// L64
      v2[v3][v4] = v7;	// L62
    }
  }
}

void forward_node1(
  float v8[32][10],
  float v9[1][32],
  float v10[1][10]
) {	//
  gemm<float,Int,32,2,1024>(1,10,32,1.000000e+00,0.000000e+00,v8,v9,v10,v10);	// L61
}

void forward_node2(
  float v11[1][32],
  float v12[32],
  float v13[1][32]
) {	//
  for (int v14 = 0; v14 < 1; v14 += 1) {	// L48
    for (int v15 = 0; v15 < 32; v15 += 1) {	// L48
      float v16 = v11[0][v15];	// L48
      float v17 = v12[v15];	// L48
      float v18 = v16 + v17;	// L45
      bool v19 = v18 > (float)0.000000;	// L50
      float v20 = v19 ? v18 : (float)0.000000;	// L51
      v13[v14][v15] = v20;	// L48
    }
  }
}

void forward_node3(
  float v21[64][32],
  float v22[1][64],
  float v23[1][32]
) {	//
  gemm<float,Int,32,2,1024>(1,32,64,1.000000e+00,0.000000e+00,v21,v22,v23,v23);	// L42
}

void forward_node4(
  float v24[1][64],
  float v25[64],
  float v26[1][64]
) {	//
  for (int v27 = 0; v27 < 1; v27 += 1) {	// L29
    for (int v28 = 0; v28 < 64; v28 += 1) {	// L29
      float v29 = v24[0][v28];	// L29
      float v30 = v25[v28];	// L29
      float v31 = v29 + v30;	// L26
      bool v32 = v31 > (float)0.000000;	// L31
      float v33 = v32 ? v31 : (float)0.000000;	// L32
      v26[v27][v28] = v33;	// L29
    }
  }
}

void forward_node5(
  float v34[3072][64],
  float v35[1][3072],
  float v36[1][64]
) {	//
  gemm<float,Int,32,2,1024>(1,64,3072,1.000000e+00,0.000000e+00,v34,v35,v36,v36);	// L23
}

void forward_node6(
  float v37[1][3][32][32],
  float v38[1][3072]
) {	//
  for (int v39 = 0; v39 < 3072; v39 += 1) {	//
    float v40 = v37[0][(v39 / 1024)][((v39 % 1024) / 32)][(v39 % 32)];	//
    v38[0][v39] = v40;	//
  }
}

/// This is top function.
void forward(
  float v41[1][3][32][32],
  float v42[1][10],
  float v43[3072][64],
  float v44[64][32],
  float v45[1][3072]
) {	// L7
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v41
  #pragma HLS interface s_axilite port=v41 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v42
  #pragma HLS interface s_axilite port=v42 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v43
  #pragma HLS interface s_axilite port=v43 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v44
  #pragma HLS interface s_axilite port=v44 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v45
  #pragma HLS interface s_axilite port=v45 bundle=ctrl
  float v46[64] = {(float)-0.001955, (float)0.011174, (float)0.001674, (float)0.002678, (float)0.009552, (float)-0.015826, (float)0.015927, (float)0.012297, (float)0.010800, (float)0.016591, (float)-0.000114, (float)0.009117, (float)-0.012822, (float)0.000113, (float)-0.017518, (float)-0.011519, (float)0.006964, (float)0.013570, (float)-0.013244, (float)-0.006899, (float)0.014993, (float)-0.014426, (float)0.003189, (float)-0.013369, (float)0.014686, (float)-0.011838, (float)-0.014204, (float)-0.011192, (float)-0.001201, (float)0.007789, (float)-0.013943, (float)0.002083, (float)-0.015791, (float)-0.000562, (float)-0.002591, (float)0.003947, (float)0.014394, (float)0.007132, (float)-0.002350, (float)-0.000693, (float)-0.009396, (float)-0.006507, (float)-0.011584, (float)-0.013653, (float)-0.007305, (float)0.017976, (float)-0.015170, (float)0.000566, (float)0.007541, (float)-0.016309, (float)-0.007319, (float)-0.004227, (float)-0.008109, (float)-0.004363, (float)0.012179, (float)-0.002256, (float)0.000546, (float)0.017111, (float)-0.012889, (float)-0.013611, (float)0.013343, (float)0.004915, (float)-0.012791, (float)0.000757};	// L12
  #pragma HLS resource variable=v46 core=ram_t2p_bram

  float v47[32] = {(float)-0.008860, (float)0.115749, (float)-0.034385, (float)-0.061210, (float)0.094797, (float)-0.113364, (float)0.064446, (float)-0.033063, (float)-0.037931, (float)0.021521, (float)-0.081418, (float)-0.019784, (float)-0.061197, (float)0.117470, (float)0.011904, (float)-0.021691, (float)0.123594, (float)0.095335, (float)-0.095611, (float)-0.020915, (float)-0.119396, (float)0.009513, (float)0.051908, (float)-0.078564, (float)-0.016565, (float)0.087703, (float)0.068503, (float)0.041308, (float)0.045557, (float)0.020815, (float)-0.000197, (float)0.006185};	// L10
  #pragma HLS resource variable=v47 core=ram_t2p_bram

  float v48[10] = {(float)0.071049, (float)-0.000239, (float)-0.051904, (float)-0.115049, (float)-0.114932, (float)-0.083458, (float)-0.146045, (float)0.021647, (float)0.089696, (float)0.142750};	// L8
  #pragma HLS resource variable=v48 core=ram_t2p_bram

  float v49[32][10] = {(float)0.139939, (float)0.066636, (float)-0.114347, (float)0.019265, (float)0.060346, (float)0.146702, (float)-0.158678, (float)0.119587, (float)0.076539, (float)-0.047116, (float)0.168133, (float)-0.138212, (float)-0.043244, (float)-0.044337, (float)-0.073823, (float)-0.117075, (float)-0.132275, (float)0.111507, (float)-0.111834, (float)0.055722, (float)-0.012651, (float)-0.052915, (float)-0.014774, (float)0.131302, (float)0.161625, (float)-0.045227, (float)-0.045083, (float)-0.167736, (float)-0.175287, (float)-0.038700, (float)0.039108, (float)-0.022533, (float)-0.166176, (float)0.106186, (float)-0.135089, (float)-0.131323, (float)0.096010, (float)0.174946, (float)0.162072, (float)-0.114960, (float)-0.029705, (float)0.062461, (float)-0.082857, (float)0.108623, (float)0.067588, (float)-0.020148, (float)-0.032541, (float)0.024483, (float)-0.024663, (float)0.140442, (float)-0.061585, (float)0.104988, (float)0.110366, (float)0.047533, (float)0.167348, (float)0.048007, (float)-0.078161, (float)-0.035533, (float)0.069183, (float)-0.104021, (float)-0.024241, (float)-0.152660, (float)-0.116111, (float)-0.025040, (float)0.107621, (float)-0.015031, (float)-0.019081, (float)-0.167956, (float)-0.040429, (float)-0.073821, (float)0.144268, (float)0.108940, (float)-0.036289, (float)0.136207, (float)-0.154374, (float)-0.102755, (float)0.148688, (float)0.067128, (float)0.069483, (float)-0.095179, (float)-0.066486, (float)0.030023, (float)0.084097, (float)-0.119006, (float)0.164504, (float)-0.140673, (float)-0.120542, (float)0.094117, (float)0.153948, (float)0.002638, (float)-0.006643, (float)-0.169200, (float)-0.067265, (float)-0.149927, (float)0.147179, (float)0.039738, (float)-0.046794, (float)0.080099, (float)0.137174, (float)0.101340, (float)0.053607, (float)-0.046040, (float)0.131861, (float)0.055102, (float)0.042640, (float)0.139923, (float)-0.079843, (float)0.089981, (float)-0.175261, (float)0.039619, (float)0.048765, (float)0.013411, (float)-0.058334, (float)0.096860, (float)0.119728, (float)-0.111044, (float)-0.003339, (float)0.119749, (float)0.008115, (float)0.106985, (float)0.033324, (float)-0.117806, (float)-0.143333, (float)-0.068876, (float)-0.056111, (float)-0.171487, (float)0.155585, (float)-0.058022, (float)0.124965, (float)-0.014522, (float)0.055691, (float)0.028389, (float)0.070547, (float)-0.040789, (float)-0.135090, (float)-0.114779, (float)-0.114903, (float)-0.115842, (float)0.056135, (float)-0.151118, (float)-0.070862, (float)0.060621, (float)-0.039249, (float)0.032503, (float)-0.117713, (float)-0.158447, (float)0.082962, (float)0.117991, (float)0.064830, (float)0.143695, (float)-0.155920, (float)-0.022810, (float)-0.115364, (float)0.153582, (float)0.040134, (float)-0.067074, (float)-0.118584, (float)-0.108602, (float)-0.143459, (float)0.104282, (float)-0.144074, (float)0.174514, (float)0.048894, (float)-0.175946, (float)-0.142496, (float)-0.044899, (float)-0.163843, (float)0.116437, (float)-0.078267, (float)0.173426, (float)0.024130, (float)-0.026055, (float)0.084007, (float)-0.136717, (float)-0.116864, (float)0.166552, (float)-0.061372, (float)-0.161007, (float)0.080868, (float)0.082543, (float)0.121707, (float)-0.120094, (float)0.090888, (float)0.176085, (float)0.113765, (float)0.118774, (float)0.175867, (float)0.091520, (float)-0.083703, (float)-0.153944, (float)-0.065545, (float)0.113772, (float)-0.173604, (float)0.129531, (float)-0.025175, (float)0.170607, (float)0.118368, (float)-0.153479, (float)0.071245, (float)0.066956, (float)0.102471, (float)0.097530, (float)-0.096791, (float)0.140324, (float)-0.024743, (float)0.015644, (float)0.123730, (float)-0.014537, (float)-0.010439, (float)0.173721, (float)0.117173, (float)-0.067923, (float)0.082637, (float)-0.009661, (float)0.045027, (float)0.118857, (float)-0.084797, (float)-0.171456, (float)-0.075976, (float)-0.135935, (float)-0.009071, (float)-0.048184, (float)0.158578, (float)-0.168997, (float)-0.171046, (float)0.027997, (float)-0.047601, (float)-0.128926, (float)0.088116, (float)-0.016263, (float)-0.147152, (float)-0.025003, (float)0.047760, (float)-0.121231, (float)-0.031690, (float)0.080240, (float)0.123170, (float)-0.161410, (float)-0.079618, (float)0.048958, (float)-0.164128, (float)-0.024656, (float)-0.151823, (float)-0.043043, (float)-0.000004, (float)-0.072392, (float)-0.168766, (float)0.067563, (float)0.070176, (float)0.131556, (float)-0.149734, (float)-0.083014, (float)-0.138083, (float)0.166909, (float)0.119410, (float)-0.061186, (float)0.113563, (float)-0.003542, (float)-0.061277, (float)-0.099041, (float)-0.131668, (float)0.076647, (float)-0.078048, (float)-0.051194, (float)-0.005089, (float)0.103013, (float)-0.023451, (float)-0.095754, (float)0.013203, (float)-0.002818, (float)-0.099202, (float)-0.149613, (float)-0.121122, (float)0.020417, (float)0.066062, (float)-0.111318, (float)0.111961, (float)0.021016, (float)0.159875, (float)-0.146005, (float)-0.011785, (float)-0.039765, (float)0.094390, (float)-0.090794, (float)-0.101999, (float)0.152018, (float)-0.020046, (float)0.006079, (float)0.110868, (float)-0.145507, (float)-0.009122, (float)0.152336, (float)0.090687, (float)0.081548, (float)-0.137771, (float)-0.024779, (float)-0.154941, (float)-0.165008, (float)0.096489, (float)-0.078730, (float)0.091184, (float)0.077132, (float)0.060118, (float)-0.000992, (float)0.090852, (float)0.002612, (float)0.117818, (float)0.060050, (float)-0.043069, (float)-0.104871, (float)-0.060504, (float)0.114298, (float)0.078576, (float)-0.109336, (float)-0.108824, (float)0.130799, (float)0.112276, (float)0.047651, (float)-0.126745, (float)-0.084180};	// L55
  #pragma HLS resource variable=v49 core=ram_t2p_bram

  forward_node6(v41, v45);	//
  float v50[1][64];	// L21
  #pragma HLS resource variable=v50 core=ram_t2p_bram

  float v51[1][64];	// L22
  #pragma HLS resource variable=v51 core=ram_t2p_bram

  forward_node5(v43, v45, v51);	//
  forward_node4(v51, v46, v50);	//
  float v52[1][32];	// L40
  #pragma HLS resource variable=v52 core=ram_t2p_bram

  float v53[1][32];	// L41
  #pragma HLS resource variable=v53 core=ram_t2p_bram

  forward_node3(v44, v50, v53);	//
  forward_node2(v53, v47, v52);	//
  float v54[1][10];	// L60
  #pragma HLS resource variable=v54 core=ram_t2p_bram

  forward_node1(v49, v52, v54);	//
  forward_node0(v54, v48, v42);	//
}


