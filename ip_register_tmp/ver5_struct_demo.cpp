
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

using namespace std;

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
  float v8[1][32],
  float v9[32][10],
  float v10[1][10]
) {	//
struct my_struct {
  typedef float dummy_type;
static const unsigned dummy_int = 15;
typedef int t_IndexType;
const unsigned t_ParEntries = 0
};

  gemm<float, int, 32, 0, 1024, my_struct>((int)1, (int)10, (int)32, (float)1.000000, (float)0.000000, v8, v9, v10, v10);	// L61
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
  float v21[1][64],
  float v22[64][32],
  float v23[1][32]
) {	//
struct my_struct {
  typedef float dummy_type;
static const unsigned dummy_int = 15;
typedef int t_IndexType;
const unsigned t_ParEntries = 4
};

  gemm<float, int, 32, 4, 1024, my_struct>((int)1, (int)32, (int)64, (float)1.000000, (float)0.000000, v21, v22, v23, v23);	// L42
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
  float v34[1][3072],
  float v35[3072][64],
  float v36[1][64]
) {	//
struct my_struct {
  typedef float dummy_type;
static const unsigned dummy_int = 15;
typedef int t_IndexType;
const unsigned t_ParEntries = 1
};

  gemm<float, int, 32, 1, 1024, my_struct>((int)1, (int)64, (int)3072, (float)1.000000, (float)0.000000, v34, v35, v36, v36);	// L23
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
  float v46[64] = {(float)-0.010047, (float)-0.003354, (float)-0.007101, (float)0.011887, (float)-0.014247, (float)-0.006167, (float)0.002377, (float)-0.004694, (float)-0.005608, (float)-0.002240, (float)0.000600, (float)-0.004456, (float)0.002327, (float)0.007333, (float)0.007090, (float)-0.008290, (float)-0.016916, (float)-0.011762, (float)-0.003848, (float)0.016287, (float)0.005583, (float)0.007146, (float)-0.008655, (float)-0.006062, (float)0.012997, (float)0.002466, (float)0.009051, (float)0.009565, (float)-0.011723, (float)-0.010591, (float)0.005450, (float)-0.014420, (float)0.009039, (float)0.015399, (float)-0.013379, (float)-0.014565, (float)-0.005855, (float)-0.014900, (float)0.001202, (float)-0.006037, (float)0.015935, (float)-0.011398, (float)0.001731, (float)0.015660, (float)-0.010107, (float)0.012912, (float)-0.006645, (float)-0.009755, (float)0.000615, (float)0.012089, (float)0.013344, (float)0.010572, (float)-0.007958, (float)0.002671, (float)-0.006330, (float)0.005422, (float)0.018002, (float)0.011401, (float)-0.016999, (float)0.016193, (float)-0.012557, (float)0.015066, (float)-0.013088, (float)-0.014423};	// L12
  #pragma HLS resource variable=v46 core=ram_t2p_bram

  float v47[32] = {(float)-0.033540, (float)-0.105796, (float)-0.064525, (float)-0.023584, (float)-0.104795, (float)-0.083218, (float)0.077449, (float)0.077984, (float)-0.040234, (float)-0.097676, (float)0.078633, (float)0.035520, (float)0.060536, (float)0.085562, (float)-0.006924, (float)0.011438, (float)0.097318, (float)-0.054592, (float)0.003820, (float)-0.050471, (float)0.016732, (float)-0.036165, (float)0.113177, (float)-0.049221, (float)-0.051352, (float)-0.098881, (float)0.071452, (float)0.028659, (float)0.049576, (float)0.113972, (float)0.064220, (float)0.108985};	// L10
  #pragma HLS resource variable=v47 core=ram_t2p_bram

  float v48[10] = {(float)-0.035857, (float)-0.062991, (float)0.012919, (float)0.169219, (float)-0.056584, (float)-0.122447, (float)-0.061099, (float)-0.130252, (float)0.012375, (float)-0.118935};	// L8
  #pragma HLS resource variable=v48 core=ram_t2p_bram

  float v49[32][10] = {(float)0.098469, (float)-0.124036, (float)0.073921, (float)0.061476, (float)-0.005849, (float)-0.079511, (float)-0.089707, (float)0.155325, (float)-0.006183, (float)-0.164726, (float)0.070717, (float)0.142314, (float)-0.091380, (float)0.016112, (float)0.089677, (float)-0.153027, (float)-0.041551, (float)-0.150028, (float)0.020417, (float)-0.159054, (float)-0.068789, (float)-0.126896, (float)0.102489, (float)0.106789, (float)-0.038483, (float)-0.106408, (float)-0.150808, (float)0.101447, (float)0.114546, (float)-0.072322, (float)0.116895, (float)0.158406, (float)-0.021754, (float)-0.085703, (float)-0.103204, (float)-0.165236, (float)0.138743, (float)-0.025879, (float)-0.028456, (float)-0.059307, (float)-0.078345, (float)0.023190, (float)0.106217, (float)0.112418, (float)-0.077126, (float)0.117901, (float)0.168591, (float)0.157817, (float)0.142533, (float)-0.053413, (float)-0.017054, (float)-0.034997, (float)-0.117241, (float)0.095175, (float)-0.019942, (float)-0.162616, (float)0.086758, (float)-0.031068, (float)0.131918, (float)-0.167800, (float)-0.097500, (float)-0.127601, (float)0.168104, (float)0.093820, (float)-0.154785, (float)0.086836, (float)0.110087, (float)-0.019132, (float)-0.173660, (float)-0.169921, (float)0.145256, (float)0.163366, (float)-0.147627, (float)0.079016, (float)-0.103413, (float)0.089350, (float)-0.169634, (float)0.157771, (float)0.052053, (float)0.000045, (float)-0.128005, (float)-0.029174, (float)0.016215, (float)-0.166611, (float)-0.150510, (float)-0.054077, (float)-0.045410, (float)-0.015525, (float)0.030991, (float)-0.091435, (float)0.162453, (float)-0.057125, (float)0.017950, (float)-0.135363, (float)0.067258, (float)-0.051334, (float)0.079420, (float)0.002215, (float)-0.087090, (float)0.166290, (float)0.062560, (float)-0.027473, (float)0.025984, (float)-0.119876, (float)-0.121007, (float)0.151675, (float)0.127743, (float)0.040198, (float)-0.024079, (float)0.009365, (float)-0.025042, (float)-0.105771, (float)-0.148598, (float)-0.155588, (float)0.055581, (float)-0.035585, (float)-0.064288, (float)0.131610, (float)0.093038, (float)-0.020057, (float)-0.116349, (float)0.156421, (float)-0.035398, (float)0.046187, (float)-0.160377, (float)0.041789, (float)0.156071, (float)0.075200, (float)-0.085293, (float)-0.047224, (float)0.164200, (float)-0.162348, (float)-0.032459, (float)0.123086, (float)0.145851, (float)0.019480, (float)-0.065686, (float)0.116847, (float)0.167591, (float)-0.150365, (float)-0.039977, (float)-0.029996, (float)0.014960, (float)0.028754, (float)-0.101476, (float)-0.130142, (float)0.071679, (float)-0.099733, (float)-0.037462, (float)-0.137468, (float)-0.120623, (float)0.043041, (float)-0.174914, (float)0.150121, (float)0.133763, (float)-0.176670, (float)0.009177, (float)-0.057483, (float)0.081811, (float)0.140213, (float)-0.027153, (float)0.125964, (float)-0.151184, (float)0.059591, (float)0.070425, (float)0.035744, (float)0.039474, (float)-0.109530, (float)-0.109859, (float)0.052543, (float)0.045957, (float)0.176407, (float)-0.118359, (float)0.008503, (float)-0.054615, (float)-0.130007, (float)-0.067083, (float)-0.081242, (float)0.156822, (float)0.023831, (float)0.101086, (float)0.028676, (float)0.127935, (float)-0.042395, (float)0.023761, (float)0.027486, (float)-0.083677, (float)0.130548, (float)-0.175281, (float)0.040216, (float)-0.172463, (float)0.149715, (float)0.117330, (float)0.141073, (float)-0.120988, (float)0.057894, (float)0.049216, (float)0.033434, (float)-0.170702, (float)0.000510, (float)-0.000089, (float)0.005667, (float)0.123502, (float)0.083301, (float)-0.162751, (float)0.028413, (float)0.013153, (float)-0.043778, (float)0.056778, (float)0.003336, (float)-0.001596, (float)-0.147684, (float)-0.027722, (float)0.099598, (float)0.090308, (float)-0.063219, (float)0.160057, (float)-0.172432, (float)0.109679, (float)-0.035038, (float)-0.117727, (float)0.026616, (float)-0.130752, (float)-0.148603, (float)-0.072696, (float)0.080525, (float)-0.000827, (float)-0.107213, (float)-0.022218, (float)-0.018029, (float)-0.149959, (float)0.150089, (float)-0.158048, (float)0.150105, (float)-0.059252, (float)0.127697, (float)0.055706, (float)0.169860, (float)-0.124950, (float)-0.082005, (float)0.037791, (float)0.073434, (float)-0.140367, (float)-0.092255, (float)0.158185, (float)-0.072319, (float)-0.011338, (float)0.090668, (float)-0.151271, (float)-0.096129, (float)0.070055, (float)0.030262, (float)0.061061, (float)0.089293, (float)-0.011039, (float)-0.018908, (float)0.031595, (float)0.018715, (float)0.140323, (float)0.068220, (float)0.122304, (float)-0.045463, (float)0.165920, (float)-0.024524, (float)-0.151519, (float)-0.094567, (float)0.119191, (float)-0.014666, (float)-0.130850, (float)0.010397, (float)0.055476, (float)-0.165557, (float)-0.034794, (float)0.094469, (float)-0.013954, (float)-0.172709, (float)0.078577, (float)-0.159674, (float)-0.078913, (float)0.089212, (float)-0.089140, (float)0.146977, (float)0.100863, (float)0.160985, (float)0.076122, (float)0.034604, (float)0.129893, (float)0.073258, (float)0.093519, (float)0.172431, (float)-0.089305, (float)0.124820, (float)0.176469, (float)0.011308, (float)0.113213, (float)-0.098441, (float)-0.070091, (float)-0.037335, (float)0.111167, (float)-0.131319, (float)0.065995, (float)-0.115058, (float)-0.061398, (float)-0.147168, (float)0.061043, (float)-0.065308, (float)0.143914, (float)-0.117370, (float)0.146063, (float)-0.011514, (float)-0.077439, (float)-0.035993, (float)-0.101023, (float)0.025857, (float)-0.144799, (float)-0.101431, (float)-0.125577, (float)0.173617, (float)0.088949, (float)-0.006710};	// L55
  #pragma HLS resource variable=v49 core=ram_t2p_bram

  forward_node6(v41, v45);	//
  float v50[1][64];	// L21
  #pragma HLS resource variable=v50 core=ram_t2p_bram

  float v51[1][64];	// L22
  #pragma HLS resource variable=v51 core=ram_t2p_bram

  forward_node5(v45, v43, v51);	//
  forward_node4(v51, v46, v50);	//
  float v52[1][32];	// L40
  #pragma HLS resource variable=v52 core=ram_t2p_bram

  float v53[1][32];	// L41
  #pragma HLS resource variable=v53 core=ram_t2p_bram

  forward_node3(v50, v44, v53);	//
  forward_node2(v53, v47, v52);	//
  float v54[1][10];	// L60
  #pragma HLS resource variable=v54 core=ram_t2p_bram

  forward_node1(v52, v49, v54);	//
  forward_node0(v54, v48, v42);	//
}


