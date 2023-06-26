
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

using namespace std;

void forward_node0(
  float v0[1][10],
  float v1[10],
  float v2[1][10]
) {	//
  for (int v3 = 0; v3 < 1; v3 += 1) {	// L36
    for (int v4 = 0; v4 < 10; v4 += 1) {	// L36
      float v5 = v0[0][v4];	// L36
      float v6 = v1[v4];	// L36
      float v7 = v5 + v6;	// L38
      v2[v3][v4] = v7;	// L36
    }
  }
}

void forward_node1(
  float v8[1][16384],
  float v9[16384][10],
  float v10[1][10]
) {	//
  for (int v11 = 0; v11 < 1; v11 += 1) {	// L35
    for (int v12 = 0; v12 < 10; v12 += 1) {	// L35
      for (int v13 = 0; v13 < 16384; v13 += 1) {	// L35
        float v14 = v8[v11][v13];	// L35
        float v15 = v9[v13][v12];	// L35
        float v16 = v10[v11][v12];	// L35
        float v17 = v14 * v15;	//
        float v18 = v16 + v17;	//
        v10[v11][v12] = v18;	// L35
      }
    }
  }
}

void forward_node2(
  float v19[1][16][32][32],
  float v20[1][16384]
) {	//
  for (int v21 = 0; v21 < 16384; v21 += 1) {	//
    float v22 = v19[0][(v21 / 1024)][((v21 % 1024) / 32)][(v21 % 32)];	//
    v20[0][v21] = v22;	//
  }
}

void forward_node3(
  float v23[1][1][34][34],
  float v24[16][1][3][3],
  float v25[1][16][32][32]
) {	//
  for (int v26 = 0; v26 < 1; v26 += 1) {	// L26
    for (int v27 = 0; v27 < 16; v27 += 1) {	// L26
      for (int v28 = 0; v28 < 32; v28 += 1) {	// L26
        for (int v29 = 0; v29 < 32; v29 += 1) {	// L26
          for (int v30 = 0; v30 < 1; v30 += 1) {	// L26
            for (int v31 = 0; v31 < 3; v31 += 1) {	// L26
              for (int v32 = 0; v32 < 3; v32 += 1) {	// L26
                float v33 = v23[v26][v30][(v28 + v31)][(v29 + v32)];	// L26
                float v34 = v24[v27][v30][v31][v32];	// L26
                float v35 = v25[v26][v27][v28][v29];	// L26
                float v36 = v33 * v34;	//
                float v37 = v35 + v36;	//
                v25[v26][v27][v28][v29] = v37;	// L26
              }
            }
          }
        }
      }
    }
  }
}

void forward_node4(
  float v38[16],
  float v39[1][16][32][32]
) {	//
  for (int v40 = 0; v40 < 1; v40 += 1) {	// L22
    for (int v41 = 0; v41 < 16; v41 += 1) {	// L22
      for (int v42 = 0; v42 < 32; v42 += 1) {	// L22
        for (int v43 = 0; v43 < 32; v43 += 1) {	// L22
          float v44 = v38[v41];	// L22
          v39[v40][v41][v42][v43] = v44;	// L22
        }
      }
    }
  }
}

void forward_node5(
  float v45[1][1][32][32],
  float v46[1][1][34][34]
) {	//
  for (int v47 = 0; v47 < 1; v47 += 1) {	// L17
    for (int v48 = 0; v48 < 1; v48 += 1) {	// L17
      for (int v49 = 0; v49 < 32; v49 += 1) {	// L17
        for (int v50 = 0; v50 < 32; v50 += 1) {	// L17
          float v51 = v45[v47][v48][v49][v50];	// L17
          v46[v47][v48][(v49 + 1)][(v50 + 1)] = v51;	// L17
        }
      }
    }
  }
}

/// This is top function.
void forward(
  float v52[1][1][32][32],
  float v53[1][10],
  float v54[16384][10],
  float v55[1][1][34][34],
  float v56[1][16][32][32],
  float v57[1][16384]
) {	// L9
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v52
  #pragma HLS interface s_axilite port=v52 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v53
  #pragma HLS interface s_axilite port=v53 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v54
  #pragma HLS interface s_axilite port=v54 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v55
  #pragma HLS interface s_axilite port=v55 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v56
  #pragma HLS interface s_axilite port=v56 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v57
  #pragma HLS interface s_axilite port=v57 bundle=ctrl
  float v58[16][1][3][3] = {(float)0.169618, (float)0.182195, (float)-0.242287, (float)0.204157, (float)-0.200759, (float)0.181613, (float)0.165327, (float)-0.210858, (float)-0.331973, (float)-0.125700, (float)-0.045925, (float)-0.232262, (float)-0.131083, (float)-0.317107, (float)0.307145, (float)0.316887, (float)-0.236655, (float)0.206485, (float)0.295711, (float)-0.132614, (float)-0.223665, (float)-0.029344, (float)-0.160834, (float)-0.038972, (float)-0.161814, (float)0.125304, (float)-0.138740, (float)0.122585, (float)0.141517, (float)-0.104231, (float)0.220686, (float)-0.209981, (float)-0.254468, (float)0.124534, (float)-0.162870, (float)0.026290, (float)-0.303588, (float)0.036394, (float)0.003434, (float)0.155152, (float)-0.035388, (float)-0.130481, (float)-0.279868, (float)0.222376, (float)0.198518, (float)0.107173, (float)-0.250201, (float)0.180789, (float)-0.078264, (float)0.115065, (float)0.105617, (float)0.156855, (float)0.089778, (float)-0.252934, (float)0.222178, (float)0.113076, (float)0.291991, (float)0.141331, (float)-0.310892, (float)0.212485, (float)-0.297037, (float)-0.125210, (float)0.095715, (float)0.054236, (float)-0.136358, (float)0.018744, (float)0.059341, (float)-0.006189, (float)0.271646, (float)0.045616, (float)-0.113468, (float)-0.248712, (float)-0.026717, (float)-0.190999, (float)-0.061027, (float)-0.063333, (float)0.063725, (float)-0.099104, (float)0.156107, (float)-0.240131, (float)-0.042076, (float)0.079215, (float)0.225617, (float)-0.131458, (float)0.278936, (float)0.268940, (float)-0.246653, (float)0.262431, (float)0.058470, (float)-0.180038, (float)0.036567, (float)0.327586, (float)0.244487, (float)0.325084, (float)0.034669, (float)-0.173393, (float)-0.238151, (float)0.171832, (float)-0.261707, (float)-0.092032, (float)0.176975, (float)0.042896, (float)-0.052065, (float)-0.209438, (float)-0.256908, (float)-0.055802, (float)-0.173312, (float)-0.077840, (float)-0.189004, (float)0.301812, (float)-0.266313, (float)-0.236144, (float)-0.114392, (float)-0.271915, (float)0.237879, (float)-0.044573, (float)0.023777, (float)0.123969, (float)0.269478, (float)0.077499, (float)-0.076571, (float)0.153735, (float)-0.128184, (float)0.044745, (float)0.077827, (float)-0.125887, (float)0.307027, (float)-0.127651, (float)0.237414, (float)-0.074888, (float)-0.263762, (float)-0.306796, (float)0.304069, (float)-0.133626, (float)0.097383, (float)0.158240, (float)-0.037120, (float)0.093434, (float)0.202442, (float)0.211825, (float)0.208184, (float)-0.177164, (float)0.047904, (float)-0.319041};	// L13
  #pragma HLS resource variable=v58 core=ram_t2p_bram

  float v59[16] = {(float)0.175285, (float)0.175341, (float)-0.121469, (float)-0.019936, (float)0.259413, (float)-0.022718, (float)0.124151, (float)0.271973, (float)-0.060168, (float)0.029482, (float)-0.244315, (float)0.134206, (float)-0.082869, (float)-0.099012, (float)-0.299801, (float)0.160506};	// L12
  #pragma HLS resource variable=v59 core=ram_t2p_bram

  float v60[10] = {(float)-0.004213, (float)-0.002942, (float)0.001845, (float)-0.003048, (float)0.001205, (float)-0.007714, (float)-0.000827, (float)-0.006617, (float)0.000875, (float)0.001452};	// L10
  #pragma HLS resource variable=v60 core=ram_t2p_bram

  forward_node5(v52, v55);	//
  forward_node4(v59, v56);	//
  forward_node3(v55, v58, v56);	//
  forward_node2(v56, v57);	//
  float v61[1][10];	// L34
  #pragma HLS resource variable=v61 core=ram_t2p_bram

  forward_node1(v57, v54, v61);	//
  forward_node0(v61, v60, v53);	//
}


