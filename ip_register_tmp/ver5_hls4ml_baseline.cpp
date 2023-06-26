
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
  for (int v3 = 0; v3 < 1; v3 += 1) {	// L30
    for (int v4 = 0; v4 < 10; v4 += 1) {	// L30
      float v5 = v0[0][v4];	// L30
      float v6 = v1[v4];	// L30
      float v7 = v5 + v6;	// L32
      v2[v3][v4] = v7;	// L30
    }
  }
}

void forward_node1(
  float v8[1][14400],
  float v9[14400][10],
  float v10[1][10]
) {	//
  for (int v11 = 0; v11 < 1; v11 += 1) {	// L29
    for (int v12 = 0; v12 < 10; v12 += 1) {	// L29
      for (int v13 = 0; v13 < 14400; v13 += 1) {	// L29
        float v14 = v8[v11][v13];	// L29
        float v15 = v9[v13][v12];	// L29
        float v16 = v10[v11][v12];	// L29
        float v17 = v14 * v15;	//
        float v18 = v16 + v17;	//
        v10[v11][v12] = v18;	// L29
      }
    }
  }
}

void forward_node2(
  float v19[1][16][30][30],
  float v20[1][14400]
) {	//
  for (int v21 = 0; v21 < 14400; v21 += 1) {	//
    float v22 = v19[0][(v21 / 900)][((v21 % 900) / 30)][(v21 % 30)];	//
    v20[0][v21] = v22;	//
  }
}

void forward_node3(
  float v23[1][1][32][32],
  float v24[16][1][3][3],
  float v25[1][16][30][30]
) {	//
  for (int v26 = 0; v26 < 1; v26 += 1) {	// L20
    for (int v27 = 0; v27 < 16; v27 += 1) {	// L20
      for (int v28 = 0; v28 < 30; v28 += 1) {	// L20
        for (int v29 = 0; v29 < 30; v29 += 1) {	// L20
          for (int v30 = 0; v30 < 1; v30 += 1) {	// L20
            for (int v31 = 0; v31 < 3; v31 += 1) {	// L20
              for (int v32 = 0; v32 < 3; v32 += 1) {	// L20
                float v33 = v23[v26][v30][(v28 + v31)][(v29 + v32)];	// L20
                float v34 = v24[v27][v30][v31][v32];	// L20
                float v35 = v25[v26][v27][v28][v29];	// L20
                float v36 = v33 * v34;	//
                float v37 = v35 + v36;	//
                v25[v26][v27][v28][v29] = v37;	// L20
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
  float v39[1][16][30][30]
) {	//
  for (int v40 = 0; v40 < 1; v40 += 1) {	// L16
    for (int v41 = 0; v41 < 16; v41 += 1) {	// L16
      for (int v42 = 0; v42 < 30; v42 += 1) {	// L16
        for (int v43 = 0; v43 < 30; v43 += 1) {	// L16
          float v44 = v38[v41];	// L16
          v39[v40][v41][v42][v43] = v44;	// L16
        }
      }
    }
  }
}

/// This is top function.
void forward(
  float v45[1][1][32][32],
  float v46[1][10],
  float v47[14400][10],
  float v48[1][16][30][30],
  float v49[1][14400]
) {	// L9
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v45
  #pragma HLS interface s_axilite port=v45 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v46
  #pragma HLS interface s_axilite port=v46 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v47
  #pragma HLS interface s_axilite port=v47 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v48
  #pragma HLS interface s_axilite port=v48 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v49
  #pragma HLS interface s_axilite port=v49 bundle=ctrl
  float v50[16][1][3][3] = {(float)0.193695, (float)0.054736, (float)-0.217778, (float)0.192036, (float)0.267184, (float)-0.090886, (float)0.269298, (float)-0.034672, (float)-0.112264, (float)-0.127171, (float)0.003456, (float)0.315432, (float)-0.025175, (float)0.139313, (float)-0.010147, (float)0.326608, (float)-0.250921, (float)-0.005831, (float)-0.298626, (float)-0.157163, (float)-0.162817, (float)-0.066016, (float)-0.055630, (float)0.218518, (float)-0.016430, (float)-0.031168, (float)0.061049, (float)0.291784, (float)0.094298, (float)0.216651, (float)0.186540, (float)-0.028967, (float)0.259833, (float)0.081514, (float)-0.186813, (float)-0.264199, (float)-0.229723, (float)-0.017827, (float)0.143622, (float)-0.082465, (float)0.188502, (float)-0.032680, (float)0.198683, (float)-0.244972, (float)-0.219570, (float)-0.285165, (float)-0.042494, (float)0.164213, (float)-0.271539, (float)-0.317421, (float)0.130995, (float)0.127682, (float)0.285428, (float)-0.102764, (float)-0.221756, (float)-0.102683, (float)0.273939, (float)-0.298301, (float)0.311205, (float)-0.060476, (float)0.157170, (float)0.268651, (float)0.068806, (float)0.220611, (float)0.108747, (float)0.128920, (float)0.024597, (float)-0.242550, (float)-0.062080, (float)-0.001395, (float)0.252042, (float)-0.287345, (float)-0.327252, (float)0.298467, (float)-0.024882, (float)-0.055722, (float)0.159660, (float)0.264598, (float)-0.136443, (float)-0.282375, (float)-0.145982, (float)0.314045, (float)-0.088401, (float)0.224888, (float)-0.164555, (float)0.117379, (float)-0.265722, (float)-0.049906, (float)0.268550, (float)-0.001689, (float)0.067304, (float)-0.037651, (float)-0.047394, (float)0.030853, (float)0.061133, (float)-0.229651, (float)-0.056410, (float)-0.085895, (float)0.131567, (float)-0.187649, (float)0.026868, (float)0.043494, (float)-0.153502, (float)0.155656, (float)-0.168729, (float)-0.295107, (float)-0.268492, (float)0.272129, (float)0.190087, (float)-0.319858, (float)-0.053659, (float)-0.217100, (float)0.312077, (float)0.070006, (float)0.204766, (float)-0.232074, (float)-0.269277, (float)-0.115107, (float)-0.238373, (float)-0.051255, (float)0.242331, (float)0.315974, (float)0.228198, (float)0.091422, (float)-0.137775, (float)-0.203918, (float)0.328893, (float)0.077591, (float)-0.286678, (float)-0.065939, (float)0.177057, (float)0.124285, (float)-0.165596, (float)-0.086805, (float)-0.209271, (float)0.304822, (float)-0.215936, (float)0.322131, (float)0.104216, (float)-0.206937, (float)0.009454, (float)0.106956, (float)0.114722, (float)0.201500};	// L13
  #pragma HLS resource variable=v50 core=ram_t2p_bram

  float v51[16] = {(float)-0.198921, (float)-0.273219, (float)0.202328, (float)-0.078605, (float)0.125979, (float)0.061520, (float)-0.137337, (float)0.181695, (float)-0.133327, (float)0.288195, (float)0.056903, (float)0.273306, (float)0.049878, (float)0.182747, (float)0.107954, (float)-0.304992};	// L12
  #pragma HLS resource variable=v51 core=ram_t2p_bram

  float v52[10] = {(float)0.005304, (float)0.003062, (float)-0.000763, (float)-0.005677, (float)0.005218, (float)-0.004528, (float)-0.001889, (float)-0.006977, (float)0.006989, (float)-0.004053};	// L10
  #pragma HLS resource variable=v52 core=ram_t2p_bram

  forward_node4(v51, v48);	//
  forward_node3(v45, v50, v48);	//
  forward_node2(v48, v49);	//
  float v53[1][10];	// L28
  #pragma HLS resource variable=v53 core=ram_t2p_bram

  forward_node1(v49, v47, v53);	//
  forward_node0(v53, v52, v46);	//
}


