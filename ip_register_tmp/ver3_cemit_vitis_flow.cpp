
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

using namespace std

void forward_node0(
  float v0[1][10],
  float v1[10],
  float v2[1][10]
) {	//
  for (int v3 = 0; v3 < 1; v3 += 1) {	// L25
    for (int v4 = 0; v4 < 10; v4 += 1) {	// L25
      float v5 = v0[0][v4];	// L25
      float v6 = v1[v4];	// L25
      float v7 = v5 + v6;	// L22
      bool v8 = v7 > (float)0.000000;	// L27
      float v9 = v8 ? v7 : (float)0.000000;	// L28
      v2[v3][v4] = v9;	// L25
    }
  }
}

void forward_node1(
  float v10[1][1024],
  float v11[1024][10],
  float v12[1][10]
) {	//
  for (int v13 = 0; v13 < 1; v13 += 1) {	// L19
    for (int v14 = 0; v14 < 10; v14 += 1) {	// L19
      for (int v15 = 0; v15 < 1024; v15 += 1) {	// L19
        float v16 = v10[v13][v15];	// L19
        float v17 = v11[v15][v14];	// L19
        float v18 = v12[v13][v14];	// L19
        float v19 = v16 * v17;	//
        float v20 = v18 + v19;	//
        v12[v13][v14] = v20;	// L19
      }
    }
  }
}

void forward_node2(
  float v21[1][1][32][32],
  float v22[1][1024]
) {	//
  for (int v23 = 0; v23 < 1024; v23 += 1) {	//
    float v24 = v21[0][0][(v23 / 32)][(v23 % 32)];	//
    v22[0][v23] = v24;	//
  }
}

/// This is top function.
void forward(
  float v25[1][1][32][32],
  float v26[1][10],
  float v27[1024][10]
) {	// L7
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v25
  #pragma HLS interface s_axilite port=v25 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v26
  #pragma HLS interface s_axilite port=v26 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v27
  #pragma HLS interface s_axilite port=v27 bundle=ctrl
  float v28[10] = {(float)-0.006301, (float)0.029061, (float)-0.006143, (float)-0.016538, (float)0.003510, (float)-0.016316, (float)-0.029194, (float)-0.002508, (float)-0.028033, (float)0.000132};	// L8
  #pragma HLS resource variable=v28 core=ram_t2p_bram

  float v29[1][1024];	//
  #pragma HLS resource variable=v29 core=ram_t2p_bram

  forward_node2(v25, v29);	//
  float v30[1][10];	// L18
  #pragma HLS resource variable=v30 core=ram_t2p_bram

  forward_node1(v29, v27, v30);	//
  forward_node0(v30, v28, v26);	//
}


