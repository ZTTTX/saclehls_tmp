
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

Struct my_test_struct {
typedef float t_DataType;
;
static const unsigned t_MaxSizeC = 1024
};
void forward_node0(
  float v0[1][10],
  float v1[10],
  float v2[1][10]
) {	//
  for (int v3 = 0; v3 < 1; v3 += 1) {	// L20
    for (int v4 = 0; v4 < 10; v4 += 1) {	// L20
      float v5 = v0[0][v4];	// L20
      float v6 = v1[v4];	// L20
      float v7 = v5 + v6;	// L22
      v2[v3][v4] = v7;	// L20
    }
  }
}

void forward_node1(
  float v8[1][1024],
  float v9[1024][10],
  float v10[1][10]
) {	//
  gemm<float, int, 32, 4, 1024, my_test_struct>(v8, v9, v10, v10, (int)1, (int)10, (int)1024, (float)1.000000, (float)0.000000);	// L19
}

void forward_node2(
  float v11[1][1][32][32],
  float v12[1][1024]
) {	//
  for (int v13 = 0; v13 < 1024; v13 += 1) {	//
    float v14 = v11[0][0][(v13 / 32)][(v13 % 32)];	//
    v12[0][v13] = v14;	//
  }
}

/// This is top function.
void forward(
  float v15[1][1][32][32],
  float v16[1][10],
  float v17[1024][10]
) {	// L7
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v15
  #pragma HLS interface s_axilite port=v15 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v16
  #pragma HLS interface s_axilite port=v16 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v17
  #pragma HLS interface s_axilite port=v17 bundle=ctrl
  float v18[10] = {(float)0.010223, (float)-0.029616, (float)0.008847, (float)0.025677, (float)0.027571, (float)0.024401, (float)0.006311, (float)0.030089, (float)-0.018667, (float)-0.013252};	// L8
  #pragma HLS resource variable=v18 core=ram_t2p_bram

  float v19[1][1024];	//
  #pragma HLS resource variable=v19 core=ram_t2p_bram

  forward_node2(v15, v19);	//
  float v20[1][10];	// L18
  #pragma HLS resource variable=v20 core=ram_t2p_bram

  forward_node1(v19, v17, v20);	//
  forward_node0(v20, v18, v16);	//
}


