
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
  for (int v11 = 0; v11 < 1; v11 += 1) {	// L61
    for (int v12 = 0; v12 < 10; v12 += 1) {	// L61
      for (int v13 = 0; v13 < 32; v13 += 1) {	// L61
        float v14 = v8[v11][v13];	// L61
        float v15 = v9[v13][v12];	// L61
        float v16 = v10[v11][v12];	// L61
        float v17 = v14 * v15;	//
        float v18 = v16 + v17;	//
        v10[v11][v12] = v18;	// L61
      }
    }
  }
}

void forward_node2(
  float v19[1][32],
  float v20[32],
  float v21[1][32]
) {	//
  for (int v22 = 0; v22 < 1; v22 += 1) {	// L48
    for (int v23 = 0; v23 < 32; v23 += 1) {	// L48
      float v24 = v19[0][v23];	// L48
      float v25 = v20[v23];	// L48
      float v26 = v24 + v25;	// L45
      bool v27 = v26 > (float)0.000000;	// L50
      float v28 = v27 ? v26 : (float)0.000000;	// L51
      v21[v22][v23] = v28;	// L48
    }
  }
}

void forward_node3(
  float v29[1][64],
  float v30[64][32],
  float v31[1][32]
) {	//
  for (int v32 = 0; v32 < 1; v32 += 1) {	// L42
    for (int v33 = 0; v33 < 32; v33 += 1) {	// L42
      for (int v34 = 0; v34 < 64; v34 += 1) {	// L42
        float v35 = v29[v32][v34];	// L42
        float v36 = v30[v34][v33];	// L42
        float v37 = v31[v32][v33];	// L42
        float v38 = v35 * v36;	//
        float v39 = v37 + v38;	//
        v31[v32][v33] = v39;	// L42
      }
    }
  }
}

void forward_node4(
  float v40[1][64],
  float v41[64],
  float v42[1][64]
) {	//
  for (int v43 = 0; v43 < 1; v43 += 1) {	// L29
    for (int v44 = 0; v44 < 64; v44 += 1) {	// L29
      float v45 = v40[0][v44];	// L29
      float v46 = v41[v44];	// L29
      float v47 = v45 + v46;	// L26
      bool v48 = v47 > (float)0.000000;	// L31
      float v49 = v48 ? v47 : (float)0.000000;	// L32
      v42[v43][v44] = v49;	// L29
    }
  }
}

void forward_node5(
  float v50[1][3072],
  float v51[3072][64],
  float v52[1][64]
) {	//
  for (int v53 = 0; v53 < 1; v53 += 1) {	// L23
    for (int v54 = 0; v54 < 64; v54 += 1) {	// L23
      for (int v55 = 0; v55 < 3072; v55 += 1) {	// L23
        float v56 = v50[v53][v55];	// L23
        float v57 = v51[v55][v54];	// L23
        float v58 = v52[v53][v54];	// L23
        float v59 = v56 * v57;	//
        float v60 = v58 + v59;	//
        v52[v53][v54] = v60;	// L23
      }
    }
  }
}

void forward_node6(
  float v61[1][3][32][32],
  float v62[1][3072]
) {	//
  for (int v63 = 0; v63 < 3072; v63 += 1) {	//
    float v64 = v61[0][(v63 / 1024)][((v63 % 1024) / 32)][(v63 % 32)];	//
    v62[0][v63] = v64;	//
  }
}

/// This is top function.
void forward(
  float v65[1][3][32][32],
  float v66[1][10],
  float v67[3072][64],
  float v68[64][32],
  float v69[1][3072]
) {	// L7
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v65
  #pragma HLS interface s_axilite port=v65 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v66
  #pragma HLS interface s_axilite port=v66 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v67
  #pragma HLS interface s_axilite port=v67 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v68
  #pragma HLS interface s_axilite port=v68 bundle=ctrl
  #pragma HLS interface m_axi offset=slave port=v69
  #pragma HLS interface s_axilite port=v69 bundle=ctrl
  float v70[64] = {(float)0.016423, (float)-0.008255, (float)-0.004849, (float)0.006518, (float)-0.003171, (float)0.017682, (float)-0.016032, (float)0.000939, (float)-0.005378, (float)-0.000583, (float)-0.000993, (float)0.009526, (float)-0.000168, (float)0.004914, (float)0.007087, (float)0.005972, (float)0.003032, (float)-0.007525, (float)0.002431, (float)0.006497, (float)-0.009755, (float)-0.004363, (float)-0.000260, (float)-0.003340, (float)-0.007238, (float)-0.010057, (float)0.016153, (float)-0.011403, (float)-0.006492, (float)-0.010710, (float)-0.003187, (float)0.000402, (float)0.017504, (float)-0.013853, (float)-0.007468, (float)0.012077, (float)-0.008613, (float)0.006713, (float)-0.004459, (float)-0.006195, (float)0.017835, (float)0.005339, (float)0.015139, (float)-0.013613, (float)0.009440, (float)0.013917, (float)-0.011650, (float)0.002180, (float)-0.008807, (float)0.009809, (float)-0.012655, (float)0.010592, (float)-0.005050, (float)0.014573, (float)0.011526, (float)0.010348, (float)-0.004051, (float)-0.003163, (float)-0.011527, (float)-0.002979, (float)0.004722, (float)-0.008962, (float)0.016897, (float)0.001274};	// L12
  #pragma HLS resource variable=v70 core=ram_t2p_bram

  float v71[32] = {(float)0.032505, (float)-0.116573, (float)0.028106, (float)-0.074597, (float)0.067316, (float)0.010908, (float)0.028476, (float)0.045281, (float)0.026288, (float)0.026379, (float)-0.065236, (float)-0.104761, (float)-0.023071, (float)0.066525, (float)0.103410, (float)-0.011623, (float)-0.094852, (float)-0.107791, (float)0.118643, (float)-0.024602, (float)0.109491, (float)-0.076219, (float)0.029535, (float)-0.069198, (float)0.124473, (float)0.038320, (float)0.075790, (float)-0.117245, (float)0.023290, (float)0.060530, (float)-0.063785, (float)0.103022};	// L10
  #pragma HLS resource variable=v71 core=ram_t2p_bram

  float v72[10] = {(float)0.095820, (float)-0.138620, (float)-0.004332, (float)-0.125428, (float)-0.038754, (float)-0.120286, (float)-0.153298, (float)-0.114444, (float)-0.113734, (float)0.156155};	// L8
  #pragma HLS resource variable=v72 core=ram_t2p_bram

  float v73[32][10] = {(float)0.069871, (float)-0.049291, (float)-0.063212, (float)-0.077843, (float)-0.029710, (float)-0.012605, (float)0.094407, (float)0.070469, (float)-0.153680, (float)0.132511, (float)-0.019365, (float)0.069375, (float)0.105882, (float)0.164792, (float)0.098792, (float)-0.025869, (float)-0.083303, (float)0.052734, (float)-0.149366, (float)-0.133511, (float)0.071491, (float)-0.137743, (float)-0.086452, (float)-0.064632, (float)-0.086276, (float)-0.095333, (float)-0.169237, (float)-0.037978, (float)0.077141, (float)0.095459, (float)-0.113666, (float)0.049199, (float)0.173353, (float)-0.168489, (float)0.001005, (float)0.151210, (float)-0.017929, (float)0.083671, (float)-0.118882, (float)0.055781, (float)-0.002879, (float)0.017146, (float)0.102607, (float)-0.060301, (float)-0.067228, (float)-0.132020, (float)-0.020183, (float)0.075615, (float)-0.069481, (float)-0.016345, (float)0.136961, (float)0.026708, (float)0.016157, (float)0.023736, (float)0.094616, (float)0.106907, (float)-0.128367, (float)0.073712, (float)-0.127250, (float)0.100140, (float)-0.067752, (float)0.093805, (float)-0.074161, (float)-0.033881, (float)0.088021, (float)-0.087767, (float)-0.071649, (float)-0.004298, (float)0.075923, (float)-0.121646, (float)0.048898, (float)-0.148057, (float)-0.035279, (float)0.018213, (float)-0.174278, (float)0.077309, (float)0.016636, (float)0.104117, (float)-0.055951, (float)0.001285, (float)-0.076519, (float)0.025985, (float)-0.115093, (float)0.110285, (float)-0.091730, (float)-0.067686, (float)0.154227, (float)-0.030736, (float)0.142252, (float)-0.114743, (float)0.068977, (float)0.056928, (float)-0.067746, (float)0.147192, (float)0.124352, (float)0.019065, (float)-0.084415, (float)-0.148924, (float)-0.173454, (float)-0.102178, (float)0.013129, (float)0.024711, (float)0.171175, (float)0.114504, (float)0.174154, (float)0.062213, (float)-0.008331, (float)0.023118, (float)0.077185, (float)-0.158766, (float)0.154707, (float)-0.116525, (float)-0.041516, (float)-0.166213, (float)-0.106694, (float)0.006308, (float)-0.153261, (float)0.096060, (float)0.007478, (float)0.113936, (float)-0.169037, (float)-0.060117, (float)0.171155, (float)0.120530, (float)0.165850, (float)-0.021724, (float)0.052610, (float)-0.010588, (float)0.007537, (float)-0.171455, (float)0.003162, (float)-0.018284, (float)-0.150239, (float)-0.025886, (float)0.140512, (float)-0.163353, (float)-0.149990, (float)0.005268, (float)0.043042, (float)0.149248, (float)-0.006796, (float)0.141662, (float)0.111130, (float)-0.122526, (float)0.022991, (float)0.065387, (float)0.146590, (float)-0.140321, (float)-0.170027, (float)0.117190, (float)0.046444, (float)0.074792, (float)-0.164496, (float)-0.042652, (float)0.059605, (float)-0.029879, (float)0.154482, (float)0.169874, (float)-0.057790, (float)-0.001179, (float)-0.064940, (float)-0.073836, (float)0.112215, (float)0.166348, (float)0.112687, (float)-0.155826, (float)0.120759, (float)0.113893, (float)-0.076524, (float)-0.164576, (float)0.137042, (float)0.034297, (float)-0.122641, (float)-0.095675, (float)-0.026209, (float)0.101748, (float)-0.124525, (float)0.149705, (float)-0.101220, (float)-0.007081, (float)0.109247, (float)0.012529, (float)-0.170422, (float)-0.075627, (float)0.153505, (float)-0.054352, (float)0.174842, (float)0.104350, (float)0.054955, (float)-0.065708, (float)-0.147464, (float)0.044725, (float)-0.012602, (float)-0.114226, (float)-0.113592, (float)-0.116146, (float)0.051897, (float)-0.069587, (float)0.112198, (float)-0.165791, (float)0.007960, (float)-0.109902, (float)0.100304, (float)-0.092957, (float)0.157529, (float)-0.134290, (float)0.059260, (float)0.144113, (float)-0.165684, (float)-0.110207, (float)0.113314, (float)0.125608, (float)-0.148450, (float)0.004517, (float)-0.155482, (float)-0.066646, (float)-0.140179, (float)0.160758, (float)0.095212, (float)-0.101660, (float)0.012867, (float)-0.171172, (float)0.115404, (float)-0.018044, (float)0.009134, (float)-0.056641, (float)-0.172333, (float)-0.165929, (float)0.059277, (float)0.172394, (float)-0.171442, (float)-0.048841, (float)0.142126, (float)0.040264, (float)0.000371, (float)-0.045191, (float)0.078406, (float)-0.152258, (float)0.032807, (float)-0.007737, (float)0.120706, (float)-0.125630, (float)0.141301, (float)-0.138614, (float)0.076254, (float)0.012397, (float)-0.119024, (float)0.046956, (float)0.175773, (float)-0.002813, (float)0.134873, (float)0.046490, (float)0.083605, (float)0.161923, (float)-0.091554, (float)-0.012088, (float)0.132404, (float)-0.139553, (float)0.168222, (float)0.084067, (float)0.027414, (float)-0.011371, (float)-0.172068, (float)-0.097586, (float)0.162748, (float)-0.125512, (float)0.163505, (float)-0.038432, (float)-0.031325, (float)0.041139, (float)-0.123771, (float)0.156205, (float)-0.117383, (float)-0.119103, (float)0.103756, (float)-0.145633, (float)0.009587, (float)-0.170394, (float)0.115991, (float)0.121406, (float)-0.153208, (float)0.161746, (float)-0.160266, (float)0.101221, (float)0.107487, (float)0.016112, (float)0.077119, (float)-0.058945, (float)0.114779, (float)0.133446, (float)-0.016069, (float)0.159528, (float)-0.064966, (float)0.166992, (float)0.000019, (float)0.065854, (float)0.037481, (float)-0.008671, (float)-0.050384, (float)0.098945, (float)0.113950, (float)0.055157, (float)0.046701, (float)0.057511, (float)0.163920, (float)-0.164612, (float)-0.003161, (float)0.062222, (float)-0.043385, (float)0.053606, (float)0.119594, (float)-0.025584, (float)-0.062138, (float)-0.153917, (float)0.029414, (float)0.116618, (float)0.078182, (float)-0.014784, (float)0.133140, (float)-0.023214};	// L55
  #pragma HLS resource variable=v73 core=ram_t2p_bram

  forward_node6(v65, v69);	//
  float v74[1][64];	// L21
  #pragma HLS resource variable=v74 core=ram_t2p_bram

  float v75[1][64];	// L22
  #pragma HLS resource variable=v75 core=ram_t2p_bram

  forward_node5(v69, v67, v75);	//
  forward_node4(v75, v70, v74);	//
  float v76[1][32];	// L40
  #pragma HLS resource variable=v76 core=ram_t2p_bram

  float v77[1][32];	// L41
  #pragma HLS resource variable=v77 core=ram_t2p_bram

  forward_node3(v74, v68, v77);	//
  forward_node2(v77, v71, v76);	//
  float v78[1][10];	// L60
  #pragma HLS resource variable=v78 core=ram_t2p_bram

  forward_node1(v76, v73, v78);	//
  forward_node0(v78, v72, v66);	//
}


