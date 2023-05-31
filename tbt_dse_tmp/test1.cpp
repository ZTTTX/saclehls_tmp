
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
  float v70[64] = {(float)0.001404, (float)-0.014552, (float)-0.004244, (float)-0.007011, (float)0.008354, (float)-0.005338, (float)-0.012872, (float)0.008876, (float)0.005472, (float)0.012783, (float)0.012675, (float)0.006894, (float)-0.008400, (float)-0.017160, (float)0.002031, (float)-0.003672, (float)0.005345, (float)0.001402, (float)0.017291, (float)0.014407, (float)-0.013328, (float)-0.016039, (float)-0.002896, (float)0.007712, (float)0.006504, (float)-0.005207, (float)-0.017407, (float)0.015493, (float)0.003617, (float)0.006178, (float)0.002520, (float)0.011934, (float)-0.004973, (float)-0.000262, (float)-0.017608, (float)-0.013478, (float)0.003866, (float)0.000551, (float)-0.010122, (float)-0.006023, (float)-0.012147, (float)0.010956, (float)-0.011427, (float)0.011457, (float)-0.002030, (float)0.008011, (float)-0.003451, (float)0.001756, (float)0.014427, (float)0.017641, (float)-0.002068, (float)-0.003187, (float)0.012609, (float)0.006325, (float)-0.006680, (float)0.013330, (float)0.018003, (float)-0.016161, (float)0.005390, (float)-0.003562, (float)0.010571, (float)0.017895, (float)-0.014140, (float)0.003963};	// L12
  #pragma HLS resource variable=v70 core=ram_t2p_bram

  float v71[32] = {(float)0.008104, (float)-0.088047, (float)0.119145, (float)-0.038037, (float)0.022011, (float)-0.094520, (float)0.110638, (float)-0.104408, (float)0.057440, (float)-0.088278, (float)-0.004654, (float)-0.082745, (float)-0.024990, (float)0.054389, (float)0.053908, (float)-0.039088, (float)-0.059527, (float)-0.094114, (float)0.107706, (float)-0.010976, (float)0.042037, (float)-0.027994, (float)0.059926, (float)0.039378, (float)-0.030635, (float)-0.046260, (float)0.092807, (float)0.016732, (float)-0.063849, (float)-0.109027, (float)-0.023512, (float)-0.120032};	// L10
  #pragma HLS resource variable=v71 core=ram_t2p_bram

  float v72[10] = {(float)0.051000, (float)0.082908, (float)0.041335, (float)-0.130734, (float)0.028168, (float)0.094225, (float)0.155344, (float)0.157340, (float)-0.023406, (float)-0.035035};	// L8
  #pragma HLS resource variable=v72 core=ram_t2p_bram

  float v73[32][10] = {(float)0.052754, (float)-0.135347, (float)-0.159047, (float)0.054662, (float)-0.121237, (float)0.037384, (float)0.043122, (float)-0.081001, (float)-0.073054, (float)-0.137362, (float)-0.034183, (float)0.099898, (float)-0.094591, (float)0.113747, (float)-0.019766, (float)-0.014141, (float)0.082227, (float)0.083669, (float)0.110225, (float)-0.087174, (float)-0.107678, (float)-0.162771, (float)0.052139, (float)-0.072359, (float)0.129967, (float)0.060217, (float)0.159688, (float)0.019722, (float)0.025660, (float)0.027298, (float)0.156803, (float)-0.057327, (float)0.029169, (float)-0.093275, (float)-0.004757, (float)0.119784, (float)0.030076, (float)-0.060152, (float)-0.073678, (float)-0.101073, (float)0.144444, (float)0.162334, (float)0.089110, (float)-0.000716, (float)0.159686, (float)0.046644, (float)-0.168450, (float)0.169223, (float)-0.080797, (float)0.113563, (float)0.176633, (float)-0.174289, (float)0.107310, (float)0.098172, (float)-0.009252, (float)-0.106635, (float)0.087083, (float)-0.095412, (float)-0.149569, (float)-0.156239, (float)-0.153325, (float)0.049075, (float)0.171489, (float)0.154588, (float)0.075567, (float)-0.154516, (float)-0.000405, (float)-0.039087, (float)-0.045453, (float)-0.003688, (float)-0.037110, (float)0.098022, (float)0.163966, (float)-0.018867, (float)0.146910, (float)0.152935, (float)0.016378, (float)-0.027941, (float)-0.041244, (float)-0.134337, (float)0.165153, (float)0.162951, (float)-0.142236, (float)-0.122431, (float)-0.175831, (float)0.118242, (float)0.122993, (float)0.120236, (float)0.076914, (float)0.090486, (float)-0.116402, (float)0.165250, (float)0.076165, (float)0.086752, (float)-0.172398, (float)0.156175, (float)-0.147289, (float)-0.011504, (float)-0.107207, (float)0.000406, (float)-0.033036, (float)-0.010783, (float)-0.071354, (float)-0.170941, (float)-0.155236, (float)0.056537, (float)0.043352, (float)0.122247, (float)-0.067461, (float)-0.084200, (float)0.093653, (float)0.025434, (float)-0.075222, (float)-0.112332, (float)0.026777, (float)0.047515, (float)-0.006508, (float)0.105606, (float)-0.127662, (float)-0.014353, (float)-0.052131, (float)0.132834, (float)-0.157583, (float)0.134827, (float)-0.145016, (float)-0.043010, (float)0.168210, (float)0.167561, (float)0.000133, (float)0.015554, (float)0.096086, (float)-0.074124, (float)-0.057809, (float)-0.153805, (float)0.057194, (float)0.168442, (float)0.164181, (float)0.069343, (float)0.134428, (float)-0.061226, (float)0.035453, (float)0.108267, (float)-0.156921, (float)-0.043658, (float)-0.131789, (float)0.064393, (float)-0.047082, (float)-0.025603, (float)0.023172, (float)0.037880, (float)-0.029574, (float)0.047808, (float)0.131543, (float)0.076911, (float)-0.049948, (float)0.039069, (float)-0.125176, (float)-0.042338, (float)0.150554, (float)-0.170278, (float)-0.029403, (float)-0.142015, (float)-0.159734, (float)-0.008482, (float)-0.146240, (float)0.104745, (float)-0.113279, (float)0.097940, (float)-0.029541, (float)0.040086, (float)-0.174086, (float)-0.173137, (float)0.165603, (float)0.117927, (float)0.171651, (float)0.009242, (float)-0.081685, (float)-0.007908, (float)-0.013048, (float)-0.126269, (float)-0.111641, (float)-0.018508, (float)-0.067884, (float)-0.175299, (float)0.147977, (float)0.011608, (float)0.052470, (float)-0.125168, (float)0.064045, (float)0.120785, (float)-0.105666, (float)0.037874, (float)0.119313, (float)0.020571, (float)-0.077517, (float)0.156119, (float)0.074232, (float)0.080844, (float)0.113094, (float)0.138668, (float)0.171192, (float)-0.129451, (float)-0.067589, (float)0.001096, (float)-0.083461, (float)-0.095418, (float)0.104469, (float)-0.091125, (float)-0.159085, (float)0.114776, (float)0.078029, (float)0.059579, (float)0.092620, (float)0.013604, (float)-0.167940, (float)0.105858, (float)0.160320, (float)0.055474, (float)0.164771, (float)-0.066256, (float)0.094493, (float)-0.032798, (float)0.128954, (float)-0.037599, (float)0.135976, (float)-0.053151, (float)0.042676, (float)-0.142715, (float)-0.044560, (float)-0.040314, (float)-0.108612, (float)-0.145295, (float)-0.032055, (float)-0.074253, (float)-0.090640, (float)-0.157378, (float)-0.173669, (float)0.055023, (float)-0.070347, (float)0.128004, (float)0.174990, (float)-0.151003, (float)-0.010208, (float)-0.141819, (float)-0.030295, (float)0.118883, (float)0.084754, (float)-0.089723, (float)0.077977, (float)0.020049, (float)-0.075862, (float)0.134479, (float)0.071383, (float)-0.105552, (float)-0.015611, (float)-0.107461, (float)-0.093393, (float)0.165965, (float)-0.025712, (float)0.150909, (float)0.175837, (float)0.170971, (float)-0.088082, (float)-0.067283, (float)-0.135948, (float)0.007439, (float)0.107886, (float)0.096462, (float)0.034299, (float)-0.048746, (float)0.032521, (float)-0.140744, (float)-0.089270, (float)0.042348, (float)-0.038198, (float)0.051242, (float)0.019208, (float)0.092086, (float)0.008899, (float)0.045574, (float)-0.172410, (float)-0.023943, (float)0.014663, (float)-0.158696, (float)0.001730, (float)-0.055800, (float)-0.120545, (float)-0.164107, (float)-0.002639, (float)-0.095657, (float)0.172699, (float)-0.010735, (float)-0.132853, (float)0.030152, (float)0.118174, (float)-0.160482, (float)0.003784, (float)-0.134530, (float)0.059789, (float)0.051599, (float)0.019400, (float)-0.055439, (float)0.166212, (float)0.078590, (float)0.162624, (float)-0.103050, (float)-0.104583, (float)-0.114391, (float)0.175843, (float)-0.111460, (float)-0.028180, (float)0.173202, (float)0.128566, (float)-0.008681, (float)-0.116443, (float)-0.150334, (float)-0.138218, (float)-0.040687, (float)0.022195, (float)0.055328};	// L55
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


