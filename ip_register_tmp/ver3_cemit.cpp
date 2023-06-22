
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
  float v8[32][10],
  float v9[1][32],
  float v10[1][10]
) {	//
  gemm<float, int, 32, 2, 1024>((int)1, (int)10, (int)32, (float)1.000000, (float)0.000000, v8, v9, v10, v10);	// L61
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
  gemm<float, int, 32, 2, 1024>((int)1, (int)32, (int)64, (float)1.000000, (float)0.000000, v21, v22, v23, v23);	// L42
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
  gemm<float, int, 32, 2, 1024>((int)1, (int)64, (int)3072, (float)1.000000, (float)0.000000, v34, v35, v36, v36);	// L23
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
  float v46[64] = {(float)0.015979, (float)-0.001881, (float)0.012370, (float)0.002388, (float)0.004341, (float)-0.005831, (float)-0.011558, (float)0.008170, (float)-0.003894, (float)0.013174, (float)0.016848, (float)0.002663, (float)0.002709, (float)-0.012877, (float)0.015068, (float)0.007045, (float)0.017811, (float)0.009427, (float)-0.001765, (float)0.001178, (float)-0.011101, (float)-0.010298, (float)0.006494, (float)0.000555, (float)0.012140, (float)-0.003069, (float)-0.006850, (float)-0.004328, (float)0.005173, (float)0.007982, (float)-0.011012, (float)0.006247, (float)-0.017558, (float)0.004380, (float)0.006805, (float)0.005386, (float)-0.009687, (float)-0.006997, (float)0.008088, (float)0.000674, (float)-0.003616, (float)0.009046, (float)0.000543, (float)-0.012593, (float)0.016256, (float)-0.017793, (float)0.016954, (float)0.014829, (float)0.012510, (float)0.008069, (float)-0.006313, (float)-0.016808, (float)-0.008682, (float)0.011358, (float)-0.013513, (float)0.010859, (float)0.014739, (float)0.008179, (float)-0.011728, (float)-0.000129, (float)0.014977, (float)-0.006594, (float)-0.007643, (float)0.000630};	// L12
  #pragma HLS resource variable=v46 core=ram_t2p_bram

  float v47[32] = {(float)-0.015740, (float)0.055946, (float)0.009910, (float)0.021014, (float)-0.029059, (float)0.052992, (float)-0.068075, (float)-0.025126, (float)0.010747, (float)0.019079, (float)-0.108115, (float)-0.005438, (float)-0.061112, (float)-0.018815, (float)-0.108980, (float)0.085644, (float)0.033987, (float)-0.095739, (float)0.055903, (float)0.061709, (float)-0.009467, (float)-0.033842, (float)0.041786, (float)0.061696, (float)0.050446, (float)-0.079458, (float)-0.106536, (float)0.004055, (float)-0.085559, (float)-0.110034, (float)0.103636, (float)-0.057630};	// L10
  #pragma HLS resource variable=v47 core=ram_t2p_bram

  float v48[10] = {(float)0.024408, (float)0.058111, (float)-0.050718, (float)-0.063940, (float)0.153019, (float)0.173915, (float)0.131704, (float)-0.140654, (float)0.170694, (float)-0.108716};	// L8
  #pragma HLS resource variable=v48 core=ram_t2p_bram

  float v49[32][10] = {(float)0.050623, (float)0.115691, (float)0.075309, (float)-0.065686, (float)0.113149, (float)0.029452, (float)0.042692, (float)0.032905, (float)0.044428, (float)0.090740, (float)-0.022065, (float)0.107658, (float)0.062138, (float)-0.136649, (float)0.011032, (float)-0.043292, (float)-0.155068, (float)-0.009683, (float)0.081525, (float)-0.152549, (float)-0.143773, (float)-0.133332, (float)-0.118255, (float)-0.030102, (float)-0.110121, (float)0.038732, (float)-0.059872, (float)-0.010531, (float)0.081103, (float)-0.088082, (float)0.076703, (float)0.043250, (float)0.170665, (float)0.148152, (float)0.020479, (float)0.025013, (float)0.129506, (float)-0.100294, (float)0.063021, (float)0.131097, (float)-0.153519, (float)0.003711, (float)0.005115, (float)0.170854, (float)-0.140320, (float)0.016012, (float)0.140796, (float)0.116914, (float)0.148963, (float)-0.053321, (float)0.071707, (float)-0.043703, (float)0.155009, (float)0.059215, (float)-0.160953, (float)-0.071314, (float)-0.054048, (float)0.155619, (float)0.095176, (float)-0.170337, (float)-0.136935, (float)0.161072, (float)0.010838, (float)0.012919, (float)-0.120315, (float)-0.060229, (float)0.018349, (float)-0.119089, (float)-0.153137, (float)0.003915, (float)0.162854, (float)-0.062352, (float)0.147616, (float)0.020504, (float)-0.081946, (float)-0.045644, (float)-0.140314, (float)-0.125781, (float)-0.031357, (float)0.004504, (float)-0.132588, (float)0.102066, (float)0.058159, (float)0.086634, (float)0.112317, (float)0.000015, (float)-0.110164, (float)0.033872, (float)-0.141153, (float)0.153409, (float)-0.086606, (float)0.155457, (float)0.043404, (float)-0.022117, (float)-0.048686, (float)-0.055056, (float)-0.026336, (float)-0.064957, (float)-0.152350, (float)0.100784, (float)-0.173551, (float)0.095185, (float)-0.088967, (float)0.096219, (float)-0.120780, (float)0.125589, (float)0.128035, (float)-0.169493, (float)0.016193, (float)-0.043408, (float)-0.176350, (float)0.038639, (float)-0.006798, (float)0.162606, (float)-0.118826, (float)0.044017, (float)0.031014, (float)0.017592, (float)0.096177, (float)0.029794, (float)-0.115827, (float)0.153684, (float)-0.088814, (float)-0.145409, (float)-0.091063, (float)-0.150165, (float)0.068085, (float)-0.043665, (float)0.118708, (float)0.057384, (float)0.039112, (float)0.071238, (float)-0.032440, (float)0.030259, (float)0.000457, (float)0.063208, (float)-0.129834, (float)0.157552, (float)-0.140963, (float)-0.007072, (float)0.165754, (float)-0.080792, (float)0.118837, (float)-0.003455, (float)-0.000169, (float)0.085122, (float)0.123642, (float)-0.034874, (float)-0.169888, (float)0.135895, (float)-0.109074, (float)-0.176202, (float)-0.002101, (float)0.071297, (float)-0.077675, (float)0.016421, (float)0.140674, (float)0.027718, (float)0.038743, (float)0.037973, (float)-0.007973, (float)-0.042177, (float)0.103496, (float)-0.162417, (float)0.108851, (float)-0.021781, (float)0.167769, (float)-0.161297, (float)0.031330, (float)-0.120510, (float)0.093295, (float)0.126380, (float)-0.123868, (float)0.122853, (float)-0.147970, (float)0.121019, (float)0.113587, (float)0.007830, (float)0.042291, (float)-0.166682, (float)0.121999, (float)0.036221, (float)-0.085083, (float)-0.046666, (float)-0.091555, (float)-0.145592, (float)-0.083588, (float)0.121005, (float)-0.143558, (float)-0.168526, (float)0.005961, (float)0.091554, (float)-0.041476, (float)0.130457, (float)-0.172534, (float)-0.078372, (float)0.132189, (float)0.055364, (float)-0.138705, (float)0.072173, (float)-0.091678, (float)0.109822, (float)0.127982, (float)0.045790, (float)0.132792, (float)-0.009810, (float)0.091473, (float)-0.080744, (float)0.079311, (float)-0.069598, (float)-0.134976, (float)0.082363, (float)0.091903, (float)0.133971, (float)-0.068928, (float)-0.143912, (float)0.161192, (float)0.154015, (float)-0.024919, (float)0.136735, (float)-0.147555, (float)0.028250, (float)0.119721, (float)-0.050302, (float)0.100310, (float)0.001323, (float)-0.043086, (float)-0.142814, (float)0.075838, (float)0.136555, (float)0.021683, (float)0.025839, (float)-0.167838, (float)0.174532, (float)-0.100282, (float)0.102067, (float)0.034281, (float)0.100420, (float)0.114180, (float)0.046005, (float)0.063535, (float)-0.083606, (float)-0.094095, (float)-0.163094, (float)-0.130699, (float)-0.034475, (float)-0.091525, (float)-0.090971, (float)-0.114477, (float)0.059590, (float)0.002494, (float)0.028891, (float)-0.038800, (float)0.135693, (float)-0.110202, (float)-0.138374, (float)0.045844, (float)-0.155150, (float)-0.144207, (float)0.039274, (float)0.101782, (float)0.009706, (float)-0.095646, (float)-0.028917, (float)-0.056922, (float)0.053323, (float)0.045331, (float)-0.045954, (float)-0.056689, (float)-0.101698, (float)0.058535, (float)0.133926, (float)-0.153525, (float)0.119299, (float)-0.058299, (float)-0.101522, (float)-0.101725, (float)-0.150122, (float)-0.145766, (float)0.130610, (float)0.059164, (float)0.003232, (float)0.098596, (float)-0.036254, (float)0.162603, (float)0.159218, (float)-0.002393, (float)0.062512, (float)-0.052506, (float)-0.142490, (float)0.025706, (float)0.164980, (float)0.144072, (float)-0.151740, (float)0.009610, (float)-0.160249, (float)-0.083293, (float)-0.022607, (float)-0.007723, (float)-0.100509, (float)-0.056415, (float)0.035225, (float)-0.166490, (float)-0.158227, (float)-0.023745, (float)0.110556, (float)0.035441, (float)-0.081187, (float)0.114308, (float)-0.021996, (float)0.080784, (float)0.071244, (float)0.056489, (float)0.043271, (float)-0.045833, (float)-0.073286, (float)-0.052361, (float)0.113185, (float)0.058244, (float)-0.164662};	// L55
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


