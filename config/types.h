#pragma once

/*
 * Public HLS data types for the LeNet accelerator.
 *
 * MEMBER 3 V1:
 *   Default float preserves existing functional C Simulation.
 *   Define LENET_USE_FIXED for the verified Python fixed-point configuration.
 *   Generated constants live in quant_params.h; rules in docs/member3_quantization.md.
 *
 * IMPORTANT:
 *   When Member 3 freezes the fixed-point rules, update the typedefs in THIS
 *   file only. Individual HLS modules must not define their own data types.
 *
 *   AP_RND_CONV (nearest/ties-even), AP_SAT (signed saturation).
 *   Keep the full product in acc_t. Cast once at the end of each dot product.
 *   HLS tool verification remains a separate integration step.
 */

// Enable with -DLENET_USE_FIXED after agreeing with Member 1.
// Default remains float so existing Float C Simulation is not silently changed.
#ifdef LENET_USE_FIXED
#include <ap_fixed.h>
#include "quant_params.h"
typedef ap_fixed<LENET_DATA_W, LENET_DATA_I, AP_RND_CONV, AP_SAT> data_t;
typedef ap_fixed<LENET_WEIGHT_W, LENET_WEIGHT_I, AP_RND_CONV, AP_SAT> weight_t;
typedef ap_fixed<LENET_ACC_W, LENET_ACC_I, AP_RND_CONV, AP_SAT> acc_t;
typedef data_t output_t;
// Products are NOT narrowed to data_t before addition.
typedef ap_fixed<LENET_DATA_W + LENET_WEIGHT_W,
                 LENET_DATA_I + LENET_WEIGHT_I, AP_RND_CONV, AP_SAT> product_t;
#else
// Input / intermediate feature-map element type.
typedef float data_t;

// Neural-network weight type.
typedef float weight_t;

// MAC accumulation type.
// This will usually need a wider fixed-point format than data_t/weight_t.
typedef float acc_t;

// Final network output (logits) type.
typedef float output_t;
#endif
