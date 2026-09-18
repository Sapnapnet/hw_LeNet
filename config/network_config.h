#pragma once

/*
 * LeNet5-MNIST network configuration
 *
 * This file is the single source of truth for all frozen network dimensions
 * used by the HLS implementation. Do not redefine these values locally in
 * individual modules.
 */

// -----------------------------------------------------------------------------
// Global
// -----------------------------------------------------------------------------

static const int NUM_CLASSES = 10;

// -----------------------------------------------------------------------------
// Input
// -----------------------------------------------------------------------------

static const int INPUT_C = 1;
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W;   // 784

// -----------------------------------------------------------------------------
// Conv1: [1, 28, 28] -> [6, 24, 24]
// -----------------------------------------------------------------------------

static const int CONV1_CIN = 1;
#if defined(LENET_AGGRESSIVE5)
static const int CONV1_COUT = 5;
#else
static const int CONV1_COUT = 6;
#endif
static const int CONV1_KH = 5;
static const int CONV1_KW = 5;
static const int CONV1_STRIDE = 1;
static const int CONV1_PADDING = 0;

static const int CONV1_IN_H = 28;
static const int CONV1_IN_W = 28;
static const int CONV1_OUT_H = 24;
static const int CONV1_OUT_W = 24;

static const int CONV1_IN_SIZE =
    CONV1_CIN * CONV1_IN_H * CONV1_IN_W;

static const int CONV1_OUT_SIZE =
    CONV1_COUT * CONV1_OUT_H * CONV1_OUT_W;

static const int CONV1_WEIGHT_COUNT =
    CONV1_COUT * CONV1_CIN * CONV1_KH * CONV1_KW;            // 150

// -----------------------------------------------------------------------------
// Pool1: [6, 24, 24] -> [6, 12, 12]
// -----------------------------------------------------------------------------

#if defined(LENET_AGGRESSIVE5)
static const int POOL1_C = 5;
#else
static const int POOL1_C = 6;
#endif
static const int POOL1_KH = 2;
static const int POOL1_KW = 2;
static const int POOL1_STRIDE = 2;

static const int POOL1_IN_H = 24;
static const int POOL1_IN_W = 24;
static const int POOL1_OUT_H = 12;
static const int POOL1_OUT_W = 12;

static const int POOL1_IN_SIZE =
    POOL1_C * POOL1_IN_H * POOL1_IN_W;

static const int POOL1_OUT_SIZE =
    POOL1_C * POOL1_OUT_H * POOL1_OUT_W;

// -----------------------------------------------------------------------------
// Conv2: [6, 12, 12] -> [16, 8, 8]
// -----------------------------------------------------------------------------

 #if defined(LENET_AGGRESSIVE5)
static const int CONV2_CIN = 5;
#else
static const int CONV2_CIN = 6;
#endif
#if defined(LENET_AGGRESSIVE5) || defined(LENET_AGGRESSIVE12)
static const int CONV2_COUT = 12;
#elif defined(LENET_AGGRESSIVE)
// Aggressive latency experiment: two low-impact Conv2 channels are removed
// and the following FC input is compressed from 256 to 224 values.
static const int CONV2_COUT = 14;
#else
static const int CONV2_COUT = 16;
#endif
static const int CONV2_KH = 5;
static const int CONV2_KW = 5;
static const int CONV2_STRIDE = 1;
static const int CONV2_PADDING = 0;

static const int CONV2_IN_H = 12;
static const int CONV2_IN_W = 12;
static const int CONV2_OUT_H = 8;
static const int CONV2_OUT_W = 8;

static const int CONV2_IN_SIZE =
    CONV2_CIN * CONV2_IN_H * CONV2_IN_W;

static const int CONV2_OUT_SIZE =
    CONV2_COUT * CONV2_OUT_H * CONV2_OUT_W;

static const int CONV2_WEIGHT_COUNT =
    CONV2_COUT * CONV2_CIN * CONV2_KH * CONV2_KW;            // 2400

// -----------------------------------------------------------------------------
// Pool2: [16, 8, 8] -> [16, 4, 4]
// -----------------------------------------------------------------------------

#if defined(LENET_AGGRESSIVE5)
static const int POOL2_C = 12;
#elif defined(LENET_AGGRESSIVE12)
static const int POOL2_C = 12;
#elif defined(LENET_AGGRESSIVE)
static const int POOL2_C = 14;
#else
static const int POOL2_C = 16;
#endif
static const int POOL2_KH = 2;
static const int POOL2_KW = 2;
static const int POOL2_STRIDE = 2;

static const int POOL2_IN_H = 8;
static const int POOL2_IN_W = 8;
static const int POOL2_OUT_H = 4;
static const int POOL2_OUT_W = 4;

static const int POOL2_IN_SIZE =
    POOL2_C * POOL2_IN_H * POOL2_IN_W;

static const int POOL2_OUT_SIZE =
    POOL2_C * POOL2_OUT_H * POOL2_OUT_W;

// -----------------------------------------------------------------------------
// Flatten
// -----------------------------------------------------------------------------

static const int FLATTEN_C = POOL2_C;
static const int FLATTEN_H = 4;
static const int FLATTEN_W = 4;
static const int FLATTEN_SIZE =
    FLATTEN_C * FLATTEN_H * FLATTEN_W;

// -----------------------------------------------------------------------------
// FC1: 256 -> 120
// -----------------------------------------------------------------------------

 #if defined(LENET_AGGRESSIVE5)
static const int FC1_IN = 192;
#elif defined(LENET_AGGRESSIVE12)
static const int FC1_IN = 192;
#elif defined(LENET_AGGRESSIVE)
static const int FC1_IN = 224;
#else
static const int FC1_IN = 256;
#endif
static const int FC1_OUT = 120;
static const int FC1_WEIGHT_COUNT = FC1_OUT * FC1_IN;          // 30720

// -----------------------------------------------------------------------------
// FC2: 120 -> 84
// -----------------------------------------------------------------------------

static const int FC2_IN = 120;
static const int FC2_OUT = 84;
static const int FC2_WEIGHT_COUNT = FC2_OUT * FC2_IN;          // 10080

// -----------------------------------------------------------------------------
// FC3: 84 -> 10
// -----------------------------------------------------------------------------

static const int FC3_IN = 84;
static const int FC3_OUT = NUM_CLASSES;
static const int FC3_WEIGHT_COUNT = FC3_OUT * FC3_IN;          // 840

// -----------------------------------------------------------------------------
// Total weight count
// Bias is disabled in all Conv/FC layers.
// -----------------------------------------------------------------------------

static const int TOTAL_WEIGHT_COUNT =
    CONV1_WEIGHT_COUNT +
    CONV2_WEIGHT_COUNT +
    FC1_WEIGHT_COUNT +
    FC2_WEIGHT_COUNT +
    FC3_WEIGHT_COUNT;                                         // 44190

// -----------------------------------------------------------------------------
// Layout helpers
// -----------------------------------------------------------------------------

// Feature map layout: CHW
// index = c * H * W + h * W + w
static inline int chw_index(int c, int h, int w, int H, int W)
{
    return c * H * W + h * W + w;
}

// Conv weight layout: [OUT_CHANNEL][IN_CHANNEL][KH][KW]
static inline int conv_weight_index(
    int oc, int ic, int kh, int kw,
    int CIN, int KH, int KW)
{
    return ((oc * CIN + ic) * KH + kh) * KW + kw;
}

// FC weight layout: [OUT][IN]
static inline int fc_weight_index(int out, int in, int IN)
{
    return out * IN + in;
}
