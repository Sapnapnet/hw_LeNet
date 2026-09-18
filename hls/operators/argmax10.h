#ifndef MEMBER5_ARGMAX10_H
#define MEMBER5_ARGMAX10_H

/*
 * Argmax over the NUM_CLASSES logits; on ties the smallest index wins
 * (member3 frozen rule: strict '>' keeps the first maximum).
 * Post-processing only; the top level still outputs logits.
 */

#include "../../config/network_config.h"
#include "../../config/types.h"

inline int argmax10(const data_t logits[NUM_CLASSES]) {
#pragma HLS INLINE
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

#endif // MEMBER5_ARGMAX10_H
