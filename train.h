#ifndef TRAIN_H
#define TRAIN_H

#include <stdlib.h>

/* input vector & classification 
 * stores 1 at values[0] to allow perceptron weights[0] to store bias */
struct tset_pair {
    float *values;
    int c, n;
};

struct tset_pair *
build_tset_pair(float *v, int c, int n);

void
destroy_tset_pair(struct tset_pair *tp);

#endif
