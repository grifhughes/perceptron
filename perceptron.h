#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "train.h"
#include <time.h>

/* single layer perceptron, weights[0] stores bias */
struct perceptron {
    float *weights;
    float lrate;
    int inputs;
};

/* construct perceptron with i inputs & l learning rate*/
struct perceptron *
build_perceptron(int i, float l);

/* cleanup */
void
destroy_perceptron(struct perceptron *p); 

/* train perceptron */
void
learn(struct perceptron *p, struct tset_pair **tp, int D, float k);

static inline __attribute__((always_inline)) 
float dot_p(float *a, float *b, int l)
{
    float r = 0.0f;
    for(int i = 0; i < l; ++i)
        r += a[i] * b[i]; 
    return r;
}

static inline __attribute__((always_inline))
int classify(struct perceptron *p, struct tset_pair *tp)
{
    return dot_p(p->weights, tp->values, p->inputs) >= 0 ? 1 : 0;
}

#endif
