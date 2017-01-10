#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "train.h"
#include <time.h>
#include <math.h>

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

/* train perceptron for k epochs */
void 
learn(struct perceptron *p, struct tset_pair **tp, int D, float k);

/* computes dot product, compiles to AVX instructions on my Skylake processor */
static inline __attribute__((always_inline)) 
float dot_p(float *a, float *b, int l)
{
    float r = 0.0f;
    for(int i = 0; i < l; ++i)
        r += a[i] * b[i]; 
    return r;
}

/* updates weights if a misclassification is made while training */
static inline __attribute__((always_inline))
void update_weights(struct perceptron *p, struct tset_pair *tp, float scalar)
{
    p->weights[0] += scalar;
    for(int i = 1; i < p->inputs; ++i)
        p->weights[i] += scalar * tp->values[i];
}

/* classifies given test instance */
static inline __attribute__((always_inline)) 
int classify(struct perceptron *p, struct tset_pair *tp)
{
    return dot_p(p->weights, tp->values, p->inputs) > 0 ? 1 : 0;
}   

/* completes 1 epoch of training */
static inline __attribute__((always_inline))
void train(struct perceptron *p, struct tset_pair **tp, int D)
{
    for(int i = 0; i < D; ++i) {
        int output = classify(p, tp[i]);
        int error = tp[i]->c - output;
        if(!error) 
            continue;
        else 
            update_weights(p, tp[i], p->lrate * error);
    }
}

#endif
