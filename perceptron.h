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

/* seed rng before calling */
struct perceptron *
build_perceptron(int i, float l);

void
destroy_perceptron(struct perceptron *p); 

float 
dot_p(float *a, float *b, int l);

int 
classify(struct perceptron *p, struct tset_pair *tp);

void 
update_weights(struct perceptron *p, struct tset_pair *tp, float scalar);

void 
train(struct perceptron *p, struct tset_pair **tp, int D);

void 
learn(struct perceptron *p, struct tset_pair **tp, int D, float k);

#endif
