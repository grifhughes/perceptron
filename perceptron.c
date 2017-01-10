#include "perceptron.h"

struct perceptron *
build_perceptron(int i, float l)
{
    struct perceptron *tmp = malloc(sizeof(struct perceptron));
    tmp->inputs = i + 1;
    tmp->lrate = l;
    tmp->weights = malloc(sizeof(float) * tmp->inputs);
    for(int i = 0; i < tmp->inputs; ++i)
        tmp->weights[i] = (float)rand()/(float)RAND_MAX;
    return tmp;
}

void
destroy_perceptron(struct perceptron *p) 
{
    free(p->weights);
    free(p);
}

float 
dot_p(float *a, float *b, int l)
{
    float r = 0.0f;
    for(int i = 0; i < l; ++i)
        r += a[i] * b[i]; 
    return r;
}

int 
classify(struct perceptron *p, struct tset_pair *tp)
{
    return dot_p(p->weights, tp->values, p->inputs) > 0 ? 1 : 0;
}

void 
update_weights(struct perceptron *p, struct tset_pair *tp, float scalar)
{
    p->weights[0] += scalar;
    for(int i = 1; i < p->inputs; ++i)
        p->weights[i] += scalar * tp->values[i];
}

void 
train(struct perceptron *p, struct tset_pair **tp, int D)
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

void 
learn(struct perceptron *p, struct tset_pair **tp, int D, float k)
{
    for(int i = 0; i < k; ++i)
        train(p, tp, D);
}
