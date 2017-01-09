#include "perceptron.h"

struct perceptron *
build_perceptron(int i, float l)
{
    srand(time(NULL));
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

void 
learn(struct perceptron *p, struct tset_pair **tp, int D, float k)
{
    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < D; ++j) {
            int output = classify(p, tp[j]);
            float error = tp[j]->c - output;
            if(error != 0) 
                update_weights(p, tp[j], p->lrate * error);
            else 
                continue;
        }
    }
}
