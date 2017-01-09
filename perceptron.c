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
    int iterations = 0;
    while(iterations <= k) {
        ++iterations;
        for(int i = 0; i < D; ++i) {
            int output = classify(p, tp[i]);
            float error = tp[i]->c - output;
            if(error != 0) {
                float cached = p->lrate * error;
                p->weights[0] += cached;
                /* start at 1 to update weights as bias is w[0] & x[0] is 1 */ 
                for(int j = 1; j < p->inputs; ++j) 
                    p->weights[j] += cached * tp[i]->values[j];
            } else continue;
        }
    }
}
