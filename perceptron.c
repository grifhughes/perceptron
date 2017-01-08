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
    float gerr = 0.0f;
    do {
        ++iterations;
        gerr = 0;
        for(int i = 0; i < D; ++i) {
            int output = classify(sigmoid(p, tp[i]));
            float error = tp[i]->c - output;
            if(error != 0) {
                float cached = p->lrate * error;
                p->weights[0] += cached;
                for(int j = 1; j < p->inputs; ++j) 
                    p->weights[j] += cached * tp[i]->values[j];
                gerr += error * error;
            }
        }
    } while(gerr != 0 && iterations <= k);
}
