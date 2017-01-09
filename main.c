#include "perceptron.h"
#include <stdio.h>

#define MAX_ITER 100 
#define LEARN_RATE 0.125
#define NEXAMPLES 100 
#define NFEATURES 4 

int
main(void)
{
    FILE *fp;
    struct tset_pair *training_examples[NEXAMPLES]; 
    int classes[NEXAMPLES] = {0};

    fp = fopen("data.txt", "r");

    /* parse file */
    for(int i = 0; i < NEXAMPLES; ++i) {
        float tmp[NFEATURES];
        fscanf(fp, "%f,%f,%f,%f,%d", &tmp[0], &tmp[1], &tmp[2], &tmp[3], &classes[i]);
        training_examples[i] = build_tset_pair(tmp, classes[i], NFEATURES);
    }

    struct perceptron *p = build_perceptron(NFEATURES, LEARN_RATE);
    learn(p, training_examples, NEXAMPLES, MAX_ITER);

    float nwrong = 0.0f;
    for(int i = 0; i < NEXAMPLES; ++i) { 
        if(classes[i] != classify(p, training_examples[i]))
            ++nwrong;
        destroy_tset_pair(training_examples[i]);
    }
    printf("percent correct: %f\n", 100.0f - ((nwrong / NEXAMPLES) * 100.0f));
    destroy_perceptron(p);
}
