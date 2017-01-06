#include "perceptron.h"
#include <stdio.h>

#define MAX_ITER 500 
#define LEARN_RATE 0.115
#define NEXAMPLES 130 
#define NFEATURES 13 

int
main(void)
{
    FILE *fp;
    struct tset_pair *training_examples[NEXAMPLES]; 

    fp = fopen("data.txt", "r");
    int classes[NEXAMPLES] = {0};

    /* parse file */
    for(int i = 0; i < NEXAMPLES; ++i) {
        float tmp[NFEATURES];
        fscanf(fp, "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &classes[i], &tmp[0], &tmp[1], &tmp[2], &tmp[3], &tmp[4],
                &tmp[5], &tmp[6], &tmp[7], &tmp[8], &tmp[9], &tmp[10], &tmp[11], &tmp[12]); 
        training_examples[i] = build_tset_pair(tmp, classes[i], NFEATURES);
    }

    struct perceptron *p = build_perceptron(NFEATURES, LEARN_RATE);
    learn(p, training_examples, NEXAMPLES, MAX_ITER);

    float nwrong = 0.0f;
    for(int i = 0; i < NEXAMPLES; ++i) { 
        if(classes[i] != classify(p, training_examples[i]))
            nwrong++;
        destroy_tset_pair(training_examples[i]);
    }
    printf("percent correct: %f\n", 100.0f - ((nwrong / NEXAMPLES) * 100.0f));
    destroy_perceptron(p);
}
