#include "perceptron.h"
#include <stdio.h>

#define MAX_ITER 50 
#define LEARN_RATE 0.1
#define NEXAMPLES 85 
#define NFEATURES 4 

int
main(void)
{
    FILE *fp;
    struct tset_pair *training_examples[NEXAMPLES]; 

    fp = fopen("data.txt", "r");

    /* parse file */
    for(int i = 0; i < NEXAMPLES; ++i) {
        float tmp[NFEATURES];
        int class = 0;
        fscanf(fp, "%f,%f,%f,%f,%d", &tmp[0], &tmp[1], &tmp[2], &tmp[3], &class); 
        training_examples[i] = build_tset_pair(tmp, class, NFEATURES);
    }

    struct perceptron *p = build_perceptron(NFEATURES, LEARN_RATE);
    learn(p, training_examples, NEXAMPLES, MAX_ITER);

    for(int i = 0; i < NEXAMPLES; ++i) { 
        printf("class: %s\n", classify(p, training_examples[i]) ? 
                "Iris-setosa" : "Iris-virginica");
        destroy_tset_pair(training_examples[i]);
    }
    destroy_perceptron(p);
}
