#include "train.h"

struct tset_pair *
build_tset_pair(float *v, int c, int n)
{
    struct tset_pair *tmp = malloc(sizeof(struct tset_pair));
    tmp->n = n + 1;
    tmp->c = c;
    tmp->values = malloc(sizeof(float) * tmp->n);
    tmp->values[0] = 1;
    for(int i = 1; i < tmp->n; ++i)
        tmp->values[i] = v[i];
    return tmp;
}

void
destroy_tset_pair(struct tset_pair *tp)
{
    free(tp->values);
    free(tp);
}
