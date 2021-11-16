#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arb.h>
#include "acb.h"
#include "quantities_newns_anderson.h"
#include "density_of_states.h"

int main()
{
    double Vak = 1.0; 
    double eps = -2.0;
    double eps_a = 0.0;
    double eps_d = -2.0;
    double wd = 1.0;
    double Delta0 = 1.0;
    long precision = 20;

    acb_t dos;
    acb_init(dos);

    // get the density of states
    get_density_of_states(dos, Vak, eps, eps_d, wd, eps_a, Delta0, precision);
    printf("dos = ");
    acb_printd(dos, precision);
    printf("\n");

    return 0;
}
    