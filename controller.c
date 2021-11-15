#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arb.h>
#include "quantities_newns_anderson.h"
#include "density_of_states.h"

int main()
{
    float Vak=1.0; 
    float eps=-2.0;
    float eps_a=0.0;
    float eps_d=0.0;
    float wd=1.0;
    float Delta0 = 1.0;
    float Delta;
    float Lambda;
    float eps_diff;
    float dos;

    Delta = get_Delta_semiellipse(Vak, eps, eps_d, wd);
    printf("Delta = %f\n", Delta);
    Lambda = get_Lambda_semiellipse(Vak, eps, eps_d, wd);
    printf("Lambda = %f\n", Lambda);
    eps_diff = get_eps_difference(eps, eps_a);
    printf("eps_diff = %f\n", eps_diff);
    dos = density_of_states_semiellipse(Delta, Lambda, eps_diff, Delta0);
    printf("dos = %f\n", dos);
}