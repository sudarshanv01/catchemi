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
    int precision = 100;

    arb_t Vak_arb, eps_arb, eps_d_arb, wd_arb, Delta_arb;
    // Set arb quantities
//     arb_set_d(Vak_arb, Vak);
//     arb_set_d(eps_arb, eps);
//     arb_set_d(eps_d_arb, eps_d);
//     arb_set_d(wd_arb, wd);
    // Output quantities for arb

    Delta = get_Delta_semiellipse(Vak, eps, eps_d, wd);
    printf("Delta = %f\n", Delta);
//     get_Delta_semiellipse_arb(Delta_arb, Vak_arb, eps_arb, eps_d_arb, wd_arb, precision);
//     printf("Delta (arb) = %.*f\n", Delta);

    return 0;
}
    