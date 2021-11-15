#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arb.h>

float density_of_states_semiellipse(float Delta, float Lambda, float eps_diff, float Delta0)
{
    float rho; // density of states

    rho = M_PI * ( Delta + Delta0 ); 
    rho /= ( pow(eps_diff - Lambda, 2) + pow(Delta + Delta0, 2) );

    return rho;
}
