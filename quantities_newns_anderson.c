#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arb.h"

// Function declarations
float get_Delta_semiellipse(float Vak, float eps, float eps_d, float w_d);
float get_Lambda_semiellipse(float Vak, float eps, float eps_d, float w_d);
float get_eps_difference(float eps, float eps_a);

// Define all the quantities needed for the Newns-Anderson
// method. These include Delta, Lambda (Hilbert transform of Delta)
// and the energy difference between eps - eps_a

int main()
{
    float Vak=1.0; 
    float eps=-2.0;
    float eps_a=0.0;
    float eps_d=0.0;
    float wd=1.0;
    float Delta;
    float Lambda;
    float eps_diff;

    Delta = get_Delta_semiellipse(Vak, eps, eps_d, wd);
    printf("Delta = %f\n", Delta);
    Lambda = get_Lambda_semiellipse(Vak, eps, eps_d, wd);
    printf("Lambda = %f\n", Lambda);
    eps_diff = get_eps_difference(eps, eps_a);
    printf("eps_diff = %f\n", eps_diff);
}

// Define the function for Delta
float get_Delta_semiellipse(float Vak, float eps, float eps_d, float wd)
{
    float Delta;
    float norm_eps;

    // Normalised epsilon keeps track of Delta to make sure
    // that only the Delta within the eps_d +- wd region is
    // non-zero, everything else is zero.
    norm_eps = (eps - eps_d) / wd;

    if (norm_eps > 1.0)
    {
        Delta = 0.0;
    }
    else if (norm_eps < -1.0)
    {
        Delta = 0.0;
    }
    else
    {
        Delta = pow(Vak, 2) * pow( 1 - pow( norm_eps, 2), 0.5 ) ; 
        Delta /= wd / 2 ;
    }

    return Delta;
}

// Define the function for Lambda
float get_Lambda_semiellipse(float Vak, float eps, float eps_d, float wd)
{
    float Lambda;
    float norm_eps;

    // Normalised epsilon keeps track of the position
    // of Delta such that the analytical version of 
    // Lambda is altered based on the energy supplied

    norm_eps = (eps - eps_d) / wd;

    if (norm_eps > 1.0)
    {
        Lambda = M_PI * pow(Vak, 2) * ( norm_eps - pow( pow( norm_eps, 2) - 1, 0.5 ) );
    }
    else if (norm_eps < -1.0)
    {
        Lambda = M_PI * pow(Vak, 2) * ( norm_eps + pow( pow( norm_eps, 2) - 1, 0.5 ) );
    }
    else
    {
        Lambda = M_PI * pow(Vak, 2) * norm_eps;
    }

    return Lambda;
}

float get_eps_difference(float eps, float eps_a)
{
    float eps_diff;

    eps_diff = eps - eps_a;

    return eps_diff;
}