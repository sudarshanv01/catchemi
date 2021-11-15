#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arb.h"

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