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

// void get_Delta_semiellipse_arb(arb_t Delta, arb_t Vak, arb_t eps, arb_t eps_d, arb_t wd, int prec)
// {
//     arb_t norm_eps; // Normalised epsilon to keep track of where we are in energy
//     arb_t lower_energy_bound; // Lower bound for energy
//     arb_t upper_energy_bound; // Upper bound for energy

//     // Normalised epsilon keeps track of Delta to make sure
//     // that only the Delta within the eps_d +- wd region is
//     // non-zero, everything else is zero.
//     arb_sub(norm_eps, eps, eps_d, prec);
//     arb_div(norm_eps, norm_eps, wd, prec);
//     arb_set_ui(lower_energy_bound, -1);
//     arb_set_ui(upper_energy_bound, 1);

//     // If norm_eps is greater than 1, then Delta is zero
//     // If it is less than -1, then Delta is zero
//     // Otherwise, Delta is the value of the function 
//     // Delta = Vak^2 * ( 1 - norm_eps^2 )^0.5 / wd / 2
//     if (arb_gt(norm_eps, upper_energy_bound))
//     {
//         arb_zero(Delta);
//     }
//     else if (arb_lt(norm_eps, lower_energy_bound)) 
//     {
//         arb_zero(Delta);
//     }
//     else
//     {
//         arb_zero(Delta);
//     }
// }


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