#include <acb.h>
void get_Delta_semiellipse(acb_t Delta, double Vak_f, double eps_f, double eps_d_f, double wd_f, long prec);
void get_Lambda_semiellipse(acb_t Lambda, double Vak_f, double eps_f, double eps_d_f, double wd_f, long prec);
void get_energy_difference(acb_t energy_difference, double eps_f, double eps_a_f, int prec);