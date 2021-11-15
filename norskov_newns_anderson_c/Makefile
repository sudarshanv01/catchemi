ARB_PATH="/home/vijays/Documents/bin/external_packages/arb-2.21.1"
FLINT_PATH="/home/vijays/Documents/bin/external_packages/flint-2.8.3"
CC="gcc"
LIBS="-larb -lflint"

general: quantities_newns_anderson.c density_of_states.c controller.c 
	$(CC) -I$(ARB_PATH) -I$(FLINT_PATH) -L$(ARB_PATH) -L$(FLINT_PATH) \
	quantities_newns_anderson.c density_of_states.c controller.c  -lm -larb -lflint -o controller.out 

clean:
	rm -f *.o *.out 