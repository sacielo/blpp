#!/bin/bash
for i in 1 2 3 4 ; do

	export OPENBLAS_NUM_THREADS=$i

	./sweep25.sh
	./analyze25.py

	mv runs-$(hostname) thr_$i


done
