cd discoal
make discoal

cd ../msmodified
gcc -o ms ms.c streec.c rand1.c -lm

cd ../msdir
gcc -O3 -o ms ms.c streec.c rand2.c -lm

cd ../SLiM
mkdir build
cd build
cmake ..
make all

cd ../../relate/build
cmake ..
make

