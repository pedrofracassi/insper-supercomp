FILENAME=$1 # file.cpp
BASENAME=$(basename $FILENAME .cpp)

g++ -fopenmp $FILENAME -o $BASENAME && ./$BASENAME