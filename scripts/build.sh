#!/bin/sh

# build script
# might change directory 
rm -rf $HOME/dev/RecordWithRS/build
mkdir $HOME/dev/RecordWithRS/build
cd $HOME/dev/RecordWithRS/build

# CMAKE
cmake ..
make

echo "$ ./build/rs-capture"