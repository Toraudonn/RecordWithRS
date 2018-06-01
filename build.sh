#!/bin/sh

# build script
echo rm -rf $HOME/dev/RecordWithRS/build
mkdir $Home/dev/RecordWithRS/build
cd $Home/dev/RecordWithRS/build

# CMAKE
cmake ..
make
