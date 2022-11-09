#!/bin/bash

mkdir stereo_trainset/crestereo -p
cd stereo_trainset/crestereo


# for dir in tree shapenet reflective hole
for dir in hole
do
  mkdir $dir && cd $dir
  for i in $(seq 0 9)
  do
    echo $dir: $(expr $i + 1) / 10
    wget https://data.megengine.org.cn/research/crestereo/dataset/$dir/$i.tar
    tar -xvf $i.tar
    rm $i.tar
  done
  cd ..
done

