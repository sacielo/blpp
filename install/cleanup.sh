#!/bin/bash
rm pkgs -rf

for i in $(git submodule | awk '{print $2}')
do
 cd $i
 rm build -rf
 git stash
 git clean -fd
done
