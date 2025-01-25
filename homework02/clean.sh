#!/bin/sh
find . -type f -name 'HW*.ipynb' | while read file
do
  jupyter nbconvert --clear-output --inplace "$file"
  git add "$file"
done