#!/usr/bin/env bash

for f in */; do
    cd $f
    git pull
    cd ..
done
find . -name "__init__.py" -delete