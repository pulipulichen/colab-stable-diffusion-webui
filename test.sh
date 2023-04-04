#!/bin/bash

FILE=/tmp/test.txt
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE not exists."
    touch $FILE
fi