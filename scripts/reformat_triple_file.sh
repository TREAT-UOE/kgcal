#! /bin/bash

mv $1.txt $1.txt.old
awk '{print $1 "\t" $3 "\t" $2}' $1.txt.old > $1.txt