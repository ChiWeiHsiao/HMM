#!/bin/zsh
#Tune num_train_seq from 5 to 35
#Output file name is : "result-predict-5.txt"
msg="tune param is: "
param=5
while [ "$param" != "35" ]
do
  param=$(($param+1))
  echo "$msg $param \n" > "result-diff-version-""$param"".txt"
  python3 guess.py "$param">> "result-diff-version-""$param"".txt"
done
echo "Finish"
