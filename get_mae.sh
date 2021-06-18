#!/bin/bash

path=$1

function getmae() {
  TARGET_DOMAINS=($(grep "MAE" < $1 | awk 'BEGIN { OFS="," } {print $2,$3}')) > test.txt

  # 配列の各要素に対して処理を実行する
  for i in ${TARGET_DOMAINS[@]}
  do
    echo $i >> $2
  done
}

getmae $path $2

exit 0