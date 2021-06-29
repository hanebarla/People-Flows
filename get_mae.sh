#!/bin/bash

path=$1

function getmae() {
  TARGET_DOMAINS=($(grep "$3 $4" < $1 | awk 'BEGIN { OFS="," } {print $3,$4}'))

  # 配列の各要素に対して処理を実行する
  for i in ${TARGET_DOMAINS[@]}
  do
    echo $i >> $2
  done
}

getmae $path $2 $3 $4

exit 0