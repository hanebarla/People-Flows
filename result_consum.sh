#!/bin/bash

bash get_mae.sh $1 TrainMAE.csv Train MAE
bash get_mae.sh $1 Trainloss.csv Train Loss
bash get_mae.sh $1 ValMAE.csv Val MAE
bash get_mae.sh $1 Valloss.csv Val Loss

mvfold=${1%log.txt}
echo $mvfold

mv TrainMAE.csv $mvfold
mv Trainloss.csv $mvfold
mv ValMAE.csv $mvfold
mv Valloss.csv $mvfold

echo "move csv files"

title=${mvfold##*CrowdFlow/}
echo $title
tmae=${mvfold}TrainMAE.csv
tloss=${mvfold}Trainloss.csv
vmae=${mvfold}ValMAE.csv
vloss=${mvfold}Valloss.csv

python plot_learningcurv.py -t $title -tmae $tmae -tloss $tloss -vmae $vmae -vloss $vloss
mv lr_curv.png $mvfold