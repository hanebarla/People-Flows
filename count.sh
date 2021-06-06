for sc in `seq 0 19`
do
    echo $sc
    scene=$[$sc / 4]
    echo "Scene ${scene}"
    if [$sc % 4 = 0]; then
        python countpeople.py TrainData_Path.csv Scene_IM0${scene}.csv --dataset CrowdFlow --load_model CrowdFlow_weight/checkpoint_0.pth.tar
    elif [$sc % 4 = 1]; then
        python countpeople.py TrainData_Path.csv Scene_IM0${scene}.csv --dataset CrowdFlow --load_model CrowdFlow_weight/checkpoint_0.1.pth.tar
    elif [$sc % 4 = 2]; then
        python countpeople.py TrainData_Path.csv Scene_IM0${scene}_hDyn.csv --dataset CrowdFlow --load_model CrowdFlow_weight/checkpoint_0.pth.tar
    else
        python countpeople.py TrainData_Path.csv Scene_IM0${scene}_hDyn.csv --dataset CrowdFlow --load_model CrowdFlow_weight/checkpoint_0.1.pth.tar
    fi
done