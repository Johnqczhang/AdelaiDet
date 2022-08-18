#!/usr/bin/env zsh


cfg_file="configs/CondInst/mots_R_50.yaml"
base_cmd="OMP_NUM_THREADS=1 python tools/train_net.py --num-gpus 2"

mots_ids=("02" "05" "09" "11")
# condinst_models=("models/condinst_ms_r50_3x.pth" "models/condinst_ms_r50_3x_sem.pth")
condinst_models=("models/CondInst_MS_R_50_3x.pth" "models/CondInst_MS_R_50_3x_sem.pth")

# out_root="train_dir/base/multi_scale"
out_root="train_dir/pxvol/v1.0"


# args: model_weight, train_set, test_set, output_dir
def run_train() {
    # cmd=$base_cmd" MODEL.WEIGHTS $1 DATASETS.TRAIN $2 DATASETS.TEST $3 OUTPUT_DIR $4"
    cmd=$base_cmd" --eval-only MODEL.WEIGHTS $1 DATASETS.TRAIN $2 DATASETS.TEST $3 OUTPUT_DIR train_dir/debug"
    echo $cmd
    eval $cmd
    # eval "rm -rf $4/inference"
    echo "\n"
}


def train_mots() {
    # condinst_models=("models/CondInst_MS_R_50_3x.pth" "models/CondInst_MS_R_50_3x_sem.pth")
    # for model ($condinst_models) {
    model="models/CondInst_MS_R_50_3x.pth"
    for val ($mots_ids) {
        train_set="'(\"mots_train_$val\",)'"
        val_set="'(\"mots_val_$val\",)'"

        # if (( $val == 09 && $model[(I)sem])) {
        #     continue
        # }
        # if (( $val == 05 && $model[(I)sem])) {
        #     continue
        # }

        out_dir="mots$val""_R50_adam_1e-4"
        if [[ $model == *"sem"* ]] {
            out_dir+="_unsem"
        }
        out_dir="$out_root/$out_dir"
        model="$out_dir/model_final.pth"

        # run_train $model $train_set $val_set $out_dir
        run_train $model $train_set $val_set
    }
    # }
}


def run_cmd() {
    cmd=$1
    out_dir=$2

    echo $cmd
    eval $cmd
    eval "rm -rf $out_dir/inference"
    echo "\n"
}


def test_condinst() {
    condinst_models=("models/CondInst_MS_R_50_3x.pth" "models/CondInst_MS_R_50_3x_sem.pth")
    # condinst_models=("models/condinst_ms_r50_3x.pth" "models/condinst_ms_r50_3x_sem.pth")
    out_dir="train_dir/debug"
    dataset=$1
    if [[ "$1" == "mots" ]] {
        cfg_file="configs/CondInst/mots_R_50.yaml"
    } else {
        cfg_file="configs/CondInst/kitti_mots_R_50.yaml"
    }
    opts="MODEL.RESNETS.NORM GN MODEL.RESNETS.STRIDE_IN_1X1 False MODEL.FPN.NORM GN MODEL.CONDINST.MASK_BRANCH.NORM GN"
    cmd1=$base_cmd" --config-file $cfg_file --eval-only OUTPUT_DIR $out_dir $opts"

    for model ($condinst_models) {
        echo "Testing on mots using $model\n"
        # if (( $model[(I)sem] == 0 )) {
        #     continue
        # }
        if [[ "$1" == "mots" ]] {
            for val ($mots_ids) {
                val_set="'(\"mots_val_$val\",)'"
                cmd2=$cmd1" MODEL.WEIGHTS $model DATASETS.TEST $val_set"
                run_cmd $cmd2 $out_dir
            }
        } else {
            cmd2=$cmd1" MODEL.WEIGHTS $model DATASETS.TEST '(\"kitti_mots_val\",)'"
            run_cmd $cmd2 $out_dir
        }   
    }
}


test_mots() {
    models_dir=($1/mots*)
    for model_dir ($models_dir) {
        weight="$model_dir/model_final.pth"
        model_name=${model_dir:t}
        val=$model_name[5,6]
        val_set="'(\"mots_val_$val\",)'"

        run_test $weight $val_set
    }
}



# arg1: 'train' or 'test'
func=$1
if [[ "$1" == "test" ]] {
    if [[ "$2" == "condinst" ]] {
        test_condinst $3
    } elif [[ "$2" == "base" ]] {
        test_mots $3
    }
} else {
    train_mots
}
