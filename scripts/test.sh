#!/bin/sh

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -p|--exp)
      EXP="$2"
      shift
      shift
      ;;
    -l|--split)
      SPL="$2"
      shift
      shift
      ;;
    -g|--gpu)
      GPU=$2
      shift # past argument
      shift # past value 
      ;;
    -s|--start)
      START=$2
      shift
      shift
      ;;
    -e|--end)
      END=$2
      shift
      shift
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done


for((i=$START;i<=$END;i+=5000));
do
  echo "python train.py --exp_name $EXP --resume $EXP/model_$i.pth  --device_id $GPU --task.asset.splits $SPL --test_only --log.mode wandb";
  python train.py --exp_name $EXP --resume $EXP/model_$i.pth  --device_id $GPU --task.asset.splits $SPL --test_only --log.mode wandb
done