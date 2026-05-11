handle_sigint() {
    echo "Ctrl+C signal"
    exit 1
}
trap handle_sigint SIGINT



#!/bin/bash

datasets=()
trainsets=()
partsets=()
part_indices=()
testsets1=()
testsets2=()
sf=()


while (( "$#" )); do
  case "$1" in
    --datasets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        datasets+=("$1")
        shift
      done
      ;;
    --trainsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        trainsets+=("$1")
        shift
      done
      ;;
    --partsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        partsets+=("$1")
        shift
      done
      ;;
    --part_indices)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        part_indices+=("$1")
        shift
      done
      ;;
    --testsets1)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        testsets1+=("$1")
        shift
      done
      ;;
    --testsets2)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        testsets2+=("$1")
        shift
      done
      ;;
    --sf)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        sf+=("$1")
        shift
      done
      ;;
    --)
      shift
      break
      ;;
    -*|--*=)
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

# 在这里，你可以使用$datasets和$testsets数组
echo "Datasets: ${datasets[@]}"
echo "Trainsets: ${trainsets[@]}"
echo "Partsets: ${partsets[@]}"
echo "Testsets1: ${testsets1[@]}"
echo "Testsets2: ${testsets2[@]}"
echo "Part Indices: ${part_indices[@]}"
echo "SF: ${sf[@]}"


seeds=(1000)
for sf in "${sf[@]}"; do
  for part_index in "${part_indices[@]}"; do
    for seed in "${seeds[@]}"; do
      for dataset in "${datasets[@]}"; do
        for trainset in "${trainsets[@]}"; do
          for partset in "${partsets[@]}"; do
            for testset1 in "${testsets1[@]}"; do
              for testset2 in "${testsets2[@]}"; do
                output_root="../../devign_storage/$dataset"
                if [ ! -d "$output_root" ]; then
                  mkdir -p "$output_root"
                fi
                processed_train_path=../../devign_storage/shard/$trainset
                processed_part_path=../../devign_storage/shard/$partset
                processed_test1_path=../../devign_storage/shard/$testset1
                processed_test2_path=../../devign_storage/shard/$testset2
                trains=$(find ../../devign_storage/shard/$trainset -type f -name "*shard*")
                parts=$(find ../../devign_storage/shard/$partset -type f -name "*shard*")
                tests1=$(find ../../devign_storage/shard/$testset1 -type f -name "*shard*")
                tests2=$(find ../../devign_storage/shard/$testset2 -type f -name "*shard*")
                if [ "$partset" == "none" ]; then
                  exec python -u ../code/main.py \
                  --mode train \
                  --dataset_root ../../devign_storage \
                  --train_mode train_epoch \
                  --dataset $dataset \
                  --seed "$seed" \
                  --sf "$sf" \
                  --model_type devign \
                  --train_src $trains \
                  --test1_src $tests1 \
                  --test2_src $tests2 \
                  --processed_train_path $processed_train_path \
                  --processed_test1_path $processed_test1_path \
                  --processed_test2_path $processed_test2_path \
                  2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${seed}_${sf}.log"
                else
                  exec python -u ../code/main.py \
                  --mode train \
                  --dataset_root ../../devign_storage \
                  --train_mode train_epoch \
                  --dataset $dataset \
                  --seed "$seed" \
                  --sf "$sf" \
                  --model_type devign \
                  --train_src $trains \
                  --part_src $parts \
                  --part_indices $part_index \
                  --test1_src $tests1 \
                  --test2_src $tests2 \
                  --processed_train_path $processed_train_path \
                  --processed_part_path $processed_part_path \
                  --processed_test1_path $processed_test1_path \
                  --processed_test2_path $processed_test2_path \
                  2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${seed}_${sf}.log"
                fi
              done
            done
          done
        done
      done
    done
  done
done


