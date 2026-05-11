handle_sigint() {
    echo "Ctrl+C signal"
    exit 1
}
trap handle_sigint SIGINT


datasets=()
trainsets=()
partsets=()
testsets=()
seeds=()
under=()
selection=()


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
    --testsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        testsets+=("$1")
        shift
      done
      ;;
    --seeds)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        seeds+=("$1")
        shift
      done
      ;;
    --under)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        under+=("$1")
        shift
      done
      ;;
    --selection)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        selection+=("$1")
        shift
      done
      ;;
    --) # 结束参数处理
      shift
      break
      ;;
    -*|--*=) # 不支持的参数
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

echo "Datasets: ${datasets[@]}"
echo "Trainsets: ${trainsets[@]}"
echo "Partsets: ${partsets[@]}"
echo "Testsets: ${testsets[@]}"
echo "Seeds: ${seeds[@]}"
echo "Under: ${under[@]}"
echo "Selection: ${selection[@]}"


for dataset in "${datasets[@]}"; do
  for trainset in "${trainsets[@]}"; do
    for partset in "${partsets[@]}"; do
      for testset in "${testsets[@]}"; do
        for seed in "${seeds[@]}"; do
          for under in "${under[@]}"; do
            for selection in "${selection[@]}"; do
              output_root="./storage/checkpoint/$dataset"
              if [ ! -d "$output_root" ]; then
                mkdir -p "$output_root"
              fi
              python run_velvet.py \
                --model_name=velvet_model.bin \
                --do_train \
                --do_test \
                --output_dir=$output_root \
                --train_data_file=./storage/${trainset}.csv \
                --part_data_file=./storage/${partset}.csv \
                --test_data_file=./storage/${testset}.csv \
                --joern_output_dir=/home/?/storage/processed \
                --under $under \
                --selection $selection \
                --epochs 10 \
                --encoder_block_size 512 \
                --train_batch_size 64 \
                --eval_batch_size 64 \
                --learning_rate 5e-5 \
                --max_grad_norm 1.0 \
                --evaluate_during_training \
                --seed $seed  2>&1 | tee "$output_root/${dataset}_${seed}.log"
            done
          done
        done
      done
    done
  done
done