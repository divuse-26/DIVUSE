


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
              exec python ./sastvd/scripts/train.py \
                --dataset $dataset \
                --trainset $trainset \
                --partset $partset \
                --testset $testset \
                --seed $seed \
                --under $under \
                --selection $selection \
                2>&1 | tee "$output_root/${dataset}_${seed}.log"
            done
          done
        done
      done
    done
  done
done