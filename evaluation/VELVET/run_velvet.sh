handle_sigint() {
    echo "Ctrl+C signal"
    exit 1
}
trap handle_sigint SIGINT
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

trainsets="bigvul"
testsets="reveal"
seeds=("0")
partsets_list=("vgx" "vulgen" "juliet")
under_list=("0.8" "0.6" "0.4" "0.2")
selection_list=("random")
for seed in "${seeds[@]}"; do
  for partsets in "${partsets_list[@]}"; do
    for under in "${under_list[@]}"; do
      for selection in "${selection_list[@]}"; do
        datasets="${trainsets}_${partsets}_under${under}_${selection}"
        bash run.sh --trainsets $trainsets --partsets $partsets --testsets $testsets --seeds $seed --under $under --selection $selection --datasets $datasets
      done
    done
  done
done