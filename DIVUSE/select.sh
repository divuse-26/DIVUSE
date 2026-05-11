handle_sigint() {
    echo "Ctrl+C signal"
    exit 1
}
trap handle_sigint SIGINT
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"


trainsets="reveal"
testsets2="reveal"
handlesets="123470"
over="moderate"
partsets="juliet"
cd ./scripts
under_list=("0.6")

for under in "${under_list[@]}"; do
  datasets="${trainsets}_${partsets}_${handlesets}_under${under}_over_${over}"
  bash run.sh --datasets $datasets --trainsets $trainsets --validsets "none" --partsets $partsets  --ratio "1.0" --handlesets $handlesets --under $under --over $over
done
