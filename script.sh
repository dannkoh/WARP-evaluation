# Check src/evaluator.py for all the arguments you can pass
python3 src/evaluator.py \
evaluation.dataset=dannkoh/WARP-benchmark \
model.model_name=dannkoh/warp-1.0 \
model.sampling.max_tokens=32768 \
evaluation.batch_size=8 
