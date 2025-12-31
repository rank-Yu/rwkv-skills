import os
import json
import argparse
import itertools

from src.eval.evaluators.free_response import FreeResponsePipeline
from src.infer.model import ModelLoadConfig
from src.eval.metrics.free_response import evaluate_free_response
from src.infer.sampling import SamplingConfig

def dict_type(arg):
    return json.loads(arg)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV free-form grid search")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--grid-space", required=True, type=dict_type, 
                        help="Grid search parameter space as a dictionary: \n e.g. \'{\"temperature\":[0.3,0.4],\"top_k\":[50],\"top_p\":[0.3,0.4],\"alpha_presence\":[0.0,0.1],\"alpha_frequency\":[0.1],\"alpha_decay\":[0.99]}\'")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--cot-max-tokens", type=int, default=4096, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, default=64, help="Clamp final answer generation length")
    parser.add_argument("--output-path", default=os.path.join(os.getcwd(), 'results'), help="Output path (defaults to ./results)")
    return parser.parse_args()


def get_parameter_grid(param_grid: dict) -> list[dict]:
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def process(args, pipeline, params: dict):
    file_name = f'temp{params["temperature"]}_tp{params["top_p"]}_tk{params["top_k"]}_ap{params["alpha_presence"]}_af{params["alpha_frequency"]}_ad{params["alpha_decay"]}'
    pred_dir = os.path.join('pred', f'{file_name}.jsonl')
    eval_dir = os.path.join('eval', f'{file_name}.jsonl')

    cot_config = SamplingConfig(
        max_generate_tokens=args.cot_max_tokens,
        temperature=params['temperature'],
        top_k=params['top_k'],
        top_p=params['top_p'],
        alpha_presence=params['alpha_presence'],
        alpha_frequency=params['alpha_frequency'],
        alpha_decay=params['alpha_decay'],
        stop_tokens=(0, 261, 24281),
    )

    final_config = SamplingConfig(
        max_generate_tokens=args.final_max_tokens,
        temperature=params['temperature'],
        top_k=params['top_k'],
        top_p=params['top_p'],
        alpha_presence=params['alpha_presence'],
        alpha_frequency=params['alpha_frequency'],
        alpha_decay=params['alpha_decay'],
        stop_tokens=(0, 2402, 4910),
    )

    pipeline.run(
        dataset_path=args.dataset,
        output_path=pred_dir,
        cot_sampling=cot_config,
        final_sampling=final_config,
        batch_size=args.batch_size
        )
    
    results = evaluate_free_response(completions_path=pred_dir,
                                     dataset_path=args.dataset,
                                     eval_output_path=eval_dir)
    return results.exact_accuracy


def main():
    args = parse_args()
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    pipeline = FreeResponsePipeline(ModelLoadConfig(weights_path=args.model_path, device=args.device))
    params_grid = get_parameter_grid(args.grid_space)

    with open('GridSearch_results.txt', 'w') as f:
        for params in params_grid:
            accuracy = process(args, pipeline, params)
            f.write(f'Params: {params}, Accuracy: {accuracy}\n')
            f.flush()
    
    pass

if __name__ == "__main__":
    main()