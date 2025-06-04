import argparse
import json
from glob import glob

def collect_test_results(directory):
    result = {'test_acc_top1': {}, 'test_acc_top5': {}}
    import glob

    files = glob.glob(f"{directory}/**/test_result.json", recursive=True) + \
            glob.glob(f"{directory}/**/lin_ft_linear/test_result.json", recursive=True)
    for json_file in files:
        model_name = json_file.split('/')[-2] if 'lin_ft_linear' not in json_file else json_file.split('/')[-3]
        with open(json_file, 'r') as f:
            data = json.load(f)[0]
            result['test_acc_top1'][model_name] = data.get('test_acc_top1')
            result['test_acc_top5'][model_name] = data.get('test_acc_top5')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/results')
    parser.add_argument('--output', default='./data/all_test_results.json')
    args = parser.parse_args()

    aggregated = collect_test_results(args.dir)
    with open(args.output, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"saved to {args.output}")

if __name__ == '__main__':
    main()