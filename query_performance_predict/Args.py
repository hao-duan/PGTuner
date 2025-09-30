'''
experimental modes:main, dataset_change, ds_change, and qd_change
The 'main' mode corresponds to the individual tuning on different datasets (main experiment).
The 'dataset_change' mode corresponds to the successive transfer tuning across different datasets.
The 'ds_change' and 'qd_change' modes corresponds to the successive transfer tuning in dynamic scenarios where DS or QD changes.
'''

import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset-name', type=str, default='tiny')
args.add_argument('--last-dataset-name', type=str, default='')
args.add_argument('--experiment-mode', type=str, default='main')
args.add_argument('--dipredict-n-epochs', type=int, default=3000)
args.add_argument('--dipredict-batch-size', type=int, default=4096)
args.add_argument('--dipredict-layer-sizes', type=str, default='[14, 128, 256, 64, 3]')
args.add_argument('--dipredict-layer-sizes-nsg', type=str, default='[17, 128, 256, 64, 2]')
args.add_argument('--dipredict-lr', type=float, default=0.001)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--dipredict-valid-epoch', type=int, default=2900)
args.add_argument('--max-count', type=int, default=5)
args.add_argument('--max-selected-num', type=int, default=14)
args.add_argument('--seed', type=int, default=42)

args, unknown = args.parse_known_args()
print(args)
