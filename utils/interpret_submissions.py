from collections import defaultdict
import gzip
import json
import argparse
from typing import Union


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()

    return args


def stagewise(path, stage = None):
    assert not (
        path is None
        and stage is None
    ), f"Both path and stage cannot be None"
    if path is None:
        path = f"subtask_expert_res_{stage}.json.gz"

    with gzip.open(path, 'r') as fin:
        jbytes = fin.read()

    jstr = jbytes.decode('utf-8')
    res = json.loads(jstr)

    num_episodes = len(res.keys())
    metric_dict = defaultdict(list)
    for epi_id, metrics in res.items():
        for k, v in metrics.items():
            if (
                (
                    isinstance(v, float) or isinstance(v, int)
                ) and k.startswith('unshuffle')
            ):
                metric_dict[k].append(v)
    
    averages = {
        k: sum(v) / num_episodes
        for k, v in metric_dict.items()
    }
    if stage is not None:
        print(f"stage: {stage}")
    print(averages)
    import pdb; pdb.set_trace()


def main():
    args = parse_args()
    if args.path is not None and len(args.path) > 0:
        stagewise(args.path)
    else:
        for stage in ("train", "val", "test", "combined"):
            stagewise(path=None, stage=stage)
    

if __name__ == "__main__":
    main()
