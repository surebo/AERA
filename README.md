## Requirements

- python
- torch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)
- [GRF](https://github.com/google-research/football)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)
+ [GRF](https://github.com/google-research/football)



## Quick Start

```shell
$ python main.py --alg=qmix --map=8m --replay_alg=aera --result_dir='result' 
```

```shell
$ python main.py --alg=qmix --map=academy_3_vs_1_with_keeper --replay_alg=aera --result_dir='result' 
```

## Replay

If you want to see the replay, make sure the `replay_dir` is an absolute path, which can be set in `./common/arguments.py`. Then the replays of each evaluation will be saved, you can find them in your path.
