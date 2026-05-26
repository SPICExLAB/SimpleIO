## SimpleIO 

IMUs suck. Writing code is hard. This repository serves as a one-stop shop to train a simple model that predicts motion trajectory from IMUs alone. This is pretty much an impossible task, so you can only hope to better than baseline. Close enough is sometimes good enough. 

### Train 

```
python train.py --config config/nymeria.conf
```

### Evaluate
Generate results:
```
python inference_motion.py --config configs/nymeria.conf
```

Evaluate results
```
python evaluate.py --exp experiments/nymeria --dataconf configs/nymeria.conf --device cuda:1
```

### Visualize
```
python visualize.py --dataconf configs/nymeria.conf --seq 0
```