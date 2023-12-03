# GateLoop Language Model
An implementation of [GateLoop](https://arxiv.org/abs/2311.01927) in Flax based on [this discussion](https://github.com/lucidrains/gateloop-transformer/discussions/1) with modifications like 
1) add dropout after token embedding
2) add dropouts after sub layers of gateloop blocks
3) make group norm optional 

# train
an example command to train the model
```
$ python train.py \
--config-dir configs \
--data-path data.txt \
--sp-model m.model \
--num-epochs 5 \
--checkpoint-dir gl-ckpt
```

`config-dir` is the directory that contains `model_config.json` and `train_config.json`. \
`model_config.json` is for setting the model hyper-parameters such as the model dimension, the number of layers etc. see [configs/model_config.json](configs/model_config.json).\
`train_config.json` is for choosing the hyper-parameters like the learning rate, batch size etc. see [configs/train_config.json](configs/train_config.json)\
`sp-model` is a [sentencepiece](https://github.com/google/sentencepiece) model file for text tokenization.

# generate texts
after the training, generate texts by running this command

```
$ python generate.py \
--config-dir configs \
--sp-model m.model \
--checkpoint-dir gl-ckpt \
--prompt "YOUR PROMPT" \
--num-outputs 10
```

this outputs a csv file containing the generated texts.
