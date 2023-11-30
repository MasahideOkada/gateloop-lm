# GateLoop Language Model
An implementation of [GateLoop](https://arxiv.org/abs/2311.01927) in Flax based on [this discussion](https://github.com/lucidrains/gateloop-transformer/discussions/1) with modifications like 
1) add dropout after token embedding
2) add dropouts after sub layers of gateloop blocks
3) make group norm optional 
