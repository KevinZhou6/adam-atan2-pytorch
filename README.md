

## Adam-atan2 - Pytorch

Implementation of the proposed <a href="https://arxiv.org/abs/2407.05872">Adam-atan2</a> optimizer in Pytorch

A multi-million dollar paper out of google deepmind proposes a small change to Adam update rule (using `atan2`) to remove the epsilon altogether for numerical stability and scale invariance

It also contains some features for improving plasticity (continual learning field)