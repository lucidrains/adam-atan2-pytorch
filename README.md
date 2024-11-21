<img src="./adam-atan2.png" width="400px"></img>

## Adam-atan2 - Pytorch

Implementation of the proposed <a href="https://arxiv.org/abs/2407.05872">Adam-atan2</a> optimizer in Pytorch

A multi-million dollar paper out of google deepmind proposes a small change to Adam update rule (using `atan2`) to remove the epsilon altogether for numerical stability and scale invariance

It also contains some features for improving plasticity (continual learning field)

## Install

```bash
$ pip install adam-atan2-pytorch
```

## Usage

```python
import torch
from torch import nn

# toy model

model = nn.Linear(10, 1)

# import AdamAtan2 and instantiate with parameters

from adam_atan2_pytorch import AdamAtan2

opt = AdamAtan2(model.parameters(), lr = 1e-4)

# forward and backwards

for _ in range(100):
  loss = model(torch.randn(10))
  loss.backward()

  # optimizer step

  opt.step()
  opt.zero_grad()
```

## Citations

```bibtex
@inproceedings{Everett2024ScalingEA,
    title   = {Scaling Exponents Across Parameterizations and Optimizers},
    author  = {Katie Everett and Lechao Xiao and Mitchell Wortsman and Alex Alemi and Roman Novak and Peter J. Liu and Izzeddin Gur and Jascha Narain Sohl-Dickstein and Leslie Pack Kaelbling and Jaehoon Lee and Jeffrey Pennington},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271051056}
}
```

```bibtex
@inproceedings{Kumar2023MaintainingPI,
    title   = {Maintaining Plasticity in Continual Learning via Regenerative Regularization},
    author  = {Saurabh Kumar and Henrik Marklund and Benjamin Van Roy},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:261076021}
}
```

```bibtex
@article{Lewandowski2024LearningCB,
    title   = {Learning Continually by Spectral Regularization},
    author  = {Alex Lewandowski and Saurabh Kumar and Dale Schuurmans and Andr'as Gyorgy and Marlos C. Machado},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.06811},
    url     = {https://api.semanticscholar.org/CorpusID:270380086}
}
```

```bibtex
@inproceedings{Taniguchi2024ADOPTMA,
    title   = {ADOPT: Modified Adam Can Converge with Any \$\beta\_2\$ with the Optimal Rate},
    author  = {Shohei Taniguchi and Keno Harada and Gouki Minegishi and Yuta Oshima and Seong Cheol Jeong and Go Nagahara and Tomoshi Iiyama and Masahiro Suzuki and Yusuke Iwasawa and Yutaka Matsuo},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273822148}
}
```
