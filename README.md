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

```bibtex
@inproceedings{Liang2024CautiousOI,
    title   = {Cautious Optimizers: Improving Training with One Line of Code},
    author  = {Kaizhao Liang and Lizhang Chen and Bo Liu and Qiang Liu},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274234738}
}
```

```bibtex
@misc{jordan2024muon,
    author       = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and Franz Cesista and Laker Newhouse and Jeremy Bernstein},
    title        = {Muon: An optimizer for hidden layers in neural networks},
    year         = {2024},
    url          = {https://kellerjordan.github.io/posts/muon/}
}
```

```bibtex
@misc{kimiteam2025kimik2openagentic,
    title   = {Kimi K2: Open Agentic Intelligence}, 
    author  = {Kimi Team and Yifan Bai and Yiping Bao and Guanduo Chen and Jiahao Chen and Ningxin Chen and Ruijue Chen and Yanru Chen and Yuankun Chen and Yutian Chen and Zhuofu Chen and Jialei Cui and Hao Ding and Mengnan Dong and Angang Du and Chenzhuang Du and Dikang Du and Yulun Du and Yu Fan and Yichen Feng and Kelin Fu and Bofei Gao and Hongcheng Gao and Peizhong Gao and Tong Gao and Xinran Gu and Longyu Guan and Haiqing Guo and Jianhang Guo and Hao Hu and Xiaoru Hao and Tianhong He and Weiran He and Wenyang He and Chao Hong and Yangyang Hu and Zhenxing Hu and Weixiao Huang and Zhiqi Huang and Zihao Huang and Tao Jiang and Zhejun Jiang and Xinyi Jin and Yongsheng Kang and Guokun Lai and Cheng Li and Fang Li and Haoyang Li and Ming Li and Wentao Li and Yanhao Li and Yiwei Li and Zhaowei Li and Zheming Li and Hongzhan Lin and Xiaohan Lin and Zongyu Lin and Chengyin Liu and Chenyu Liu and Hongzhang Liu and Jingyuan Liu and Junqi Liu and Liang Liu and Shaowei Liu and T. Y. Liu and Tianwei Liu and Weizhou Liu and Yangyang Liu and Yibo Liu and Yiping Liu and Yue Liu and Zhengying Liu and Enzhe Lu and Lijun Lu and Shengling Ma and Xinyu Ma and Yingwei Ma and Shaoguang Mao and Jie Mei and Xin Men and Yibo Miao and Siyuan Pan and Yebo Peng and Ruoyu Qin and Bowen Qu and Zeyu Shang and Lidong Shi and Shengyuan Shi and Feifan Song and Jianlin Su and Zhengyuan Su and Xinjie Sun and Flood Sung and Heyi Tang and Jiawen Tao and Qifeng Teng and Chensi Wang and Dinglu Wang and Feng Wang and Haiming Wang and Jianzhou Wang and Jiaxing Wang and Jinhong Wang and Shengjie Wang and Shuyi Wang and Yao Wang and Yejie Wang and Yiqin Wang and Yuxin Wang and Yuzhi Wang and Zhaoji Wang and Zhengtao Wang and Zhexu Wang and Chu Wei and Qianqian Wei and Wenhao Wu and Xingzhe Wu and Yuxin Wu and Chenjun Xiao and Xiaotong Xie and Weimin Xiong and Boyu Xu and Jing Xu and Jinjing Xu and L. H. Xu and Lin Xu and Suting Xu and Weixin Xu and Xinran Xu and Yangchuan Xu and Ziyao Xu and Junjie Yan and Yuzi Yan and Xiaofei Yang and Ying Yang and Zhen Yang and Zhilin Yang and Zonghan Yang and Haotian Yao and Xingcheng Yao and Wenjie Ye and Zhuorui Ye and Bohong Yin and Longhui Yu and Enming Yuan and Hongbang Yuan and Mengjie Yuan and Haobing Zhan and Dehao Zhang and Hao Zhang and Wanlu Zhang and Xiaobin Zhang and Yangkun Zhang and Yizhi Zhang and Yongting Zhang and Yu Zhang and Yutao Zhang and Yutong Zhang and Zheng Zhang and Haotian Zhao and Yikai Zhao and Huabin Zheng and Shaojie Zheng and Jianren Zhou and Xinyu Zhou and Zaida Zhou and Zhen Zhu and Weiyu Zhuang and Xinxing Zu},
    year    = {2025},
    eprint  = {2507.20534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2507.20534}, 
}
```
