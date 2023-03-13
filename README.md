<div align="center">
  
# Hierarchical Domain-Adapted Feature Learning for Video Saliency Prediction
  Giovanni Bellitto, Federica Proietto Salanitri, Simone Palazzo, Francesco Rundo, Daniela Giordano, Concetto Spampinato
 
[![Paper](http://img.shields.io/badge/paper-arxiv.2010.01220-B31B1B.svg)](https://arxiv.org/abs/2010.01220)
[![Conference](http://img.shields.io/badge/IJCV-2021-4b44ce.svg)](https://link.springer.com/article/10.1007/s11263-021-01519-y)
</div>

# Overview
Official PyTorch implementation for paper: <b>"Hierarchical Domain-Adapted Feature Learning for Video Saliency Prediction"</b>

# Abstract
In this work, we propose a 3D fully convolutional architecture for video saliency prediction that employs hierarchical supervision on intermediate maps (referred to as conspicuity maps) generated using features extracted at different abstraction levels. We provide the base hierarchical learning mechanism with two techniques for domain adaptation and domain-specific learning. For the former, we encourage the model to unsupervisedly learn hierarchical general features using gradient reversal at multiple scales, to enhance generalization capabilities on datasets for which no annotations are provided during training. As for domain specialization, we employ domain-specific operations (namely, priors, smoothing and batch normalization) by specializing the learned features on individual datasets in order to maximize performance. The results of our experiments show that the proposed model yields state-of-the-art accuracy on supervised saliency prediction. When the base hierarchical model is empowered with domain-specific modules, performance improves, outperforming state-of-the-art models on three out of five metrics on the DHF1K benchmark and reaching the second-best results on the other two. When, instead, we test it in an unsupervised domain adaptation setting, by enabling hierarchical gradient reversal layers, we obtain performance comparable to supervised state-of-the-art.

# Code and models

- The code for train and test is an adapted version of the original available [here](https://github.com/MichiganCOG/TASED-Net).

- As Feature Extractor, HD2S employs S3D pretained on Kinetics-400 dataset. The S3D weights can be downloaded [here](https://github.com/kylemin/S3D).

- The whole HD2S weights trained on [DHF1K](https://mmcheng.net/videosal/) dataset can be downloaded [here](https://studentiunict-my.sharepoint.com/:u:/g/personal/uni307680_studium_unict_it/EVyDIERfwcdOnAF84v1b1VQBlDNxxhOdI-nAIafqwVV7Lg?download=1)


# Citation

```bibtex
@article{bellitto2021hierarchical,
  title={Hierarchical domain-adapted feature learning for video saliency prediction},
  author={Bellitto, G and Proietto Salanitri, F and Palazzo, S and Rundo, F and Giordano, D and Spampinato, C},
  journal={International Journal of Computer Vision},
  pages={1--17},
  year={2021},
  publisher={Springer}
}
```

# Examples
Examples of video saliency prediction in complex scenes (from DHF1K dataset):
![](gif/0648.gif)

![](gif/0692.gif)

![](gif/0685.gif)

![](gif/0609.gif)

![](gif/0605.gif)

![](gif/0622.gif)

![](gif/0690.gif)

![](gif/0652.gif)

![](gif/0674.gif)

# License

This code is released under [CC BY-NC 4.0 license](LICENSE.txt).




