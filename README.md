# HD2S
[Video Saliency Detection with Domain Adaption using Hierarchical Gradient Reversal Layers](https://arxiv.org/abs/2010.01220)

## Examples
Examples of video saliency prediction in complex scenes (from DHF1K dataset):
![](gif/0648.gif)

![](gif/0692.gif)

![](gif/0685.gif)

![](gif/0609.gif)

## Notes

- The code for train and test is an adapted version of the original available [here](https://github.com/MichiganCOG/TASED-Net).

- As Feature Extractor, HD2S employs S3D pretained on Kinetics-400 dataset. The S3D weights can be downloaded [here](https://github.com/kylemin/S3D).

- The whole HD2S weights trained on [DHF1K](https://mmcheng.net/videosal/) dataset can be downloaded [here](https://studentiunict-my.sharepoint.com/:u:/g/personal/uni307680_studium_unict_it/EVyDIERfwcdOnAF84v1b1VQBlDNxxhOdI-nAIafqwVV7Lg?download=1)

## Citation
```bibtex
@article{bellitto2020video,
  title={Video Saliency Detection with Domain Adaption using Hierarchical Gradient Reversal Layers},
  author={Bellitto, Giovanni and Salanitri, Federica Proietto and Palazzo, Simone and Rundo, Francesco and Giordano, Daniela and Spampinato, Concetto},
  journal={arXiv preprint arXiv:2010.01220},
  year={2020}
}
```


