# autolens

## Purpose
Effortlessly implementation of pre-existing code from open-source Augmented ML and Automated ML tools that support image classification tasks

## Usage
This project can be both used as a library and CLI tool. 
### Dataset preparation
- Main folder with sub-folders, each named with an integer number
  
### Library Setup
- clone repo: `git clone https://github.com/AnaMiguelRodrigues1/autolens.git`
- move to the root of the project
- install the library: `python3.9 setup.py install`
- install python virtual environment on the root folder: `python -m venv {automl_tool}_venv`
- run `source -m {automl_tool}_venv/bin/activate`

````python

from autolens.LUDWIG.run import main #prior selection of automl tool

main(
    "../../chest_xray/", #dataset path 
    1, #bigger steps for less computational resources
    (255, 255), #target size
    0.2, #size of testing dataset
    0.1 #size of validation dataset
)

````

### CLI Interaction
- clone repo: `git clone https://github.com/AnaMiguelRodrigues1/autolens.git`
- move to the root of the project

````python

python3.9 autolens.py "ludwig" "../../chest_xray"
  --target_size "(255, 255)"
  --test_percentage "0.2"
  --val_percentage "0.1"
  --clean_metadata "store_true"
  --cache_dir "{home_dir}/.cache/autolens"

````

### Configuration Details
**S.F.** - Supported Framework
**I.S.** - Interface Solutions
**Lang.** - Programming Language
**O.S.** - Operative System

|                  | **Fastai v2.7.12** | **Ktrain v0.37.2** | **Ludwig v0.8.1.post1** | **Autogluon[^2] v0.8.2**| **Autokeras v1.1.0** |
|------------------|--------------------|--------------------|-------------------------|-------------------------|----------------------|
| **S.F.**         | Pytorch v1.13.1| Tensorflow v2.11 | Tensorflow[^1] | Pytorch v1.13.1   | Tensorflow v>=2.8.0[^3] |
| **I.S.**         | API           | API          | API/CLI          | API               | API           |
| **Lang.**        | Python v3.7-v3.10 | Python v3.6-v3.10 | Python v>=3.8 | Python v3.8-v3.10 | Python v3.8-v3.11 |
| **O.S.**         | Linux, Windows | Linux        | Linux, Windows   | Linux, Windows[^4] | Linux, Windows[^5], MacOS |

[^1]: Uses Tensorboard v2.14, a visualization toolkit from Tensorflow.
[^2]: Uses **Fastai** as one of the installation requirements.
[^3]: Tensorflow v2.9.1 most compatible with the remaining dependencies.
[^4]: Advisable to use Anaconda.
[^5]: Requires Microsoft Visual C++ and v>7.

### More Information
- [AutoKeras: ImageClassifier](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [AutoGluon: AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [Ludwig: LudwigModel](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [Ktrain: vision](https://amaiya.github.io/ktrain/vision/index.html)
- [Fastai: vision_learner](https://docs.fast.ai/tutorial.vision.html)
