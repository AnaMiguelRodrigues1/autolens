# autolens

## Purpose
Effortlessly implementation of pre-existing code from open-source Augmented ML and Automated ML tools that support image classification tasks

## Resources

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
![Technical requisites](/assets/config_details.png)


### More Information
- [AutoKeras: ImageClassifier](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [AutoGluon: AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [Ludwig: LudwigModel](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/beginner_multimodal.html)
- [Ktrain: vision](https://amaiya.github.io/ktrain/vision/index.html)
- [Fastai: vision_learner](https://docs.fast.ai/tutorial.vision.html)
