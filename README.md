# EchoTracker: Advancing Myocardial Point Tracking in Echocardiography

This is the official code repository for the EchoTracker model (accepted within the top 11% of MICCAI2024).

**[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_60)] [[Project Page](https://riponazad.github.io/echotracker/)] [[ArXiv(submitted version)](https://arxiv.org/abs/2405.08587)]**

![UltraTRacking.png](https://github.com/riponazad/echotracker/blob/main/assets/UltraTRacking.png)

## Updates
- (19.11.24): Fine-tuning and inference codes for TAPIR is added. Hence, the licensing got updated.
- (19.11.24): Now the interactive demo (demo2) visualizes tracking of both EchoTracker and TAPIR together.

---

## Table of Contents
- [Requirement](#required-steps)
- [Demos](#run-demos)
- [Train/Evaluation](#trainevaluation)
- [Citation](#citation)
- [License](#license-summary)

---

## Required steps
1. Clone the project: `git clone https://github.com/riponazad/echotracker.git`
2. Switch to the project directory: `cd echotracker`
3. Create a conda/virtual environment with python=3.11: `conda create --name echotracker python=3.11`
4. Activate the environment: `conda activate echotracker`
5. Install all the required packages: `pip install -r requirements.txt`
6. Add the project directory (echotracker) to **PYTHONPATH**: ``export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH``

Here you go!!!!
## Run demos
First, download the [weight](https://drive.google.com/file/d/1FJmNlfE5lNPSpBbtnc3KBOgC7cp6DCpl/view?usp=drive_link) and ensure before running, the model weight(model-000000000.pth file) is placed inside the 'model' directory.
Now run demos:
1. `python demo1.py`: It will load the EchoTracker model and the provided ultrasound sample data, estimate the trajectories for the given query points, and print performance metrics.
2. `python demo2.py`: This is an interactive demo. It will allow you to choose any query points using your mouse on the first of the provided video and then track those points throughout the entire video.
    - Press `q` after watching the video.
    - The first frame will appear.
    - Select query points (max 20) on any physical space and press `q` again. Of course, you can increase the maximum number of points by editing demp2.py file.
    - EchoTracker will estimate the trajectories in no time and then visualize the video.
    - See the video.

    ![output.gif](https://github.com/riponazad/echotracker/blob/main/assets/output.gif)


## Train/Evaluation
### EchoTracker
Since we are unable to publish the datasets, we are leaving this part mostly to you. However, we provided training and inference code as class methods [def train](https://github.com/riponazad/echotracker/blob/main/model/net.py) so that you can train/finetune/evaluate EchoTracker on your datasets as simple as just the following lines.

```
from model.net import EchoTracker
B=1, # batchsize 
S=24, # seqlen
#you should have your own load_datasets() method before executing the following line    
dataloaders, dataset_size = load_datasets(B=B, S=S, debug=debug, use_aug=True)

model = EchoTracker(device_ids=[0])
#model.load(path=configs.WEIGHTS_PATH.echotracker, eval=False)  #uncomment to fine-tune the model instead training
model.load(eval=False) #load the model to train
model.train(
        dataloaders=dataloaders, 
        dataset_size=dataset_size, 
        log_dir=configs.LOG_PATH.echotracker, 
        ckpt_path=configs.SAVE_PATH.echotracker,
        epochs=100
)
``` 

### TAPIR
Similarly, you can fine-tune/evaluate with TAPIR.

```
from model.net import TAPIR
B=1, # batchsize 
S=24, # seqlen
#you should have your own load_datasets() method before executing the following line    
dataloaders, dataset_size = load_datasets(B=B, S=S, debug=debug, use_aug=True)

model = TAPIR(pyramid_level=0)
#model.load(path=configs.WEIGHTS_PATH.tapir, eval=False)  #uncomment to fine-tune the model instead training
model.load(eval=False) #load the model to train
model.finetune(
        dataloaders=dataloaders, 
        dataset_size=dataset_size, 
        log_dir=configs.LOG_PATH.tapir, 
        ckpt_path=configs.SAVE_PATH.tapir,
        epochs=100
)
``` 


## Citation

If you use this code for your research, please cite our paper:

Bibtex:
```
@InProceedings{azad2024echo,
author="Azad, Md Abulkalam
and Chernyshov, Artem
and Nyberg, John
and Tveten, Ingrid
and Lovstakken, Lasse
and Dalen, H{\aa}vard
and Grenne, Bj{\o}rnar
and {\O}stvik, Andreas",
title="EchoTracker: Advancing Myocardial Point Tracking in Echocardiography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
volume="IV",
publisher="Springer Nature Switzerland",
pages="645--655"
}
```


## License Summary

This project is licensed under a dual-license structure as some codes are directly adopted from third-party repositories.

- **EchoTracker**: MIT License
- **Third-Party Code**: Apache License 2.0

### MIT License

The MIT License applies to all original code written for this repository. For full details, see the `LICENSE` file.

### Apache License 2.0 (Third-Party Code)

This project includes code from third-party repository licensed under the Apache License 2.0. Specifically:
- [TAPIR](https://github.com/google-deepmind/tapnet)

See the `LICENSE` file for the full Apache License 2.0 terms also on their repository.

### How to Comply

When using this project:
- You are free to use, modify, and distribute the code under the respective licenses.
- Retain the MIT license notice for original contributions.
- Retain the Apache 2.0 license notice and adhere to its requirements for third-party components.
