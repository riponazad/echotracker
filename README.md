# EchoTracker: Advancing Myocardial Point Tracking in Echocardiography

This is the official code repository for the EchoTracker model provisionally (early) accepted (top 11%) to MICCAI2024.

**[[Paper](https://arxiv.org/abs/2405.08587)] [[Project Page](https://riponazad.github.io/echotracker/)]**

![UltraTRacking.png](https://github.com/riponazad/echotracker/blob/main/assets/UltraTRacking.png)


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
Since we are unable to publish the datasets, we are leaving this part to you. However, we provided training and inference code as class methods (check model/net.py) so that you can easily train/finetune/evaluate EchoTracker on your datasets.


## Citation

If you use this code for your research, please cite our paper:

Bibtex:
```
@inproceedings{azad2024echo,
 author = {Azad, Md Abulkalam and Chernyshov, Artem and Nyberg, John and Tveten, Ingrid and Lovstakken, Lasse and Dalen, Havard and Grenne, Bjørnar and {\O}stvik, Andreas},
 title = {EchoTracker: Advancing Myocardial Point Tracking in Echocardiography},
 booktitle = {MICCAI2024},
 year = {2024}
}
```

