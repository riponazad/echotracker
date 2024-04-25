# EchoTracker: Advancing Myocardial Point Tracking in Echocardiography.

![UltraTRacking.png](https://github.com/riponazad/echotracker/blob/main/assets/UltraTRacking.png)

(Full repo with EchoTracker and a sample will be uploaded soon.)

## Requirements
1. Clone the project: `git clone https://github.com/riponazad/echotracker.git`
2. Switch to the project directory: `cd echotracker`
3. Create a conda/virtual environment with python=3.11: `conda create --name echotracker python=3.11`
4. Activate the environment: `conda activate echotracker`
5. Install all the required packages: `pip install -r requirements.txt`
6. Add the project directory (echotracker) to **PYTHONPATH**: ``export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH``

Here you go!!!!
## Run demos
Ensure before running the model weight(.pth file) are placed inside the 'model' directory.
Now run demos:
1. `python demo1.py`: It will load the EchoTracker model and the provided ultrasound sample data, estimate the trajectories for the given query points, and print performance metrics.
2. `python demo2.py`: This is an interactive demo. It will allow you to choose any query points using your mouse on the first of the provided video and then track those points throughout the etire video.
    - Press `q` after watching the video.
    - The first frame will appear.
    - Select query points (max 20) on any physical space and press `q` again. Of course you can increase the maximum number of points by editing demp2.py file.
    - EchoTracker will estimate the trajectories in no time and then visualize video.
    - See the video.

    ![output.gif](https://github.com/riponazad/echotracker/blob/main/assets/output.gif)


