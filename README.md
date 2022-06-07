# SGF

This version aims to demonstrate the key features of the SGF.

## intall
```
conda create -n sgf python=3.7
conda activate sgf
```

```
conda install numpy=1.20 -y
conda install opencv=3.4.2 -y
conda install h5py=2.8.0 -y
conda install matplotlib=3.3.4 -y
conda install imageio=2.9.0 -y
conda install scikit-learn=0.24 -y
conda install xlwt=1.3.0 -y
conda install xlrd=2.0.1 -y
conda install xlutils=2.0.0 -y
```

```
pip install opencv-python   
pip install opencv-contrib-python 
```

## Dataset
1. Download DVS Gesture dataset from https://research.ibm.com/interactive/dvsgesture/

2. change the data_path and code_path in cfg.py

3. Preprocess the event data into npy files. (This may take a few hours and need ~70GB free space on disk)
   
```python
cd ${your_path}/SGF_submmit
python process_dvs_gesture.py
python dvsgesture_t.py
```

## Run

```python
cd ${your_path}/SGF_submmit
python main.py --train_test --iter 36 --selected_events 2+8+9+10+1_3_4+5_6+7
```
36 equals to training/test sample ratio 1.5:1.
