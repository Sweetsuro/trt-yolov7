# Using Custom YOLO7-tiny Model with TensorRT in Python
##  Prepare Environment
For a general system:
```
pip install --upgrade setuptools pip
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda
pip install cuda-python
```
Steps to follow for Jetson Orin Nano:
```
pip install --upgrade setuptools pip
pip install nvidia-pyindex
sudo apt-get install tensorrt nvidia-tensorrt-dev python3-libnvinfer-dev
```
Jetson Orin Nano comes with cuda already installed. Make sure the path environment variables are set correctly. tensorrt should also already be installed depending on flashed JetPack version.

## Python Demo
Using default YOLOv7-tiny Model:
```python
git clone https://github.com/WongKinYiu/yolov7.git
```


```python
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```


```python
pip install -r yolov7/requirements.txt
```

Replace `yolov7-tiny.pt` weights with your own custom weights for the rest of the demo if so desired:
```python
python yolov7/export.py --weights yolov7-tiny.pt --grid  --simplify
```

### include  NMS Plugin


```python
python export.py -o yolov7-tiny.onnx -e yolov7-tiny.trt --end2end
```


```python
python trt.py -e yolov7-tiny.trt  -i src/1.jpg -o yolov7-tiny-1.jpg --end2end
```

###  Exclude NMS Plugin


```python
python export.py -o yolov7-tiny.onnx -e yolov7-tiny-norm.trt
```


```python
python trt.py -e yolov7-tiny-norm.trt  -i src/1.jpg -o yolov7-tiny-norm-1.jpg
```

## Citing 
Heavily modified from:
```bibtex
@Misc{yolotrt2022,
  author =       {Jian Lin},
  title =        {YOLOTRT: tensorrt for yolo series},
  howpublished = {\url{[https://github.com/Linaom1214/TensorRT-For-YOLO-Series]}},
  year =         {2022}
}
```