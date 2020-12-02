# Sitting Posture Detection
基于openpifpaf项目的人体坐姿检测，帮助家长实时监控孩子坐姿。
## 使用说明
共分两步：
- 姿态标定，指定参数-c或--calibration时生效

    1、指定参数--image，则需要传入待标定的图片路径，并指定是从左侧标定还是右侧（默认为左侧）
    
        从右侧标定的例子：python main.py -c --image=./data/img/baby.png -r
    
    2、不指定参数--image，则需要通过实时摄像头做标定，先选择“左侧”（按键l）标定还是“右侧”（按键r）标定（默认左侧，可重选），选择完毕后按键q退出
        
        例子：python main.py -c

- 姿态识别

    1、指定参数--source，0为第一个摄像头，以此类推
    
    2、指定参数--show，则摄像头画面会显示出来，否则后台运行不显示，如为显示模式，鼠标任意点击绘图区域退出程序，否则ctl+c退出
        
        python main.py --source=0 --show

## 参数说明
- c, --calibration:是否执行坐姿标定，标定后得到的参数为后续判定坐姿的依据
- --image:指定图片做标定
- -r,--right:指定图片标定时，从左侧标定还是右侧
- --version:依赖的OpenPifPaf的版本
- --source:指定使用那个摄像头，0为第一个摄像头，以此类推
- --show:是否显示摄像头画面
- --disable-cuda:是否禁用CUDA
- -q,--quiet:是否只显示警告及以上日志信息
- --debug:是否显示所有日志信息

## 依赖说明
- torch~=1.7.0

- baidu-aip

- playsound~=1.2.2

- sudo apt-get install python3-gst-1.0

- argparse~=1.4.0

- Pillow~=8.0.0

- opencv-python~=4.4.0.46

- openpifpaf~=0.11.9

- matplotlib~=3.3.2

## 例子
- 图片标定+检测
python main.py -c --image=/data/img/baby.png --source=0 --show
- 摄像头标定+检测
python main.py -c --source=0 --show
- 默认配置+检测
python main.py --source=0 --show
