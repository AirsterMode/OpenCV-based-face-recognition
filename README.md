`
心血来潮，简单用opencv实现一个人脸识别。


使用说明：

1、运行main文件

python main.py

2、生成data文件夹后，将需要录入的人脸图像加入，文件组织如下：

```plaintext
root/
├── data/
│   ├── person001/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── person002/
│       └── ...
└── main.py
```
  
3、再次运行main.py文件即可会打开摄像头进行人脸识别，如果你变更了data文件夹，需要删除yml和npy文件重新运行。
