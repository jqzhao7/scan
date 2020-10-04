# scan
希望通过学习实现扫描全能王的功能

# 一. 学习步骤:
## 1.1 opencv-python环境
- 安装python
- 插入opencl包 `pip install opencv-python==4.4.0`
`cv2.__version__`
- 绘图工具matplotlib `pip install matplotlib`

## 1.2 基础问题和解决
- print时显示全部数据`np.set_printoptions(threshold=np.inf)`
- 解决中文问题
```python
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
- win10下打不开jupyter 使用命令`jupyter-notebook` 注意中间有横线, 同时将python和Scripts文件路径放到系统中
- plt显示的图片是RGB格式 , 需要将cv2读取到的图片转化为RGB才能显示
- numpy没有None如何判断空 
```python
need_find = np.array(need_find)
if (need_find==None).all():pass
```
## 1.3 OpenCV基础知识
- 读取图片`img = cv2.imread(img_path) `
- 图片大小 `img_size = img.shape`
- 图片转化格式 `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);` 可以转化灰度值, RGB值
- 高斯平滑 `img_guassian = cv2.GaussianBlur(img_gray, (5,5), 0) # 高斯平滑`
- 边缘提取 `img_canny = cv2.Canny(img_dst, 30, 120) `
- 最大面积索引`max_idx = np.argmax(np.array(area))`
- 得到全部轮廓 `cv2.drawContours(img_canny,)`
- 得到轮廓的线 `img_houghlines = cv2.HoughLinesP(img_canny)`
- 显示图片
```python
cv2.namedWindow('findCorners', 0)    
cv2.resizeWindow('findCorners', 700, 900)   # 自己设定窗口图片的大小
cv2.imshow("findCorners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 二. 提取图片中的重要数据和轮廓
## 2.1 思路
图像读取->灰度图->阈值分割->轮廓提取->计算最大轮廓->得到最大轮廓线条->
线条分成东南西北四组->每组线条合并为一条线条(最小二乘法)->计算四条线条的交点(交叉计算为6个,去掉2个)->   
根据四个交点按照设定好大小进行透视变换
- 中间三个部分得到交点是难点, 起初是想根据每一侧的多个线, 对线上的点求均值得到均衡后的线,但是同侧的线有的也不是很顺溜,因此就想到了深度学习课上学习的最小二乘法进行线性拟合

















