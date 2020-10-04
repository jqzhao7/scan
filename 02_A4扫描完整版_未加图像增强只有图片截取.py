import cv2
import numpy as np
import os
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt # 用来解决不能显示的问题
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEBUG = True
img_path = r"imgs/04.jpg"
img_distinguishability = (2480,3508) # 300像素A4纸张, 目标像素

img = cv2.imread(img_path) # 
img_size = img.shape  
print("当前图片的大小是:",img_size)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化
img_guassian = cv2.GaussianBlur(img_gray, (5,5), 0) # 高斯平滑
ret, img_threshold = cv2.threshold(img_gray, 0, 150, cv2.THRESH_OTSU | cv2.THRESH_BINARY) 
#二值化后的图转化为和白图
# 第一个参数 矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
img_dst = cv2.dilate(img_threshold, kernel) # 正式膨胀
img_canny = cv2.Canny(img_dst, 30, 120)  # 边缘提取常用函数


# 通过边缘得到每一个轮廓
contours, _ = cv2.findContours(img_canny, 
                               cv2.RETR_EXTERNAL, # 只检测外围轮廓
                               cv2.CHAIN_APPROX_SIMPLE) # 保存轮廓上的所有点
# 轮廓的点集(contours), 各层轮廓的索引(hierarchy)
area = [cv2.contourArea(contour) for contour in contours] # 得到每一个轮廓的面积
max_idx = np.argmax(np.array(area))  # 找到最大面积的索引
for i in range(len(img_canny)): # 先清空图
    for j in range(len(img_canny[i])):
        img_canny[i][j] = 0
for i in contours[max_idx]: # 将轮廓点插入到图中
    img_canny[i[0][1]][i[0][0]]  = 255

img_draw = cv2.drawContours(img_canny,  # 要绘制轮廓的图像, 里面填充255颜色
                            contours, # 所有轮廓
                            max_idx, # 轮廓编号
                            255,  # 轮廓颜色
                            cv2.FILLED)# 填充最大的轮
# 到这里img_canny已经是只有外围轮廓的图片了


# 用于显示
# display_img = [img, img_gray, img_guassian, img_threshold, img_dst, img_canny,img_draw]
# display_title = ["原图", "灰度图", "高斯平滑", "阈值化", "膨胀", "边缘提取canny", "最大边缘填充"]

# for i in range(len(display_img)):
#   plt.subplot(3,3,i+1)
#   img_show = cv2.cvtColor(display_img[i],cv2.COLOR_BGR2RGB)
#   plt.imshow(img_show)
#   plt.title(display_title[i])
# plt.show()



# 这里开始寻找四条边线, 方法是, 每一条边线都会有斜率和偏移, 
# 然后将所有先都规划到四个边, 然后在将这些点进行线性规划,既可将所有的边规划到四个边, 
# 在计算四个边的交点, 得到四个点, 然后根据四个点, 进行透视变换, 得到形状变化好的图片, 然后在对图片进行图像增强即可
img_canny2 = cv2.Canny(img_draw, 30, 120) 
img_houghlines = cv2.HoughLinesP(img_canny2, 
                                 5, # 像素点精度
                                 (np.pi / 180),
                                 150, 
                                 minLineLength=100, # 线长
                                 maxLineGap=20)   # 共线两点之间最短距离
img_houghlines = np.array(img_houghlines)
if (img_houghlines==None).all():
    print("img_houghlines为空 未找到线条, 直接退出")
    exit(1)
img_houghlines = list(img_houghlines[:, 0, :])  # 将数据转换到二维
print("共有线条",len(img_houghlines))
# print("转化1维度",img_houghlines)
for x1,y1,x2,y2 in img_houghlines: # 在原图上绘制提取到的线条
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),20)

from src import get_k

EastWestSouthNorth = {"东西":[],"东":[],"南":[],"西":[],"北":[],"东_index":[], "西_index":[], "南_index":[], "北_index":[],   "南北":[]} # 东西南北


westEast = {}
southNorth = {}
# 分两级
for i,img_houghline in enumerate(img_houghlines): # 确定东南或者西北
    k,b = get_k(img_houghline)
    if 60 < k <120: # 归为东西方向, 在分东西
        EastWestSouthNorth["东西"].append(img_houghline)
    else: # 归于南北方向, 在分南北
        EastWestSouthNorth["南北"].append(img_houghline)

# 定四象
# 计算东西方向的线只看x, 以mean_x分东西, 垂直方向的x都近似,因此取x的最大值和最小值, y差异很大, 也取最大值和最小值? 
mean_x = [ line[0] for line in EastWestSouthNorth["东西"] ]
for i,_ in enumerate(mean_x):
    if _ > img_size[1]/2:# 在s东
        EastWestSouthNorth["东"].append(EastWestSouthNorth["东西"][i])
    else:
        EastWestSouthNorth["西"].append(EastWestSouthNorth["东西"][i])



# 计算南北方向的线只看y, 以mean_y分南北
mean_y = [ line[1] for line in EastWestSouthNorth["南北"] ]
for i,_ in enumerate(mean_y):
    if _ > img_size[0]/2:# 在南
        EastWestSouthNorth["南"].append(EastWestSouthNorth["南北"][i])
    else:
        EastWestSouthNorth["北"].append(EastWestSouthNorth["南北"][i])

print(EastWestSouthNorth)

from src import *
# 稳四线
from src import line_cross_point_from_kb
line_result = []
for direction in ["东", "南", "西", "北"]:
    _ = []
    for x1,y1,x2,y2 in EastWestSouthNorth[direction]:
        _.extend([[x1,y1],[x2,y2]])
    _ = np.array(_)
    x = _[:,0]
    y = _[:,1]
    w,b = fit(_)
    line_result.append([w,b])
    if w:
        pred_y= w*x+b
        plt.plot(x,pred_y,c='r')

if DEBUG:print("使用最小二乘法拟合直线为",line_result)
# plt.show()


# 得四点
from src import line_cross_point_from_kb
img_houghlines =  line_result
cross_point = []
for i, (poin1) in enumerate(img_houghlines): # 0,1,2
    if i+1==len(img_houghlines):
        break
    for j,poin2 in enumerate(img_houghlines[i+1:]):
        print(f"计算{i},{j+i+1}的交点")
        _ = line_cross_point_from_kb(poin1, poin2)
        if _:cross_point.append(_)
print("得到的交点的结果是",cross_point)
if len(cross_point) != 4:
    print(f"怎么能交点是{len(cross_point)}个呢, 你太神了")




# rows,cols,_ = img.shape
points1 = np.float32(cross_point) 
print(cross_point)
from src import get_points2
points2 = get_points2(cross_point, img_distinguishability[0],img_distinguishability[1])
print("points2",points2)
points2 = np.float32(points2)
matrix = cv2.getPerspectiveTransform(points1,points2)
# 将四个点组成的平面转换成另四个点组成的一个平面
# output = cv2.warpPerspective(img, matrix, (cols, rows))
img = cv2.imread(img_path)
output = cv2.warpPerspective(img, matrix, img_distinguishability)
# 通过warpPerspective函数来进行变换
img_show = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)  
plt.imshow(img_show), plt.title("output")
cv2.imwrite("swap/"+os.path.split(img_path)[1],output)
plt.show()

# 用于显示

# img_show = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img_show)
# plt.title("原图加边缘线")
# plt.show()
# 
# 显示
# cv2.namedWindow('findCorners', 0)    
# cv2.resizeWindow('findCorners', 700, 900)   # 自己设定窗口图片的大小
# cv2.imshow("findCorners", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

