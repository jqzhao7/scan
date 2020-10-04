import numpy as np
import math
# 计算直线交点
def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c


def get_line_cross_point(line1, line2):
    # x1y1x2y2
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    if x>300 or x<0 or y>400 or y<0:
        return None
    else:
        return [int(x),int(y)]

def get_points2(cross_point, X_max, Y_max):
    x = []
    y = []
    points2 = []
    for point in cross_point:
        x.append(point[0])
        y.append(point[1])
    x_mean = int(np.mean(x))
    y_mean = int(np.mean(y))

    for i,point in enumerate(cross_point):
        if point[0] > x_mean:
            points2.append([X_max,0])
        else:
            points2.append([0,0])
        if point[1] > y_mean:
            points2[i][1] = Y_max
    return points2

import math
def get_k(img_houghline:list):
    # print(img_houghline)
    x1,y1,x2,y2 = img_houghline
    if x1==x2:
        return 90, x1
    return abs(math.atan((y1-y2)/(x1-x2))*180/math.pi), (x1*y2-x2*y1)/(x1-x2)

# a = 181, 1512,  187, 1411
# print(get_k(a))


def line_cross_point_from_kb(p1, p2):
    (k1,b1),(k2,b2) = p1,p2
    if k1==k2: # 平行
        return None
    elif k1 == None and None== k2: # p1和p2是垂直的
        return None 
    elif k1 == None :
        x = b1
        y = b1*k2+b2
    elif k2 == None:
        x = b2
        y = b2*k1+b1
    else:
        x = (b1-b2)/(k2-k1)
        y = (b1*k2-b2*k1)/(k2-k1)

    if x<0 or y<0 or x>1600 or y>2000: 
        return None
    else: 
        return (x,y)


# 最小二乘法将每个方向上面的点拟合为一条直线
# 损失函数是系数的函数，另外还要传入数据的x,y
def compute_cost(w,b,points):
    total_cost=0
    M =len(points)
    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        total_cost += (y-w*x-b)**2
    return total_cost/M #一除都是浮点 两个除号是地板除，整型。 如 3 // 4

def fit(points):
    if len(set(points[:,0]))==1: # 数据去重判断X轴相等
        return None,points[0][0]
    # print(points,"==========")
    M = len(points) # 点数
    print(M)
    x_bar=np.mean(points[:,0])
    sum_yx= 0
    sum_x2=0
    sum_delta =0
    for i in range(M): # 2
        x=points[i,0]
        y=points[i,1]
        sum_yx += y*(x-x_bar)
        sum_x2 += x**2
    #根据公式计算w
    # print(sum_yx, sum_x2, x_bar)
    w = sum_yx/(sum_x2-M*(x_bar**2))
    
    for i in range(M):
        x=points[i,0]
        y=points[i,1] 
        sum_delta += (y-w*x)
    b = sum_delta / M
    return w,b



a = None
# a = np.array([2,3,4,5])
a = np.array(a)
if (a == None).all():
    print("空")
