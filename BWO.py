import numpy as np
import matplotlib.pyplot as plt
import random
import math

class population:
    def __init__(self,x_max,x_min,Gmax,sizepop,obj_fun,dim):
        self.pp=0.6#生育率
        self.CR=0.54#同类相食率
        self.pm=0.4#突变率
        self.x_min=x_min
        self.x_max=x_max
        self.sizepop=sizepop
        self.dim=dim
        self.get_fun=obj_fun
        self.Gmax=Gmax
        self.g=1
        self.best_value = []

    def init_population(self):
        #种群初始化
        self.initialization=np.random.uniform(self.x_min,self.x_max,[self.sizepop,self.dim])
        self.objectvalue=[self.get_fun(i) for i in self.initialization]

    def procreate(self):
        obj_sorted=sorted(self.objectvalue) # 对适应度值进行排序，但并不改变原来的顺序
        parents_num = round(self.sizepop * self.pp) # 计算父母个体数目
        parent_index = [] #记录父代的索引位置
        self.parent=[] # 父代蜘蛛个体
        self.children=[] # 存储同类相食后的个体

        parent_obj=obj_sorted[0:parents_num]  # 取出适应度排名靠前的个体

        for i in range(len(parent_obj)):
            temp=parent_obj[i]
            parent_index.append(self.objectvalue.index(temp))  # 找出该个体适应度对应的索引

        for i in range(len(parent_index)):
            temp1=parent_index[i]
            self.parent.append(self.initialization[temp1]) # 根据适应度，找出该适应度对应种群个体索引
        # 随机选择一对父母(雌雄黑寡妇)进行生育后代
        self.parent=np.random.permutation(self.parent)
        for i in range(0,len(self.parent),2):
            child_obj = []  # 需要进行同类相食个体，适应度的值
            eat = []
            eat_index = []  # 记录同类相食，保留下来优秀个体索引位置
            alpha = random.random()
            child = []  # 存储即将同类相食的个体
            temp2=self.get_fun(self.parent[i])
            temp3=self.get_fun(self.parent[i+1])
            if temp2 < temp3:
                x1=self.parent[i]
                x2=self.parent[i + 1]
                child.append(x1)
            else:
                x2 = self.parent[i]
                x1 = self.parent[i + 1]
                child.append(x1)
             # 每对父母生育D个后代
            if self.dim==2:
                y1=alpha*x1+(1-alpha)*x2
                child.append(y1)

                y2=alpha*x2 +(1-alpha)*x1
                child.append(y2)

            if self.dim>2:
                for j in range(0,round(self.dim/2)+1,2):
                    temp=alpha*x1+(1-alpha)*x2
                    child.append(temp)

                    temp1=alpha*x2 +(1-alpha)*x1
                    child.append(temp1)

                child=child[0:self.dim]
             # 每一对生育完成后，开始同类相食
            child_obj=[self.get_fun(i) for i in child] # 计算适应度的值
            child_obj_sort=sorted(child_obj)  # 按照适应度排序
            eat=child_obj_sort[0:round(len(child)*self.CR)]  # 按照同类相食率，取出适应度排名靠前的个体

            for i in range(len(eat)):
                temp4=eat[i]
                eat_index.append(child_obj.index(temp4))

            for i in range(len(eat_index)):
                temp5=eat_index[i]
                self.children.append(child[temp5])

    def mutation(self): # 从父代蜘蛛种群中进行变异
        self.parent = np.random.permutation(self.parent) # 随机排列
        self.mutat=[] # 存储即将突变的个体
        self.mutat=self.parent[0:round(self.sizepop*self.pp*self.pm)]# 根据突变率pm,随机选择多个黑寡妇

        # 每个黑寡妇随机交换数组中的两个维度的值
        if self.dim==2:
            for i in range(len(self.mutat)):
                for j in range(0,self.dim-1,2):
                    temp = self.mutat[i][j]
                    self.mutat[i][j] = self.mutat[i][j+1]
                    self.mutat[i][j+1] = temp
        if self.dim>2:
            for i in range(len(self.mutat)):
                d1 = np.random.sample(range(0, self.dim - 1), 1)
                d2 = np.random.sample(range(0, self.dim - 1), 1)
                if d1 != d2:
                    temp = self.mutat[i][d1]
                    self.mutat[i][d1] = self.mutat[i][d2]
                    self.mutat[i][d2] = temp

    # 一次迭代之后，将同类相食阶段保留下来的黑寡妇以及突变阶段得到的黑寡妇作为下一次迭代的初始种群。
    def update_population(self):
        self.initialization1=[]
        self.objectvalue1=[]
        self.initialization2=[]
        self.initialization1 = np.append(self.children, self.mutat, axis=0)

        if len(self.initialization1)>self.sizepop:
            self.objectvalue1_sort_index=[]

            for i in range(len(self.initialization1)): # 计算initialization1适应度值
                temp=self.get_fun(self.initialization1[i])
                self.objectvalue1.append(temp)
            init1_sort = sorted(self.objectvalue1) # initialization1适应度值排序

            self.objectvalue1_sort=init1_sort[0:self.sizepop] # 只选择initialization1适应度值排序前sizepop的个体

            for i in range(len(self.objectvalue1_sort)):
                temp=self.objectvalue1_sort[i]
                self.objectvalue1_sort_index.append(self.objectvalue1.index(temp))

            for i in range(len(self.objectvalue1_sort_index)):
                temp1=self.objectvalue1_sort_index[i]
                self.initialization2.append(self.initialization1[temp1])
            self.initialization=self.initialization2

            for i in range(len(self.initialization)): # 计算initialization适应度值
                temp=self.get_fun(self.initialization[i])
                self.objectvalue[i]=temp

        else:
            self.initialization2=np.append(self.initialization1,self.parent,axis=0)
            self.initialization=self.initialization2[0:self.sizepop]
            for i in range(len(self.initialization)):
                temp=self.get_fun(self.initialization[i])
                self.objectvalue[i]=temp

    def best(self):
        f_min = min(self.objectvalue)
        self.best_value.append(f_min)
        x=self.objectvalue.index(f_min)

        print("当前迭代次数为" + str(self.g))
        print("最好的个体是：" + str(self.initialization[x]))
        print("目标函数值为：'" + str(f_min))

    def BWO(self):
        self.init_population()
        while self.g < self.Gmax:
            self.procreate()
            self.mutation()
            self.update_population()
            self.best()
            self.g = self.g + 1

    def draw(self):
        plt.plot(self.best_value)
        plt.show()

if __name__=="__main__":
    def f(x):
        # 测试函数采用  Cross-in-Tray function
        return -0.0001 * math.pow(math.fabs(math.sin(x[0] * math.sin(x[1]) * math.exp(math.fabs(100 - math.sqrt(x[0] ** 2 + x[1] ** 2) / math.pi)))) + 1,0.1)

    p = population(x_min=-10, x_max=10, dim=2, Gmax=200, sizepop=100, obj_fun=f)
    p.BWO()
    p.draw()

