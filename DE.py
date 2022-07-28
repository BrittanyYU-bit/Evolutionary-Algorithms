import numpy as np
import random
import matplotlib.pyplot as plt
import math
class population:
    def __init__(self,dim,sizepop,x_min,x_max,obj_fun,Gmax,):
        self.F=0.5
        self.CR=0.8
        self.dim=dim#维度
        self.sizepop=sizepop#种群个数
        self.x_min=x_min#种群中最大值
        self.x_max=x_max#种群中最小值
        self.Gmax=Gmax#种群最大迭代次数
        self.g=1#设置迭代初始
        self.get_fun=obj_fun

        # 种群初始化
        # self.initialization = self.x_min + (self.x_max - self.x_min) * random.random((self.sizepop, self.dim))
        self.initialization=np.random.uniform(self.x_min,self.x_max,[self.sizepop,self.dim])
        # 计算适应度的值
        self.objectvalue=[self.get_fun(i) for i in self.initialization]
        self.best_finess=[]

    def mutate(self):
        self.mutant=[]
        for i in range(self.sizepop):
            r0, r1, r2=0, 0, 0
            while r0==r1 or r1==r2 or r2==r0 or r0==i:
                r0=random.randint(0,self.sizepop-1)
                r1=random.randint(0,self.sizepop-1)
                r2=random.randint(0,self.sizepop-1)
            temp=self.initialization[r0]+self.F*(self.initialization[r1]-self.initialization[r2])
            for j in range(self.dim):
                if temp[j]>self.x_max or temp[j]<self.x_min:
                    temp[j]=random.uniform(self.x_min,self.x_max)
            self.mutant.append(temp)

    def crossover(self):
        for i in range(self.sizepop):
            jrand = random.randint(0, self.dim)
            for j in range(self.dim):
                R=random.random()
                if R>self.CR and jrand!=j:#j=jrand
                    self.mutant[i][j]=self.initialization[i][j]
                else:
                    self.mutant[i][j]=self.mutant[i][j]

    def select(self):
        for i in range(self.sizepop):
            tep = self.get_fun(self.mutant[i])
            if tep < self.objectvalue[i]:
                self.initialization[i] = self.mutant[i]
                self.objectvalue[i] = tep

    def print_best(self):
        f_min=min(self.objectvalue)
        x=self.objectvalue.index(f_min)
        print("当前迭代次数为"+str(self.g))
        print("最好的个体为："+str(self.initialization[x]))
        print("目标函数值为：'"+str(f_min))

    def DE(self):
        while self.g<self.Gmax:
            self.mutate()
            self.crossover()
            self.select()
            self.print_best()
            self.g=self.g+1

            self.best_finess.append(min(self.objectvalue))

    def draw(self):
        plt.plot(self.best_finess)
        plt.show()

if __name__=="__main__":
    def f(x):
        #测试函数采用  Cross-in-Tray function
        return -0.0001*math.pow(math.fabs(math.sin(x[0]*math.sin(x[1])*math.exp(math.fabs(100-math.sqrt(x[0]**2+x[1]**2)/math.pi))))+1,0.1)

    p=population(x_min=-10,x_max=10,dim=2,Gmax=200,sizepop=100,obj_fun=f)
    p.DE()
    p.draw()
