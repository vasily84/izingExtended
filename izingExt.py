import numpy as np
import matplotlib.pyplot as plt
import os,subprocess
import cProfile,pstats
import imageio
import numba
import math

# pylint: disable=E1101 # игнор линтинга numpy

Slides = None

def func_Brillouin(x,J=1):
    """ функция Блиллюэна. """
    C1 = np.tanh(x*(2J+1)/(2*J))
    C2 = np.tanh(x/(2*J))
    Bj = (2J+1)/(2J*C1)-1/(2*J*C2)
    return Bj

def func_Langevin(x):
    """ функция Ланжевена """
    L = 1/np.tanh(x)-1/x
    return L
    

@numba.njit
def test_Metropolis(E1,E2,kT):
    """ разыграть вероятность перехода E1->E2 для заданной kT. По алгоритму Метрополиса .
    True - переход делаем, False - не делаем """
    dE = E2-E1
    if dE<=0:
        return True
    W = np.exp(-dE/kT)
    p = np.random.random()
    if p<=W:
        return True
    return False   

    
@numba.njit
def test_HeatBath(E1,E2,kT):
    """ разыграть вероятность перехода E1->E2 для заданной kT. По алгоритму тепловой ванны.
    True - переход делаем, False - не делаем """
    A = np.exp(-E2/kT)
    B = np.exp(-E1/kT)
    W = A/(A+B)
    p = np.random.random()
    if p<=W:
        return True
    return False   

    
@numba.njit
def inverseSpin(lattice,i,j):
    """ перевернуть спин с индексом i,j. Изменяет переменную spinSumm"""
    if lattice[i,j]>0:
        lattice[i,j]=-1
        return -1
    else:
        lattice[i,j]=1
        return +1

class CLattice2d():
    """ двумерная решетка Изинга """
    def __init__(self,XSIZE=200,YSIZE=200,Aij=0.1,Hext=0,kT=0.1):
        """ размер X, размер Y, константа обменного взаимодействия, поле, температура """
        self.XSIZE = XSIZE
        self.YSIZE = YSIZE
        self.SPIN_CNT = self.XSIZE*self.YSIZE
        self.lattice = np.ndarray(shape=(self.XSIZE,self.YSIZE))
        self.kT_lattice = np.zeros_like(self.lattice)
        self.kT_lattice[:] = kT # поле температур
        self.Aij_lattice = np.zeros_like(self.lattice)
        self.Aij_lattice[:] = Aij # поле констант обменного взаимодействия
        self.Hext = Hext # внешнее поле
        self.fileDir = 'workdir'
        self.init_lattice()

    def init_lattice(self,kind=0):
        """ инициализировать решетку """
        self.init_izing(kind)

    def init_izing(self,kind):
        """ инициализировать решетку """
        self.lattice[:]=-1. #
        if kind==0:
            for _ in range(self.XSIZE*20): # инициализируем часть спинов в противоположную сторону
                i = np.random.randint(0,self.XSIZE)
                j = np.random.randint(0,self.YSIZE)
                self.lattice[i,j] = 1
        elif kind==1:
            for i in range(self.XSIZE):
                for j in range(self.YSIZE):
                    r = self.XSIZE/3
                    r2 = r*r
                    a = i-self.XSIZE/2
                    b = j-self.YSIZE/2
                    r2_ab = a*a+b*b
                    if r2_ab<r2:
                        self.lattice[i,j] = 1
        elif kind==2:
            for i in range(self.XSIZE):
                for j in range(self.YSIZE):
                    if (i%10<5)and(j%10<5):
                        self.lattice[i,j] = 1

        self.spinSumm = np.sum(self.lattice)
    
    def init_subLattice(self,subLattice,x0,x1,y0,y1):
        """ инициализировать подрешетку по линейному закону.
        x0 - значение верхний левый, x1 -верхний правый, 
        y0 - нижний левый, y1 - нижний правый """
        for i in range(self.XSIZE): 
            A = x0+i*(x1-x0)/self.XSIZE # уровень верхний
            B = y0+i*(y1-y0)/self.XSIZE # уровень нижний
            for j in range(self.YSIZE):
                subLattice[i,j] = A+j*(B-A)/self.YSIZE


    def shake_lattice(self,Ncount=50):
        A = CLattice2d._izing2d_shake_lattice(Ncount,self.Hext,self.lattice,self.Aij_lattice,self.kT_lattice)
        self.spinSumm = np.sum(self.lattice)
        print("Adoptation = {}, Mtotal={} ,Energy={}".format(A/Ncount,self.spinSumm/self.SPIN_CNT,self.calculate_full_energy()))
        return A/Ncount # возвращаем коеффициент принятия


    @staticmethod
    @numba.njit(cache=True)
    def _izing2d_shake_lattice(Ncount,Hext,spinLattice,Aij_lattice,kT_lattice):
        """ разыграть Ncount случайных узлов решетки"""
        XSIZE,YSIZE = spinLattice.shape
        SPIN_CNT = XSIZE*YSIZE
        spinSumm = np.sum(spinLattice)
        acceptedCount = 0

        for _ in range(Ncount):
            i = np.random.randint(0,XSIZE)
            j = np.random.randint(0,YSIZE)

            ip = (i+1)%XSIZE # индексы с периодическим продолжением
            im = (i-1+XSIZE)%XSIZE
            jp = (j+1)%YSIZE
            jm = (j-1+YSIZE)%YSIZE
            
            # сумма спинов ближайших соседей, участвующих в обменном взаимодействии 
            NearestSpinSumm = spinLattice[i,jp]+spinLattice[i,jm]+spinLattice[im,j]+spinLattice[ip,j]  
            Aij = Aij_lattice[i,j]
            
            # вычисляем 
            if spinLattice[i,j]>0: # пробуем переход '+1'->'-1'
                M1 = spinSumm/SPIN_CNT 
                #M1 = +1.  
                E1 = -Aij*NearestSpinSumm-(Hext-M1)*M1 # Энергия в состоянии '+1'
                M2 = (spinSumm-1)/SPIN_CNT  
                #M2 = -1.
                E2 = Aij*NearestSpinSumm-(Hext-M2)*M2 # Энергия в состоянии '-1'
            else: # пробуем переход '-1'->'+1'
                M1 = (spinSumm)/SPIN_CNT #-1.  
                #M1 = -1.
                E1 = Aij*NearestSpinSumm-(Hext-M1)*M1 # Энергия в состояниии '+1'
                M2 = (spinSumm+1)/SPIN_CNT #+1. 
                #M2 = +1.
                E2 = -Aij*NearestSpinSumm-(Hext-M2)*M2 # Энергия в состоянии '-1'

            #if test_HeatBath(E1,E2,kT_lattice[i,j]): 
            if test_Metropolis(E1,E2,kT_lattice[i,j]): 
                spinSumm += inverseSpin(spinLattice,i,j)
                acceptedCount += 1 # коеффициент принятия
            
        return acceptedCount
        
    def task_video_to_equilibrium(self,kind=0):
        """ снять фильм с эволюцией доменной структуры к равновесию """
        Acoef = [] # список хранит коеффициенты принятия
        Slides.begin_show()
        Slides.show(self.lattice)
        try:
            if kind==0: # просто эволюция со временем
                for _ in range(400):
                    a = self.shake_lattice(self.SPIN_CNT/10)
                    Acoef.append(a)
                    Slides.show(self.lattice)
            elif kind==1: # эволюция со временем, температура меняется скачком
                kT = 0.1
                dkT = 0.03
                self.kT_lattice[:] = kT
                self.shake_lattice(self.SPIN_CNT*50) # начальное равновесие
                for _ in range(100):
                    for _ in range(10):
                        a = self.shake_lattice(self.SPIN_CNT*5)
                        Acoef.append(a)
                        Slides.show(self.lattice)
                    kT += dkT

                    self.kT_lattice[:] = kT
            Slides.show(self.lattice,'result.png') # картинка с финишем эволюции
        except KeyboardInterrupt: # cntr^C чтобы прекратить длительные вычисления досрочно
            pass
        finally: 
            Slides.end_show()
            plt.plot(Acoef)
            plt.xlabel('evolution')
            plt.ylabel('adoption')
            plt.show()


    def calculate_full_energy(self):
        """ измерить полную энергию системы """
        E_lattice = np.zeros_like(self.lattice)

        for i in range(self.XSIZE):
            for j in range(self.YSIZE):
                ip = (i+1)%self.XSIZE # индексы с периодическим продолжением
                im = (i-1+self.XSIZE)%self.XSIZE
                jp = (j+1)%self.YSIZE
                jm = (j-1+self.YSIZE)%self.YSIZE
                Aij = self.Aij_lattice[i,j]

                # сумма спинов ближайших соседей, участвующих в обменном взаимодействии 
                NearestSpinSumm = self.lattice[i,jp]+self.lattice[i,jm]+self.lattice[im,j]+self.lattice[ip,j]  
 
                E1 = self.lattice[i,j]*Aij*NearestSpinSumm # Энергия спина
                E_lattice[i,j] = E1

        M1 = self.spinSumm/self.SPIN_CNT
        # суммирование повышенной точности
        return math.fsum(E_lattice.reshape((self.SPIN_CNT)))/self.SPIN_CNT+self.Hext*M1 

    def task_heat_capacity(self):
        """ измерить теплоемкость """
        temperature = np.linspace(0.1,3)
        self.kT_lattice[:] = 0.1
        self.shake_lattice(self.SPIN_CNT*100) # стартовые разыгрывания, чтобы придти в равновесие

        fullEnergy = np.zeros_like(temperature)
    
        for i in range(len(temperature)):
            self.kT_lattice[:] = temperature[i]
            self.shake_lattice(self.SPIN_CNT*10)
            fullEnergy[i] = self.calculate_full_energy()
        
        plt.plot(temperature,fullEnergy)
        plt.xlabel('температура')
        plt.ylabel('энергия')
        plt.show()


class CSlideShow():
    def __init__(self):
        self.fileDir = 'workdir'

    def begin_show(self):
        """ начать деятельность по покадровой записи видео """
        if os.path.isdir(self.fileDir)==False:
            os.mkdir(self.fileDir)
        os.chdir(self.fileDir)
        self.pngFiles = []
        self.pngCount = 0

    def show(self,lattice,fileName=None):
        """ сохранить картинку решетки в файл """
        if fileName is None:
            fname = "framefile_%08d.png" % self.pngCount
            imageio.imwrite(fname, lattice)
            self.pngFiles.append(fname)
            self.pngCount += 1
        else:
            imageio.imwrite(fileName, lattice)
            
    def end_show(self,fileName='untitled.mp4'):
        """ собрать картинки в один видеофайл, удалить временные картинки """
        if os.path.isfile(fileName):
            os.remove(fileName)

        subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'framefile_%08d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        fileName])

        for file_name in self.pngFiles:
            os.remove(file_name)


def main():
    global Slides
    Slides = CSlideShow()

    Izing = CLattice2d(kT=1)
    #Izing.init_lattice(2)
    #Izing.init_subLattice(Izing.Aij_lattice,-1,1,-1,1)
    #Izing.init_subLattice(Izing.kT_lattice,0.1,0.1,3,3)
    Izing.task_video_to_equilibrium(kind=1)
    #Izing.task_heat_capacity()


def main_profile():
    cProfile.run('main()','profile.bin')
    file = open('profile.txt', 'w')
    profile = pstats.Stats('profile.bin', stream=file)
    profile.sort_stats('cumulative') 
    profile.print_stats(50) 
    file.close()
    

if __name__=='__main__':
    #main()
    main_profile()