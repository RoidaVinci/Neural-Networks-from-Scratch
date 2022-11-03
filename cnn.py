import numpy as np
import math
import time
import cv2
from PIL import Image

class Convolucion:
    def __init__(self,dimfiltro,salto):
        self.dimfiltro=dimfiltro #tupla contento as dimensions do filtro
        self.salto=salto
    
    def conv(self,imaxe,filtro):
        output=np.zeros(np.shape(imaxe))
        for indice in np.ndindex(np.shape(imaxe)):
            seccion=np.zeros(self.dimfiltro)
            for i in np.ndindex(self.dimfiltro):
                itilde=tuple(math.floor(p+q-(r-1)/2) for p, q, r in zip(i,indice,self.dimfiltro))
                if self.indexexists(itilde,imaxe):
                    seccion[i]=imaxe[itilde] #a sección é un array contendo unha subimaxe do tamaño do filtro centrada en indice
            output[indice]=np.sum(np.concatenate(np.multiply(seccion,filtro)))
        return output
    
    def indexexists(self,index,imaxe):
        ranges=np.shape(imaxe)
        for i in range(len(ranges)):
            if index[i] not in range(ranges[i]):
                return False
        return True

class Pooling:
    def __init__(self,cuadricula,metodo):
        self.metodo=metodo
        self.cuadricula=cuadricula

    def pool(self,imaxe):
        dim=tuple(math.ceil(n/self.cuadricula[i]) for i,n in enumerate(np.shape(imaxe)))
        if self.metodo=='max':
            output=np.full(dim,-math.inf)
        elif self.metodo=='avg':
            output=np.zeros(dim)
            N=math.prod(self.cuadricula)
        for index in np.ndindex(np.shape(imaxe)):
            indextilde=tuple(math.floor(n/self.cuadricula[i]) for i,n in enumerate(index))
            if self.metodo=='max':
                output[indextilde]=max(output[indextilde],imaxe[index])
            elif self.metodo=='avg':
                output[indextilde]+=imaxe[index]/N
        return output

class FCLayer:
    def __init__(self,num_nodos,num_nodossaida):
        self.weights=np.random.randn(num_nodossaida,num_nodos)
    
    def forward(self,x):
        return np.matmul(self.weights,x)

class ADAM:
    def __init__

def ReLu(input):
    f=np.vectorize(lambda x: max(x,0))
    return f(input)

def Sigmoid(input):
    f=np.vectorize(lambda x: math.exp(x))
    s=np.sum(f(input))
    return f(input)/s

def Vectorize(image):
    return np.concatenate(image)

imaxe = np.random.rand(10,10)
print(imaxe)
convolucion=Convolucion((3,3),1)
pooling=Pooling((2,2),'max')
layer1=FCLayer(25,5)
filtro=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

start_time=time.time()
imaxe=convolucion.conv(imaxe,filtro)
imaxe=pooling.pool(imaxe)
imaxe=convolucion.conv(imaxe,filtro)
imaxe=Vectorize(imaxe)
imaxe=layer1.forward(imaxe)
print(imaxe)
imaxe=ReLu(imaxe)
print(time.time()-start_time)
print(imaxe)