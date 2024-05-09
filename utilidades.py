#Algoritmos que se utilizan continuamente durante el proyecto
import numpy as np

def fusion(M1, M2, x, y):
    "Pega el arreglo M2 en M1 empezando en la posicion x,y"
    m = M1.shape
    mm = M2.shape

    if mm[-2] > m[-2] or mm[-1] > m[-1]:
        raise ValueError('La matriz a insertar debe ser menor mas pequenia que la original')
    MC = np.copy(M1)

    MC[..., y : y + mm[-2], x : x + mm[-1]] = M2
    return MC

def MSEloss(A, B):
    "Compara el error cuadratico medio entre A y B, donde A y B son arreglos complejos"
    a = A.shape
    b = B.shape
    if a != b:
        raise ValueError('Las matrices deben ser de tamanios iguales')
    "Calcula el error cuadratico medio entre A y B"
    out = np.sum(np.abs(B - A) ** 2)
    out /= np.prod(B.shape)
    return out

def tfourier(x):
    "Transformada rápida de Fourier"
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def itfourier(x):
    "Transformada inversa rápida de Fourier"
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def propft(uin, *args):
    "Transformada de Fresnel -- Propagacion libre en el espacio una distancia z"
    Uin = np.fft.fft2(np.fft.fftshift(uin))
    z = args[0] #Distancia de propagacion
    wl = args[1] #Longitud de onda
    dx = args[2] #Tamano de pixeles
    m = args[3] #Numero de pixeles en cada lado
    Lx = m * dx
    fx = np.arange(-1/(2*dx),1/(2*dx),1/Lx)
    FX,FY = np.meshgrid(fx,fx)
    H = np.exp(-1j*wl*z*np.pi*(FX**2+FY**2))
    H = np.fft.fftshift(H)
    
    Uout = Uin * H
    uout = np.fft.ifftshift(np.fft.ifft2(Uout))
    return uout