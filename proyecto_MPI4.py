import numpy as np
from PIL import Image as im 
from mpi4py import MPI
from time import process_time
from utilidades import *
 
def DBS(Hf, division, rank):
    """Hf es la fase de un holograma complejo que genera la escena deseada
    division es el numero de secciones en los cuales se debe dividir cada lado"""
    sy, sx = Hf.shape

    HO = np.exp(1j * Hf * 2 * np.pi) #Calcula el holograma complejo
    C = tfourier(HO) #Propaga el holograma para generar la escena objetivo
    C = C[:, : sx // 2 - 5] #Toma una seccion a la izquierda de la escena objetivo

    Is = np.arange(0, HO.shape[1], HO.shape[1] // division) #Calcula los puntos en donde se deben dividir la secciones del holograma
    if HO.shape[0] % division != 0:
        Is = Is[:-1] #Modifica los puntos que definen la divisiones en caso de que el numero de divisiones solicitado no permita una particion exacta del holograma
    Is = np.append(Is, HO.shape[1]) 

    Ix0 = Is[rank % division]
    Iy0 = Is[rank // division]
    Ixf = Is[rank % division + 1]
    Iyf = Is[rank // division + 1] #Calcula las coordenadas de las esquinas de la seccion que cada procesador optimizara de acuerdo a su rango

    subx = Ixf - Ix0
    suby = Iyf - Iy0 #Calculo del tamano de la seccion a optimizar

    tajada = np.zeros((suby, subx))
    tajada[Hf[Iy0 : Iyf, Ix0 : Ixf] >= 0.5] = 1 #Binariza la fase del holograma original en la seccion respectiva y la toma como holograma de amplitud binaria. 
    #Esto proporciona una aproximacion de la escena objetio y lo tomamos como la aproximacion inicial que sera optimizado.

    H = fusion(HO, tajada, Ix0, Iy0) #Reemplaza la respectiva seccion del holograma complejo por el holograma binario de amplitud

    R_prev = tfourier(H) #Propaga el nuevo holograma
    R_prev = R_prev[:, : sx // 2 - 5] #Toma una seccion a la izquierda de la escena reconstruida

    MSE_prev = np.mean(np.abs(C - R_prev) ** 2) #Evalua el error cuadratico medio entre la escena objetivo y la escena reconstruida. Donde ambas son escenas complejas

    Indexes = np.arange(subx * suby) #Define los indices para acceder a los pixeles de la seccion a optimizar
    np.random.shuffle(Indexes) #Randomiza el orden en que se accede a los pixeles de la seccion
    
    for i in np.arange(len(Indexes)): #Por cada pixel de la seccion a optimizar
        Iy = Indexes[i] // suby + Iy0
        Ix = Indexes[i] % subx + Ix0 #Calcula las coordenadas del pixel a evaluar
    
        H[Iy][Ix] = not(H[Iy][Ix]) #Cambia el valor del pixel a evaluar

        R_new = tfourier(H) #Calcula la reconstruccion de la nueva escena reconstruida
        R_new = R_new[:, : sx // 2 - 5] #Toma una seccion a la izquierda de la nueva escena reconstruida
        MSE_new = np.mean(np.abs(C - R_new) ** 2) #Evalua el error cuadratico medio entre la escena objetivo y la neuva escena reconstruida. Donde ambas son escenas complejas
        if MSE_new <= MSE_prev: #Determina si el cambio realizado disminuyo el error cuadratico medio y, por tanto, mejoro la reconstruiccion de la escena
            MSE_prev = MSE_new #En caso afirmativo dejamos el cambio y nos quedamos con el nuevo error cuadratico medio
        else:
            H[Iy][Ix] = not(H[Iy][Ix]) #En caso contrario revertimos el cambio
            
    Hsave = H[Iy0 : Iyf, Ix0 : Ixf] #Toma la tajada optimizada
    Hsave = abs(Hsave) * 255 
    Hsave = im.fromarray(Hsave).convert('L')
    Hsave.save("".join(["Por recuadros/recuadros/", str(division), "x", str(division), "/", str(rank), "rank.png"])) #Guarda la tajada de acuerdo a su rango y al numero de divisiones que se hizo sobre el holograma original

if __name__ == "__main__":

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    HO = im.open("".join(["Imagenes originales/multiplane_hologram.bmp"])) #Carga la fase del holograma de fase cuya reconstruccion se desea obtener mediante hologramas binarios 
    HO = HO.convert('L')
    HO = np.asarray(HO)
    HO = HO / 255. #Se pone en un rango entre 0 y 1

    n_division = int(np.sqrt(world_size))

    t0 = process_time()
    DBS(HO, n_division, my_rank)
    tf = process_time() #Calcula el tiempo requerido
    print("El tiempo: ", str(tf - t0), "s")
    folder = "".join(["Recuadros/", str(n_division), "x", str(n_division), "/"])

    with open("".join("".join([folder, 'Tiempos.txt'])), 'a') as f:
        f.write("time: {:.1f} \n".format(tf - t0)) #Toma el tiempo requerido por cada procesador y lo registra en una archivo de texto