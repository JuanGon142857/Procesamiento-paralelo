"Este algoritmo toma los recuadros optimizados anteriormente, los pega para obtener el holograma binario completo, propaga el holograma y evalua los resultados obtenidos comparando con las iamgenes originales"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im 
from utilidades import *
import matplotlib.pyplot as plt

wl = 532e-9 #Longitud de onda
dx = 8e-6 #Tamano de los pixeles del holograma en metros
m = 512 #Numero de pixeles en cada lado
L = m * dx
zc = L * dx / wl
z_apart = zc * 0.2 #Distancia de propagacion para cada plano del holograma multiplano
print("La distancia de propagacion entre planos es " + str(z_apart), "m")
planes = 4
s_objeto = 100 #Tamanho objeto

divisiones_evaluadas = np.arange(1, 6)

Holograma_completo = np.zeros((m, m)) 
Pupsize = 10
Pup = fusion(np.ones((m, m)), np.zeros((Pupsize, Pupsize)), (Holograma_completo.shape[0] - Pupsize) // 2, (Holograma_completo.shape[1] - Pupsize) // 2) #Funcion pupila para retirar el orden central que se forma al reconstruir un holograma de amplitud binaria

Imagenes_originales = np.zeros((planes, s_objeto, s_objeto), dtype = np.complex64)
MSEs = np.zeros((planes, len(divisiones_evaluadas)))
MSEs_complex = np.zeros((planes, len(divisiones_evaluadas)))

W = im.open("".join(['Imagenes originales/multiplane_hologram.bmp'])) #Carga el holograma complejo de la escena que se uso como la escena objetivo
W = W.convert('L')
W = np.asarray(W) / 255.
W = np.exp(1j * 2 * np.pi * W)
W = tfourier(W) #Reconstruye la escena objetivo a partir del holograma

for i in np.arange(planes):
    Imagenes_originales[i] = W[(m - s_objeto) // 2  : (m + s_objeto) // 2, (m // 2 - s_objeto) // 2: (m // 2 + s_objeto) // 2] #Recorta los objetos en el campo objetivp
    W = propft(W, z_apart, wl, dx, m) #Propaga el campo una distancia z_apart para llegar al plano donde se forma la siguiente imagen 

for division in divisiones_evaluadas:
    folder = "".join([str(division), "x", str(division)])

    for rank in np.arange(division ** 2):  
        Is = np.arange(0, m, m // division) #Define las coordenadas de las esquina de las secciones del holograma que se optimizo en cada procesador
        if m % division != 0: #Modifica los puntos que definen la divisiones en caso de que el numero de divisiones solicitado no permita una particion exacta del holograma
            Is = Is[:-1]
        Is = np.append(Is, m)

        Ix0 = Is[rank % division]
        Iy0 = Is[rank // division]
        Ixf = Is[rank % division + 1]
        Iyf = Is[rank // division + 1] #Calcula las coordenadas de las esquinas de la seccion que cada procesador optimizara de acuerdo a su rango

        I = im.open("".join(['Recuadros/', folder, "/", str(rank), 'rank.png'])) #Carga cada seccion optimizada del holgorama
        I = I.convert('L')
        I = np.asarray(I) / 255.
        Holograma_completo[Iy0: Iyf, Ix0: Ixf] = I #Coloca la seccion optimizada en su respectiva posicion

    F = tfourier(Holograma_completo) #Reconstruye la escena a partir del holograma
    F = F * Pup #Elimina el orden central que se forma a partir de hologramas de amplitud binaria
    for i in np.arange(planes):
        Hsave = abs(F[(m - s_objeto) // 2  : (m + s_objeto) // 2, (m // 2 - s_objeto) // 2: (m // 2 + s_objeto) // 2]) #Recorta los objetos en el campo reconstruido
        MSEs_complex[i, division- 1] = MSEloss(Hsave, Imagenes_originales[i]) #Calcula el error cuadratico medio teniendo en cuenta el objeto complejo

        Hsave = np.abs(Hsave)
        MSEs[i, division- 1] = MSEloss(Hsave, abs(Imagenes_originales[i])) #Calcula el error cuadratico medio teniendo en cuenta el objeto complejo
        Hsave = Hsave / np.max(Hsave)
        Hsave = Hsave * 255
        Hsave = np.asarray(Hsave)
        Hsave = im.fromarray(Hsave).convert('L')
        Hsave.save("".join(["Resultados/",str(i + 1), "Plano_", folder, ".png"]))
        F = propft(F, z_apart, wl, dx, m)

print(np.transpose(MSEs) / 10e5)

plt.figure(1, figsize = (12,9))
plt.rcParams.update({'font.size': 22})
plt.plot(np.arange(1, planes + 1), MSEs, label = ["1x1 bloque", "2x2 bloques", "3x3 bloques", "4x4 bloques", "5x5 bloques"], marker = "o", linewidth = 3, linestyle = "dashed")
plt.xticks([1,2,3,4])
plt.title("Reconstruccion del objeto real")
plt.xlabel("Plano")
plt.ylabel("MSE")
plt.legend()
plt.savefig("Resultados/MSE_objetoreal.png")
plt.show()

plt.figure(2, figsize = (12,9))
plt.rcParams.update({'font.size': 22})
plt.plot(np.arange(1, planes + 1), MSEs_complex, label = ["1x1 bloque", "2x2 bloques", "3x3 bloques", "4x4 bloques", "5x5 bloques"], marker = "o", linewidth = 3, linestyle = "dashed")
plt.xticks([1,2,3,4])
plt.title("Reconstruccion del objeto complejo")
plt.xlabel("Plano")
plt.ylabel("MSE")
plt.legend()
plt.savefig("Resultados/MSE_objetocomplejo.png")
plt.show()