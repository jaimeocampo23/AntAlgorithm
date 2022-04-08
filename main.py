import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import pickle
from numpy import random
from tkinter import *
import PIL.ImageTk
import tkinter

with open("nodosmetro.dat", "rb") as f:
    nodos = pickle.load(f)
with open("distancias.dat", "rb") as f:
    dist = pickle.load(f)
with open("hormonas.dat", "rb") as f:
    hormonas = pickle.load(f)

nom = np.load('namestations.npy')
coordenadas = np.load('coordstations.npy')
coordenadas = np.array(coordenadas)


def actualizacionhormona(camino, disttotal):
    olvido = 0.1
    for i in range(1, len(nom)):
        name = nom[i]
        r = 0
        for t in range(np.size(camino) - 1):
            est1 = camino[t]
            if name == camino[t]:
                est1 = camino[t]
                est = nodos[est1]
                hl = hormonas[est1][:]
                for ha in range(np.size(hl)):
                    if est[ha] == camino[t + 1]:

                        ests = est.index(est[ha])
                        hormonas[est1][ests] = (1 - olvido) * hormonas[est1][ests] + 1 / disttotal
                        r = 1
                    else:

                        hormonas[est1][ha] = (1 - olvido) * hormonas[est1][ha]

        if r == 0:
            kl = np.array(hormonas[name][:])
            hormonas[name][:] = (1 - olvido) * kl


def leftclick(event):
    print("left")
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))
    pixel = np.array([[y, x]])
    euclx = coordenadas[:, 0] - pixel[0, 0]
    euclx = pow(euclx, 2)
    eucly = coordenadas[:, 1] - pixel[0, 1]
    eucly = pow(eucly, 2)
    eucl = euclx + eucly
    eucl = pow(eucl, 1 / 2)
    mini = np.min(eucl)
    posmin = np.argwhere(eucl == mini)

    station = nom[posmin]
    st = str(station[0][0])
    print('estacion inicio', st)
    prompt = f'estacion inicio {st}'
    L1.config(text=prompt)
    np.save('st1', st)


def rightclick(event):
    print("right")
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))
    #######estacion destino
    pixel2 = np.array([[y, x]])
    euclx = coordenadas[:, 0] - pixel2[0, 0]
    euclx = pow(euclx, 2)
    eucly = coordenadas[:, 1] - pixel2[0, 1]
    eucly = pow(eucly, 2)
    eucl = euclx + eucly
    eucl = pow(eucl, 1 / 2)
    mini = np.min(eucl)
    posmin = np.argwhere(eucl == mini)
    station2 = nom[posmin]
    st2 = str(station2[0][0])
    print('estacion final', st2)

    #################################
    ##########estacion siguiente#########        
    st = str(np.load('st1.npy'))
    a = nodos[st]
    D = dist[st]
    distancias = np.array(dist[st])
    hormona = np.array(hormonas[st])
    sz = np.size(a)
    vis = 1 / distancias
    # mult1=vis*hormona
    ab = 2

    mult1 = (pow(np.array(vis), 1)) * (pow(np.array(hormona), ab))
    sum1 = sum(mult1)
    P = mult1[:] / sum1
    prompt = f'estacion final {st2}'
    L2.config(text=prompt)
    # L2 = Label(window, text=prompt, width=len(prompt))
    # L2.pack(expand=True,fill = BOTH,side=TOP)
    Vdist = []
    Vest = {}

    y = 0
    for i in range(150):  ### hormigas

        random1 = random.rand(1)
        minimo = 0
        for t in range(sz):
            if minimo < random1 and P[t] + minimo > random1:
                pos = t
                # print(pos)
                break
            else:
                minimo = P[t] + minimo
        sig = a[pos]
        D1 = D[pos]
        camino = [st, sig]
        camdist = [D1]
        p = 1
        anterior = st
        while p == 1:  ######para definir cuando llega
            if st2 == sig:
                p = 2
                # print('fail1')
                break
            sign = nodos[sig][:]
            sigd = dist[sig][:]
            sigf = hormonas[sig][:]

            posanterior = sign.index(anterior)

            sign.pop(posanterior)
            sigd.pop(posanterior)
            sigf.pop(posanterior)

            if sigd:
                p = 1
            else:
                # si tiene una trayectoria en la que llega a donde ya no puede avanzar
                p = 3
                break
            #########calculo de P###########
            sigd = np.array(sigd, dtype=float)
            sz1 = np.size(sigd)
            vis2 = 1 / sigd[:]
            # ab=6
            mult2 = (pow(np.array(vis2), 1)) * (pow(np.array(sigf), ab))
            sum2 = sum(mult2)
            P2 = mult2[:] / sum2
            random1 = random.rand(1)
            minimo = 0
            #########definimos estacion siguiente######
            for j in range(sz1):
                if minimo < random1 and P2[j] + minimo > random1:
                    pos = j
                    break
                else:
                    minimo = P2[j] + minimo
            anterior = sig
            sig = sign[pos]
            D2 = sigd[pos]
            camino.append(sig)
            camdist.append(D2)
            #########################################
        ###############si el camino es valido y llego a la estacion final##########
        if p == 2:
            disttotal = sum(camdist)

            ####almacenamos trayectoria de cada hormiga
            Hn = f'hormiga {y}'
            Vest[Hn] = [camino]
            Vdist.append(disttotal)
            y = y + 1

            #####modificar hormonas de trayecto recorrido
            actualizacionhormona(camino, disttotal)
            #####modificar hormonas de no recorridos

    ##################camino minimo####################
    distmin = np.min(Vdist)
    posdist = np.argwhere(distmin == Vdist)
    Hnm = f'hormiga {posdist[0][0]}'
    Hg = Vest[Hnm]
    prompt = (f'{Hg}')
    L4.config(text=prompt)
    prompt = (f'distancia minima recorrida{distmin}')
    L5.config(text=prompt)
    print('el camino mas corto es', Hg)
    print('distancia minima recorrida', distmin)

    ####camino mas visitado
    repeticiones = 0
    for i in Vdist:
        apariciones = Vdist.count(i)
        if apariciones > repeticiones:
            repeticiones = apariciones

    modas = []
    for i in Vdist:
        apariciones = Vdist.count(i)
        if apariciones == repeticiones and i not in modas:
            modas.append(i)

    print("moda:", modas)

    for i in modas:
        posdist = np.argwhere(i == Vdist)
        Hnm = f'hormiga {posdist[0][0]}'
        Hg = Vest[Hnm]
        prompt = f'{Hg}'
        L7.config(text=prompt)
        prompt = f'por {np.size(posdist)} hormigas'
        L8.config(text=prompt)
        print('el camino mas recorrido es', Hg)
        print(f'por {np.size(posdist)} hormigas')


# Create a window
window = tkinter.Tk()
window.title("Mapa metro")
# Load an image using OpenCV
cv_img = cv2.cvtColor(cv2.imread('MapaMetro.bmp'), cv2.COLOR_BGR2RGB)

# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = cv_img.shape

# Create a canvas that can fit the above image
canvas = tkinter.Canvas(window, width=width, height=height)
canvas.pack(side=LEFT)

# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))

# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

prompt = f'click izquierdo para seleccionar estacion de inicio'
L12 = Label(window, text=prompt, width=len(prompt))
L12.pack(expand=True, fill=BOTH, side=TOP)
L12.config(fg="red")
prompt = f'estacion inicio'
L1 = Label(window, text=prompt, width=len(prompt))
L1.pack(expand=True, fill=BOTH, side=TOP)
prompt = f'click derecho para seleccionar estacion final'
L13 = Label(window, text=prompt, width=len(prompt))
L13.pack(expand=True, fill=BOTH, side=TOP)
L13.config(fg="red")
prompt = f'estacion final'
L2 = Label(window, text=prompt, width=len(prompt))
L2.pack(expand=True, fill=BOTH, side=TOP)
prompt = f'el camino mas corto es:'
L3 = Label(window, text=prompt, width=len(prompt))
L3.pack(expand=True, fill=BOTH, side=TOP)
prompt = f'...'
L4 = Label(window, text=prompt, width=len(prompt), wraplength=250)
L4.pack(expand=True, fill=BOTH, side=TOP)
L4.config(fg="green")
prompt = f'distancia minima recorrida:'
L5 = Label(window, text=f'distancia minima recorrida:', width=len(prompt))
L5.pack(expand=True, fill=BOTH, side=TOP)
prompt = f'el camino mas recorrido es'
L6 = Label(window, text=prompt, width=len(prompt))
L6.pack(expand=True, fill=BOTH, side=TOP)
prompt = f'...'
L7 = Label(window, text=prompt, width=len(prompt), wraplength=250)
L7.pack(expand=True, fill=BOTH, side=TOP)
L7.config(fg="green")
prompt = f'...'
L8 = Label(window, text=prompt, width=len(prompt))
L8.pack(expand=True, fill=BOTH, side=TOP)
window.bind("<Button-1>", leftclick)
window.bind("<Button-3>", rightclick)

# Run the window loop
window.mainloop()