import numpy as np

nx = 10
nGhost = 1
xmin = 0
xmax = 1.0


dX_wide = 2 * (xmax-xmin) / (2*nx - 9)
dX_small = dX_wide / 4
xGhost_start = xmin - dX_small * (np.flip(np.arange(nGhost))+0.5)
xFirst = xmin + (np.arange(3)+0.5)*dX_small
xCentral = xmin + xFirst[-1] + (np.arange(nx-6)+1/8+1/2)*dX_wide
xEnd = xmin + xCentral[-1] + (1/8+1/2)*dX_wide + dX_small*(np.arange(3))
xGhost_end = xmax + dX_small * (np.arange(nGhost)+0.5)

xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))

for n in range(1, nGhost +4):
    xWalls = xCentres[:n] - 0.5*dX_small
for n in range(nGhost + 4, nx-3+nGhost):
    xWalls = xCentres[:n] - 0.5*dX_wide
for n in range (nx-3, nx+2*nGhost+2):
    xWalls = xCentres[:n] - 0.5*dX_small

xWalls = np.append(xWalls, xCentres[nx+2*nGhost-1] + 0.5*dX_small)

xL = xWalls[1:nx+2*nGhost]
xR = xWalls[2:nx+2*nGhost+1]

print("xCentres = ", xCentres)
print("xWalls = ", xWalls)
print("xL = ", xL)
print("xR = ", xR)