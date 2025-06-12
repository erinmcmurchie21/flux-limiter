import numpy as np


def create_non_uniform_grid():
    nx = 30
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

    xWalls_s = np.arange(-nGhost,4)*dX_small
    xWalls_m = xWalls_s[-1] + np.arange(1,nx-5)*dX_wide
    xWalls_e = xWalls_m[-1] + np.arange(1,3+nGhost+1)*dX_small
    xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))
    deltaZ = xWalls[1:nx+2*nGhost+1] - xWalls[:nx+2*nGhost]

    column_grid = {
        "num_cells": nx,
        "nGhost": nGhost,
        "zCentres": xCentres,
        "zWalls": xWalls,
        "deltaZ": deltaZ
    }

    return column_grid