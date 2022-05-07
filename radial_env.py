import numpy as np
import pygame as pyg
import pyautogui as pya
import sys

pyg.init

white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

gameDisplay = pyg.display.set_mode((800,600))
gameDisplay.fill(black)

pixAr = pyg.PixelArray(gameDisplay)
pixAr[10:100][20:200] = green







