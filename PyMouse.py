import pyautogui
import random
import time

scrollSpeed = 20
pyautogui.FAILSAFE = False

def moveTo(x, y, duration=0):
    pyautogui.moveTo(x, y, duration)

def getMousePosition():
    return pyautogui.position()

def getScreenSize():
    return pyautogui.size()

def scrollUp():
    pyautogui.scroll(scrollSpeed)
    #print('Scrolling up')

def scrollDown():
    pyautogui.scroll(-scrollSpeed-1)
    #print('Scrolling down')

def singleLeftClick():
    pyautogui.mouseDown(button='left')
    time.sleep(random.randint(20, 81)/1000)
    pyautogui.mouseUp(button='left')
    #print('singleLeftClick at', getMousePosition())

def doubleLeftClick():
    singleLeftClick()
    time.sleep(random.randint(30, 91)/1000)
    singleLeftClick()
    #print('doubleLeftClick at', getMousePosition())