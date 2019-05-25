import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from pygame import *
from math import sqrt

mnist = tf.keras.datasets.mnist.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist
trainImages, testImages = trainImages/255, testImages/255

retrain = False

def newModel():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(784, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)    
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train(nn):
    model.fit(trainImages, trainLabels, epochs=30, batch_size=128)
    model.save("digit_classifier_NN")

# return true if a point (x,y) is within a rectangle (x,y,w,h)
def inRect(pos, rect, inclusive=True):
    x, y = pos
    rX, rY, rW, rH = rect
    if inclusive:
        return rX <= x <= rX+rW and rY <= y <= rY+rH
    else:
        return rX < x < rX+rW and rY < y < rY+rH

if retrain:
    model = newModel()
    train(model)
else:
    model = tf.keras.models.load_model("digit_classifier_NN")

init()
SIZE = 448,600
screen = display.set_mode(SIZE)
clock = time.Clock()

gridW, gridH = 28,28
pixelSize = 16
pixelSpacing = 0

canvas = Surface((gridW*pixelSize, gridH*pixelSize))

BLACK = 0,0,0
WHITE = 255,255,255
RED = 255,0,0
GREEN = 0,255,0

class Pixel:
    def __init__(self, x=0, y=0):
        self.x, self.y = x,y
        self.rect = x*pixelSize + x*pixelSpacing, y*pixelSize + y*pixelSpacing, pixelSize, pixelSize
        self.brightness = 0
    def draw(self):
        if self.brightness > 255:
            self.brightness = 255
        colour = self.brightness, self.brightness, self.brightness
        draw.rect(canvas, colour, self.rect)

class IndexBar:
    def __init__(self, x=0, y=0, index=0, amount=0, maxSizePixels=100, w=20):
        self.x, self.y = x, y
        self.amount = 100
        self.rect = Rect(x,y,w,(self.amount/100)*maxSizePixels)
        self.width = w
        self.maxSizePixels = maxSizePixels
        
        self.colour = RED
        self.index = index
        self.text = Text(self.rect, str(self.index), 32, BLACK)
    def update(self):
        self.rect[3] = (self.amount/100)*self.maxSizePixels
    def draw(self):
        draw.rect(screen, self.colour, self.rect)
        self.text.draw()

# faster and easier
class Text:
    def __init__(self, rect, text, fontSize=14, colour=(0,0,0)):
        self.rect = rect
        self.font = font.SysFont("lucida console", fontSize) # initialize font
        self.text = text
        self.textImg = self.font.render (text,False,colour)
        self.colour = colour
        
    def update (self, newText=""):
        self.text = newText
        self.textImg = self.font.render (newText,1, self.colour)  
    def draw(self):
        screen.blit(self.textImg, self.rect)


def dist (p1, p2):
    diffX = p2[0] - p1[0]
    diffY = p2[1] - p1[1]
    return sqrt(diffX ** 2 + diffY ** 2)

def getHovered ():
    for col in pixels:
        for pixel in col:
            if inRect((mouseX, mouseY), pixel.rect):
                return pixel

def getValueArray (): # converts surface into 2D array of pixel values
    values = []
    for x in range (gridW):
        values.append([])
        for y in range (gridH):
            values[x].append(pixels[y][x].brightness/255)
    return np.asarray([values], dtype="float64")

def brush (brushSize=2): # draws at mouse pos
    
    for x in range (hovered.x-brushSize//2, hovered.x+brushSize//2, 1):
        for y in range (hovered.y-brushSize//2, hovered.y+brushSize//2, 1):
            if 0 <= x <= gridW-1 and 0 <= y <= gridH-1:
                
                d = dist((x,y), (hovered.x,hovered.y))
                if d == 0:
                    pixels[x][y].brightness += 255
                else:
                    pixels[x][y].brightness += 255/d

def resetPixels():
    for col in pixels:
        for pixel in col:
            pixel.brightness = 0

def updateBars ():
    for i in range (len(bars)):
        
        if i == np.argmax(prediction):
            bars[i].colour = GREEN
        else:
            bars[i].colour = RED
        
        bars[i].amount = prediction[i]*100
        bars[i].update()

def drawBars():
    for bar in bars:
        bar.draw()


pixels = [[Pixel(x,y) for y in range (gridH)] for x in range (gridW)]

bars = [IndexBar(i*45,448,i) for i in range (10)]
print(len(bars))

mouseDrag = False
hovered = None
run = True
while run:
    screen.fill(WHITE)
    mouseX, mouseY = mouse.get_pos()
    hovered = getHovered()
    
    for col in pixels:
        for pixel in col:
            pixel.draw()
    screen.blit(canvas, (0,0))
    
    prediction = model.predict(getValueArray())[0]
    if mouseDrag and hovered != None:
        brush()
        updateBars()
    drawBars()

    for e in event.get():
        if e.type == MOUSEBUTTONDOWN:
            if e.button == 1:
                mouseDrag = True
            elif e.button == 3:
                resetPixels()
        elif e.type == MOUSEBUTTONUP:
            mouseDrag = False
        elif e.type == QUIT:
            run = False
    display.update()