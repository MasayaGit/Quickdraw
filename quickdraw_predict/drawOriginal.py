# -*- coding:utf-8 -*-

from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tkinter
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from modelpredict import judgeAndPlt
# size
window_width = 400
window_height = 400
canvas_width = 400
canvas_height = 400
button_width = 5
button_height = 1
BAR_SPACE = 3
BAR_WIDTH = 30

num_classes = 5

#airplane
#apple
#cat
#flower
#bus

# color depth
draw_depth = int(10. / 100. * 255)  # 10%


class Scribble(object):

    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y

    def on_dragged(self, event):
        # draw surface canvas
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                                width=2,
                                tag="draw")

      
        self.draw.line(
            ((self.sx, self.sy), (event.x, event.y)),
            (draw_depth, draw_depth, draw_depth),8)

        # store the position in the buffer
        self.sx = event.x
        self.sy = event.y

    def judge(self):
        imagePIL = ImageOps.grayscale(self.image1)
        imageCV = np.asarray(imagePIL)
        judgeAndPlt(imageCV,self.resnet18,self.resnet34)
       
    def clear(self):
        # clear the surface canvas
        self.canvas.delete("draw")

        # clear(initialize) the hidden canvas
        self.image1 = Image.new(
            "RGB", (window_width, window_height), (255, 255, 255)
        )
        self.draw = ImageDraw.Draw(self.image1)


    def create_window(self):
        window = tkinter.Tk()

        # canvas frame
        canvas_frame = tkinter.LabelFrame(
            window, bg="white",
            text="canvas",
            width=window_width, height=window_height,
            relief='groove', borderwidth=4
        )
        canvas_frame.pack(side=tkinter.LEFT)
        self.canvas = tkinter.Canvas(canvas_frame, bg="white",
                                     width=canvas_width, height=canvas_height,
                                     relief='groove', borderwidth=4)
        self.canvas.pack()
        quit_button = tkinter.Button(canvas_frame, text="exit",
                                     command=window.quit)
        quit_button.pack(side=tkinter.RIGHT)
        
        judge_button = tkinter.Button(canvas_frame, text="judge",
                                      width=button_width, height=button_height,
                                     command=self.judge)
        judge_button.pack(side=tkinter.LEFT)
        
        clear_button = tkinter.Button(canvas_frame, text="clear",
                                      command=self.clear)
        clear_button.pack(side=tkinter.LEFT)
        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

       

        return window

    def __init__(self):

        #self.modelName = modelName

        self.window = self.create_window()

        # set canvas
        self.image1 = Image.new(
            "RGB", (window_width, window_height), (255, 255, 255)
        )
        self.draw = ImageDraw.Draw(self.image1)

        model18 = models.resnet18(pretrained=False)
        num_features =  model18.fc.in_features
        model18.fc = nn.Linear(num_features, num_classes)
        model18.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        model18.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        model34 = models.resnet34(pretrained=False)
        num_features = model34.fc.in_features
        model34.fc = nn.Linear(num_features, num_classes)
        model34.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        model34.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
       
        #Resnet18
        param18 = torch.load('quickdrawResnet18.model')
        self.resnet18 = model18.to('cpu')
        self.resnet18.load_state_dict(param18)

        #Resnet34
        param34 = torch.load('quickdrawResnet34.model')
        self.resnet34 = model34.to('cpu')
        self.resnet34.load_state_dict(param34)


    def run(self):
        self.window.mainloop()
    

def main():
    Scribble().run()


if __name__ == '__main__':
    main()
