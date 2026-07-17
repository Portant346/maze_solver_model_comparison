import fileinput
import numpy as np
import cv2
import datetime

def str2img(d):
  imglist=[]
  charlist=list(d)
  j=1
  while j < len(charlist)-1:
    c=charlist[j]
    if c == '\\':
      n1=charlist[j+1]
      n2=charlist[j+2]
      n3=charlist[j+3]
      if n1<'0' or n1>'7' or n2<'0' or n2>'7' or n3<'0' or n3>'7':
        val = 0
      else:
        val = int(n1)*8*8+int(n2)*8+int(n3)
      j=j+3
    else:
      val = ord(c)
    imglist.append(val)
    j=j+1

  return imglist

def bgr2rgb(imga):
  imgarray=np.zeros(imga.shape,dtype=np.uint8)
  imgarray[:,:,0] = imga[:,:,2]
  imgarray[:,:,1] = imga[:,:,1]
  imgarray[:,:,2] = imga[:,:,0]
  return imgarray

def process_image(imgarray):
  # Show image in window
  cv2.imshow("image",imgarray)
  cv2.waitKey(500)
  return 0

  # Save image
  # now=datetime.datetime.now()
  # fn="image"+str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)+str(now.microsecond)+".png"
  # cv2.imwrite(fn,imgarray)
  # return 0

  # Classify image
  # - Pre-process image 
  # - Call classifier
  # - return class

for line in fileinput.input():

  # Line to BGR image
  tokens=line.split()
  for i in range(len(tokens)):
    t=tokens[i]
    if t == "width:":
      w = int(tokens[i+1])
    if t == "height:":
      h = int(tokens[i+1])
    if t == "data:":
      d = tokens[i+1]
      imglist=str2img(d)
      if h*w*3 == len(imglist):
        imga=np.array(imglist,dtype=np.uint8).reshape(h,w,3)
      else:
        imga=np.zeros((h,w,3),dtype=np.uint8)
      # Channel exchange (BGR -> RGB)
      imgarray=bgr2rgb(imga)

      # Process image
      cl = process_image(imgarray)

      # Process classification result
      # ...