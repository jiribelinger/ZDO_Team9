#import of libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import scipy.signal
import skimage.exposure
import skimage.filters
import skimage.io
import skimage.color
import skimage.measure
import skimage.morphology
from skimage.morphology import skeletonize, binary_erosion, binary_dilation, binary_closing, binary_opening, rectangle
import statistics
import sys


#image rescale according to min and max value (coef - ratio of lower values reduction)
def rescaling(im,coef):
  mn=np.min(im)
  mx=np.max(im)
  im= skimage.exposure.rescale_intensity(im, in_range=(mn+(mx-mn)*coef, mx), out_range=(0, 1))

  return im


#frequency (fourier) transformation with given mask
def fourier(im,mask):
  ft = np.fft.fft2(im)
  ft_amp = np.abs(ft)
  ftshift = np.fft.fftshift(ft)
  ftshift_mask = ftshift*(mask)
  ftishift_back = np.fft.ifftshift(ftshift_mask)
  im = np.fft.ifft2(ftishift_back)
  im = np.abs(im)

  return im


#automatic threshold calculation using image statistics (coef - std tolerance, ori - orientation of thresholding - upper/lower )
def threshold(im,coef,ori):
  mhigh=statistics.median_high(im.ravel())
  pstdev=statistics.pstdev(im.ravel())
  if ori=='l':
    im=im<mhigh-pstdev*coef
  elif ori=='u':
    im=im>mhigh+pstdev*coef
  else:
    im=im

  return im


#reducing image to maximal nonzero extent
def focus(im):
  x, y = np.nonzero(im)
  xl,xr = x.min(),x.max()
  yl,yr = y.min(),y.max()
  im=im[xl:xr, yl:yr].astype('float64')
    
  return im,xl,xr,yl,yr


#relative cropping of image (vertical/horizontal)
def crop(im,x,y):
  #zero - no cropping, otherwise 1/x of full extent
  if x==0:
    marginx=0
  else:
    marginx=round(im.shape[0]/x)
  if y==0:
    marginy=0
  else:
    marginy=round(im.shape[1]/y)
  im=im[0+marginx:im.shape[0]-marginx,0+marginy:im.shape[1]-marginy]

  return im


#incision segmentation
def incision(im):
  #horizontal filtering
  im=scipy.ndimage.prewitt(im,0)

  im=rescaling(im,0)
  im=threshold(im,0.75,'u')

  #joining of possibly disrupted lines
  krn = rectangle(1,round(im.shape[0]/10)+1)
  im = binary_closing(im, krn)

  return im


#distinction of intersections between incision and stitches
def intersections(im):
  ver=im
  hor=im

  #suppression of horizontal (ver) and vertical (hor) lines
  #mv,mh - vertical/horizontal convolution kernels
  mv = np.array([[1],[1],[1],[1],[1]])
  mh = np.array([[1,1,1,1,1]])
  for i in range(10):
    ver=scipy.signal.convolve2d(ver, mv, mode='same', boundary='symm')
    hor=scipy.signal.convolve2d(hor, mh, mode='same', boundary='symm')

  ver=threshold(ver,1,'l').astype('float64')
  ver=skeletonize(ver)
  hor=threshold(hor,1,'l').astype('float64')
  hor=skeletonize(hor)

  #reduction of vertical lines extension caused by convolution
  krne = rectangle(round(ver.shape[0]/30)+1,1)
  ver = binary_erosion(ver, krne)

  ver=skimage.filters.gaussian(ver,2)
  hor=skimage.filters.gaussian(hor,2)

  #aditional frequency filtration of horizontal and vertical lines
  x, y = np.indices(ver.shape)
  mask = abs((x-ver.shape[0]/2)) <= 0.5*abs(y-ver.shape[1]/2)+1
  ver=fourier(ver,mask)

  x, y = np.indices(hor.shape)
  mask = abs((y-hor.shape[1]/2)) <= 0.5*abs(x-hor.shape[0]/2)+1
  hor=fourier(hor,mask)

  #overlay of vertical and horizontal lines (higher values in intersections)
  com=ver+hor

  #supressing non overlaying parts
  com=rescaling(com,0.5)

  #residual horizontal lines filtered out (vertical lines convenient)
  x, y = np.indices(com.shape)
  mask = abs((x-com.shape[0]/2)) <= 0.5*abs(y-com.shape[1]/2)+1
  com=fourier(com,mask)

  com=skimage.filters.gaussian(com,1)
  com=rescaling(com,0)

  return com


#identification of thin stitches
def thin_stitches(im):
  im=threshold(im,1.25,'l').astype('float64')
  im=skeletonize(im)

  #suppression of horizontal lines
  mvs = np.array([[1],[1]])
  im=scipy.signal.convolve2d(im, mvs, mode='same', boundary='symm')

  im=skimage.filters.gaussian(im,1)

  #filtration of vertical lines
  x, y = np.indices(im.shape)
  mask = abs((x-im.shape[0]/2)) <= 0.5*abs(y-im.shape[1]/2)
  im=fourier(im,mask)

  im=rescaling(im,0)

  return im


#basic contours of stitches
def contours(im):
  #frequency filtration (vertical)
  x, y = np.indices(im.shape)
  mask = abs((x-im.shape[0]/2)) <= 0.5*abs(y-im.shape[1]/2)+1
  im=fourier(im,mask)

  #gradient filtration (vertical)
  im=scipy.ndimage.prewitt(im,1)
  im=scipy.ndimage.gaussian_gradient_magnitude(im, 0.25)
  im=rescaling(im,0)

  return im


#function for detection and visualisation of stitches
def main(output,vis,content,path,folder):
  cnt=np.zeros(len(content)).astype('int32')
  results=[]

  #iteration trough all given images
  for k in range(len(content)):
    filepath = os.path.join(path,content[k])
    img = skimage.io.imread(filepath)
    imgg = skimage.color.rgb2gray(img)
    
    #copy of image with original size for visualization
    ims=imgg
    
    #identification of incision lines
    cut=incision(imgg)

    cut,xl,xr,yl,yr=focus(cut)
    
    cut=skeletonize(cut)
    cutlabel = skimage.measure.label(cut, background=0)
    cutprops = skimage.measure.regionprops(cutlabel)

    pole=[]

    #horizontal extent of major incision line
    for i in range(len(cutprops)):
      pole.append(cutprops[i].bbox[3]-cutprops[i].bbox[1])

    cutlen=np.max(pole)
    
    #path for storing of visualised figure (for both usable and unusable images)
    fig=os.path.join(folder,content[k])

    #decision - incision presence and sufficient contrast (if not - unusable image)
    if cutlen>img.shape[1]/3 and (np.max(imgg)-np.min(imgg))>0.2:
      
      #identification of incision/stitches intersections
      cross=intersections(imgg)
    
      #identification of thin stitches
      thin=thin_stitches(imgg)

      #reducing image heights (relative reduction)
      #denominators (fine,rough reduction)
      denomf=50
      denomr=10
      
      #upper,lower,left,right - margin sizes for restoration of original image size (before focusing/cropping)
      upper=round(imgg.shape[0]/denomf)
      lower=-round(imgg.shape[0]/denomf)
      
      cross=crop(cross,denomf,0)
      thin=crop(thin,denomf,0)
      imgg=crop(imgg,denomf,0)
      
      left=yl
      right=-(imgg.shape[1]-yr)

      #reducing image widths to incision extent
      cross=cross[0:cross.shape[0], yl:yr]
      thin=thin[0:thin.shape[0], yl:yr]
      imgg=imgg[0:imgg.shape[0], yl:yr]
      
      #identification of basic cntours of stitches
      con=contours(imgg)

      #combination of segmented contours (higher priority-double value), intersections and thin lines
      stitches=2*con+cross+thin

      stitches=rescaling(stitches,0)
      stitches=threshold(stitches,1.25,'u')
      
      upper=upper+round(stitches.shape[0]/denomr)
      lower=lower-round(stitches.shape[0]/denomr)
      left=left+round(stitches.shape[1]/denomf)
      right=right-round(stitches.shape[1]/denomf)

      #reduction of margins
      stitches_crop=crop(stitches,denomr,denomf)

      stitches,xl,xr,yl,yr=focus(stitches_crop)

      upper=upper+xl
      lower=lower-(stitches_crop.shape[0]-xr)
      left=left+yl
      right=right-(stitches_crop.shape[1]-yr)
   
      stitches=skeletonize(stitches)

      #detection of individual objects
      stlabel = skimage.measure.label(stitches, background=0)
      stprops = skimage.measure.regionprops(stlabel)
      
      #identification of assumed stitches (according to object vertical extent)
      cnt[k]=0
      obj=[]
      for i in range(len(stprops)):
        if ((stprops[i].bbox[2]-stprops[i].bbox[0]))>stitches.shape[0]/2:
          #saving of object ID
          obj.append(i+1)
          #stitches counting
          cnt[k]=cnt[k]+1

      print(cnt[k])

      text=(content[k],cnt[k])
      results.append(text)

      #deletion of residual objects in image
      obj=set(obj)

      for i in range(stlabel.shape[0]):
        for j in range(stlabel.shape[1]):
            if (stlabel[i,j] in obj):
              stlabel[i,j]=stlabel[i,j]
            else:
              stlabel[i,j]=0
    
      #reshaping output image
      ims[0:ims.shape[0],0:ims.shape[1]]=0
      ims[upper:ims.shape[0]+lower,left:ims.shape[1]+right]=stlabel[:,:]

      stcol = skimage.color.label2rgb(ims)

      #visualisation (usable image)
      if vis:
        stnum=str(cnt[k])
        filename=str(content[k])
        plt.figure()
        plt.subplot(211)
        plt.title(filename)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(stcol)
        plt.xlabel("Number of stitches: "+''+stnum)
        plt.savefig(fig)
      else:
        continue

    #incision not found/too low contrast of image
    else:
      #count set to value for "unknown" number of stitches
      cnt[k]=-1
      print(cnt[k])

      text=(content[k],cnt[k])
      results.append(text)

      #visualisation (unusable image)
      if vis:
        filename=str(content[k])
        plt.figure()
        plt.subplot(211)
        plt.title(filename)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(cut, cmap='gray')
        plt.xlabel("Unusable image")
        plt.savefig(fig)
      else:
        continue
        
  #save results to csv file
  output = os.path.join(folder,output)
  df = pd.DataFrame(results, columns = ["filename", "n_stitches"])
  df.to_csv(output, index=False)

if __name__ == "__main__":
  #relative path to images
  path="images"

  #folder name for storing of results (csv,visualisation)
  folder="results"

  dir=os.path.exists(folder)

  if dir:
    print("Folder already exists")
  else:
    os.mkdir(folder)

  #reading of arguments
  output=sys.argv[1]
  
  #decision - visualise?
  if sys.argv[2] == "-v":
    vis=True
    #image names
    content=sys.argv[3:]
  else:
    vis=False
    content=sys.argv[2:]

  main(output,vis,content,path,folder)