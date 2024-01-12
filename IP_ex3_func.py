import numpy as np

def addBorder(img, br, bc):
    row,col = img.shape
    imgOut = np.zeros((row +2*br,col+2*bc),np.uint8)
    r = 0
    c = 0
    for px in np.nditer(imgOut[:,:], op_flags=["readwrite"]):
        rI = br - r
        cI = bc - c
        if rI < 0:
            rI = r - br 
        if rI >= row:
            rI = row - (rI - row) - 2
        if cI < 0:
            cI = c - bc
        if cI >= col:
            cI = col - (cI - col) - 2
        px[...] = img[rI,cI]
        c+=1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut

def convolution(img,kernel):  
    rows,cols = kernel.shape
    rI,cI = img.shape
    imgOut = np.zeros((rI,cI),np.float32)
    startC = int(cols/2 )
    startR = int(rows/2 )
    imgBorder = addBorder(img, startR, startC)
    imgBorder.astype(np.float32)
    r = 0
    c = 0
    for pxOut in np.nditer(imgOut[:,:], op_flags =["writeonly"]):
        it = np.nditer([imgBorder[r : r+2*startR+1 , c : c+2*startC+1],kernel[:,:]],flags=["buffered","external_loop"],op_flags =["readonly"], op_dtypes=["float64","float64"])
        val = 0.0
        for i,k in it:
            val += np.sum(i*k)
        pxOut[...] = val
        c+= 1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut

def sobel(img):
    ksx = np.zeros((3,3),np.float32)  
    #        [-1 0 1]
    # ksx =  [-2 0 2]
    #        [-1 0 1]
    ksx[0,0] = -1.0
    ksx[1,0] = -2.0
    ksx[2,0] = -1.0
    ksx[0,2] = 1.0
    ksx[1,2] = 2.0
    ksx[2,2] = 1.0
    ksy = np.zeros((3,3),np.float32)
    #       [-1 -2 -1]
    # ksy = [0   0   0]
    #       [1   2   1]
    ksy[0,0] = -1.0
    ksy[0,1] = -2.0
    ksy[0,2] = -1.0
    ksy[2,0] = 1.0
    ksy[2,1] = 2.0
    ksy[2,2] = 1.0
    Sx = convolution(img,ksx)
    Sy = convolution(img,ksy)
    Sdir = np.zeros(img.shape,np.float64)
    Smag = np.zeros(img.shape,np.float64)
    for x,y,d,m in np.nditer([Sx[:,:],Sy[:,:],Sdir[:,:],Smag[:,:]],op_flags=["readwrite"],op_dtypes=["float32","float32","float64","float64"]):
        # if x==0 and y ==0:
        #d[...] = -10.0
        # else: 
        d[...] = np.arctan2(y,x)
        # output of d :
        #    ++++++++
        #  0 <- x -> +-Pi
        #    --------  
        m[...] = np.sqrt(x*x + y*y)
    return Sx,Sy,Sdir,Smag


def non_max_suppresion(Sdir,sMag, size=4):
    """
    creates a binary image of break lines.
    input is the Sobel direction and magnitude image from the Sobel method
    """
    out = np.zeros(sMag.shape,np.float32)
    rows,cols = sMag.shape
    for r in range(0,rows):
        for c in range(0,cols):
             maxV = -1.0
             pxDir = Sdir[r,c]
             pxMag = sMag[r,c]     
             if pxMag < 0.5:
                 continue # not a prominent edge
             for n in range(-size,size):
                 rtemp = int(round(r - float(n)*np.sin(pxDir)))
                 ctemp = int(round(c - float(n)*np.cos(pxDir)))
                 if rtemp < 0 or rtemp >= rows \
                     or ctemp < 0 or ctemp >= cols:
                         continue # only find neighbour pixels in the same dir
                 valCheck = sMag[rtemp,ctemp]
                 if valCheck > maxV:
                    maxV = valCheck # keeps replacing max value
             # end for n
             if maxV - pxMag < 1.0 :
                    out[r,c] = 1.0 # found the max pixel
    # end for images
    return out

def colorTransform(img, mat):
    """
    apply a color transformation (e.g. color to gray)
    """
    
    if len(img.shape) !=3:
        print("can't apply color transform wrong img shape!")
        return None
    
    if mat.shape[0] !=3 or mat.shape[1] !=3:
        print("can't apply color transform wrong kernel shape!")
        return None
    
    imOut = np.zeros(img.shape,np.uint8)    
    rows,cols,channel = imOut.shape
    
    for row in range(0,rows):
        for col in range(0,cols):
            colArray = img[row,col,:].astype(float)
            imOut[row,col,:] = np.minimum(np.dot(mat,colArray).astype(int),[255,255,255])
            
    return imOut


