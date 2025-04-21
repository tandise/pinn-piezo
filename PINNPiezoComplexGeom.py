#!/usr/bin/env python
# coding: utf-8

# # PINN for linear elasticity in complex 2-D geometries
# This notebook aims at solving PDEs within a 2-D rectangular domain as a model of linear elastic problem of a plane under shear stress. 
# Data are taken from OpenFOAM. A custom utility called 'geomGenerator' must be called for an OpenFOAM case to generate required data for this network.
# A trick for drawing contours in complex geometries with unstructured mesh is implemented in this network. 

# In[1]:
epc = 50
import sys
# Number of collocation points inside and at boundaries of the domain
nPoint = int(sys.argv[1])
nBcPoint = int(sys.argv[2])
nNeron = int(sys.argv[3])
nLayer = int(sys.argv[4])
#print(nPoint)
case = sys.argv[5]
#case ='plateSector-piezo'
#case = 'disk-piezo'
#case = 'hexa-piezo'
#case = 'plateHole-piezo'
#case = 'plate-piezo'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import scipy.optimize as sopt
#from autograd_minimize.tf_wrapper import tf_function_factory
#from autograd_minimize import minimize
from IPython.display import clear_output
import os

import pandas as pd
from glob import glob

plt.rcParams['font.size'] = 14  # Set the font size to 14




# In[2]:


def setColors(CI):
    if uuid.getnode() != CI:
                colourIDS=np.zeros([uuid.getnode(),uuid.getnode()])
class TrainingPlot(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss4 = []
        self.loss5 = []
        self.lossBcw = []
        self.lossBcthetax = []
        self.lossBcthetay = []
        self.lossBcphi = []
        self.lossBcdeltaN = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss_tot'))
        self.loss1.append(logs.get('loss1'))
        self.loss2.append(logs.get('loss2'))
        self.loss3.append(logs.get('loss3'))
        self.loss4.append(logs.get('loss4'))
        self.loss5.append(logs.get('loss5'))
        self.lossBcw.append(logs.get('lossBcw'))
        self.lossBcthetax.append(logs.get('lossBcthetax'))
        self.lossBcthetay.append(logs.get('lossBcthetay'))
        self.lossBcphi.append(logs.get('lossBcphi'))
        self.lossBcdeltaN.append(logs.get('lossBcdeltaN'))
                    
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses)%epc==0:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            #gpuCheck()
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(5, 3.5),dpi=200)
            plt.plot(N, self.losses, label="loss$_{tot}$")        
            plt.plot(N, self.loss1, label="loss$_{Q}$")             
            plt.plot(N, self.loss2, label="loss$_{Mx}$") 
            plt.plot(N, self.loss3, label="loss$_{My}$")           
            plt.plot(N, self.loss4, label="loss$_{D}$")              
            plt.plot(N, self.loss5, label="loss$_{J}$")             
            plt.plot(N, self.lossBcw, label="loss$_{BC,w}$")        
            plt.plot(N, self.lossBcthetax, label="loss$_{BC,\\theta x}$")  
            plt.plot(N, self.lossBcthetay, label="loss$_{BC,\\theta y}$")  
            plt.plot(N, self.lossBcphi, label="loss$_{BC,\\phi}$")      
            plt.plot(N, self.lossBcdeltaN, label="loss$_{BC,\\Delta N}$")  

            #plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.yscale('log')
            #plt.show()
            plt.savefig(
            case_save + '/convergence.pdf', 
                   format='pdf', dpi=200, bbox_inches='tight'
                   )

            # Write loss information to 'final_loss.txt'
            with open(case_save + '/data.txt', 'w') as f:
                  f.write(f"Final Loss: {self.losses[-1]:.1e}\n")
                  f.write(f"Lw: {self.loss1[-1]:.1e}\n")
                  f.write(f"Lm: {self.loss2[-1]:.1e}\n")
                  f.write(f"Lq: {self.loss3[-1]:.1e}\n")
                  f.write(f"Ld: {self.loss4[-1]:.1e}\n")
                  f.write(f"Lj: {self.loss5[-1]:.1e}\n")
                  f.write(f"BCw: {self.lossBcw[-1]:.1e}\n")
                  f.write(f"BCtheta_x: {self.lossBcthetax[-1]:.1e}\n")
                  f.write(f"BCtheta_y: {self.lossBcthetay[-1]:.1e}\n")
                  f.write(f"BCphi: {self.lossBcphi[-1]:.1e}\n")
                  f.write(f"BCdelta_N: {self.lossBcdeltaN[-1]:.1e}\n")

            
plot_losses = TrainingPlot()


tf.config.list_physical_devices('GPU')



# ### Reading computational domain, boundary conditions 
# A rectangular domain (L=1 m and H=1 m) with 200 x 200 points is generated. Also, three vectors related to BCs are defined: "normal" and "trac_true" vectors for traction BC and "y_true" for fixed-value BC. 

# In[3]:




path_save = "./"

case_path = path_save + case +'/inputs/'

case_save = path_save+case+'/intP'+str(nPoint)+'_bcP'+str(nBcPoint)+ '_nLay'+ str(nLayer)+'_nNeo' + str(nNeron)
if not os.path.isdir(case_save):
    os.makedirs(case_save)

scaleValue = 1.0


intPoints = pd.read_csv(glob(case_path + '*intern*')[0],delim_whitespace = True).values[:,:2].astype(np.float32)
try:
    trBc = pd.read_csv(glob(case_path + '*Tractio*')[0],delim_whitespace = True).values.astype(np.float32)
    trBcPoints = trBc[:,:2]
    trBcNormal = trBc[:,3:5]
    trBcTrac = trBc[:,6:8]
    traction = True
except:
    trBcPoints = 0
    trBcNormal = 0
    trBcTrac = 0
    traction = False
    

fixBc = pd.read_csv(glob(case_path + '*Displace*')[0],delim_whitespace = True).values
fixBcPoints = fixBc[:,:2]
fixBcNormal = fixBc[:,3:5]
fixBcDisp = fixBc[:,6:8]


# ### Generating random points for at inside and boundary of the domain

# In[4]:




# Random select of BC points from each group (fixed-value or traction) proportianl to their size
try:
    nFixBcPoint = np.min(
        [np.math.floor(fixBcPoints.shape[0]*0.999/
                       (0.99*trBcPoints.shape[0]+
                        fixBcPoints.shape[0]
                       )*nBcPoint
                      ),
         fixBcPoints.shape[0]])
    nTrBcPoint = np.min([nBcPoint - nFixBcPoint,trBcPoints.shape[0]])
except:
    nFixBcPoint = nBcPoint
    nTrBcPoint = 0
    
idxBcFix = np.random.choice(fixBcPoints.shape[0],
                            nFixBcPoint,replace=False)
fixBcPointsRnd = (fixBcPoints[idxBcFix])
fixBcNormalRnd = (fixBcNormal[idxBcFix])
fixBcDispRnd = tf.constant(fixBcDisp[idxBcFix])
try:
    idxBcTrac = np.random.choice(trBcPoints.shape[0],nTrBcPoint,replace=False)
    trBcPointsRnd = trBcPoints[idxBcTrac]
    trBcNormalRnd = trBcNormal[idxBcTrac]
    trBcTracRnd = trBcTrac[idxBcTrac]
except:
    print("no traction bc!")
    trBcPointsRnd = 0
    trBcNormalRnd = 0
    trBcTracRnd = 0
    
    

# Random select of internal point (BC points are excluded from selection)
nIntPoints = np.min([nPoint, intPoints.shape[0]])
indexDomain = np.random.choice(intPoints.shape[0],
                               nIntPoints, replace=False)
intPointsRnd = intPoints[indexDomain]
inData = intPoints
y_true=np.zeros([inData.shape[0],5],np.float32)
idx=indexDomain


# In[5]:


nX = 500
nY = nX
try:
    allBcPoints = np.concatenate([trBcPoints,fixBcPoints],0)
    allBcNormals = np.concatenate([trBcNormal,fixBcNormal],0)
except:
    allBcPoints = fixBcPoints
    allBcNormals = fixBcNormal
    
xMax = np.max(allBcPoints[:,0])
xMin = np.min(allBcPoints[:,0])
yMax = np.max(allBcPoints[:,1])
yMin = np.min(allBcPoints[:,1])
L = xMax - xMin
H = yMax - yMin

x_data=np.arange(xMin,xMax,(xMax-xMin)/nX,np.float32)
y_data=np.arange(yMin,yMax,(yMax-yMin)/nY,np.float32)


XData, YData = np.meshgrid(x_data,y_data)
XYStructured = np.stack([np.ndarray.flatten(XData),
                         np.ndarray.flatten(YData)],axis=1)




tileBc = np.tile(allBcPoints,(XYStructured.shape[0],1,1))
tileXYStructured = np.tile(np.expand_dims(XYStructured,1),(1,allBcPoints.shape[0],1))

pointToBc = tileXYStructured-tileBc
distFromBc= np.linalg.norm(pointToBc,axis=-1)
minDistFromBcInex = np.argmin(distFromBc,-1)

pointToNearBc = XYStructured - allBcPoints[minDistFromBcInex]
pointToNearBcDotNormal = np.sum(allBcNormals[minDistFromBcInex]* pointToNearBc, axis=-1)
maskXYStructured = pointToNearBcDotNormal > 0


fig = plt.subplots(1,1,figsize=(6, 6),dpi=200)

plt.scatter(allBcPoints[:,0],allBcPoints[:,1],s=1,
            c='black',label ='All boundary points')
plt.scatter(intPointsRnd[:,0],intPointsRnd[:,1],s=8,
            c='red', cmap='viridis', label = "Random internal points")
plt.scatter(fixBcPointsRnd[:,0],fixBcPointsRnd[:,1],s=8,
            c='blue', cmap='viridis', label = "Random fixed points")
try:
    plt.scatter(trBcPointsRnd[:,0],trBcPointsRnd[:,1],s=8,
            c='green', cmap='viridis', label = "Random traction points")
except:
    print("No traction points!")
plt.autoscale()
# Show the plot with all points
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper center',
           bbox_to_anchor=(0.5,1.2))
plt.savefig(case_save +'/points.pdf',
            format = 'pdf', dpi=200,bbox_inches='tight')
#plt.show()


# In[6]:



#print(fixBcPointsRnd)
#print('trP:\n',trBcPointsRnd)
#print('n::\n', trBcNormal)
## normalize
allPoints = np.concatenate([intPoints,allBcPoints],0)
Lx = np.std(allPoints[:,0])
Ly = np.std(allPoints[:,1])
std_x=np.std(allPoints[:,0])
std_y=np.std(allPoints[:,1])
mean_x=np.mean(allPoints[:,0])
mean_y=np.mean(allPoints[:,1])

intPointsRnd[:,0] = tf.constant((intPointsRnd[:,0] - mean_x ) / std_x)
intPointsRnd[:,1] = tf.constant((intPointsRnd[:,1] - mean_y ) / std_y)

fixBcPointsRnd[:,0] = tf.constant((fixBcPointsRnd[:,0] - mean_x ) / std_x)
fixBcPointsRnd[:,1] = tf.constant((fixBcPointsRnd[:,1] - mean_y ) / std_y)
try:
    trBcPointsRnd[:,0] = tf.constant((trBcPointsRnd[:,0] - mean_x ) / std_x)
    trBcPointsRnd[:,1] = tf.constant((trBcPointsRnd[:,1] - mean_y ) / std_y)
except:
    print("No taction points!")


# ### Implemtation of governing equation and boundary condition by defining a CustomeModel with a custom training step

# In[7]:



    
h=1e-6
c11=6.297e10
c12=3.037e10
c33=9.380e10
c44=1.504e10
c66=1.630e10
e15=-0.21
e31=-0.24
e33=0.44
k11=7.9865e-11
k33=8.4381e-11
Mu11=0.034
Mu33=0.034
d11=8.7897e-4
d33=8.7897e-4
q=1.6e-19
k=np.pi/(12**(0.5))
f=1.0e4
n0=1.0e19

B=.001000

@tf.function
def constitutiveModels(NN, xy):
    
        pred = NN(xy, training=True)  # Forward pass
        #loss = keras.losses.mean_squared_error(y, y_pred)
        
        w = pred[:,0]
        thetaX = pred[:,1]
        thetaY = pred[:,2]
        phi = pred[:,3]
        deltaN = pred[:,4]
        w_x = pred[:,0]
        thetaX_xx = pred[:,1]
        thetaY_yy = pred[:,2]
        phi_x = pred[:,3]
        deltaN_x = pred[:,4]

        
        dwDxy = tf.gradients(w, xy)[0]
        dwDx = dwDxy[:,0]
        dwDy = dwDxy[:,1]
        dwDxy_x = tf.gradients(w, xy)[0]
        dwDx_y = dwDxy[:,0]
        dwDy_xy = dwDxy[:,1]
          
        
        dthetaXDxy = tf.gradients(thetaX, xy)[0]
        dthetaXDx = dthetaXDxy [:,0]
        dthetaXDy = dthetaXDxy [:,1]
        dthetaXDxy_x = tf.gradients(thetaX, xy)[0]
        dthetaXDx_y = dthetaXDxy [:,0]
        dthetaXDy_xy = dthetaXDxy [:,1]
        
        dthetaYDxy = tf.gradients(thetaY, xy)[0]
        dthetaYDx = dthetaYDxy [:,0]
        dthetaYDy = dthetaYDxy [:,1]
        
        
        dphiDxy = tf.gradients(phi, xy)[0]
        dphiDx = dphiDxy[:,0]
        dphiDy = dphiDxy[:,1]
        dphiDxy_x = tf.gradients(phi, xy)[0]
        dphiDx_y = dphiDxy[:,0]
        dphiDy_xy = dphiDxy[:,1]
        
        
        ddeltaNDxy = tf.gradients(deltaN, xy)[0]
        ddeltaNDx = ddeltaNDxy[:,0]
        ddeltaNDy = ddeltaNDxy[:,1]
        ddeltaNDxy_x = tf.gradients(deltaN, xy)[0]
        ddeltaNDx_y = ddeltaNDxy[:,0]
        ddeltaNDy_xy = ddeltaNDxy[:,1]
        
        Mxx_x = 4*L**3/(3*Lx**3)*(dthetaXDx+(c12/Ly)*(Lx/c11)*dthetaYDy)
        Myy_x = 4*L**3/(3*Ly**3)*((c12/Lx)*(Ly/c11)*dthetaXDx+dthetaYDy)
        Mxy_y = 4*L**3*c66/(3.0*c11*Lx*Ly)*(dthetaXDy/Ly+dthetaYDx/Lx)
        Mxx = (c11*h**2)/(3*Lx**2*k**2*c44) * dthetaXDx + (c12*h**2)/(3*Lx**2*k**2*c44) * dthetaYDy
        Myy = (c12*h**2)/(3*Ly**2*k**2*c44) * dthetaXDx + (c11*h**2)/(3*Ly**2*k**2*c44) * dthetaYDy
        Mxy = (c66*h**2)/(3*Ly**2*k**2*c44) * dthetaXDy + (c66*h**2)/(3*Lx**2*k**2*c44) * dthetaYDx
          
            
        Qx_x = ((2*k**2*c44*L**4)/(Lx**2*c11*h**2))*(dwDx +(2*Lx/L)*thetaX+((e15*d11*h**3*c11)/(Mu11*k**2*c44*f*L**4))*dphiDx)   
        Qy_y = ((2*k**2*c44*L**4)/(Lx**2*c11*h**2))*(dwDy +(2*Ly/L)*thetaY+((e15*d11*c11*h**3)/(k**2*Mu11*c44*f*L**4))*dphiDy)
        Qx = dwDx + thetaX + (2*h*e15*d11)/(f*Mu11*Lx**2)*dphiDx
        Qy = (Lx/Ly)**2*dwDy + thetaY + (2*h*e15*d11)/(f*Mu11*Ly**2)*dphiDy
        
        Dx_x = ((k*e15*f*L**4)/(q*n0*c11*Lx**2*h**3))*(dwDx +thetaX*(2*Lx/L)-((k11*d11*c11*h**3)/(Mu11*k*e15*f*L**4))*dphiDx) 
        Dy_y = ((k*e15*f*L**4)/(q*n0*c11*Ly**2*h**3))*(dwDy +thetaY*(2*Ly/L)-((k33*d11*c11*h**3)/(Mu11*k*e15*f*L**4))*dphiDy)
        Dx = (f*e15)/(2*h*k*c44*q*n0) * (dwDx + thetaX) - (k11*d11)/(Mu11*Lx**2*q*n0) * dphiDx
        Dy = (f*e15)/(2*h*k*c44*q*n0) * ((Lx/Ly)**2*dwDy + thetaY) - (k33*d11)/(Mu11*Ly**2*q*n0) * dphiDy
        
        Jx = -dphiDx+ddeltaNDx
        Jy = -dphiDy+ddeltaNDy
        return Mxx, Myy, Mxy, Qx, Qy, Dx, Dy, Jx, Jy, deltaN, dwDx
        
@tf.function
def basicLawsRes(NN, xy):
    
        Mxx, Myy, Mxy, Qx, Qy, Dx, Dy, Jx, Jy, deltaN, dwDx= constitutiveModels(NN, xy)
        dQxDxy=tf.gradients(Qx, xy)[0]
        dQxDx = dQxDxy[:,0]
        dQxDy = dQxDxy[:,1]
        
        dQyDxy=tf.gradients(Qy, xy)[0]
        dQyDx = dQyDxy[:,0]
        dQyDy = dQyDxy[:,1]

        dMxxDxy=tf.gradients(Mxx, xy)[0]
        dMxxDx = dMxxDxy[:,0]
      
        
        dMxyDxy=tf.gradients(Mxy, xy)[0]
        dMxyDx = dMxyDxy[:,0]
        dMxyDy = dMxyDxy[:,1]
        
        dMyyDxy=tf.gradients(Myy, xy)[0]
        dMyyDy = dMyyDxy[:,1]
        
        dDxDxy=tf.gradients(Dx, xy)[0]
        dDxDx = dDxDxy[:,0]
     
        dDyDxy=tf.gradients(Dy, xy)[0]
        dDyDy = dDyDxy[:,1]
        
        
        dJxDxy=tf.gradients(Jx, xy)[0]
        dJxDx = dJxDxy[:,0]
        
        dJyDxy=tf.gradients(Jy, xy)[0]
        dJyDy = dJyDxy[:,1]
        
        res1 = (dQxDx + dQyDy + 1)
        res2 = dMxxDx + dMxyDy - Qx
        res3 = dMxyDx + dMyyDy - Qy
        res4 = dDxDx + dDyDy + deltaN
        #res5 = 2*h*q*n0*(d11/Lx**2*dJxDx + d33/Ly**2*dJyDy)
        res5 = dJxDx + dJyDy

        
        return res1, res2, res3, res4, res5

@tf.function
def tracBcRes(NN, xyTrac, normalVec, trac):
    
        Mxx, Myy, Mxy, Qx, Qy, Dx, Dy, Jx, Jy, deltaN, dwDx= constitutiveModels(NN, xyTrac)
        nx = normalVec[:,0]
        ny = normalVec[:,1]
        
        res1 = nx * nx * Mxx + 2 * ny * nx * Mxy + ny * ny * Myy 
        res2 = -nx * ny * Mxx  + (nx * nx - ny * ny) * Mxy + ny * nx * Myy 
        res3 = Qx * nx + Qy * ny
        
        return res1, res2, res3
    
@tf.function
def evaluateLossAndGrad(NN, data):  #data = tuple indata,outputData = data  trainstep   
        xy = data[0][0][0]
        xFix = data[0][1][0]
        yFix = data[0][2][0]
        xTrac = data[0][3][0]
        trBcNorVec = data[0][4][0]
        trBcTrac = data[0][5][0]
        #Trac = tf.expand_dims(tf.squeeze(data[0][4]),-1)
        #tracBc = tf.squeeze(data[0][5])
        #refs = data[0][6][0]
        #print(xy.shape)
        #print(xFix.shape)
        #print(yFix.shape)
        #print(xTrac.shape)
        
        #xy, output = data # xy = inData  , output= outputData
        #desired = output[0] #y_true
        
        #ref1 = output[1][0,0]
        #ref2 = output[1][0,1]
        #ref3 = output[1][0,2]
        #ref4 = output[1][0,3]
        #ref5 = output[1][0,4]
        #refW = output[1][0,5]
        #refThX = output[1][0,6]
        #refThY = output[1][0,7]
        #refPhi = output[1][0,8]
        #refDelN = output[1][0,9]


        xyBc = xFix #tf.stack(xFix,xTrac,-1) #tf.gather(xy,idxBcFix)#input of bc
        desiredBc = tf.zeros([xFix.shape[0],5]) #tf.gather(desired,idxBcFix) #output of bc... acces to bundary condition 
        #try:
        #    xyInt= tf.concat([xy,xTrac],0) # tf.gather(xy, idx) # acces to interior points
        #except:
        xyInt = xy
 
        with tf.GradientTape() as tape:
            predBc = NN(xyBc, training=True)# predBc depends on weight of network and is changing
            
            #predInt = NN(xyInt, training=False)
            #predAvg = tf.math.reduce_max(tf.math.abs(predInt),axis=0)
           # print("Avg",predAvg)
            res1, res2, res3, res4, res5 = basicLawsRes(NN,xyInt)
            
            #if traction:
            #    resTrBcMx, resTrBcMy, resTrBcQ = tracBcRes(NN, xTrac, trBcNorVec, trBcTrac)
            #    res1= tf.concat([res1,resTrBcMx],0)
            #    res2= tf.concat([res2,resTrBcMy],0)
            #    res3= tf.concat([res3,resTrBcQ],0)
                
            
            loss1 = tf.keras.losses.mse(tf.zeros_like(res1),res1)  
            loss2 = tf.keras.losses.mse(tf.zeros_like(res2),res2)
            loss3 = tf.keras.losses.mse(tf.zeros_like(res3),res3)
            loss4 = tf.keras.losses.mse(tf.zeros_like(res4),res4)
            loss5 = tf.keras.losses.mse(tf.zeros_like(res5),res5)
            lossBcW = tf.keras.losses.mse(desiredBc[:,0],predBc[:,0])
            lossBcThX = tf.keras.losses.mse(desiredBc[:,1],predBc[:,1])
            lossBcThY = tf.keras.losses.mse(desiredBc[:,2],predBc[:,2])
            lossBcPhi = tf.keras.losses.mse(desiredBc[:,3],predBc[:,3])
            lossBcDelN = tf.keras.losses.mse(desiredBc[:,4],predBc[:,4])
                   
            loss_tot = loss1 + loss2 + loss3 + loss4 + loss5 + lossBcW + lossBcThX + lossBcThY + lossBcPhi + lossBcDelN
            if traction:
                desiredBcTrac = tf.zeros([xTrac.shape[0],5])
                resTrBcMx, resTrBcMy, resTrBcQ = tracBcRes(NN, xTrac, trBcNorVec, trBcTrac)
                lossBcTrMx = tf.keras.losses.mse(tf.zeros_like(resTrBcMx),resTrBcMx)
                lossBcTrMy = tf.keras.losses.mse(tf.zeros_like(resTrBcMy),resTrBcMy)
                lossBcTrQ = tf.keras.losses.mse(tf.zeros_like(resTrBcQ),resTrBcQ)
                predBcTrac = NN(xTrac, training=True)
                lossBcPhiTrac = tf.keras.losses.mse(desiredBcTrac[:,3],predBcTrac[:,3])
                lossBcDelNTrac = tf.keras.losses.mse(desiredBcTrac[:,4],predBcTrac[:,4])
                #print(xFix.shape)
                #print(resTrBcMx.shape)
                loss_tot = loss_tot  + lossBcPhiTrac + lossBcDelNTrac + (lossBcTrMx + lossBcTrMy + lossBcTrQ) 

        # Compute gradients

        trainable_vars = NN.trainable_variables 

        loss_tracker.update_state(loss_tot)
        loss1_tracker.update_state(loss1)
        loss2_tracker.update_state(loss2)
        loss3_tracker.update_state((loss3))
        loss4_tracker.update_state((loss4))
        loss5_tracker.update_state((loss5))
        lossBcW_tracker.update_state((lossBcW))
        lossBcthetax_tracker.update_state((lossBcThX))
        lossBcthetay_tracker.update_state((lossBcThY))
        lossBcphi_tracker.update_state((lossBcPhi))
        lossBcdeltaN_tracker.update_state((lossBcDelN))

        
        grad = tape.gradient(loss_tot, trainable_vars)
        return loss_tot, grad 
    

loss_tracker = keras.metrics.Mean(name="loss_tot")
loss1_tracker = keras.metrics.Mean(name="loss1")
loss2_tracker = keras.metrics.Mean(name="loss2")
loss3_tracker = keras.metrics.Mean(name="loss3")
loss4_tracker = keras.metrics.Mean(name="loss4")
loss5_tracker = keras.metrics.Mean(name="loss5")
lossBcW_tracker = keras.metrics.Mean(name="lossBcw")
lossBcthetax_tracker = keras.metrics.Mean(name="lossBcthetax")
lossBcthetay_tracker = keras.metrics.Mean(name="lossBcthetay")
lossBcphi_tracker = keras.metrics.Mean(name="lossBcphi")
lossBcdeltaN_tracker = keras.metrics.Mean(name="lossBcdeltaN")



class CustomModel(keras.Model):#custommodel class from keras
     

    def train_step(self, data):   #data= tuple(indata , outputdata )    
        
        loss, gradients = evaluateLossAndGrad(self,data)
        trainable_vars = self.trainable_variables 

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Compute our own metrics
        
        return {
                "loss_tot": loss_tracker.result(),
                "loss1": loss1_tracker.result(),
                "loss2": loss2_tracker.result(),
                "loss3": loss3_tracker.result(),
                "loss4": loss4_tracker.result(),
                "loss5": loss5_tracker.result(),
                "lossBcw": lossBcW_tracker.result(),
                "lossBcthetax": lossBcthetax_tracker.result(),
                "lossBcthetay": lossBcthetay_tracker.result(),
                "lossBcphi": lossBcphi_tracker.result(),
                "lossBcdeltaN": lossBcdeltaN_tracker.result(),

               }
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker,loss1_tracker, loss2_tracker,loss3_tracker,loss4_tracker,loss5_tracker,lossBcW_tracker,lossBcthetax_tracker,lossBcthetay_tracker, lossBcphi_tracker, lossBcdeltaN_tracker ]


# ### Create a fully connected netwrok.
# The input is 2-D spatial point (x) and the output is a 2-D displacement vector. 
# 

# In[8]:
if os.path.exists(case_save + '/model.h5'):
    # Load the model
    custom_objects = {'CustomModel': CustomModel}
    model = load_model(case_save + '/model.h5', custom_objects=custom_objects)
    print(f"Model loaded!")
    model.summary()
    nWeight = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
else:
    print(f"No model found!")

    input_layer=Input(2)
    dense1=Dense(nNeron,activation='tanh')(input_layer)
    for i in range(1,nLayer):
    	dense1=Dense(nNeron,activation='tanh')(dense1)

    output_layer = Dense(5,activation=None,name='output')(dense1)

    model = CustomModel(input_layer,output_layer)
    model.summary()
    nWeight = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print("Total number of trainable weights:", nWeight)

    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt,loss='mse')#tf.keras.losses.LogCosh(reduction="auto", 




    start = time.time()
    colorID=4929744357368
    outputData = y_true#tuple
    inputData = [tf.expand_dims(intPointsRnd,0),
             tf.expand_dims(fixBcPointsRnd,0),
             tf.expand_dims(fixBcDispRnd,0),
             tf.expand_dims(trBcPointsRnd,0),
             tf.expand_dims(trBcNormalRnd,0),
             tf.expand_dims(trBcTracRnd,0),
             ]
    model.fit(inputData, batch_size=intPointsRnd.shape[0], epochs=epc,shuffle=False,callbacks=[plot_losses])

    end = time.time()
    print(' Time = ',end - start)

    model.save(case_save + '/model.h5')  # Saves in HDF5 format




scale_w =f*Lx**2/(2*h*k**2*c44)
XYStructured = np.stack([np.ndarray.flatten(XData),
                         np.ndarray.flatten(YData)],axis=1)
XYStructured_scaled = np.zeros_like(XYStructured)
XYStructured_scaled[:,0] = ((XYStructured[:,0]-mean_x)/std_x)
XYStructured_scaled[:,1] = ((XYStructured[:,1]-mean_y)/std_y)
out = model.predict(XYStructured_scaled)[:,0]*scale_w

mask = maskXYStructured #np.stack([maskXYStructured, maskXYStructured], -1)
makedOut = np.ma.MaskedArray(out,mask)


phiX=makedOut.reshape(XData.shape)

fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

              

#pylab.xlim([np.min(XData),np.max(XData)])
#pylab.ylim([np.min(YData),np.max(YData)])


contourX = plt.contourf(XData,YData, phiX,30, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title('w [m]',pad=18)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.autoscale()

plt.savefig(case_save +'/w_contour.pdf',
	 format = 'pdf', dpi=200) 
 


# In[12]:


scale_phi = (d11/Mu11)
out = model.predict(XYStructured_scaled)[:,3]*scale_phi

mask = maskXYStructured #np.stack([maskXYStructured, maskXYStructured], -1)
makedOut = np.ma.MaskedArray(out,mask)


phiX=makedOut.reshape(XData.shape)

fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

              

#pylab.xlim([np.min(XData),np.max(XData)])
#pylab.ylim([np.min(YData),np.max(YData)])

from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


contourX = plt.contourf(XData,YData, phiX,30, cmap='jet')
clb = plt.colorbar()
clb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
clb.ax.yaxis.get_major_formatter().set_scientific(True)
clb.ax.yaxis.get_major_formatter().set_powerlimits((0, 0)) 


clb.ax.set_title(r'$\phi$ [V]', pad=18)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.autoscale()

plt.savefig(case_save +'/phi_contour.pdf',
	 format = 'pdf', dpi=200)   
 


# In[13]:




out = model.predict(XYStructured_scaled)[:,4]*n0

mask = maskXYStructured #np.stack([maskXYStructured, maskXYStructured], -1)
makedOut = np.ma.MaskedArray(out,mask)


phiX=makedOut.reshape(XData.shape)

fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

              

#pylab.xlim([np.min(XData),np.max(XData)])
#pylab.ylim([np.min(YData),np.max(YData)])


contourX = plt.contourf(XData,YData, phiX,30, cmap='jet')
clb = plt.colorbar()
clb.ax.set_title(r'$\Delta n$', pad=18)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.autoscale()

plt.savefig(case_save +'/delN_contour.pdf',
	 format = 'pdf', dpi=200)  
 
 


# In[14]:


from scipy.interpolate import griddata

def mapField(df, targetMesh):
    """
    Maps the scalar field 'A' from a source DataFrame to target unstructured points.

    Parameters:
    df (pd.DataFrame): Source DataFrame with columns ['x', 'y', 'A'] representing the data points and values.
    targetMesh (np.ndarray): Target points array of shape (num_points, 2) where A needs to be interpolated.

    Returns:
    pd.DataFrame: DataFrame with columns ['x', 'y', 'A'] representing target points and their mapped A values.
    """
    # Extract source data
    x = df.values[:,0]
    y = df.values[:,1]
    A = df.values[:,2]

    # Extract target mesh coordinates
    target_x = targetMesh[:, 0]
    target_y = targetMesh[:, 1]

    # Perform linear interpolation
    A_interp = griddata(points=(x, y), values=A, xi=targetMesh, method='linear')

    # Handle NaN values by using nearest interpolation
    nan_mask = np.isnan(A_interp)
    if np.any(nan_mask):
        A_interp_nearest = griddata(points=(x, y), values=A, xi=targetMesh, method='nearest')
        A_interp = np.where(nan_mask, A_interp_nearest, A_interp)

    # Create a DataFrame with interpolated results
    df_result = pd.DataFrame({
        'x': target_x,
        'y': target_y,
        'A': A_interp
    })
    
    return df_result


# In[15]:


def comsolVsPINN(model, dim, cml_file, std_x, std_y, mean_x, mean_y, scale_cml,scale_nonDim,variable):
    df = pd.read_csv(cml_file)
    df_mapped=mapField(df,allPoints*1e6)
    www=np.abs(df_mapped.values[:,-1]*scale_cml)
    xxx=df_mapped.values[:,0]*1e-6
    yyy=df_mapped.values[:,1]*1e-6
    xx = (xxx-mean_x)/std_x
    yy = (yyy-mean_y)/std_y
    cml_xy_data = np.stack([np.ndarray.flatten(xx),np.ndarray.flatten(yy)],axis=1)
    predictions = model.predict(cml_xy_data)
    w_pred=np.abs(predictions[:,dim]*scale_nonDim)

    # Calculate the squared errors
    squared_errors = (w_pred - www) ** 2  
    # Compute the mean of the squared errors
    mse = np.sqrt(np.mean(squared_errors))
    dev = round(mse/np.max(np.abs(www))*100,2)
    numerator = np.linalg.norm(w_pred - www, ord=2)
    denominator = np.linalg.norm(www, ord=2)
    dev2 = numerator / denominator

    print('\n' + cml_file[2:-4])
    print("    max comsol is = ",np.max((www)))
    print("    max predict is = ", np.max(w_pred))
    print("    overall mean error is: %", dev)
    print("    relative L2 is: %", dev2)
    df_mapped=mapField(df,XYStructured*1e6)
    www=np.abs(df_mapped.values[:,-1]*scale_cml)
    xxx=df_mapped.values[:,0]*1e-6
    yyy=df_mapped.values[:,1]*1e-6
    xx = (xxx-mean_x)/std_x
    yy = (yyy-mean_y)/std_y
    cml_xy_data = np.stack([np.ndarray.flatten(xx),np.ndarray.flatten(yy)],axis=1)
    predictions = model.predict(cml_xy_data)
    w_pred=np.abs(predictions[:,dim]*scale_nonDim)

    mask = maskXYStructured #np.stack([maskXYStructured, maskXYStructured], -1)
    makedPredictions = np.ma.MaskedArray(w_pred,mask)
    makedCml = np.ma.MaskedArray(www,mask)


    output_masked=np.abs(makedCml.reshape(XData.shape)-
                         makedPredictions.reshape(XData.shape))/(np.max(www)-np.min(www))*100

    fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

    contour_out = plt.contourf(XData,YData, output_masked,30, cmap='jet')
    plt.autoscale()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    clb = plt.colorbar()
    if variable == 'w':
        var = 'w'
    elif variable == 'delN':
        var = '\Delta n'
    elif variable == 'phi':
        var = '\phi'
    clb.ax.set_title(r'$\varepsilon_{ ' + var + r'}\,[\%]$', pad=15)
    plt.savefig(case_save +'/Err_'+ variable +'.pdf',
	 format = 'pdf', dpi=200) 
    
    return xxx, yyy, www, w_pred, dev2


# In[16]:


x_map, y_map, w_cml, w_pred_cml, err_w = comsolVsPINN(model,0, path_save + case + '/w_'+ case + '.csv',
                                   std_x, std_y, mean_x, mean_y,1e-6,f*Lx**2/(2*h*k**2*c44),'w')
x_map, y_map, phi_cml, phi_pred_cml, err_phi = comsolVsPINN(model,3,path_save + case + '/phi_'+ case + '.csv',
                                   std_x, std_y, mean_x, mean_y,1,d11/Mu11,'phi')
x_map, y_map, delN_cml, delN_pred_cml, err_delN = comsolVsPINN(model,4,path_save + case + '/delN_'+ case + '.csv',
                                   std_x, std_y, mean_x, mean_y, 1, n0,'delN')


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata





def plotLineThroughPoints(w, intPoints, line_points):
    """
    Interpolates and plots the field values along a user-defined line.

    Parameters:
    w (np.ndarray): Array of field values corresponding to intPoints.
    intPoints (np.ndarray): Array of shape (number of points, 2) representing the points in 2D space.
    line_points (np.ndarray): Array of shape (num_line_points, 2) defining the line points in 2D space.

    Returns:
    None: Displays a plot of field values along the user-defined line.
    """
    # Interpolate the field values along the user-defined line
    w_interp = griddata(points=intPoints, values=w, xi=line_points, method='linear')

    # Handle NaN values using nearest interpolation
    nan_mask = np.isnan(w_interp)
    if np.any(nan_mask):
        w_nearest = griddata(points=intPoints, values=w, xi=line_points, method='nearest')
        w_interp = np.where(nan_mask, w_nearest, w_interp)

    # Extract coordinates along the line for plotting
    line_distances = np.linalg.norm(np.diff(line_points, axis=0), axis=1).cumsum()
    line_distances = np.insert(line_distances, 0, 0)  # Add the starting point (distance = 0)
     
    return line_distances, w_interp
   


# In[26]:



if  "plateHole" in  case:
    x_values = np.linspace(-5e-6, -2e-6, 100)
    y_constant = 0
    line_points = np.column_stack((x_values, np.full_like(x_values, y_constant)))
elif  "disk" in  case:
    x_values = np.linspace(-5e-6, 5e-6, 100)
    y_constant = 0
    line_points = np.column_stack((x_values, np.full_like(x_values, y_constant)))
elif  "plateSector" in  case:
    x_values = np.linspace(0.353e-5, 1.0605e-5, 100)
    y_values = np.linspace(0.353e-5, 1.0605e-5, 100)
    line_points = np.stack([x_values, y_values],-1)



x_map_scaled = ( x_map - mean_x ) / std_x
y_map_scaled = ( y_map - mean_y ) / std_y
XYMapped_scaled =np.stack([ x_map_scaled, y_map_scaled ] , -1)
XYMapped = np.stack([ x_map, y_map ] , -1)


out = model.predict(XYMapped_scaled)

fig = plt.subplots(1,1,figsize=(4, 4),dpi=200)
plt.scatter(allBcPoints[:,0],allBcPoints[:,1],s=1,
            c='black',label ='All boundary points')
plt.plot(line_points[:,0],line_points[:,1],
            c='red',label ='Line')
plt.autoscale()
# Show the plot with all points
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig(case_save +'/line.pdf',
             format = 'pdf', dpi=200,bbox_inches='tight')


# In[28]:


fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

x_line_pinn, pred_pinn = plotLineThroughPoints(out[:,0]*scale_w, XYMapped, line_points)
x_line_cml, pred_cml = plotLineThroughPoints(w_cml, XYMapped, line_points)

 # Plot the interpolated field values along the line
plt.plot(x_line_pinn, pred_pinn, '-', label='PINN')
plt.plot(x_line_cml, pred_cml, label='COMSOL')
plt.xlabel('Distance along the line [m]')
plt.ylabel('w [m]')
plt.legend()
plt.savefig(case_save +'/w_line.pdf',
           format = 'pdf', dpi=200)  


# In[34]:


fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

x_line_pinn, pred_pinn = plotLineThroughPoints(-out[:,3]*scale_phi, XYMapped, line_points)
x_line_cml, pred_cml = plotLineThroughPoints(phi_cml, XYMapped, line_points)

 # Plot the interpolated field values along the line
plt.plot(x_line_pinn, pred_pinn, '-', label='PINN')
plt.plot(x_line_cml, pred_cml, label='COMSOL')
plt.xlabel('Distance along the line [m]')
plt.ylabel(r'$\phi$ [V]')
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(ScalarFormatter())
formatter = ax.yaxis.get_major_formatter()  # Get the ScalarFormatter instance
formatter.set_scientific(True)  # Enable scientific notation
formatter.set_powerlimits((0, 0))  # Always use scientific notation
plt.legend()
plt.savefig(case_save +'/phi_line.pdf',
           format = 'pdf', dpi=200)   


# In[30]:


fig = plt.subplots(1,1,figsize=(6, 5),dpi=200)

x_line_pinn, pred_pinn = plotLineThroughPoints(-out[:,4]*n0, XYMapped, line_points)
x_line_cml, pred_cml = plotLineThroughPoints(delN_cml, XYMapped, line_points)

 # Plot the interpolated field values along the line
plt.plot(x_line_pinn, pred_pinn, '-', label='PINN')
plt.plot(x_line_cml, pred_cml, label='COMSOL')
plt.xlabel('Distance along the line [m]')
plt.ylabel(r'$\Delta n$')
plt.legend()

plt.savefig(case_save +'/delN_line.pdf',
           format = 'pdf', dpi=200)   

with open(case_save + '/data.txt', 'a') as f:
                  f.write(f"\nRelative Error with respect to COMSOL data: \n")
                  f.write(f"delN_err: {err_delN:.2e}\n")
                  f.write(f"phi_err: {err_phi:.2e}\n")
                  f.write(f"w_err: {err_w:.2e}\n")
                  
# In[ ]:





# In[ ]:





# In[ ]:




