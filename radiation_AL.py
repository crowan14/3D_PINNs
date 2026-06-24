
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import sympy as sym
import time
from matplotlib.cm import gray

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.close('all')
plt.style.use('default')

#pipe https://www.desmos.com/3d/pukwjsdfew
#helicopter https://www.desmos.com/3d/iu3cwfmfky

#%%

#GEOMETRY AND INTEGRATION

def level_set(X):
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    #tabletop https://www.desmos.com/3d/ohn8ifglcf
    phi = ( torch.cos(2*np.pi*x) - (1-5*z**2) )**2 + (3*(y-0.5))**2 + ( torch.cos(2*np.pi*y) - (1-5*z**2) )**2 + (3*(x-0.5))**2 - 3.00
    return phi

#grid to generate data (number of points per direction in slice plane)
pts = 75
grid = torch.linspace(0,1,pts)

#in-plane spacing
dx = grid[1] - grid[0]
dV = dx**3

#all the points to evaluate level set at in 3D (put into vector)
zd , xd , yd  = torch.meshgrid( grid , grid , grid  )
Xd = torch.zeros( ( pts**3 , 3 ) )
Xd[:,0] = torch.reshape( xd , ( 1 , pts**3 ) )
Xd[:,1] = torch.reshape( yd , ( 1 , pts**3 ) )
Xd[:,2] = torch.reshape( zd , ( 1 , pts**3 ) )

#store the grayscale color, pre-evaluate level set
phase = torch.zeros(len(Xd))
evals = level_set(Xd)

#assign phase
for i in range(len(Xd)):
    #inside the level set
    if evals[i] < 0:
        phase[i] = 1
        
#get indices corresponding to inside and outside
inside = ( phase > 0.5 ).nonzero(as_tuple=True)[0]
outside = ( phase < 1 - 0.5 ).nonzero(as_tuple=True)[0]

#position of data corresponding to inside vs. outside
X = Xd[inside]

#get marching cubes to have desired orientation
phi = torch.reshape( level_set(Xd) , (pts,pts,pts) ).permute(1,2,0)

#marching cubes
verts , faces , _ , _ = measure.marching_cubes( phi.numpy() , level=0.0 )
verts = torch.tensor( verts.copy() , dtype=torch.float32 ) / (pts-1)
faces = torch.tensor( faces.copy() , dtype=torch.int )
nhat = torch.zeros( ( len(faces) , 3 ) )

#get surface points at center of faces and corresponding area element
Xs = torch.zeros( ( len(faces) , 3 ) )
dS = torch.zeros( ( len(faces) , 1 ) )
for i in range(len(faces)):
    V = verts[faces[i]]
    Xs[i,:] = torch.mean( V , axis=0 )
    a , b , c = V
    u = b - a
    v = c - a
    dS[i] = 0.5 * torch.linalg.norm(torch.cross(u, v))
    
    #facet normal vector, unsigned
    nhat_local = torch.cross(u,v) / torch.linalg.norm(torch.cross(u, v))
    
    #compute the sign
    if level_set( (Xs[i,:] + 0.025 * nhat_local).unsqueeze(0) ) < level_set( Xs[i,:].unsqueeze(0) ):
        nhat_local = -nhat_local
    
    nhat[i,:] = nhat_local

#get points corresponding to two boundaries
cutoff = 0.5

mask = Xs[:,2] < cutoff                 
idx_d = torch.where(mask)[0]            
idx_r = torch.where(~mask)[0] 

#points on surface belonging to the two boundaries
XD = Xs[idx_d]
dSD = dS[idx_d]

XR = Xs[idx_r]
dSR = dS[idx_r]
nhatR = nhat[idx_r]

dSs = torch.cat((dSD,dSR))

pts = len(X)
ptsd = len(XD)
ptsr = len(XR)
ptsb = ptsd + ptsr

# fig = plt.figure(figsize=(12, 10), constrained_layout=True)
# plt.rcParams.update({'font.size': 16})
# fig.set_constrained_layout_pads(
#     w_pad=0.15,  # padding between axes and figure edge (width)
#     h_pad=0.15,  # padding between axes and figure edge (height)
#     wspace=0.1,  # space between subplots (width)
#     hspace=0.3   # space between subplots (height)
# )

# ax = fig.add_subplot(121, projection='3d')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)
# ax.set_title('Tabletop geometry')
# ax.scatter(X[:,0], X[:,1], X[:,2])

# ax = fig.add_subplot(122, projection='3d')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)
# ax.set_title('Boundary conditions')
# ax.scatter(XD[:,0], XD[:,1], XD[:,2],color='red',label='Dirichlet boundary')
# ax.scatter(XR[:,0], XR[:,1], XR[:,2],color='blue',label='Robin boundary')
# ax.legend()

# # step = 0.1
# # ax = fig.add_subplot(133, projection='3d')
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set_zticks([])
# # ax.set_xlabel('$x_1$')
# # ax.set_ylabel('$x_2$')
# # ax.set_zlabel('$x_3$')
# # ax.set_xlim(0,1)
# # ax.set_ylim(0,1)
# # ax.set_zlim(0,1)
# # ax.set_title('Normal vectors')
# # ax.scatter(XR[:,0], XR[:,1], XR[:,2],color='blue')
# # ax.scatter(XR[:,0]+step*nhatR[:,0], XR[:,1]+step*nhatR[:,1], XR[:,2]+step*nhatR[:,2],color='red')

sigma = 0.1

#%%

#EXACT SOLUTION

#symbolic variables
xx , yy , zz = sym.symbols('xx yy zz')

#graded conductivity
kappa = 1 + 1 * zz

#numerical function
kappa_num = sym.lambdify( ( xx , yy , zz ) , kappa , 'numpy' )

#assumed solution
u_sym = 2 * zz * ( 1 + sym.sin( 2 * sym.pi * xx ) * sym.sin( 2 * sym.pi * yy ) )

#numerical function
u_num = sym.lambdify( ( xx , yy , zz ) , u_sym , 'numpy' )

#corresponding source term
f_sym = - ( sym.diff( kappa * sym.diff( u_sym , xx , 1 ) , xx , 1 ) + sym.diff( kappa * sym.diff( u_sym , yy , 1 ) , yy , 1 ) + sym.diff( kappa * sym.diff( u_sym , zz , 1 ), zz , 1 ) )

#numerical function
f_num = sym.lambdify( ( xx , yy , zz ) , f_sym , 'numpy' )

#heat flux
x_flux_sym = -kappa * sym.diff( u_sym , xx , 1 ) 
y_flux_sym = -kappa * sym.diff( u_sym , yy , 1 ) 
z_flux_sym = -kappa * sym.diff( u_sym , zz , 1 ) 

#numerical function
x_flux_num = sym.lambdify( ( xx , yy , zz ) , x_flux_sym , 'numpy' )
y_flux_num = sym.lambdify( ( xx , yy , zz ) , y_flux_sym , 'numpy' )
z_flux_num = sym.lambdify( ( xx , yy , zz ) , z_flux_sym , 'numpy' )

#convenient for NN input
def f(X):
    return f_num( X[:,0] , X[:,1] , X[:,2] ).reshape(-1,1)

#convenient for NN input
def kappa(X):
    return kappa_num( X[:,0] , X[:,1] , X[:,2] ).reshape(-1,1)

#convenient for NN input
def flux(X):
    q = torch.zeros((len(X),3))
    q[:,0] = x_flux_num( X[:,0] , X[:,1] , X[:,2] )  
    q[:,1] = y_flux_num( X[:,0] , X[:,1] , X[:,2] )  
    q[:,2] = z_flux_num( X[:,0] , X[:,1] , X[:,2] )  
    return q

#convenient for NN input
def u_check(X):
    return u_num( X[:,0] , X[:,1] , X[:,2] ).reshape(-1,1)

#precompute forcing
F = f(X)

#precompute exact solution
u_ex = u_check(X)
unorm = dV * torch.sum(torch.abs(u_ex) )

#solution along dirichlet boundary
g = u_check(XD)

#along whole boundary
gs = u_check(Xs)
gnorm = torch.sum( dS * torch.abs(gs) )

#heat flux along neumann boundary
Q = torch.einsum( 'xi,xi->x' , flux(XR) , nhatR ).reshape(-1,1) - sigma * u_check(XR)**4

#%%

class network(nn.Module):
    def __init__( self ):
        super().__init__()
        
        N = 30
        
        #define layers and activation
        self.layer_1 = nn.Linear( 3 , N )
        self.layer_2 = nn.Linear( N , N )
        self.output = nn.Linear( N , 1 , bias=False )
        self.act = nn.Tanh()
        
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        y = self.output(y)
        
        #enforce boundary conditions
        D = x[:,2]
        
        y = D.reshape(-1,1) * y
        
        return y
    
    def sf( self , X ):
        
        X = X.clone().detach().requires_grad_(True)
        u = self.forward(X)
        
        #automatic differentiation for spatial gradient (two components)
        grad_u = torch.autograd.grad( u , X , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        #two components of gradient
        u_x = grad_u[:,0]
        u_y = grad_u[:,1]
        u_z = grad_u[:,2]
        
        grad_u_x = torch.autograd.grad( kappa(X).squeeze() * u_x , X , grad_outputs=torch.ones_like(u_x) , create_graph=True )[0]
        grad_u_y = torch.autograd.grad( kappa(X).squeeze() * u_y , X , grad_outputs=torch.ones_like(u_y) , create_graph=True )[0]
        grad_u_z = torch.autograd.grad( kappa(X).squeeze() * u_z , X , grad_outputs=torch.ones_like(u_z) , create_graph=True )[0]
        
        u_xx = grad_u_x[:,0].reshape(-1,1)
        u_yy = grad_u_y[:,1].reshape(-1,1)
        u_zz = grad_u_z[:,2].reshape(-1,1)
        
        #integrand of strong form loss
        integrand = ( u_xx + u_yy + u_zz + F )**2
        
        #integrate to form it
        loss = 0.5 * dV * torch.sum(integrand)
        
        return loss
    
    def bc( self , XD , XR ):
        
        uD = self.forward(XD)
        dloss = ( uD - g )
        
        XR = XR.clone().detach().requires_grad_(True)
        uR = self.forward(XR)
        
        grad_uR = torch.autograd.grad( uR , XR , grad_outputs=torch.ones_like(uR) , create_graph=True )[0]
        flux = -kappa(XR) * torch.einsum( 'xi,xi->x' , grad_uR , nhatR ).reshape(-1,1)
        
        rloss = ( flux - Q - sigma * uR**4 )
        
        loss = torch.cat((rloss,dloss))
        
        return loss

     
#initialize networks
u = network()

#number of steps in gradient descent
epochs = 15000

#size of gradient descent step
lr = 5e-3

#update 
gamma = 2

#initialize "two optimizers
optimizer = torch.optim.Adam( u.parameters() , lr=lr )

#store values of objective at each step
losses = torch.zeros(epochs)
I = torch.zeros(epochs)
B = torch.zeros(epochs)

#store lagrange multipliers and penalty parameters
lam = torch.zeros(ptsb)
beta = torch.ones(ptsb)

start = time.time()
count = 0
#training loop
for i in range(epochs):
    
    optimizer.zero_grad()
    sf_loss = u.sf(X)
    bc_loss = u.bc(XD,XR)
    loss = sf_loss + torch.sum( dSs * bc_loss * lam.reshape(-1,1) ) + 0.5 * torch.sum( dSs * bc_loss**2 * beta.reshape(-1,1) )
    loss.backward()

    #compute norm of gradient as convergence criterion
    total_norm = 0.0
    for p in u.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)  # L2 norm of this parameter's gradient
            total_norm += param_norm.item()**2
            
    #initial gradient and loss value
    if i==0:
        norm0 = total_norm
        loss0 = loss.item()
    
    optimizer.step()
    losses[i] = loss.item()
    I[i] = dV * torch.sum( torch.abs( u.forward(X).detach() - u_ex ) ) / unorm
    B[i] = torch.sum( dS * torch.abs( u.forward(Xs).detach() - gs ) ) / gnorm
    
    #convergene when gradient drops by fixed amount, and update parameters
    if total_norm / norm0 < 1e-2:
        print('converged')
        
        if B[i] > 7.5e-3:
            lam = lam + beta * bc_loss.detach().squeeze()
            beta = gamma * beta
            count += 1
            
        elif B[i] < 7.5e-3 and abs(loss.item())/loss0 < 5e-3:
            break
    
    if i % 100 == 0:
        print(f'Epoch {i}, Objective {round(losses[i].item(),3)}, Interior {round(I[i].item(),3)}, Boundary {round(B[i].item(),3)}')   

end = time.time()

#time the run
run = end - start

#trim zeros for early convergence
epochs = np.arange(i+1)
nplosses = np.trim_zeros( losses.numpy() , trim='b' ) 
npI = np.trim_zeros( I.numpy() , trim='b' ) 
npB = np.trim_zeros( B.numpy() , trim='b' ) 

#save results
np.savetxt('epochs.txt', epochs , fmt='%.6f' )
np.savetxt('nplosses.txt', nplosses , fmt='%.6f' )
np.savetxt('npI.txt', npI , fmt='%.6f' )
np.savetxt('npB.txt', npB , fmt='%.6f' )
np.savetxt('u.txt', u(X).detach().numpy() , fmt='%.6f' )

#%%

#read results
path = "/home/rowan/Documents/PYTHON/3D_PINNs/ex2/"

epochs   = np.loadtxt(path + "epochs.txt")
nplosses = np.loadtxt(path + "nplosses.txt")
npI      = np.loadtxt(path + "npI.txt")
npB      = np.loadtxt(path + "npB.txt")
u        = np.loadtxt(path + "u.txt") 

nplosses2 = np.loadtxt(path + "lra_nplosses.txt")
npI2 = np.loadtxt(path + "lra_npI.txt")
npB2 = np.loadtxt(path + "lra_npB.txt")

loss_min = min( [ np.min(npI), np.min(npB) , np.min(npI2) , np.min(npB2) ] )
loss_max = max(np.max(nplosses), np.max(nplosses2))

skip = 50
fig = plt.figure(figsize=(14, 5), constrained_layout=True)
plt.rcParams.update({'font.size': 16})
fig.set_constrained_layout_pads(
    w_pad=0.15,  # padding between axes and figure edge (width)
    h_pad=0.15,  # padding between axes and figure edge (height)
    wspace=0.2,  # space between subplots (width)
    hspace=0.3   # space between subplots (height)
)

ax1 = fig.add_subplot(131, projection='3d')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$x_3$')

ax2 = fig.add_subplot(132, projection='3d')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')


ax3 = fig.add_subplot(133)

c1 = np.asarray(u_ex)
c2 = np.asarray(u)

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=min(c1.min(), c2.min()),
                            vmax=max(c1.max(), c2.max()))

sc1 = ax1.scatter(X[:,0], X[:,1], X[:,2], c=c1, cmap='viridis', norm=norm)
sc2 = ax2.scatter(X[:,0], X[:,1], X[:,2], c=c2, cmap='viridis', norm=norm)

for ax in (ax1, ax2):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
ax1.set_title('Exact solution')
ax2.set_title('PINN solution')

# one colorbar for both subplots
cbar = fig.colorbar(sc2, ax=[ax1, ax2], location='right', pad=0.02, fraction=0.05)
cbar.set_label('$u(x)$')

# losses
ax3.plot(epochs[::skip], np.log(np.abs(nplosses))[::skip], label='Objective ($\log(\mathcal{L})$)')
ax3.plot(epochs[::skip], np.log(npI)[::skip], label='Interior ($\log(\mathcal{I})$)')
ax3.plot(epochs[::skip], np.log(npB)[::skip], label='Boundary ($\log(\mathcal{B})$)')
ax3.set_xlabel('epoch')

ax3.set_ylim( [ np.log(loss_min) , np.log(loss_max) ] )

ax3.set_title('Convergence')
ax3.legend()
    
    
    
    