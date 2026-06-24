    
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

#%%

#GEOMETRY AND INTEGRATION
R1 = 0.2
R2 = 0.1
def level_set(X):
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    #pipe with notch https://www.desmos.com/3d/pukwjsdfew
    phi = ( ( z - 0.5 )**2 + ( y - 0.5 )**2 - ( R1**2 + 0.2 * torch.exp(-100*(x-0.5)**2) ) ) * ( ( z - 0.5 )**2 + ( y - 0.5 )**2 - R2**2 )
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
ds = torch.zeros( ( len(faces) , 1 ) )
for i in range(len(faces)):
    V = verts[faces[i]]
    Xs[i,:] = torch.mean( V , axis=0 )
    a , b , c = V
    u = b - a
    v = c - a
    ds[i] = 0.5 * torch.linalg.norm(torch.cross(u, v))
    
    #facet normal vector, unsigned
    nhat_local = torch.cross(u,v) / torch.linalg.norm(torch.cross(u, v))
    
    #compute the sign
    if level_set( (Xs[i,:] + 0.025 * nhat_local).unsqueeze(0) ) < level_set( Xs[i,:].unsqueeze(0) ):
        nhat_local = -nhat_local
    
    nhat[i,:] = nhat_local


pts = len(X)
ptsb = len(Xs)

dS = torch.zeros((ptsb,3))
dS[:,0] = ds.squeeze()
dS[:,1] = ds.squeeze()
dS[:,2] = ds.squeeze()

dS = dS.flatten().reshape(-1,1)

#constitutive relation
E = 1
nu = 0.3
lam = E * nu / ( (1 + nu) * (1 - 2 * nu) )
mu = E / ( 2 * (1 + nu) )
delta = torch.eye(3)
C = torch.zeros( 3 , 3 , 3 , 3 )
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C[i,j,k,l] = lam*delta[i,j]*delta[k,l] + mu*( delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k] )

# fig = plt.figure(figsize=(14, 6), constrained_layout=True)
# plt.rcParams.update({'font.size': 16})
# fig.set_constrained_layout_pads(
#     w_pad=0.15,  # padding between axes and figure edge (width)
#     h_pad=0.15,  # padding between axes and figure edge (height)
#     wspace=0.2,  # space between subplots (width)
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
# ax.set_title('Pipe geometry')
# ax.scatter(X[:,0], X[:,1], X[:,2])


# step = 0.1
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
# ax.set_title('Normal vectors')
# ax.scatter(Xs[:,0], Xs[:,1], Xs[:,2],color='blue')
# ax.scatter(Xs[:,0]+step*nhat[:,0], Xs[:,1]+step*nhat[:,1], Xs[:,2]+step*nhat[:,2],color='red')

#%%

xx , yy , zz = sym.symbols('xx yy zz', real=True)

#distance from axis of cylinder
r = ( ( yy - 0.5 )**2 + ( zz - 0.5 )**2 )**0.5

#assumed solution
u = sym.zeros(1,3)
u0 = 25
u[1] = u0 * ( yy - 0.5 ) * r * sym.sin( sym.pi * xx )
u[2] = u0 * ( zz - 0.5 ) * r * sym.sin( sym.pi * xx )

#numerical function
u_num2 = sym.lambdify( ( xx , yy , zz ) , u[1] , 'numpy' )
u_num3 = sym.lambdify( ( xx , yy , zz ) , u[2] , 'numpy' )

def u_check(X):
    u = torch.zeros((len(X),3))
    u[:,1] = u_num2( X[:,0] , X[:,1] , X[:,2] )
    u[:,2] = u_num3( X[:,0] , X[:,1] , X[:,2] )
    return u

#strain tensor
grad_u = u.jacobian( [ xx , yy , zz ] )                          
eps = 0.5 * (grad_u + grad_u.T) 

#convenient way to write stress
sigma = lam * sym.trace(eps) * sym.eye(3) + 2 * mu * eps 

#stress equilibrium
f1_sym = - ( sym.diff( sigma[0,0] , xx ) + sym.diff( sigma[0,1] , yy ) + sym.diff( sigma[0,2] , zz ) )
f2_sym = - ( sym.diff( sigma[1,0] , xx ) + sym.diff( sigma[1,1] , yy ) + sym.diff( sigma[1,2] , zz ) )
f3_sym = - ( sym.diff( sigma[2,0] , xx ) + sym.diff( sigma[2,1] , yy ) + sym.diff( sigma[2,2] , zz ) )

#numerical functions
f1_num = sym.lambdify( ( xx , yy , zz ) , f1_sym , 'numpy' )
f2_num = sym.lambdify( ( xx , yy , zz ) , f2_sym , 'numpy' )
f3_num = sym.lambdify( ( xx , yy , zz ) , f3_sym , 'numpy' )

def f(X):
    force = torch.zeros((len(X),3))
    force[:,0] = f1_num( X[:,0] , X[:,1] , X[:,2] )
    force[:,1] = f2_num( X[:,0] , X[:,1] , X[:,2] )
    force[:,2] = f3_num( X[:,0] , X[:,1] , X[:,2] )
    return force

#precompute forcing
F = f(X)

#precompute exact solution
u_ex = u_check(X)
unorm = dV * torch.sum( ( u_ex[:,0]**2  + u_ex[:,1]**2 + u_ex[:,2]**2 )**0.5 )

#solution along dirichlet boundary
g = u_check(Xs)
gnorm = torch.sum( ds.squeeze() * (  g[:,0]**2 + g[:,1]**2 + g[:,2]**2 )**0.5 )

#%%

class network(nn.Module):
    def __init__( self ):
        super().__init__()
        
        N = 50
        
        #define layers and activation
        self.layer_1 = nn.Linear( 3 , N )
        self.layer_2 = nn.Linear( N , N )
        self.output = nn.Linear( N , 3 , bias=False )
        self.act = nn.Tanh()
        
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        y = self.output(y)
        
        #enforce boundary conditions
        D = torch.sin( np.pi * x[:,0] )
        
        y = D.reshape(-1,1) * y
        
        return y
    
    def sf( self , X ):
        
        X = X.clone().detach().requires_grad_(True)
        u = self.forward(X)
        
        #gradient and hessian matrices at all integration points
        G = torch.zeros( len(X) , 3 , 3 )
        H = torch.zeros( len(X) , 3 , 3 , 3 )
        
        #loop over displacement components
        for k in range(3):
            grad_uk = torch.autograd.grad( u[:,k] , X , grad_outputs=torch.ones_like(u[:,k]) , create_graph=True )[0]
            G[ : , k , : ] = grad_uk
            for j in range(3):
                grad_gradj_uk = torch.autograd.grad( grad_uk[:,j] , X , grad_outputs=torch.ones_like(grad_uk[:,j]) , create_graph=True )[0]
                H[ : , k , j , : ] = grad_gradj_uk
        
        comps = torch.einsum( 'ijkl,xkjl->xi' , C , H )
        res = comps + F
        loss = 0.5 * dV * torch.sum( res**2 )
       
        return loss
    
    def bc( self , Xs ):
        
        #flattens bcs into a vector
        u = self.forward(Xs)
        loss = ( u - g ).flatten().reshape(-1,1)

        return loss

#initialize networks
u = network()

#number of steps in gradient descent
epochs = 15000

#size of gradient descent step
lr = 1e-3

#update 
gamma = 2

#initialize "two optimizers
optimizer = torch.optim.Adam( u.parameters() , lr=lr )

#store values of objective at each step
losses = torch.zeros(epochs)
I = torch.zeros(epochs)
B = torch.zeros(epochs)

#store lagrange multipliers and penalty parameters
lam = torch.zeros(3*ptsb)
beta = torch.ones(3*ptsb)

start = time.time()
count = 0
#training loop
for i in range(epochs):
    
    optimizer.zero_grad()
    sf_loss = u.sf(X)
    bc_loss = u.bc(Xs)
    loss = sf_loss + torch.sum( dS * bc_loss * lam.reshape(-1,1) ) + 0.5 * torch.sum( dS * bc_loss**2 * beta.reshape(-1,1) )
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
    
    dif_u = u.forward(X).detach() - u_ex
    dif_us = u.forward(Xs).detach() - g
    
    I[i] = dV * torch.sum(  ( dif_u[:,0]**2 + dif_u[:,1]**2 + dif_u[:,2]**2 )**0.5 ) / unorm
    B[i] = torch.sum( ds.squeeze() * ( dif_us[:,0]**2 + dif_us[:,1]**2 + dif_us[:,2]**2 )**0.5 ) / gnorm
    
    #convergene when gradient drops by fixed amount, and update parameters
    if total_norm / norm0 < 1e-2:
        print('converged')
        
        if B[i] > 5e-3:
            lam = lam + beta * bc_loss.detach().squeeze()
            beta = gamma * beta
            count += 1
            
        elif B[i] < 5e-3 and abs(loss.item())/loss0 < 2.5e-3:
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
epochs = np.loadtxt('epochs.txt')
nplosses = np.loadtxt('nplosses.txt')
npI = np.loadtxt('npI.txt')
npB = np.loadtxt('npB.txt')
u = np.loadtxt('u.txt') / u0

path = "/home/rowan/Documents/PYTHON/3D_PINNs/ex3/"

epochs   = np.loadtxt(path + "epochs.txt")
nplosses = np.loadtxt(path + "nplosses.txt")
npI      = np.loadtxt(path + "npI.txt")
npB      = np.loadtxt(path + "npB.txt")
u        = np.loadtxt(path + "u.txt") / u0

nplosses2 = np.loadtxt(path + "lra_nplosses.txt")
npI2 = np.loadtxt(path + "lra_npI.txt")
npB2 = np.loadtxt(path + "lra_npB.txt")

loss_min = min( [ np.min(npI), np.min(npB) , np.min(npI2) , np.min(npB2) ] )
loss_max = max(np.max(nplosses), np.max(nplosses2))

    
# import matplotlib as mpl
# skip = 50
# plt.rcParams.update({'font.size': 16})

# fig = plt.figure(figsize=(12, 12), layout='constrained')  # use new API
# fig.set_constrained_layout_pads(
#     w_pad=0.15,   # padding between axes and figure edge (inches)
#     h_pad=0.15,
#     wspace=0.1,  # space between columns
#     hspace=0.12   # space between rows
# )
# gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])

# # 2×2 grid for plots; thin last column for colorbar
# ax1 = fig.add_subplot(gs[0,0], projection='3d')
# ax2 = fig.add_subplot(gs[0,1], projection='3d')
# ax3 = fig.add_subplot(gs[1,0])
# ax4 = fig.add_subplot(gs[1,1] , projection='3d')
# cax = fig.add_subplot(gs[1,2])  # colorbar spans both rows

# for ax in (ax1, ax2, ax4):
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
#     ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$x_3$')
#     ax.set_box_aspect((1, 1, 1))  # keep 3D panes cubic

# # data (as in your code)
# c1 = np.asarray(u_ex)
# c2 = np.asarray(u)
# dif_u = u0 * u - u_ex.numpy()
# c4 = np.sqrt(dif_u[:,0]**2 + dif_u[:,1]**2 + dif_u[:,2]**2)

# norm = mpl.colors.Normalize(vmin=c4.min(), vmax=c4.max())

# # plots
# sc1 = ax1.scatter(X[:,0], X[:,1], X[:,2])  # ref
# sc2 = ax2.scatter(X[:,0]+u[:,0], X[:,1]+u[:,1], X[:,2]+u[:,2])  # deformed
# sc4 = ax4.scatter(X[:,0], X[:,1], X[:,2], c=c4, cmap='viridis', norm=norm)  # error

# for ax in (ax1, ax2):
#     ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)

# ax1.set_title('Reference configuration')
# ax2.set_title('Deformed configuration')
# ax4.set_title('Error distribution')

# # colorbar in its own axes -> grid stays aligned
# cbar = fig.colorbar(sc4, cax=cax)
# # cbar.set_label('$u(x)$')  # optional

# # losses
# ax3.plot(epochs[::skip], np.log(np.abs(nplosses))[::skip], label='Objective ($\\log\\mathcal{L}$)')
# ax3.plot(epochs[::skip], np.log(npI)[::skip], label='Interior ($\\log\\mathcal{I}$)')
# ax3.plot(epochs[::skip], np.log(npB)[::skip], label='Boundary ($\\log\\mathcal{B}$)')
# ax3.set_xlabel('epoch'); ax3.set_title('Convergence'); ax3.legend()


# plt.show()


#UPDATED PLOTS


import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(16, 5), layout='constrained')
fig.set_constrained_layout_pads(
    w_pad=0.15,
    h_pad=0.15,
    wspace=0.25,
    hspace=0.1
)

# 1 row, 4 columns: iso | error | cbar | losses
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1])

ax1 = fig.add_subplot(gs[0,0], projection='3d')   # isosurface
ax2 = fig.add_subplot(gs[0,1], projection='3d')   # error distribution
cax = fig.add_subplot(gs[0,2])                    # colorbar (next to plot 2)
ax3 = fig.add_subplot(gs[0,3])                    # losses

# -------------------------------
# 1. Isosurface plot
# -------------------------------
mesh = Poly3DCollection(verts[faces], alpha=0.7)
mesh.set_facecolor([0.6, 0.75, 1.0])
mesh.set_edgecolor('k')
mesh.set_linewidth(0.1)
ax1.add_collection3d(mesh)

ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.set_zlim(0,1)
ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$'); ax1.set_zlabel('$x_3$')
ax1.set_title("Pipe geometry")
ax1.set_box_aspect((1,1,1))
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

# -------------------------------
# 2. Error distribution
# -------------------------------
dif_u = u0 * u - u_ex.numpy()
c4 = np.sqrt(np.sum(dif_u**2, axis=1))

#MANUALLY SET LIMITS SO SCALES AGREE ON TWO PLOTS
#norm = mpl.colors.Normalize(vmin=c4.min(), vmax=c4.max())
norm = mpl.colors.Normalize(vmin=0, vmax=0.071)


sc2 = ax2.scatter(X[:,0], X[:,1], X[:,2], c=c4, cmap='viridis', norm=norm)

ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_zlim(0,1)
ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$'); ax2.set_zlabel('$x_3$')
ax2.set_title("Error distribution")
ax2.set_box_aspect((1,1,1))
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

# Colorbar positioned *right next to plot 2*
fig.colorbar(sc2, cax=cax)

# -------------------------------
# 3. Loss curves
# -------------------------------
skip = 50
ax3.plot(epochs[::skip], np.log(np.abs(nplosses))[::skip], label='Objective')
ax3.plot(epochs[::skip], np.log(npI)[::skip], label='Interior')
ax3.plot(epochs[::skip], np.log(npB)[::skip], label='Boundary')

ax3.set_ylim( [ np.log(loss_min) , np.log(loss_max) ] )

ax3.set_xlabel('Epoch')
ax3.set_title('Convergence')
ax3.legend()

plt.show()



#%%


# skip = 50
# fig = plt.figure(figsize=(12, 12), constrained_layout=True)
# plt.rcParams.update({'font.size': 16})
# fig.set_constrained_layout_pads(
#     w_pad=0.15,  # padding between axes and figure edge (width)
#     h_pad=0.15,  # padding between axes and figure edge (height)
#     wspace=0.15,  # space between subplots (width)
#     hspace=0.1   # space between subplots (height)
# )

# ax1 = fig.add_subplot(221, projection='3d')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_zticks([])
# ax1.set_xlabel('$x_1$')
# ax1.set_ylabel('$x_2$')
# ax1.set_zlabel('$x_3$')

# ax2 = fig.add_subplot(222, projection='3d')
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax2.set_zticks([])
# ax2.set_xlabel('$x_1$')
# ax2.set_ylabel('$x_2$')
# ax2.set_zlabel('$x_3$')

# ax3 = fig.add_subplot(223, projection='3d')
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax3.set_zticks([])
# ax3.set_xlabel('$x_1$')
# ax3.set_ylabel('$x_2$')
# ax3.set_zlabel('$x_3$')

# #pointwise error with exact solution
# dif_u = u - u_ex.numpy()
# c3 = ( dif_u[:,0]**2 + dif_u[:,1]**2 + dif_u[:,2]**2 )**0.5

# import matplotlib as mpl
# norm = mpl.colors.Normalize(vmin=min(c3.min(), c3.min()),
#                             vmax=max(c3.max(), c3.max()))

# sc1 = ax1.scatter(X[:,0], X[:,1], X[:,2], cmap='viridis')
# sc2 = ax2.scatter(X[:,0] + u[:,0] , X[:,1] + u[:,1] , X[:,2]+u[:,2] )
# sc3 = ax3.scatter(X[:,0], X[:,1], X[:,2], c=c3, cmap='viridis', norm=norm)

# for ax in (ax1, ax2):
#     ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
# ax1.set_title('Reference configuration')
# ax2.set_title('Deformed configuration')
# ax3.set_title('Error distribution')

# # one colorbar for both subplots
# cbar = fig.colorbar(sc3, ax=[ax3], location='left', pad=0.15, fraction=0.1 )
# #cbar.set_label('$u(x)$')

# ax4 = fig.add_subplot(224)
# ax4.plot(epochs[::skip], np.log(np.abs(nplosses))[::skip], label='Objective ($\log(\mathcal{L})$)')
# ax4.plot(epochs[::skip], np.log(npI)[::skip], label='Interior ($\log(\mathcal{I})$)')
# ax4.plot(epochs[::skip], np.log(npB)[::skip], label='Boundary ($\log(\mathcal{B})$)')
# ax4.set_xlabel('epoch')
# ax4.set_title('Convergence')
# ax4.legend()




