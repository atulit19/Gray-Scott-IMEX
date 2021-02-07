#2D gray scott based imex 1 method
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

from math import exp 

count = 0

######Diffusion Constant and other Constants #######
Du = 2E-5 ; Dv = 1E-5 ; F = 0.046 ; k = 0.063

#####89.5
###GRID_SETTINGS Considering only square########
L = 2
n = 256
x = np.linspace(0,L, n)
y = x
dx = x[1] - x[0]
dy = dx

xx , yy = np.meshgrid(x,y)

xx1 = 0 + 0.5 * (L - 0)
xx2 = 0 + 0.5 * (L - 0)

l1 = L - 0
l2 = L - 0

u_in = np.zeros((n,n))
u_in.fill(1.0)
v_in = np.zeros((n,n))
v_in.fill(0.0)

dist2 = (xx - xx1)**2 + (yy - xx2)**2
u_in = u_in - 0.5 * np.exp(-100 * dist2/l1**2)
v_in = v_in + 0.5 * np.exp(-100 * dist2/l1**2)


xx1 = 0 + 0.55* l1
xx2 = 0 + 0.6 * l2
dist2 = (xx - xx1)**2 + (yy - xx2)**2
u_in = u_in - 0.5 *np.exp(-100 * dist2/l1**2)
v_in = v_in + 0.5 *np.exp(-100 * dist2/l1**2)

plt.imshow(u_in)
plt.show()
 
# #######Time Settings############
# #dt = 0.25*(dx**2)*(1/max(Du,Dv))
# #print(dt)
#term1 = np.amax(np.power(v_in,2)) + F
#term2 = np.amax(u_in*v_in) - F - k
#term3 = 2*max(term1 , term2)
#dt = (1/term3)
#dt = (1/term3)
#print(dt)
# 
dt = .5

# # ==================We are considering a square so for y and x it's same==============
alpha_u = 0.5*Du*(dt)/(dx**2)
alpha_v = 0.5*Dv*(dt)/(dx**2)
t = 0
tmax = 500
#######################Thomas for X#################################
def Thomas_alg_X( d , n_grid , Diffusion):
    mat_D = n_grid**2
    x_n = np.empty(mat_D)
    
    a = np.empty(mat_D) ; b = np.empty(mat_D) ; c = np.empty(mat_D)
    a[:] = -Diffusion ; b[:] = 1+(2*Diffusion) ;  c[:] = -Diffusion ; a[0:n_grid] = 0 ; c[mat_D-n_grid:mat_D] = 0
 
            
    for i in range(1,mat_D):
        m = a[i]/b[i-1]
        b[i] = b[i] - m*c[i-1]
        d[i] = d[i] - m*d[i-1]
    #Backward elimination
    x_n[mat_D-1] = d[mat_D-1]/b[mat_D-1]
    for i in range(mat_D-2,-1,-1):
        temp =   d[i] - ( c[i]*x_n[i+1] ) 
        x_n[i] = temp/b[i]
        
    return x_n




def Thomas_alg_Y( d , n_grid , Diffusion):
    mat_D = n_grid**2
    x_n = np.empty(mat_D)
    
    a = np.empty(mat_D) ; b = np.empty(mat_D) ; c = np.empty(mat_D)
    a[:] = -Diffusion ; b[:] = 1+(2*Diffusion) ;  c[:] = -Diffusion ; a[0:mat_D:n_grid] = 0 ; c[n_grid-1:mat_D:n_grid] = 0
    
    
        
    for i in range(1,mat_D):
        m = a[i]/b[i-1]
        b[i] = b[i] - m*c[i-1]
        d[i] = d[i] - m*d[i-1]
    #Backward elimination
    x_n[mat_D-1] = d[mat_D-1]/b[mat_D-1]
    for i in range(mat_D-2,-1,-1):
        temp =   d[i] - ( c[i]*x_n[i+1] ) 
        x_n[i] = temp/b[i]
        
    return x_n



# # ##################Implcit right hand side############3333
# # 
def Implicit_X_RHS(E_update,b):
    r =  []
    for  i in range (0,n):
        for j in range(0,n):
            if j == 0:
                r.append(b*E_update[i,j+1] -(2*b)*E_update[i,j])
            elif j == n-1:
                r.append(-(2*b)*E_update[i,j] + b*E_update[i,j-1])
            else:
                r.append(b*E_update[i,j+1] -(2*b)*E_update[i,j] + b*E_update[i,j-1]) 
    return np.array(r)
# # 
def Implicit_Y_RHS(E_update,a):
    r =  []
    for  i in range (0,n):
        for j in range(0,n):
            if i == 0:
                r.append(a*E_update[i+1,j] - (2*a)*E_update[i,j])
            elif i == n-1:
                r.append(-(2*a)*E_update[i,j] + a*E_update[i-1,j])
            else:
                r.append(a*E_update[i+1,j] - (2*a)*E_update[i,j] + a*E_update[i-1,j])
            
    return np.array(r)

#############################Explicit terms for u and v
def Explicit_u(u,v):
    square =   np.square(v)
    return ( (-u * square) + F*(1 - u) )

def Explicit_v(u,v):
    square =   np.square(v)
    return ( (u * square) - ((F+k)*v) )


u_update  = u_in
v_update  = v_in

u_update[-1,:] = u_update[-2,:]
u_update[:,-1] = u_update[:,-2]
u_update[0,:] = u_update[1,:]
u_update[:,0] = u_update[:,1]

v_update[-1,:] = v_update[-2,:]
v_update[:,-1] = v_update[:,-2]
v_update[0,:] = v_update[1,:]
v_update[:,0] = v_update[:,1]

while t<tmax:
    
    t = t+dt 
    ########solve for n+0.5* ################### u and v both
    a = u_update + (0.5)*dt*Explicit_u(u_update, v_update)
    a = np.reshape(a,n*n)
    b = Implicit_X_RHS(u_update,alpha_u/2) + 2*Implicit_Y_RHS(u_update, alpha_u/2)   
    u_halfstr_1 = Thomas_alg_X(a+b, n, alpha_u/2)
    u_halfstr_1 = np.reshape(u_halfstr_1,(n,n))
    
    a = v_update + (0.5)*dt*Explicit_v(u_update, v_update)
    a = np.reshape(a,n*n)
    b = Implicit_X_RHS(v_update,alpha_v/2) + 2*Implicit_Y_RHS(v_update, alpha_v/2)   
    v_halfstr_1 = Thomas_alg_X(a+b, n, alpha_v/2)
    v_halfstr_1 = np.reshape(v_halfstr_1,(n,n))
     
    
    a = u_halfstr_1 + 0.5*dt*Explicit_u(u_halfstr_1, v_halfstr_1)
    a = np.reshape(a,n*n)
    b = 2*Implicit_X_RHS(u_halfstr_1,alpha_u/2) + Implicit_Y_RHS(u_halfstr_1, alpha_u/2)   
    u_str_1 = Thomas_alg_Y(a+b, n, alpha_u/2)
    u_str_1 = np.reshape(u_str_1,(n,n))
    
    a = v_halfstr_1 + 0.5*dt*Explicit_v(u_halfstr_1, v_halfstr_1)
    a = np.reshape(a,n*n)
    b = 2*Implicit_X_RHS(v_halfstr_1,alpha_v/2) + Implicit_Y_RHS(v_halfstr_1, alpha_v/2)   
    v_str_1 = Thomas_alg_Y(a+b, n, alpha_v/2)
    v_str_1 = np.reshape(v_str_1,(n,n))
    
    ###############################################33
    u1 = u_update + (0.5*dt)*(Explicit_u(u_update, v_update) + Explicit_u(u_str_1, v_str_1))
    u_avg = 0.5*(u_update + u_str_1)
    F_avg = Implicit_X_RHS(u_avg, 2*alpha_u) + Implicit_Y_RHS(u_avg,2*alpha_u)
    F_avg = np.reshape(F_avg , (n,n))
    u_update = F_avg + u1
    
    v1 = v_update + (0.5*dt)*(Explicit_v(u_update, v_update) + Explicit_v(u_str_1, v_str_1))
    v_avg = 0.5*(v_update + v_str_1)
    F_avg = Implicit_X_RHS(v_avg, 2*alpha_v) + Implicit_Y_RHS(v_avg,2*alpha_v)
    F_avg = np.reshape(F_avg , (n,n))
    v_update = F_avg + v1 
    
    t = t+dt 
    print((t))
    
    #For plotting and stuff. For arrangement purpose 
    #the plots are added to 'TRAP' folder.
    #so an empty folder 'TRAP' have to be added for the program to run
    if count%2==0:
    	plt.figure(count)
    	plt.imshow(u_update, extent=[0,2,0,2] , cmap = "Spectral") 
    	plt.xlabel("x")
    	plt.ylabel("y")
    	plt.title("u")   
    	plt.colorbar()
    	plt.savefig('TRAP/imex'+str(t)+'.png')
    	plt.close()
    count = count + 1


plt.show()






