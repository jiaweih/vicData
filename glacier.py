# !#/usr/bin/env python
'''
Simulate glacier process
Still under development
'''
from pysparse.itsolvers import krylov
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime as datetime
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy import linalg

# mat = scipy.io.loadmat(infile)
# B = mat['B']
# b_dot = mat['b_dot']
# dx = mat['dx'][0,0]
# dy = mat['dy'][0,0]
# i = mat['i'][0,0] ##
# j = mat['j'][0,0] ##
# nx = mat['nx'][0,0]
# ny = mat['ny'][0,0]

'''
Define physical parameters (here assuming EISMINT-1 values)
'''

n_GLEN = 3          # Glen's flow law exponent
A_GLEN = 7.5738e-17 #6.05904e-18; Monthly #7.5738e-17 Cuffey & Paterson (4th ed) Glen's law parameter in Pa^{-3} yr^{-1} units (same as A_GLEN=2.4e-24 Pa^{-3} s^{-1})

m_SLIDE = 2        # Sliding law exponent
C_SLIDE = 0    # 1.0e-08;  # 1.0e-06;  # Sliding coefficient in Pa, metre,(Year units)

RHO = 900   # Density (SI units)
g = 9.80    # Gravity (SI units, rho*g has units of Pa)
K_eps = 1.0e-12
OMEGA = 1.5  # 1.6

MODEL = 3
METHOD = 'BACKSLASH'   ### 'PCG'
	
def main():
	B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str = model_params(MODEL)
	print_out(METHOD,OMEGA,dt,A_GLEN,C_SLIDE,nx,ny)
	timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	N = nx * ny

	# size = np.size(B)
	B = B.T.reshape(N,1)  
	B[np.isnan(B)] = 0

	b_dot = b_dot.T.reshape(N,1)
	S = B

	### Display some diagnostics before proceeding with model integration
	t = 0
	t_SAVE = dt_SAVE
	tm = time.clock()

	while 1:
		# np.savetxt('H.txt',S)
		# print S[100:105]
		S,t,LAMBDA_max,k_LAMBDA_max = step(S, B, b_dot, dt, N, t, nx, ny, dx, METHOD)
		H_max = np.max(S-B)
		k_H_max = np.argmax(S-B)
		S_max = np.max(S)
		k_S_max = np.argmax(S)
		# H_max, k_H_max = np.max(S - B)  ##### @matlab
		# S_max, k_S_max = np.max(S)        ##### @matlab
		# print 'S_max={}'.format(S_max)
		# print 'k_S_max={}'.format(k_S_max)
		# print S

		ALPHA_I = 100*np.sum(S>B)/N
		
		print 'BKS: At t={:8.2f} yr ALPHA_I={:.2f}% and maxima are: H({:d}) = {:f} \
				S({:d})={:f} LAMBDA({:d}) = {:f}\n'.format(t, ALPHA_I, k_H_max, H_max, k_S_max, S_max, k_LAMBDA_max, LAMBDA_max)

		if t > t_STOP:
			break
		
		# I = np.zeros((ny,nx))
		# I[S>B] = 1
		
		# plt.imshow(I)
		# plt.title('Location: {:s}; t = {:.2f} yr'.format(run_str, t))

		# S_map = S.reshape(ny,nx)
		# B_map = B.reshape(ny,nx)
		# I_map = I.reshape(ny,nx)
		now = datetime.datetime.now().strftime('%H:%M:%S')
		file_str = '{:s}_{:f}'.format(run_str, round(t))
		print 'main(): Output stored in file "{:s}" at time {:s} \n'.format('{}.mat'.format(file_str),now)
		
		t_SAVE = t_SAVE + dt_SAVE

	e = time.clock() - tm  ### @matlab
	print 'ALL DONE: Forward modelling'


def step(S, B, b_dot, dt, N, t, nx, ny, dx, METHOD):
	# A_tilde = np.empty
	# if A_tilde.size == 0:
		# A_tilde,C_tilde,nm_half,npl,mm_half,ml = isempty_A_tilde(A_GLEN,RHO,g,n_GLEN,dx,C_SLIDE,m_SLIDE)
	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = SetupIndexArrays(nx,ny) 
	D_IC_jc, D_IP_jc, D_ic_JC, D_ic_JP = diffusion_gl(S,B,nx,ny,dx)
	D_sum = D_IC_jc + D_IP_jc + D_ic_JC + D_ic_JP
	
	row = np.array([[ic_jc],[ic_jc],[ic_jc],[ic_jc],[ic_jc]]).reshape(-1)
	# row = row.T.reshape(row.size,1).T
	col = np.array([[im_jc],[ip_jc],[ic_jm],[ic_jp],[ic_jc]]).reshape(-1)
	# col = col.T.reshape(col.size,1).T
	val = np.array([[-OMEGA*D_IC_jc],[-OMEGA*D_IP_jc],[-OMEGA*D_ic_JC],[-OMEGA*D_ic_JP],[1/dt+OMEGA*D_sum]]).reshape(-1)
	# np.savetxt('val.txt',val)
	# row = row - 1
	# col = col - 1
	# print 'row={}'.format(row)
	# print 'col={}'.format(col)
	np.savetxt('row.txt',row)
	np.savetxt('col.txt',col)

	A = csr_matrix( (val,(row,col)), shape=(N,N)).todense()   ### matrix A is symmetric positive definite
	C = (1 - OMEGA) * ((D_IC_jc * S[im_jc]) + D_IP_jc * S[ip_jc] + D_ic_JC * S[ic_jm] + D_ic_JP * S[ic_jp]) + (1/dt - (1 - OMEGA) * D_sum) * S[ic_jc] + b_dot 
	# np.savetxt('A.txt',A)
	np.savetxt('C.txt',C)

	S_out = solver(A,C,METHOD)
	# S_out[S_out<B] = B[B>S_out]
	np.savetxt('S_out.txt',S_out)
	
	H_out = S_out - B
	t_n = t + dt
	
	D_max = np.max(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)
	k_LAMBDA_max = np.argmax(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)
	# D_max,k_LAMBDA_max = np.max(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)  ######### @matlab
	LAMBDA_max = 0.25 * dt * D_max
	
	return S_out,t_n,LAMBDA_max,k_LAMBDA_max


def diffusion_gl(S,B,nx,ny,dx):
	A_tilde = 2*A_GLEN*(RHO*g)**n_GLEN/(n_GLEN+2)/dx**2
	C_tilde = C_SLIDE*(RHO*g)**m_SLIDE/dx**2
	nm_half = (n_GLEN-1)/2
	npl = n_GLEN+1
	mm_half = (m_SLIDE-1)/2
	ml = m_SLIDE
	
	# SB = S-B
	SB = S
	SB[SB<0] = 0
	H = SB
	# np.savetxt('H.txt',S)
	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = SetupIndexArrays(nx,ny) 


	# np.savetxt('ic_jc.txt',ic_jc)
	# np.savetxt('im_jc.txt',im_jc)
	
	H_IC_jc = 0.5*(H[ic_jc] + H[im_jc])
	H_ic_JC = 0.5*(H[ic_jc] + H[ic_jm])

	# np.savetxt('H_IC_jc.txt',H_IC_jc)
	
	H_IC_jc_up = H[im_jc]
	H_ic_JC_up = H[ic_jm]
	
	ix = (S[ic_jc]>S[im_jc]).reshape(-1)
	H_IC_jc_up[S[ic_jc]>S[im_jc]] = H[ic_jc[ix]].reshape(-1)
	# np.savetxt('H_IC_jc_up.txt',H_IC_jc_up)

	ix = (S[ic_jc]>S[ic_jm]).reshape(-1)
	H_ic_JC_up[S[ic_jc]>S[ic_jm]] = H[ic_jc[ix]].reshape(-1)
	
	dS_dx_IC_jc = (S[ic_jc]-S[im_jc])/dx
	dS_dy_IC_jc = (S[ic_jp]+S[im_jp]-S[ic_jm]-S[im_jm])/(4*dx)
	dS_dx_ic_JC = (S[ip_jc]+S[ip_jm]-S[im_jc]-S[im_jm])/(4*dx)
	dS_dy_ic_JC = (S[ic_jc]-S[ic_jm])/dx

	# np.savetxt('dS_dx_IC_jc.txt',dS_dx_IC_jc)
	
	S2_IC_jc = np.square(dS_dx_IC_jc) + np.square(dS_dy_IC_jc) + K_eps
	S2_ic_JC = np.square(dS_dx_ic_JC) + np.square(dS_dy_ic_JC) + K_eps
	# np.savetxt('S2_IC_jc.txt',S2_IC_jc)
	
	if C_tilde == 0:    ### No sliding case
		D_IC_jc = A_tilde*H_IC_jc_up*np.power(H_IC_jc,npl)*np.power(S2_IC_jc,nm_half)
		D_ic_JC = A_tilde*H_ic_JC_up*np.power(H_ic_JC,npl)*np.power(S2_ic_JC,nm_half)
	elif C_tilde > 0:    ### Sliding case
		D_IC_jc = A_tilde*H_IC_jc_up*np.power(H_IC_jc,npl)*np.power(S2_IC_jc,nm_half) \
				+ C_tilde*H_IC_jc_up*np.power(H_IC_jc,ml)*np.power(S2_IC_jc,mm_half)
		D_ic_JC = A_tilde*H_ic_JC_up*np.power(H_ic_JC,npl)*np.power(S2_ic_JC,nm_half) \
				+ C_tilde*H_ic_JC_up*np.power(H_ic_JC,ml)*np.power(S2_ic_JC,mm_half)
	else:
		print 'diffusion(): C_tilde is undefined or incorrectly defined'
		
	D_IP_jc  = D_IC_jc[ip_jc]
	D_ic_JP  = D_ic_JC[ic_jp]
	# np.savetxt('D_IP_jc.txt',D_IP_jc)
	
	return D_IC_jc,D_IP_jc,D_ic_JC,D_ic_JP


def SetupIndexArrays(nx, ny):
	N = nx * ny

	ic_jc = np.arange(1,N+1)  
	ic_jc = ic_jc.reshape(nx,ny)

	ic = np.arange(nx)
	ip = np.append(np.array([range(1,nx)]),nx - 1)
	im = np.append(0,np.array([range(nx - 1)]))

	jc = np.arange(ny)
	jp = np.append(0,np.array([range(ny - 1)]))
	jm = np.append(np.array([range(1,ny)]),ny - 1)

	ip_jc = setupArrays(ip,jc,ic_jc) - 1
	im_jc = setupArrays(im,jc,ic_jc) - 1
	ic_jp = setupArrays(ic,jp,ic_jc) - 1
	ic_jm = setupArrays(ic,jm,ic_jc) - 1

	im_jm = setupArrays(im,jm,ic_jc) - 1
	ip_jm = setupArrays(ip,jm,ic_jc) - 1
	im_jp = setupArrays(im,jp,ic_jc) - 1
	ip_jp = setupArrays(ip,jp,ic_jc) - 1

	ic_jc = ic_jc.reshape(-1) - 1

	return ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp


def setupArrays(a,b,ic_jc):
	x,y = np.meshgrid(b,a)
	array = []
	for l in zip(y.ravel(),x.ravel()):
		array.append(ic_jc[l])
	array = np.array(array)
	return array


def solver(A,C,METHOD):
	upper = str.upper(METHOD)
	if upper == 'BACKSLASH':
		S_out = np.linalg.solve(A,C)
	elif upper == 'PCG':
		tol = 1.0e-06       ### 1.0e-09;
		MAXIT = 100
		S_out = np.empty(N*N)
		info, itern, relres = krylov.pcg(A, C, S_out, tol, MAXIT)
		if info != 0:
			if info == -1:
				print 'step(): pcg did not converge after MAXIT iterations at t={:f} yr'.format(t)
			elif info == -2:
				print 'step(): pcg preconditioner is ill-conditioned at t={:f} yr'.format(t)
			elif info == -5:
				print 'step(): pcg stagnated at t={:f} yr'.format(t)
			elif info == -6:
				print 'step(): one of the scalar quantities in pcg became too small or too large at t={:f} yr'.format(t)
			else:
				print 'step(): Unknown pcg flag raised, flat={}'.format(info)
	return S_out

def print_out(METHOD,OMEGA,dt,A_GLEN,C_SLIDE,nx,ny):
    print '=================================================================================='
    print 'LAUNCHING GLACIER SIMULATION MODEL - Ver 5.01 using the {:s} solver\n\n'.format(METHOD)
    print '  OMEGA      = {:.2f}\n'.format(OMEGA)
    print '  dt         = {:.2f} yr\n'.format(dt)
    print '  A_GLEN     = {:e}\n'.format(A_GLEN)
    print '  C_SLIDE    = {:e}\n'.format(C_SLIDE)
    print '  nx         = {:d}\n'.format(nx)
    print '  ny         = {:d}\n'.format(ny)
    if METHOD.find('ADI') != -1:
        print '  ADI METHOD = {%s}\n'.format(ADI_METHOD)
    print '=================================================================================='

def model_params(MODEL):
	case = {0: case_0, 1: case_1, 2: case_2, 3: case_3, 4: case_4}
	if MODEL in range(5):
		B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str = case[MODEL]()
	else:
		print 'main_forward(): Unprogrammed MODEL'
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str


def read_mat(infile):
	mat = io.loadmat(infile)
	B = mat['B']
	b_dot = mat['b_dot']
	dx = mat['dx'][0,0]
	dy = mat['dy'][0,0]
	i = mat['i'][0,0] 
	j = mat['j'][0,0] 
	nx = np.int_(mat['nx'][0,0])
	ny = np.int_(mat['ny'][0,0])
	return B,b_dot,dx,dy,ny,nx


def case_0():
	run_str = 'Toy model'
	dt = 1000
	dt_SAVE = 5 * dt
	t_STOP = 25000
	
	ny = 11
	nx = 11
	N = nx*ny
	
	dx = 100    ### 200
	dy = 100    ### 200
	
	x = np.linspace(0, dx*(nx - 1), nx)
	y = np.linspace(dx*(ny - 1), 0, ny)
	
	L_x = dx*(nx - 1)
	L_y = dy*(ny - 1)
	
	R0 = 0.5*L_x
	
	x_c = 0.5*L_x
	y_c = 0.5*L_y
	
	X, Y = np.meshgrid(x,y)     
	
	Z0 = 2000
	sigma_x = x_c
	sigma_y = y_c
	R2 = np.square(X-x_c) + np.square(Y-y_c)
	B = Z0*np.exp(-R2/R0**2)  ##
	
	B_min = np.min(B)   ##
	B_max = np.max(B)   ##
	b_dot_melt = -2 + 2*(B - B_min)/(B_max - B_min)
	b_dot_ppt = 1
	b_dot = b_dot_melt + b_dot_ppt
	
	#### differ from matlab
	B[4,5] = B[4,5] - 100
	B[5,5] = B[5,5] + 100
	B[6,5] = B[6,5] - 100
	
	B[5,4] = B[5,4] + 200
	B[5,6] = B[5,6] + 200
	
	B = B + 5000
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str

def case_1():
	run_str = 'problem_1'
	file_dat = '{}.mat'.format(run_str)
	B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
	
	t_STOP = 500
	dt_SAVE = 5*t_STOP
	dt = 1
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str

def case_2():
	run_str = 'problem_1 - ascii'
	file_dat = os.path.join('/', 'data', 'LettenmaierCoupledModel', 'Onestep', 'v_200', 'data', 'problem_1.dat')
	##   [B, b_dot, dx, dy, ny, nx] = LoadAsciiData(file_dat);
	#B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
	t_STOP = 500
	dt_SAVE = 5*t_STOP
	dt = 1
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str

def case_3():
	run_str = 'mb4_spin1'
	#file_dat = os.path.join('M:', 'DHSVM', 'washington', 'cascade','spin','{}.mat'.format(run_str)) #####
	file_dat = '{}.mat'.format(run_str)
	B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
	
	t_STOP = 0.07        ### 1000
	dt_SAVE = 5*t_STOP
	dt = 0.08333
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str

def case_4():
	run_str = 'manipulate9_mth'
	file_dat = os.path.join('M:', 'DHSVM', 'bolivia', 'spinup', 'spin_up','{}.mat'.format(run_str))
	B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
	
	t_STOP = 12000
	dt_SAVE = 5*t_STOP
	dt = 1.0
	return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt,run_str

	print '=================================================================================='
	print 'LAUNCHING GLACIER SIMULATION MODEL - Ver 5.01 using the {:s} solver\n\n'.format(METHOD)
	print '  OMEGA      = {:.2f}\n'.format(OMEGA)
	print '  dt         = {:.2f} yr\n'.format(dt)
	print '  A_GLEN     = {:e}\n'.format(A_GLEN)
	print '  C_SLIDE    = {:e}\n'.format(C_SLIDE)
	print '  nx         = {:d}\n'.format(nx)
	print '  ny         = {:d}\n'.format(ny)
	if METHOD.find('ADI') != -1:
		print '  ADI METHOD = {%s}\n'.format(ADI_METHOD)
	print '=================================================================================='

### if A_tilde.size == 0, then execute the following function
def isempty_A_tilde(A_GLEN,RHO,g,n_GLEN,dx,C_SLIDE,m_SLIDE):
	A_tilde = 2*A_GLEN*(RHO*g)**n_GLEN/((n_GLEN+2)*dx**2)
	C_tilde = C_SLIDE*(RHO*g)**m_SLIDE/dx**2
	nm_half = (n_GLEN-1)/2
	npl = n_GLEN+1
	mm_half = (m_SLIDE-1)/2
	ml = m_SLIDE
	return A_tilde,C_tilde,nm_half,npl,mm_half,ml

if __name__ == "__main__":
	main()