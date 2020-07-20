import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import math as mth
import scipy
import scipy.sparse as sparse
from qutip import *
from scipy import sparse, linalg
from scipy.interpolate import interp1d
import itertools as it
from itertools import *
from collections import Counter 
from timeit import default_timer as timer
import time
from mpl_toolkits.mplot3d import Axes3D
import imageio

# GENERAL HELPERS #######################################################################################

# a = number; n = number of bits in representtion
def flipBits(a,n,returnType="int"):
    if returnType=="int":
        return int(bin((a ^ (2 **(n+1) - 1)))[3:],2)
    else:
        return bin((a ^ (2 **(n+1) - 1)))[3:]

# express the state n = # as a list of 2**L bits
def bitsToList(n,L):
    return list(map(int,list(np.binary_repr(n,2**L))))

# takes in two matrices and computes direct sum
def directSum(A, B):
    dsum = np.zeros(np.add(A.shape, B.shape), dtype=complex)
    dsum[:A.shape[0], :A.shape[1]] = A
    dsum[A.shape[0]:, B.shape[1]:] = B
    return dsum

# circularly shifts lists by n elements
def circshift(arr,n=1,direction="left"):
    if direction=="left":
        return arr[n::] + arr[:n:] 
    else:
        return arr[-n:] + arr[0:-n] 

# checks if two lists have same cyclic order
def checkCyclic(l1, l2):
    if len(l1) != len(l2):
        raise Exception("Lists have different size!")
    if not (set(l1)==set(l2)):
        raise Exception("Lists have different elements!")
    
    first = l1[0]
    idx = l2.index(first)
    cycleL2 = circshift(l2,idx)
    return l1==cycleL2

# given two lists with same elements return levi civita
def levicivita(l):
    cyclic = np.arange(min(l),max(3,max(l))+1,step=1).tolist()
    setDiff = np.setdiff1d(cyclic,l)
    lorder = l.copy()
    lorder.extend(setDiff)
    returnVals = [-1,1]   # TRUE = 1, FALSE = 0
    return returnVals[checkCyclic(lorder,cyclic)]

# returns image of matrix with nonzero elements colored black
def getIm(H):
    Im = np.logical_not(abs(H - 0) < 1e-13);
    return Im

# finds the relative angle between two angular positions
def getThetaRel(th1,th2):
    return abs((th1%(2*np.pi))-th2%(2*np.pi))

# get list of coordinates as a dxN array where N is the total number of lattice points
# d = dimension
# a = lattice spacing along each axis/dimension; if is a number, then uniform spacing
# L = number of sites along each axis/dimension; if is a number, then square grid; ow L is dx1 vector giving dimensions
def getLatticeCoord(d,L,a):
    L = L*np.ones(d)
    a = a*np.ones(d)
    coordvecs = tuple([np.arange(L[i]*a[i],step=a[i]) for i in range(d)])
    coord = np.meshgrid(*coordvecs)
    return np.stack(list(map(np.ravel, coord)),axis=1)


# coord = Lx3 matrix where L = # of spins
# returns the distance and cos(th) of all pairs of ising spins from coordinates
def get3DThetaDistCoord(coord):
    xcoord = coord[:,0]
    ycoord = coord[:,1]
    zcoord = coord[:,2]
    
    # displacement coordinates
    xi,xj = np.meshgrid(xcoord,xcoord,sparse=True)
    yi,yj = np.meshgrid(ycoord,ycoord,sparse=True)
    zi,zj = np.meshgrid(zcoord,zcoord,sparse=True)
    zd = zi - zj
    r_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zd)**2)  # mag of displacement
    costh_ij = np.divide(zi-zj,r_ij + np.eye(r_ij.shape[0]),out=np.zeros_like(r_ij),where=(zi-zj)!=0)
    return r_ij,costh_ij

# returns matrix of distances between each pair of spins
def pairwiseDist(coord):
    sqdist = 0
    for i in range(coord.shape[1]):
        xi,xj = np.meshgrid(coord[:,i],coord[:,i],sparse=True)
        sqdist = sqdist + (xi-xj)**2
    r_ij = np.sqrt(sqdist)
    return r_ij


# # Define function for calculating a power law
# powerLaw = lambda x, A, b: A * (x**b)


# EXACT DIAGONALIZATION HELPERS #######################################################################################

# OPERATOR MECHANICS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Hconj(op):
    return np.conjugate(np.transpose(op))

def outerprod(v1,v2):
    return np.outer(v1,np.conjugate(v2))

# returns a PRODUCT of many-body operators, each elem in op_list acts on tensor of L sites
# opList = LIST of NP ARRAYS
def opProd(opList):
    if isinstance(opList[0],np.ndarray):
        P = np.eye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P@opList[i] 
        return P
    elif isinstance(opList[0],Qobj):
        P = qeye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P*opList[i]
        return P
    elif isinstance(opList[0],sparse.csr_matrix):
        P = sparse.eye(opList[0].shape[0])
        for i in range(len(opList)):
            P = P*opList[i]
        return P
    else:
        raise Exception("unknown type")

# define many-body operator on N qubits from list of LOCAL qubit ops at locations given by pos
# pos indexed at 0
def tensorOp(N, op, pos):
    idxCheck = len(op) == len(pos)  # make sure op and pos have same size
    if not (idxCheck):
        raise Exception('Number of operators doesn''t match number of positions')
    opList = [qeye(2)]*(pos[0]) + [op[0]]
    
    for i in range(1,len(op)):
        opList = opList +[qeye(2)]*(pos[i]-pos[i-1]-1) + [op[i]]
    opList = opList + [qeye(2)]*(N-pos[-1]-1)
    return tensor(opList)

# generates list of Paulis on N sites
def genPauliList(N,returnType = "qobj"): 
    # initialize lists
    sx_list = []
    sy_list = [] 
    sz_list = []
    sp_list = []
    sm_list = []
    
    for i in range(N):
        sx = [qeye(2)]*(i) + [sigmax()] + [qeye(2)]*(N-i-1)
        sy = [qeye(2)]*(i) + [sigmay()] + [qeye(2)]*(N-i-1)
        sz = [qeye(2)]*(i) + [sigmaz()] + [qeye(2)]*(N-i-1)
        sp = [qeye(2)]*(i) + [sigmap()] + [qeye(2)]*(N-i-1)
        sm = [qeye(2)]*(i) + [sigmam()] + [qeye(2)]*(N-i-1)
    
        if returnType=="np":
            X = np.array(tensor(sx))
            Y = np.array(tensor(sy))
            Z = np.array(tensor(sz))
            P = np.array(tensor(sp))
            M = np.array(tensor(sm))
        else:
            X = tensor(sx)
            X.dims = [[x] for x in X.shape]
            Y = tensor(sy)
            Y.dims = [[x] for x in Y.shape]
            Z = tensor(sz)
            Z.dims = [[x] for x in Z.shape]
            P = tensor(sp)
            P.dims = [[x] for x in P.shape]
            M = tensor(sm)
            M.dims = [[x] for x in M.shape]
        
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        sp_list.append(P)
        sm_list.append(M)

    return sx_list, sy_list, sz_list, sp_list, sm_list

# generates list of spin operators on N sites
def genSpinOpList(N,returnType="qobj"):
    # initialize lists
    sx_list = []
    sy_list = [] 
    sz_list = []
    sp_list = []
    sm_list = []
    
    for i in range(N):
        sx = [qeye(2)]*(i) + [(1/2)*sigmax()] + [qeye(2)]*(N-i-1)
        sy = [qeye(2)]*(i) + [(1/2)*sigmay()] + [qeye(2)]*(N-i-1)
        sz = [qeye(2)]*(i) + [(1/2)*sigmaz()] + [qeye(2)]*(N-i-1)
        sp = [qeye(2)]*(i) + [sigmap()] + [qeye(2)]*(N-i-1)
        sm = [qeye(2)]*(i) + [sigmam()] + [qeye(2)]*(N-i-1)
    
        if returnType=="np":
            X = np.array(tensor(sx))
            Y = np.array(tensor(sy))
            Z = np.array(tensor(sz))
            P = np.array(tensor(sp))
            M = np.array(tensor(sm))
        else:
            X = tensor(sx)
            X.dims = [[x] for x in X.shape]
            Y = tensor(sy)
            Y.dims = [[x] for x in Y.shape]
            Z = tensor(sz)
            Z.dims = [[x] for x in Z.shape]
            P = tensor(sp)
            P.dims = [[x] for x in P.shape]
            M = tensor(sm)
            M.dims = [[x] for x in M.shape]
        
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        sp_list.append(P)
        sm_list.append(M)

    return sx_list, sy_list, sz_list, sp_list, sm_list 


# generates two-spin interaction at k-distance away in 1D spin chain
def interactionkDist(ops, ops2=[],k=1, bc='obc'):
    N = len(ops)
    if ops2 ==[]:
        ops2 = ops
        
    H_int = Qobj(shape=ops[0].shape)
    Nmax = N if bc == 'pbc' else N-k
    for i in range(Nmax):
        H_int = H_int + ops[i]*ops2[np.mod(i+k,N)]  
    return H_int

# given a list of ops on each site, generate an onsite "field" where ops = field ops (eg sigmaz on each site)
def getOnsiteField(ops, hList):
    return sum(i[0] * i[1] for i in zip(ops, hList))


# GENERAL SYMMETRY HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# symmetry block
class block:
    def __init__(self, p=None):
        self.val = p
        self.states = []
        self.blockSize = None

# given a symmetry operator, find its eigenvalues (one for each block) and eigenvectors
class symmetry:
    def __init__(self, Op):
        self.op = Op  # actual symmetry operator
        self.COB = []   # matrix of eigenvecs that makes symm op diagonal
#         self.membership = {}  # a dictionary that says which sector each of the eigenstates belongs to; key = idx of eigenstate, value = symm eigenval
        self.blocks = {}    # dictionary containing eigenvalue of each symmetry sector and number of states
        self.blockSizes = None
        self.minBlockSize = None  # size of smallest block
        return
    
    # diagonalize symmetry operator if can't do it by construction
    def fillSymm(self):
        (eigs,states) = self.op.eigenstates()
        eigs = np.round(eigs,10) # round eigenvalues
        self.membership = dict(zip(range(len(states)), eigs)) 
        A = [np.array(x) for x in states]
        self.COB = np.asmatrix(np.array(A))
        self.blocks = dict(Counter(eigs))
        self.blockSizes = self.blocks.values()
        self.minBlockSize = min(self.blocks.values())
        

# for each symmetry of Hamiltonian, have COB. Then block diagonalize H using COB and calculate eigenstates in this basis.
# for each eigenstate in symm basis have symm sector value
# class to store a Hamiltonian and its associated symmetries and block diagonal form
class Hsymm_class:
    def __init__(self, H, symm):
        self.H = H
        self.HBD = H
        self.BDSymm = []
        self.symmetries = []
        self.addSymm(symm)
        self.eigs = []
        self.evecs = []
        self.beigs = []
        self.bvecs = []

    def addSymm(self, symm):
        comm = commutator(self.H,symm.op)
        if comm!=Qobj(np.zeros(self.H.shape)):
            raise Exception("not a symmetry!")
        self.symmetries.append(symm)
        # sort symmetries based on which one has highest "resolution" of state, aka one with the smallest minimal block
        self.symmetries.sort(key=lambda x: x.minBlockSize)

        # update block diagonal matrix
        # choose the symmetry with the "finest granularity"
        self.BDSymm = self.symmetries[0]
        S = self.BDSymm.COB
        self.HBD = S @ np.array(self.H) @ S.H


    # diagonalize block diagonal Hamiltonian
    def diagonalize(self):
        blockSizes = self.BDSymm.blockSizes  # dictionary that contains the symmetry blocks
        beigs = []  # list of lists that will contain the eigenvalues of each block
        bvecs = []  # list of lists that will contain the eigenvectors of each block
        Hdim = max(self.HBD.shape)

        end = 0
        for bs in blockSizes:
            start = end
            end = end + bs
            hBlock = np.array(self.HBD[start:end, start:end])

            # diagonalize the H block
            eigs,evecs = np.linalg.eigh(hBlock)
            
            # pad with 0's to get full eigenvectors
            before = start
            after = Hdim - end
            evecs_full = [np.pad(v,(before,after))  for v in np.transpose(evecs)]
    
            beigs.append(eigs)   
            bvecs.append(evecs_full)

        beigs_tot = np.concatenate(beigs) 
        bvecs_tot = np.concatenate(np.array(bvecs))

        bvecs_tot = np.round(bvecs_tot@np.array(self.BDSymm.COB.conjugate()),15)

        order = np.argsort(beigs_tot)
        beigs_tot = beigs_tot[order]
        bvecs_tot = bvecs_tot[order]
        
        self.beigs = beigs
        self.bvecs = bvecs
        self.eigs = beigs_tot
        self.evecs = bvecs_tot

# diagonalize block diagonal Hamiltonian given a dictionary of symmetry blocks with their values and sizes
# EIGENVECTORS ARE **ROWS**
def diagonalizeHBD(HBD,blockSizes,COB):
    beigs = []  # list of lists that will contain the eigenvalues of each block
    bvecs = []
    Hdim = max(HBD.shape)
    end = 0
    for bs in blockSizes:
        start = end
        end = end + bs
        currBlock = np.array(HBD[start:end, start:end])

        # diagonalize the block
        eigs, evecs = np.linalg.eigh(currBlock)
        beigs.append(eigs)
        
        # pad with 0's to get full eigenvectors
        before = start
        after = Hdim - end
        
        evecs_full = [np.round(np.pad(v,(before,after)),15)  for v in np.transpose(evecs)]

        bvecs.append(evecs_full)

    beigs_tot = np.concatenate(beigs) 
    bvecs_tot = np.concatenate(np.array(bvecs))
    bvecs_tot = bvecs_tot@np.array(COB.conjugate())
    order = np.argsort(beigs_tot)
    beigs_tot = beigs_tot[order]
    bvecs_tot = bvecs_tot[order] # sort rows = eigenvectors

## PARTICULAR SYMMETRY HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Parity op = prod_i s_i^{x,y,z}
def getParityOp(N,axis='x',PauliList=[]):
    axes = ['x','y','z']
    ops = [sigmax(),sigmay(),sigmaz()]
    if PauliList == []:
        p = tensor([ops[axes.index(axis)]]*N)
        p.dims = [[x] for x in p.shape]
    else:
        p = opProd(PauliList[axes.index(axis)])
    return p

def getParitySymm(L,axis='x',PauliList=[]):  # FASTEST BOI
    P = getParityOp(L,axis=axis,PauliList=PauliList)
    parity = symmetry(P)
    if axis=='z': # Z PARITY
        # prod_i(Z_i) is already diagonal
        start = timer()
        posIdx = np.where(P.diag()==1)[0]
        negIdx = np.where(P.diag()==-1)[0]
#         print(np.concatenate((posIdx,negIdx)))
        PCOB = qeye(2**L)[np.concatenate((posIdx,negIdx))]
        PCOB = np.asmatrix(PCOB)
        end = timer()
        print("time to get COB: %f" % (end-start))
    
    if axis=='x':  # X PARITY
        xp = Qobj(np.array([1,1]))
        xm = Qobj(np.array([1,-1]))
        start = timer()
        PCOB = [np.array(tensor(list(map(lambda x: xp if x=='0' else xm, np.binary_repr(n,L))))) for n in range(2**L)]
        PCOB.sort(key=lambda x: x[0]==x[-1])  # same = even = +1 = True = right; different = odd = -1 = False = left
        PCOB = np.asmatrix(np.array(PCOB))
        end = timer()
        print("time to get COB: %f" % (end-start))
        
    elif axis=='y':  # Y PARITY
        yp = Qobj(np.array([1,1j]))
        ym = Qobj(np.array([1,-1j]))
        start = timer()
        PCOB = [np.array(tensor(list(map(lambda x: yp if x=='0' else ym, np.binary_repr(n,L))))) for n in range(2**L)]
        PCOB.sort(key=lambda x: x[0]==(1j**(N//2))*x[-1])  # same = even = +1 = True = right; different = odd = -1 = False = left
        PCOB = np.asmatrix(np.array(PCOB))
        end = timer()
        print("time to get COB: %f" % (end-start))

    parity.COB = PCOB   
    parity.blockSizes = [2**(L - 1),2**(L - 1)]
    parity.blocks = dict(zip([-1,1],parity.blockSizes))
    parity.minBlockSize = 2**(L - 1)
        
    return parity


## EXPECTATION VALUES AND CORREL HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Calculate time-dependent expectation value of operator by switching to energy eigenbasis
# USES NUMPY ARRAYS -- evecs = ROWS are eigenvectors
def expval(state,op,eigs,evecs,t):
    e0_state = np.conjugate(evecs)@state  # state in energy basis
    expectval = np.round(np.dot(np.conjugate(e0_state), np.diag(np.exp(1j*eigs*t)) @ np.conjugate(evecs) @ op @ np.transpose(evecs) @ np.diag(np.exp(-1j*eigs*t)) @ e0_state),15)
    if np.imag(expectval)==0:
        return np.real(expectval)
    else:
        return expectval

# uses matrix exponentials NOT in energy eigenbasis
# INPUTS ARE QOBJ
def expval_me(state,op,H,t):
    if isinstance(state,Qobj):
        return np.round(expect(op,(-1j*H*t).expm()*state),15)

    elif isinstance(state,np.ndarray):
        tstate = scipy.linalg.expm(-1j*H*t)@state
        return np.round(np.dot(np.conjugate(tstate),op@tstate),15)

    elif isinstance(op,sparse.csr_matrix):
        tstate = scipy.sparse.linalg.expm(-1j*H*t)*(state)
        return np.round(np.dot(np.conjugate(tstate),op*tstate),15)
    else:
        raise Exception("unknown type")

## INTERACTION FUNCTIONS #######################################################################################

# spatially inhomogeneous interaction function
# j = max strength; theta = angle of displacement vector with quantization axis z
# Returns a vectorized function J(theta,dist)
# J_ij (r, θ) = j(1 - 3 cosθ^2)/|r_ij|^α
def powerLawJfunc(j,alpha):
    Jfunc = lambda costh,dist: [j*(1-(3*costh**2) - np.eye(costh.shape[0]))/(dist**alpha + np.eye(dist.shape[0]))]
    return Jfunc

# J_ij (r_ij) = j* (1/(1 + (|r_ij|/rc)^α))
def RydbergJfunc(j,alpha,rc):
    Jfunc = lambda r: j/(1+((r/rc)**alpha))
    return Jfunc

# returns SYMMETRIC 2D matrix with interaction strength between each pair of spins
# calculates interaction based off of a power law interaction
def getJij(coord,j,alpha,Jfunc=[]):
    if Jfunc==[] and coord.shape[1]==3: # if no interaction function provided, use 3D power law interaction 
        Jfunc = powerLawJfunc(j,alpha)
        r_ij, cos_ij = get3DThetaDistCoord(coord)
        J_ij = Jfunc(cos_ij, r_ij)[0] 
        return J_ij 

    r_ij = pairwiseDist(coord)
    J_ij = Jfunc(r_ij)
    J_ij = J_ij - np.diag(np.diag(J_ij))
    return J_ij 

# returns an interaction matrix where each element is drawn from a random Gaussian distribution 
def randomJij(dim,mean=0,var=1):
    J_ij = np.random.normal(loc=mean, scale=np.sqrt(var), size=(dim,dim))
    J_ij = J_ij - np.diag(np.diag(J_ij))
    return J_ij

# get string Lx x Ly x Lz... representing for rectangular lattice
def getDimStr(L):
    dimStr = ""
    for i in range(len(L)):
        dimStr+= "%d"%L[i]
        if i!=len(L)-1:
            dimStr+="x"
    return dimStr


## ED HAMILTONIANS #######################################################################################

# Generates 1D NN General Heisenberg model
# Hamilt: (1/2)*sum_ij[ (Jx_ij(X_i X_j) + Jy_ij(Y_i Y_j)) +  Jz_ij(Z_i Z_j)] + hx * sum_i[X_i] + hz * sum_i[Z_i]
# if Jperp = (Jx, Jy), then have XYZ. ow have XXZ w/ Jx = Jy = Jperp/2
# hx, hz = generically a vector of on-site transverse/longitudinal fields; if these are numbers, field is assumed to be uniform
def Heisenberg_NN(N, Jperp, Jz, hx, hz, BC='pbc',PauliBool = False, opList=[],):
    if opList==[]:
        if PauliBool:
            opList = genPauliList(N)
        else:
            opList = genSpinOpList(N)
    sx, sy, sz, sp, sm = opList
        
    H_NN_x = interactionkDist(sx,k=1,bc=BC)
    H_NN_y = interactionkDist(sy,k=1,bc=BC)
    H_NN_z = interactionkDist(sz,k=1,bc=BC)
    
    if isinstance(Jperp,(float, int)):
        H_int = ((Jperp/2)*(H_NN_x + H_NN_y) + Jz*H_NN_z)
    elif isinstance(Jperp,list) and len(Jperp)==2:
        Jx = Jperp[0]
        Jy = Jperp[1]
        H_int = (Jx*H_NN_x + Jy*H_NN_y + Jz*H_NN_z)
    else:
        raise Exception("Wrong number of elements in Jperp!")
      
    Sx_tot = sum(sx) # total spin in x
    Sy_tot = sum(sy) # total spin in y
    Sz_tot = sum(sz) # total spin in Z

    # FIELD HAMILTONIANS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LONGITUDINAL FIELD
    if isinstance(hz,(float, int)):     #  Uniform Z field
        Hz = hz*Sz_tot
    else:                               # Non-uniform Z field
        Hz = sum([hz[i]*sz[i] for i in range(N)])

    # TRANSVERSE FIELD
    if isinstance(hx,(float, int)):     #  Uniform X field
        Hx = hx*Sx_tot
    else:                               # Non-uniform X field
        Hx = sum([hx[i]*sx[i] for i in range(N)])


    H = sum(H_int) + Hx + Hz  # total Hamiltonian
    
    return H, opList, (Sx_tot, Sy_tot, Sz_tot)

#### Generates general all-to-all Heisenberg model
# Hamilt: sum_ij[ (Jx_ij(X_i X_j) + Jy_ij(Y_i Y_j)) +  Jz_ij(Z_i Z_j)] + hx * sum_i[X_i] + hz * sum_i[Z_i]
# if Jperp = (Jx, Jy), then have XYZ. ow have XXZ w/ Jx = Jy = Jperp/2
# L = vector containing number of spins along each dimension
# PauliBool = (optional) True to switch to Pauli operators; False to use Spin Operators
# opList = (optional) to provide list
# Jfunc = (optional) list of functions for Jx, Jy, Jz to compute interaction between every pair of spins; may be distance-dependent
# coord = (optional) Nxd numpy arrays representing the d coordinates of each spin; default gives regular lattice with spacing a=1
# alpha = (optional) exponent of decay for interaction strength  J_ij ~ 1/|r_ij|^α
def Heisenberg_A2A(L, Jperp, Jz, hx, hz, PauliBool = False, opList=[], Jfunc=[], coord = [], alpha=[]):
    if not coord==[] and np.prod(L)!=coord.shape[0]:
        raise Exception("Number of coordinates doesn't match number of spins!")
        print(np.prod(L))
    N = np.prod(L)  # total number of spins = product of number of spins along each direction
    # print("N: %d" %N)

    if opList==[]:
        if PauliBool:
            opList = genPauliList(N)
        else:
            opList = genSpinOpList(N)
    sx, sy, sz, sp, sm = opList
    # print("number of spin ops: %d" %len(sx))
    # print("dimension of spin op: %s" %(sx[0].shape,))
    
    # Generate coupling strengths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if isinstance(Jperp,(float, int)):   # XXZ CHAIN
        Jx = Jperp/2
        Jy = Jperp/2

    elif isinstance(Jperp,(list,tuple,np.ndarray)) and len(Jperp)==2:
        Jx = Jperp[0]
        Jy = Jperp[1]
    else:
        raise Exception("Wrong number of elements in Jperp!")      
    # print("Jx: %g \t Jy: %g \t Jz: %g" % (Jx,Jy,Jz))
    

    # INTERACTION HAMITLONIAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    H_int = []
    if (alpha!=[] and alpha!=0) or Jfunc!=[]:          # ANISOTROPIC POWER LAW INTERACTIONS
        # get coordinates
        if coord==[]:
            if isinstance(L,(int,float)):
                coord = getLatticeCoord(1,L,1)
            else:
                coord = getLatticeCoord(len(L),L,1)

        # print("coord dim: %s"%(coord.shape,))

        # get J_ij INTERACTING MATRIX
        if len(Jfunc)!=3:
            raise Exception("Not enough Jfuncs provided!")
        Jx_ij = getJij(coord,Jx,alpha,Jfunc=Jfunc[0])
        Jy_ij = getJij(coord,Jy,alpha,Jfunc=Jfunc[1])
        Jz_ij = getJij(coord,Jz,alpha,Jfunc=Jfunc[2])

        # print("Jz_ij shape: %s"%(Jz_ij.shape,))
     
        for i,j in it.combinations(range(N),2):
            H_xx = Jx_ij[i,j] * sx[i]*sx[j]
            H_yy = Jy_ij[i,j] * sy[i]*sy[j]
            H_zz = Jz_ij[i,j] * sz[i]*sz[j]

            H_int.append((H_xx + H_yy + H_zz))
          
    else:               # ISOTROPIC INTERACTIONS
        
        for i,j in it.combinations(range(N),2):
            H_xx = Jx * sx[i]*sx[j]
            H_yy = Jy * sy[i]*sy[j]
            H_zz = Jz * sz[i]*sz[j]    
            H_int.append((H_xx + H_yy + H_zz))

#     #  can use einstein notation to get interaction terms but is inefficient for large system sizes
#     H_xx = np.einsum('ij,ixy,jyz -> xz',Jx_ij,sx,sx)
#     H_yy = np.einsum('ij,ixy,jyz -> xz',Jy_ij,sy,sy)
#     H_zz = np.einsum('ij,ixy,jyz -> xz',Jz_ij,sz,sz)
            
    Sx_tot = sum(sx) # total spin in X
    Sy_tot = sum(sy) # total spin in X
    Sz_tot = sum(sz) # total spin in Z

    # FIELD HAMILTONIANS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LONGITUDINAL FIELD
    if isinstance(hz,(float, int)):     #  Uniform Z field
        Hz = hz*Sz_tot
    else:                               # Non-uniform Z field
        Hz = sum([hz[i]*sz[i] for i in range(N)])

    # TRANSVERSE FIELD
    if isinstance(hx,(float, int)):     #  Uniform X field
        Hx = hx*Sx_tot
    else:                               # Non-uniform X field
        Hx = sum([hx[i]*sx[i] for i in range(N)])


    H = sum(H_int) + Hx + Hz  # total Hamiltonian

    return H, opList, (Sx_tot,Sy_tot,Sz_tot)


## SQUEEZING HELPERS #######################################################################################


# finds squeezing param evolution over time tvec as well as optimal squeezing parameter within that range
# suitable for Hamiltonian that is already diagonal, i.e. Ising Hamiltonian
def getEDIsingSqueezing(H,initState,tvec, opList=[], PauliBool = False):
    
    N = int(np.log2(len(eigs)))
    if opList==[]:
        if PauliBool:
            opList = genPauliList(N)
        else:
            opList = genSpinOpList(N)
    sx,sy,sz,sp,sm = opList
    

    Sy_tot = sum(sy)
    Sz_tot = sum(sz)
    Ssq_tot_ops = [Sy_tot*Sy_tot,Sz_tot*Sz_tot]
    
    
    Ssq_mean_ed = []
    for i in range(len(Ssq_tot_ops)):
        Ssq_mean_ed.append(np.array([expval_me(Qobj(initState),Ssq_tot_ops[i],H,t) for t in tvec]))
    Ssq_mean_ed = np.array(Ssq_mean_ed)

    W_ed = np.array([expval_me(Qobj(initState),commutator(Sy_tot,Sz_tot,kind="anti"),H,t) for t in tvec])
    Vp_ed = Ssq_mean_ed[0,:] + Ssq_mean_ed[1,:]
    Vm_ed = Ssq_mean_ed[0,:] - Ssq_mean_ed[1,:]
    squeezing_ed = (Vp_ed - np.sqrt(W_ed**2 + Vm_ed**2))/(np.prod(N)/2)
        
    if not np.allclose(squeezing_ed,squeezing_ed.real):
        raise Exception("Squeezing param imag for time range")
    
    return np.min(squeezing_ed.real), squeezing_ed.real
    
    
# returns vectorized functions for mean total spin components, variances, and squeezing param under Ising dynamics given initial x spin-polarized state
# Based on analytic results for Ising Hamiltonian: H = Jz sum_ij z_i z_j
def IsingSqueezing(N,Jz):
    Sx_func = lambda t: (N//2)*(np.cos(Jz*t/2)**(N-1))
    Afunc = lambda t: 1 - (np.cos(Jz*t))**(N-2)
    Ssqx_func = lambda t: (N/4)*(N*(1-(np.cos(Jz*t/2)**(2*(N-1))) ) - (N//2 - (1/2))*Afunc(t)) + Sx_func(t)**2
    Ssqy_func = lambda t: (N/4)*(1 + (N//2 - (1/2))*(1 - (np.cos(Jz*t))**(N-2)))
    yz_acomm_func = lambda t: (N//2)*(N-1)*np.sin(Jz*t/2)*(np.cos(Jz*t/2)**(N-2))
    Vp_func = lambda t: Ssqy_func(t) + (N/4)
    Vm_func = lambda t: Ssqy_func(t) - (N/4)
    squeezing_func = lambda t: (Vp_func(t) - np.sqrt(Vm_func(t)**2 + yz_acomm_func(t)**2))/(N/2)
    return squeezing_func,Sx_func, Ssqx_func, Ssqy_func, yz_acomm_func

