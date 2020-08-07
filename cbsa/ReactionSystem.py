import sys
import numpy as np
from .SimulationCBSA import SimulationCBSA

def extend_matrix_toroid(S,n,diffusion=True):
    frac,_ = np.modf(np.sqrt(n))
    if n<2 or frac>0:
        sys.exit("ERROR: expansion number must be equal or greater than 2 and have exact square root. Got: "+str(n))
    exS = S
    n2 = n
    n = int(np.sqrt(n2))
    m = S.shape[0]
    r = 1
    if len(S.shape)>1:
        r = S.shape[1]
    if np.sum(S!=0)==0:
        r = 0
    new_m = m*n2
    new_r = r*n2+4*n2*m
    exS = np.zeros((new_m,new_r),dtype=S.dtype)
    for i in range(n2):
        exS[i*m:i*m+m,i*r:i*r+r] = S
    if diffusion:
        counter = 0
        start = r*n2
        for j in range(n2):
            up = (j-n+n2)%n2
            down = (j+n)%n2
            right = (int(j/n)*n)+(j+1+n)%n
            left = (int(j/n)*n)+(j-1+n)%n
            #print(j,"up",up,"down",down,"left",left,"right",right)
            for i in range(m):
                exS[j*m+i,start+counter] = -1
                exS[up*m+i,start+counter] = 1
                counter+=1
                exS[j*m+i,start+counter] = -1
                exS[down*m+i,start+counter] = 1
                counter+=1
                exS[j*m+i,start+counter] = -1
                exS[left*m+i,start+counter] = 1
                counter+=1
                exS[j*m+i,start+counter] = -1
                exS[right*m+i,start+counter] = 1
                counter+=1
    return exS


def fill_reactant_substitution_matrix(S,R,subS_idx,subS_sto):
    fill_from_R = np.count_nonzero(R)
    for i in range(S.shape[1]):
        counter_reactant = 0
        for j in range(S.shape[0]):
            if S[j,i] < 0:
                subS_sto[counter_reactant,i] = -S[j,i]
                subS_idx[counter_reactant,i] = j
                counter_reactant += 1
            if fill_from_R>0:
                if R[j,i] != 0:
                    subS_sto[counter_reactant,i] = R[j,i]
                    subS_idx[counter_reactant,i] = j
                    counter_reactant += 1

    
def fill_reaction_substitution_matrix(S,subS_idx,subS_sto):
    for i in range(S.shape[0]):
        counter_reaction = 0
        for j in range(S.shape[1]):
            if S[i,j]:
                subS_sto[i,counter_reaction] = S[i,j]
                subS_idx[i,counter_reaction] = j
                counter_reaction += 1

def add_empty_row_and_column(M):
    expM = M
    expM = np.c_[np.zeros(expM.shape[0],dtype=expM.dtype),expM]
    expM = np.r_[[np.zeros(expM.shape[1],dtype=expM.dtype)],expM]
    return expM

class ReactionSystem:
    
    def __init__(self,
                 _S,                     #Stoichiometric matrix
                 _R=None,                #Regulation matrix
                 _int_type=np.intc,      #Type for integers
                 _float_type=np.float32  #Type for floats
                ):
        self.int_type = _int_type
        self.float_type = _float_type
        self.S = np.array(_S,dtype=self.int_type)
        if not _R is None:
            self.R = np.array(_R,dtype=self.int_type)
        else:
            self.R = np.zeros_like(self.S,dtype=self.int_type)
        if self.S.shape!=self.R.shape:
            sys.exit("ERROR: Matrix S and R must have the same dimensions. Got: "+str(_S.shape)+" and "+str(_R.shape))
        self.n_molecules = self.S.shape[0]
        self.n_reactions = self.S.shape[1]
        
        self.called_setup = False
        
        self.nSubspaces = 1
        self.expanded_space = None
        self.interfaces_per_subspace = 0
        
        self.expS = None
        self.expR = None
        self.exp_n_molecules = None
        self.exp_n_reactions = None
        
        self.subV_sto = None
        self.subV_idx = None

        self.subX_idx = None
        self.subX_sto = None
        
        self.x0 = None
        self.exp_x0 = None
        
        #Parameters
        
        self.k = None
        self.diffusion_k = None
        self.exp_k = None
        
        self.max_dt = None
        
        #Simulation
        self.simulation_kernel = None
        self.simulation_data = None
        
    def setup(self,
              n_subspaces=1, #Number of subspaces to expansion
              space=None     #Type of expansion
             ):
        #Get sizes
        self.nSubspaces = int(n_subspaces)
        self.expS = self.S
        self.expR = self.R
        
        self.expanded_space = space
        
        #Expand matrices to space
        if self.expanded_space == 'toroid':
            self.expS = extend_matrix_toroid(self.S,self.nSubspaces,diffusion=True)
            self.expR = extend_matrix_toroid(self.R,self.nSubspaces,diffusion=False)
            self.interfaces_per_subspace = 4
        
        #add empty row and column
        self.expS = add_empty_row_and_column(self.expS)
        self.expR = add_empty_row_and_column(self.expR)
        
        #Get expanded sizes
        self.exp_n_molecules = self.expS.shape[0]
        self.exp_n_reactions = self.expS.shape[1]
        
        #Allocate substitution matrices
        max_reactants = np.max(np.sum(np.vstack((self.expS<0,self.expR!=0)),axis=0))
        max_reactions = np.max(np.sum(self.expS!=0,axis=1))
        
        self.subV_sto = np.zeros((max_reactants,self.expS.shape[1]),dtype=self.int_type)
        self.subV_idx = np.zeros((max_reactants,self.expS.shape[1]),dtype=self.int_type)

        self.subX_idx = np.zeros((self.expS.shape[0],max_reactions),dtype=self.int_type)
        self.subX_sto = np.zeros((self.expS.shape[0],max_reactions),dtype=self.int_type)
        
        #Fill substitution matrices
        fill_reactant_substitution_matrix(self.expS,self.expR,self.subV_idx,self.subV_sto)
        fill_reaction_substitution_matrix(self.expS,self.subX_idx,self.subX_sto)
        
        self.called_setup = True
        
    def check_setup(self):
        if not self.called_setup:
            sys.exit("ERROR: 'setup' must be called before this operation.")
    
    def set_x(self, x,mol_per_subspace=False):
        self.check_setup()
        self.x0 = np.array(x,dtype=self.int_type)
        if self.x0.size!=self.n_molecules:
            sys.exit("ERROR: size of x shoud be "+str(self.n_molecules)+". Got: "+str(self.x0.size))
        
        self.exp_x0 = np.array([0],dtype=self.int_type)
        copies = self.nSubspaces
        to_copy = self.x0
        if not mol_per_subspace:
            self.exp_x0 = np.concatenate((self.exp_x0,self.x0))
            to_copy = np.zeros_like(self.x0,dtype=self.int_type)
            copies -= 1
        for i in range(copies):
            self.exp_x0 = np.concatenate((self.exp_x0,to_copy))
            
    def set_k(self, k,diff_k=None):
        self.check_setup()
        self.k = np.array(k,dtype=self.float_type)
        if not diff_k is None:
            self.diffusion_k = np.array(diff_k,dtype=self.float_type)
        if self.k.size!=self.n_reactions:
            sys.exit("ERROR: size of k shoud be "+str(self.n_reactions)+". Got: "+str(self.k.size))
        if self.nSubspaces>1 and diff_k is None:
            sys.exit("ERROR: Diffusion coefficients expected.")
        if self.nSubspaces>1 and self.diffusion_k.size!=self.n_molecules:
            sys.exit("ERROR: size of diff_k shoud be "+str(self.n_molecules)+". Got: "+str(self.diffusion_k.size))
            
        self.exp_k = np.array([0],dtype=self.float_type)
        if np.sum(self.k)>0:
            for i in range(self.nSubspaces):
                self.exp_k = np.concatenate((self.exp_k,self.k))
            
        if self.nSubspaces>1:
            for i in range(self.nSubspaces):
                for j in range(self.n_molecules):
                    self.exp_k = np.concatenate((self.exp_k,np.array([self.diffusion_k[j] for m in range(self.interfaces_per_subspace)])))
        
        
    def set_max_dt(self,dt):
        self.max_dt = np.array([dt],dtype=self.float_type)
        
    def setup_simulation(self,use_opencl=False,alpha=0.5,max_dt=0.1):
        self.simulation_kernel = SimulationCBSA(
                                             self.subV_idx,
                                             self.subV_sto,
                                             self.subX_idx,
                                             self.subX_sto,
                                             self.exp_x0,
                                             self.exp_k,
                                             np.array([max_dt],dtype=self.float_type),
                                             np.array([alpha],dtype=self.float_type),
                                             self.int_type,
                                             self.float_type,
                                             use_opencl
                                            )
        
        
    def compute_simulation(self,total_time,file_output=None,batch_steps=1):
        self.simulation_data = self.simulation_kernel.compute_simulation(total_time,file_output,batch_steps)

