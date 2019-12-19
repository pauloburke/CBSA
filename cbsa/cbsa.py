import sys
import numpy as np

kernel_src = """

int perm(int a, int b){
    if(!b) return(1);
    if(a<b) return(0);
    int perm = a;
    for(int i=1;i<b;i++){
        perm *= (a-i)/(i+1);
    }
    return(perm);
}

__kernel void replace_and_perm(__global const int *idx, __global const int *sto, __global const int *x, __global int *r){
    int i = get_global_id(0);
    r[i] = perm(x[idx[i]],sto[i]);
}

__kernel void column_prod_reduce_mult_kdt_add_dvnoise(__global const int *m,__global const float *k,__global const float *dv,__global const float *n,__global const float *max_dt,__global const int *width,__global const int *height,__global float *r){
    int i = get_global_id(0);
    int w = *width;
    int h = *height;
    float dt = *max_dt;
    float v = 1.0;
    for(int j=0;j<h;j++){
        v *= m[i+j*w];
    }
    v = v*k[i]*dt+dv[i];
    v += sqrt(v)*n[i]*sqrt(dt);
    if(v<0.) v=0.;
    r[i] = v;
}


__kernel void separate_int_float(__global const float *fv,__global int *v,__global float *dv){
    int i = get_global_id(0);
    v[i] = (int) fv[i];
    dv[i] = fv[i] - v[i];
}

__kernel void replace_and_mult(__global const int *idv, __global const int *sto, __global const int *v, __global int *r){
    int i = get_global_id(0);
    r[i] = v[idv[i]]*sto[i];
}

__kernel void row_sum_reduce(__global const int *m,__global const int *width,__global const int *heigth,__global int *r){
    int i = get_global_id(0);
    int w = *width;
    int h = *heigth;
    
    int sum = 0;
    for(int j=0;j<w;j++){
        sum += m[i*w+j];
    }
    r[i] = sum;
}

__kernel void is_valid_dx(__global const int *x,__global const int *dx,__global int *r){
    int i = get_global_id(0);
    if(-dx[i]>x[i]) *r = 0;
}

__kernel void set_true(__global int *r){
    *r = 1;
}


__kernel void sum_copy_buffer(__global const int *x,__global const int *y,__global int *r,__global int *buff){
    int i = get_global_id(0);
    r[i]=x[i]+y[i];
    buff[i] = r[i];
}


__kernel void mult_scalar(__global const float *x,__global const float *s,__global float *r){
    int i = get_global_id(0);
    float scr = *s;
    r[i] = x[i]*scr;
}

__kernel void mult_scalar_floor(__global const int *x,__global const float *alpha,__global int *r){
    int i = get_global_id(0);
    float a = *alpha;
    r[i] = (int) x[i]*a;
}


__kernel void copy(__global const int *from,__global int *to){
    int i = get_global_id(0);
    to[i]=from[i];
}

__kernel void fmult(__global const float *a,__global const float *b,__global float *r){
    int i = get_global_id(0);
    r[i] = a[i]*b[i];
}

"""

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
            self.R = np.zeros_like(R,dtype=self.int_type)
        if self.S.shape!=self.R.shape:
            sys.exit("ERROR: Matrix S and R must have the same dimensions. Got: "+str(S.shape)+" and "+str(R.shape))
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
        
        self.alpha = np.array([0.5],dtype=self.float_type)
        
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
        
    def setup_simulation(self,use_opencl=False):
        self.simulation_kernel = SimulationCBSA(
                                             self.subV_idx,
                                             self.subV_sto,
                                             self.subX_idx,
                                             self.subX_sto,
                                             self.exp_x0,
                                             self.exp_k,
                                             self.max_dt,
                                             self.alpha,
                                             self.int_type,
                                             self.float_type,
                                             use_opencl
                                            )
        
        
    def compute_simulation(self,total_time,file_output=None,batch_steps=1):
        self.simulation_data = self.simulation_kernel.compute_simulation(total_time,file_output,batch_steps)
        
class SimulationCBSA:
    
    def __init__(self,
                 _subV_idx,
                 _subV_sto,
                 _subX_idx,
                 _subX_sto,
                 _x,
                 _k,
                 _max_dt,
                 _alpha,
                 _int_type=np.intc,
                 _float_type=np.float32,
                 _use_opencl = False
                ):
        
        
        self.use_opencl = _use_opencl
        
        self.subV_idx = _subV_idx
        self.subV_sto = _subV_sto
        self.subX_idx = _subX_idx
        self.subX_sto = _subX_sto
        self.x = _x
        self.k = _k
        self.max_dt = _max_dt
        self.alpha = _alpha
        
        
        self.int_type = _int_type
        self.float_type = _float_type
        
        #Allocate memory on host
        
        self.dx = np.zeros_like(self.x,dtype=self.int_type)
        
        self.subV_replaced = np.zeros_like(self.subV_idx,dtype=self.float_type)
        self.subX_replaced = np.zeros_like(self.subX_idx,dtype=self.int_type)
        
        self.v = np.zeros_like(self.k,dtype=self.int_type)
        self.tmp_v = np.zeros_like(self.k,dtype=self.int_type)
        self.fv = np.zeros_like(self.k,dtype=self.float_type)
        self.tmp_fv = np.zeros_like(self.k,dtype=self.float_type)
        self.dv = np.zeros_like(self.k,dtype=self.float_type)
        
        self.is_valid_dx = np.array([1],self.int_type)
        
        self.dt = np.array([0.0],dtype=self.float_type)
        
        self.n_molecules = self.x.shape
        self.n_reactions = self.k.shape
        
        if self.use_opencl:
            self.setup_opencl()
        
        
    def setup_opencl(self):
        
        #OpenCl Setup
        self.cl = __import__('pyopencl',fromlist=['array','clrandom'])
        self.ctx = self.cl.create_some_context()
        self.queue1 = self.cl.CommandQueue(self.ctx)
        self.queue2 = self.cl.CommandQueue(self.ctx)
        self.queue_transfer = self.cl.CommandQueue(self.ctx)
        self.mf = self.cl.mem_flags
        
        #Build Kernel
        self.pgr = self.cl.Program(self.ctx, kernel_src).build()
        
        #Allocate memory on host        
        self.subV_replaced = self.subV_replaced.flatten()
        self.subX_replaced = self.subX_replaced.flatten()        

        self.subV_n_elements = self.subV_replaced.shape
        self.subX_n_elements = self.subX_replaced.shape
        self.n_one = self.is_valid_dx.shape
        
        #Random number generator
        self.rand_gen = self.cl.clrandom.PhiloxGenerator(self.ctx)
        self.ev_rand = None
        
        #Allocate memory on device

        self.d_subV_sto = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.subV_sto.flatten())
        self.d_subV_idx = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.subV_idx.flatten())
        self.d_subV_replaced = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.subV_replaced)
        
        self.d_subX_sto = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.subX_sto.flatten())
        self.d_subX_idx = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.subX_idx.flatten())
        self.d_subX_replaced = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.subX_replaced)
        
        self.d_x = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.x)
        self.d_x_buff = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.x)
        self.d_dx = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dx)
        
        self.d_k = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.k)
        
        self.d_v = self.cl.Buffer(self.ctx, self.mf.READ_WRITE| self.mf.COPY_HOST_PTR, hostbuf=self.v)
        self.d_tmp_v = self.cl.Buffer(self.ctx, self.mf.READ_WRITE| self.mf.COPY_HOST_PTR, hostbuf=self.tmp_v)
        self.d_fv = self.cl.Buffer(self.ctx, self.mf.READ_WRITE| self.mf.COPY_HOST_PTR, hostbuf=self.fv)
        self.d_tmp_fv = self.cl.Buffer(self.ctx, self.mf.READ_WRITE| self.mf.COPY_HOST_PTR, hostbuf=self.tmp_fv)
        self.d_dv = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dv)
        
        self.d_subV_height = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subV_sto.shape[0]))
        self.d_subV_width = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subV_sto.shape[1]))
        self.d_subX_height = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subX_sto.shape[0]))
        self.d_subX_width = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subX_sto.shape[1]))
        
        self.d_is_valid_dx = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.is_valid_dx)
        self.d_max_dt = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.max_dt)
        self.d_dt = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dt)
        self.d_alpha = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.alpha)
        self.d_rand = self.cl.array.Array(self.queue1,self.fv.shape,self.fv.dtype)
        
        #Release unecessary memory
        del self.subV_replaced
        del self.subX_replaced
        del self.v
        del self.tmp_v
        del self.fv
        del self.tmp_fv
        del self.dv
        del self.dx
        
        

    def compute_simulation(self,total_time,file_output=None,batch_steps=1):
        
        #Outputs
        log_sim = None
        f = None
        if file_output!=None:
            write_format = ['%.10f']+['%d' for i in range(self.n_molecules[0]-1)]
            f = open(file_output,'w')
            f.write("CBSA Simulation ----------------------------\n")
            f.close()
            f = open(file_output,'ab')
        
        #buffers
        x_buffer = np.zeros(self.n_molecules[0],dtype=self.int_type)        
        
        #start
        if self.use_opencl:
            self.ev_rand = self.rand_gen.fill_normal(self.d_rand,queue=self.queue2)
        time = 0.
        
        #Log fist row
        if f is None:
            log_sim = [self.x]
        else:
            np.savetxt(f,self.x.reshape((1,self.n_molecules[0])),delimiter=' ',fmt=write_format)
        #Compute
        while(time<total_time):

            dt,x = self.compute_batch_steps(batch_steps,x_buffer)
            time += dt
            #print(dt)
            x = x.astype(self.float_type)
            x[0] = time
            if f is None:
                log_sim.append(x)
            else:
                np.savetxt(f,x.reshape((1,self.n_molecules[0])),delimiter=' ',fmt=write_format)        
        
        if f is None:
            return log_sim
        else:
            f.close()
            return []
        
        
    def compute_batch_steps(self,n_steps,x_buffer):
        
        dt = np.array([self.max_dt for i in range(n_steps)])
        for i in range(n_steps):
            if self.use_opencl:
                self.compute_step_opencl(x_buffer,dt[i],i==n_steps-1)
            else:
                dt[i],x_buffer = self.compute_step()
        return np.sum(dt),x_buffer
        
    def compute_step_opencl(self,x_buffer,dt_buffer,get_vals):
        ev0 = self.pgr.copy(self.queue2,self.n_one, None,self.d_max_dt,self.d_dt)
        ev1 = self.pgr.replace_and_perm(self.queue1, self.subV_n_elements, None ,self.d_subV_idx,self.d_subV_sto,self.d_x,self.d_subV_replaced)
        ev2 = self.pgr.column_prod_reduce_mult_kdt_add_dvnoise(self.queue1, self.n_reactions, None ,self.d_subV_replaced,self.d_k,self.d_dv,self.d_rand.data,self.d_max_dt,self.d_subV_width,self.d_subV_height,self.d_fv)
        ev5 = self.pgr.separate_int_float(self.queue1, self.n_reactions, None ,self.d_fv,self.d_v,self.d_dv)        
        ev6 = self.pgr.replace_and_mult(self.queue1, self.subX_n_elements, None ,self.d_subX_idx,self.d_subX_sto,self.d_v,self.d_subX_replaced)
        ev7 = self.pgr.row_sum_reduce(self.queue1, self.n_molecules, None ,self.d_subX_replaced,self.d_subX_width,self.d_subX_height,self.d_dx)
        ev8 = self.pgr.set_true(self.queue2,self.n_one, None,self.d_is_valid_dx)
        ev9 = self.pgr.is_valid_dx(self.queue1, self.n_molecules, None ,self.d_x,self.d_dx,self.d_is_valid_dx,wait_for=[ev7,ev8])
        ev10 = self.cl.enqueue_copy(self.queue_transfer, self.is_valid_dx, self.d_is_valid_dx,wait_for=[ev9])
        self.cl.enqueue_barrier(self.queue_transfer)
        if(not self.is_valid_dx):
            ev_g0 = self.pgr.copy(self.queue1, self.n_reactions, None ,self.d_v,self.d_tmp_v)
            self.ev_rand = self.rand_gen.fill_uniform(self.d_rand,queue=self.queue2)
            while(not self.is_valid_dx):
                ev_g0 = self.pgr.fmult(self.queue2, self.n_one, None ,self.d_dt,self.d_alpha,self.d_dt)
                ev_g1 = self.pgr.mult_scalar_floor(self.queue1, self.n_reactions, None ,self.d_tmp_v,self.d_alpha,self.d_tmp_v)
                ev_g11 = self.pgr.mult_scalar(self.queue2, self.n_reactions, None ,self.d_dv,self.d_alpha,self.d_dv)
                ev_g3 = self.pgr.replace_and_mult(self.queue1, self.subX_n_elements, None ,self.d_subX_idx,self.d_subX_sto,self.d_tmp_v,self.d_subX_replaced)
                ev_g4 = self.pgr.row_sum_reduce(self.queue1, self.n_molecules, None ,self.d_subX_replaced,self.d_subX_width,self.d_subX_height,self.d_dx)
                ev_g5 = self.pgr.set_true(self.queue1,self.n_one, None,self.d_is_valid_dx)
                ev_g6 = self.pgr.is_valid_dx(self.queue1, self.n_molecules, None ,self.d_x,self.d_dx,self.d_is_valid_dx,wait_for=[ev_g4,ev_g5])
                ev_g7 = self.cl.enqueue_copy(self.queue_transfer, self.is_valid_dx, self.d_is_valid_dx,wait_for=[ev_g6])
                ev_g7.wait()
            transf = self.cl.enqueue_copy(self.queue_transfer, dt_buffer, self.d_dt,wait_for=[ev_g0])
        self.cl.enqueue_barrier(self.queue1)
        self.cl.enqueue_barrier(self.queue2)
        ev12 = self.pgr.sum_copy_buffer(self.queue1, self.n_molecules, None ,self.d_x,self.d_dx,self.d_x,self.d_x_buff)
        if get_vals:
            transf2 = self.cl.enqueue_copy(self.queue_transfer, x_buffer, self.d_x_buff,wait_for=[ev12])
        self.ev_rand = self.rand_gen.fill_normal(self.d_rand,queue=self.queue2)
    
    
    
    def compute_step(self):
        dt = self.max_dt
        self.replace_and_perm(self.subV_idx,self.subV_sto,self.x,self.subV_replaced)
        self.fv = np.prod(self.subV_replaced,axis=0).astype(self.float_type)*self.k*self.max_dt + self.dv
        noise = np.sqrt(self.fv)*np.random.normal(size=self.fv.size)*np.sqrt(self.max_dt)
        self.fv = (self.fv+noise).clip(min=0.)
        self.dv,self.tmp_fv = np.modf(self.fv)
        self.v = self.tmp_fv.astype(self.int_type)
        self.replace_and_mult(self.subX_idx,self.subX_sto,self.v,self.subX_replaced)
        self.dx = np.sum(self.subX_replaced,axis=1)
        x = (self.x+self.dx)
        self.is_valid_dx = not np.any(x<0)
        if not self.is_valid_dx:
            np.copyto(self.tmp_v, self.v)
            while not self.is_valid_dx:
                self.tmp_v = np.floor(self.tmp_v*self.alpha).astype(self.int_type)
                self.dv = self.dv*self.alpha
                dt = dt*self.alpha
                self.replace_and_mult(self.subX_idx,self.subX_sto,self.tmp_v,self.subX_replaced)
                self.dx = np.sum(self.subX_replaced,axis=1)
                x = (self.x+self.dx)
                self.is_valid_dx = not np.any(x<0)
        self.x = x
        
        return dt,x
            
    def replace_and_perm(self,idx,sto,x,r):
        for index, x_idx in np.ndenumerate(idx):
            r[index] = self.perm(x[x_idx],sto[index])
    
    def replace_and_mult(self,idx,sto,x,r):
        for index, x_idx in np.ndenumerate(idx):
            r[index] = x[x_idx]*sto[index]
        
    def perm(self,a,b):    
        if(not b): return 1
        if(a<b): return 0
        perm = 1.
        for i in range(b):
            perm *= (a-i)/(i+1)
        return perm
        
        
        