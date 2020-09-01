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

__kernel void column_prod_reduce(__global const int *m,__global const int *width,__global const int *height,__global float *r){
    int i = get_global_id(0);
    int w = *width;
    int h = *height;
    float v = 1.0;
    for(int j=0;j<h;j++){
        v *= m[i+j*w];
    }
    r[i] = v;
}

__kernel void add_noise(__global const float *fv,__global const float *n,__global float *r){
    int i = get_global_id(0);
    r[i] = fv[i]+(sqrt(fv[i])*n[i]);
}

__kernel void column_prod_reduce_mult_k(__global const int *m,__global const float *k,__global const int *width,__global const int *height,__global float *r){
    int i = get_global_id(0);
    int w = *width;
    int h = *height;
    float v = 1.0;
    for(int j=0;j<h;j++){
        v *= (float) m[i+j*w];
    }
    r[i] = v*k[i];
}


__kernel void separate_int_float(__global const float *fv,__global int *v,__global float *dv){
    int i = get_global_id(0);
    v[i] = (int) fv[i];
    dv[i] = fv[i] - (float) v[i];
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

__kernel void is_not_valid_dx(__global const int *x,__global const int *dx,__global int *r){
    int i = get_global_id(0);
    if(-dx[i]>x[i]) *r = 1;
}

__kernel void set_true(__global int *r){
    int i = get_global_id(0);
    r[i] = 1;
}

__kernel void set_false(__global int *r){
    int i = get_global_id(0);
    r[i] = 0;
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

__kernel void fsum(__global const float *a,__global const float *b,__global float *r){
    int i = get_global_id(0);
    r[i] = a[i]+b[i];
}

__kernel void fdivide(__global const float *a,__global const float *b,__global float *r){
    int i = get_global_id(0);
    r[i] = a[i]/b[i];
}

"""


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
        self.tmp_dv = np.zeros_like(self.k,dtype=self.float_type)
        
        self.is_valid_dx = np.array([1],self.int_type)
        self.is_not_valid_dx = np.array([1],self.int_type)
        
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
        self.d_tmp_dv = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dv)
        
        self.d_subV_height = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subV_sto.shape[0]))
        self.d_subV_width = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subV_sto.shape[1]))
        self.d_subX_height = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subX_sto.shape[0]))
        self.d_subX_width = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.int_type(self.subX_sto.shape[1]))
        
        self.d_is_not_valid_dx = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.is_not_valid_dx)
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
        #set variable to true
        self.is_not_valid_dx = np.array([1],self.int_type)
        #set dt to max_dt
        ev1 = self.pgr.copy(self.queue2,self.n_one, None,self.d_max_dt,self.d_dt)
        #Replace x values and permutate
        ev2 = self.pgr.replace_and_perm(self.queue1, self.subV_n_elements, None ,self.d_subV_idx,self.d_subV_sto,self.d_x,self.d_subV_replaced)
        #product reduce columns from subistituted and permutated x and multiply by constant k
        ev3 = self.pgr.column_prod_reduce_mult_k(self.queue1, self.n_reactions, None ,self.d_subV_replaced,self.d_k,self.d_subV_width,self.d_subV_height,self.d_fv)
        #while not valid dx (or the first time)
        while(self.is_not_valid_dx):
            #multiply fv by dt
            ev4 = self.pgr.mult_scalar(self.queue1, self.n_reactions, None ,self.d_fv,self.d_dt,self.d_tmp_fv)
            #add noise
            ev5 = self.pgr.add_noise(self.queue1, self.n_reactions, None ,self.d_tmp_fv,self.d_rand.data,self.d_tmp_fv)
            #sum dv to fv
            ev6 = self.pgr.fsum(self.queue1, self.n_reactions, None ,self.d_tmp_fv,self.d_dv,self.d_tmp_fv)
            #separate int and float parts
            ev7 = self.pgr.separate_int_float(self.queue1, self.n_reactions, None ,self.d_tmp_fv,self.d_v,self.d_tmp_dv)
            #replace v and muliply
            ev8 = self.pgr.replace_and_mult(self.queue1, self.subX_n_elements, None ,self.d_subX_idx,self.d_subX_sto,self.d_v,self.d_subX_replaced)
            #row sum reduce to compute dx
            ev9 = self.pgr.row_sum_reduce(self.queue1, self.n_molecules, None ,self.d_subX_replaced,self.d_subX_width,self.d_subX_height,self.d_dx)
            #set is_not_valid_dx to false (assumes it is right)
            ev10 = self.pgr.set_false(self.queue2,self.n_one, None,self.d_is_not_valid_dx)
            #check if any dx is not valid
            ev11 = self.pgr.is_not_valid_dx(self.queue1, self.n_molecules, None ,self.d_x,self.d_dx,self.d_is_not_valid_dx,wait_for=[ev10])
            #makes dt = dt*alpha
            ev12 = self.pgr.fmult(self.queue2, self.n_one, None ,self.d_dt,self.d_alpha,self.d_dt)
            #copy is_not_valid_dx to host
            ev13 = self.cl.enqueue_copy(self.queue_transfer, self.is_not_valid_dx,self.d_is_not_valid_dx,wait_for=[ev10])
            ev13.wait()
        # set dv = tmp_dv
        ev14 = self.pgr.copy(self.queue1,self.n_reactions, None,self.d_tmp_dv,self.d_dv)
        # make x = x+dx and copy to buffer
        ev15 = self.pgr.sum_copy_buffer(self.queue2, self.n_molecules, None ,self.d_x,self.d_dx,self.d_x,self.d_x_buff)
        # make dt = dt/alpha
        ev16 = self.pgr.fdivide(self.queue1, self.n_one, None ,self.d_dt,self.d_alpha,self.d_dt)
        #copy dt to host buffer
        self.cl.enqueue_copy(self.queue_transfer, dt_buffer, self.d_dt,wait_for=[ev16])
        if get_vals:
            #copy x to host buffer
            transf = self.cl.enqueue_copy(self.queue_transfer, x_buffer, self.d_x_buff)
        #generate new random numbers
        self.ev_rand = self.rand_gen.fill_normal(self.d_rand,queue=self.queue1)
        #wait all queues
        self.cl.enqueue_barrier(self.queue1)
        self.cl.enqueue_barrier(self.queue2)
    
    
    
    def compute_step(self):
        self.is_valid_dx = False
        dt = self.max_dt
        self.replace_and_perm(self.subV_idx,self.subV_sto,self.x,self.subV_replaced)
        self.fv = np.prod(self.subV_replaced,axis=0).astype(self.float_type)*self.k
        rand = np.random.normal(size=self.fv.size)
        while not self.is_valid_dx:
            self.tmp_fv = self.fv*dt
            self.tmp_fv = self.tmp_fv+np.sqrt(self.tmp_fv)*rand #add noise
            self.tmp_dv,self.tmp_fv = np.modf(self.tmp_fv+self.dv)
            self.v = self.tmp_fv.astype(self.int_type)
            self.replace_and_mult(self.subX_idx,self.subX_sto,self.v,self.subX_replaced)
            self.dx = np.sum(self.subX_replaced,axis=1)
            x = (self.x+self.dx)
            self.is_valid_dx = not np.any(x<0)
            dt = dt*self.alpha
        self.dv = self.tmp_dv
        self.x = x
        dt = dt/self.alpha
        return dt,x
            
    def replace_and_perm(self,idx,sto,x,r):
        for index, x_idx in np.ndenumerate(idx):
            r[index] = self.perm(x[x_idx],sto[index])
    
    def replace_and_mult(self,idx,sto,v,r):
        for index, x_idx in np.ndenumerate(idx):
            r[index] = v[x_idx]*sto[index]
        
    def perm(self,a,b):    
        if(not b): return 1
        if(a<b): return 0
        perm = 1.
        for i in range(b):
            perm *= (a-i)/(i+1)
        return perm
        
