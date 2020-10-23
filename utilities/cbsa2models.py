from sys import path as syspath
from os import path as ospath
import os, glob
import numpy as np
import functools
import copy
from timeit import default_timer as timer


#CBSA
syspath.append(ospath.join(ospath.expanduser("~"), 'CBSA'))
from cbsa import ReactionSystem

def cbsa2stochpy(cbsa_model,path="/home/burke/Stochpy/pscmodels/"):
    import stochpy as sp
    model_str = ""
    for i in range(1,cbsa_model.exp_n_reactions):
        reactants = np.where(cbsa_model.expS[:,i] < 0)[0]
        reactants_sto = list(cbsa_model.expS[:,i][reactants]*-1)
        modifiers = np.where(cbsa_model.expR[:,i] > 0)[0]
        modifiers_sto = list(cbsa_model.expR[:,i][modifiers])
        products = np.where(cbsa_model.expS[:,i] > 0)[0]
        products_sto = list(cbsa_model.expS[:,i][products])

        psc_reactants = []
        psc_modifiers = []
        psc_products = []
        
        if len(reactants):
            psc_reactants = ["{"+str(reactants_sto[j])+"}M"+str(reactants[j]) for j in range(len(reactants))]
        if len(modifiers):
            psc_modifiers = ["{"+str(modifiers_sto[j])+"}M"+str(modifiers[j]) for j in range(len(modifiers))]
        if len(products):
            psc_products = ["{"+str(products_sto[j])+"}M"+str(products[j]) for j in range(len(products))]
        
        psc_reactants += psc_modifiers
        psc_products += psc_modifiers
        
        if not len(psc_reactants):
            psc_reactants = ['$pool']
            
        if not len(psc_products):
            psc_products = ['$pool']
        
        k_mols = [["M"+str(reactants[j]) for k in range(reactants_sto[j])] for j in range(len(reactants))]
        k_mols += [["M"+str(modifiers[j]) for k in range(modifiers_sto[j])] for j in range(len(modifiers))]
        k_mols = [item for sublist in k_mols for item in sublist]
        
        model_str += "R"+str(i)+":\n\t"
        model_str += "+".join(psc_reactants)            
        model_str += " > "
        model_str += "+".join(psc_products)        
        model_str += "\n\t"
        model_str += "*".join(k_mols+["k"+str(i)])
        model_str += "\n\n"
    
    for i in range(1,cbsa_model.exp_n_reactions):
        model_str += "k"+str(i)+" = "+str(cbsa_model.exp_k[i])+"\n"
    
    model_str += "\n"
    
    for i in range(1,cbsa_model.exp_n_molecules):
        model_str += "M"+str(i)+" = "+str(int(cbsa_model.exp_x0[i]))+"\n"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    model_file = "model_tmp"+str(timer())+".psc"
    
    with open(path+model_file,"w") as f:
        f.write(model_str)

    smod = sp.SSA()
    smod.Model(model_file)
    
    return smod


def cbsa2gillespy(cbsa_model):
    import gillespy2 as glp
    
    gilles = glp.Model(name="Model")

    k = [glp.Parameter(name='k'+str(i), expression=cbsa_model.exp_k[i]) for i in range(1,cbsa_model.exp_n_reactions)]
    gilles.add_parameter(k)
    mols = [glp.Species(name='M'+str(i), initial_value=int(cbsa_model.exp_x0[i])) for i in range(1,cbsa_model.exp_n_molecules)]
    gilles.add_species(mols)
    
    reactions = []
    
    for i in range(1,cbsa_model.exp_n_reactions):
        reactants = list(np.where(cbsa_model.expS[:,i] < 0)[0])
        reactants_sto = list(cbsa_model.expS[:,i][reactants]*-1)
        modifiers = list(np.where(cbsa_model.expR[:,i] > 0)[0])
        modifiers_sto = list(cbsa_model.expR[:,i][modifiers])
        products = list(np.where(cbsa_model.expS[:,i] > 0)[0])
        products_sto = list(cbsa_model.expS[:,i][products])
        
        reactants += modifiers
        reactants_sto += modifiers_sto
        products += modifiers
        products_sto += modifiers_sto
        
        reactions.append(glp.Reaction(name="R"+str(i),
                                      rate=k[i-1],
                                      reactants={mols[reactants[j]-1]:reactants_sto[j] for j in range(len(reactants))},
                                      products={mols[products[j]-1]:products_sto[j] for j in range(len(products))}
                                     )
                        )
    
    gilles.add_reaction(reactions)
    
    return gilles

def cbsa2steps(cbsa_model):
    
    import steps.geom as swm
    import steps.model as smodel
    import steps.rng as srng
    import steps.solver as ssolver
    
    mdl = smodel.Model()
    vsys = smodel.Volsys('vsys', mdl)    
    mols = [smodel.Spec('M'+str(i), mdl) for i in range(1,cbsa_model.exp_n_molecules)]    
    reactions = []    
    for i in range(1,cbsa_model.exp_n_reactions):
        reactants = list(np.where(cbsa_model.expS[:,i] < 0)[0])
        reactants_sto = list(cbsa_model.expS[:,i][reactants]*-1)
        modifiers = list(np.where(cbsa_model.expR[:,i] > 0)[0])
        modifiers_sto = list(cbsa_model.expR[:,i][modifiers])
        products = list(np.where(cbsa_model.expS[:,i] > 0)[0])
        products_sto = list(cbsa_model.expS[:,i][products])
        
        reactants += modifiers
        reactants_sto += modifiers_sto
        products += modifiers
        products_sto += modifiers_sto
        
        reactants_objs = [[mols[reactants[j]-1] for k in range(reactants_sto[j])] for j in range(len(reactants))]
        reactants_objs = [item for sublist in reactants_objs for item in sublist]
        
        products_objs = [[mols[products[j]-1] for k in range(products_sto[j])] for j in range(len(products))]
        products_objs = [item for sublist in products_objs for item in sublist]
        
        reactions.append(smodel.Reac("R"+str(i), vsys, lhs=reactants_objs, rhs=products_objs, kcst=cbsa_model.exp_k[i]))
    
    wmgeom = swm.Geom()

    comp = swm.Comp('comp', wmgeom)
    comp.addVolsys('vsys')
    comp.setVol(1.6667e-21)

    r = srng.create('mt19937', 256)
    r.initialize(int(timer()))
    sim = ssolver.Wmdirect(mdl, wmgeom, r)
    sim.reset()

    for i in range(1,cbsa_model.exp_n_molecules):
        sim.setCompConc('comp', 'M'+str(i), cbsa_model.exp_x0[i]*1e-6)
    
    return sim
    


def generate_cbsa_diffusion_model(sqrt_n_spaces,init_mols,diffusion_k,max_dt=0.1,total_sim_time=100):
    
    S = [[0]]
    R = [[0]]

    x = [init_mols]
    k = [0.]
    diff_k = [diffusion_k]    
    
    cbsa = ReactionSystem(S,R)
    if sqrt_n_spaces<2:
        cbsa.setup()
    else:
        cbsa.setup(sqrt_n_spaces**2,'toroid')

    cbsa.set_x(x)
    cbsa.set_k(k,diff_k)
    
    cbsa.bench_max_dt = max_dt
    cbsa.bench_total_sim_time = total_sim_time
    
    return cbsa

def generate_cbsa_burst_model(sqrt_n_spaces=1,diffusion_k=0.1,max_dt=0.1,total_sim_time=100):    
    
    S = [[-1,1,0,0],
         [1,-1,0,0],
         [0,0,1,-1]]

    R = [[0,0,1,0],
         [0,0,0,0],
         [0,0,0,0]]

    x = [0,1,0]
    k = [0.05,0.05,200,0.5]
    diff_k = [diffusion_k for i in range(len(x))]

    cbsa = ReactionSystem(S,R)
    if sqrt_n_spaces<2:
        cbsa.setup()
    else:
        cbsa.setup(sqrt_n_spaces**2,'toroid')
    cbsa.set_x(x,mol_per_subspace=True)
    cbsa.set_k(k,diff_k)
    
    cbsa.bench_max_dt = max_dt
    cbsa.bench_total_sim_time = total_sim_time
    
    return cbsa

def generate_cbsa_n_reactions_model(n_reactions,k0,k1,max_dt=0.1,total_sim_time=100):
    
    S = np.zeros((n_reactions+1,n_reactions+1))
    S[0,0] = 1
    S[0,1:] = np.array([-1 for i in range(n_reactions)])
    S[1:,1:] = np.identity(n_reactions)

    x = [0 for i in range(n_reactions+1)]
    k = [k0]+[k1 for i in range(n_reactions)]
    
    cbsa = ReactionSystem(S)
    cbsa.setup()
    cbsa.set_x(x)
    cbsa.set_k(k)
    
    cbsa.bench_max_dt = max_dt
    cbsa.bench_total_sim_time = total_sim_time
    
    return cbsa
