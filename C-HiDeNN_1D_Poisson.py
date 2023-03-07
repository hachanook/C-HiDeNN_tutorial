import numpy as onp
from scipy.sparse import csc_matrix
# from scipy.sparse import  csc_matrix
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
import time
from itertools import combinations
import math
from functools import partial
import os,sys
import humanize,psutil,GPUtil
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import cm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this

# memory report
def mem_report(num, gpu_idx):
    print(f"-{num}-CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
    
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[gpu_idx]
    # for i, gpu in enumerate(GPUs):
    print('---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))


from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=4)

############# FEM functions ############
def GaussSet(Gauss_Num = 2, cuda=False):
    if Gauss_Num == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
       
    elif Gauss_Num == 3:
        Gauss_Weight1D = [0.55555556, 0.88888889, 0.55555556]
        Gauss_Point1D = [-0.7745966, 0, 0.7745966]
       
        
    elif Gauss_Num == 4:
        Gauss_Weight1D = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        Gauss_Point1D = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]

    elif Gauss_Num == 6: # double checked, 16 digits
        Gauss_Weight1D = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        Gauss_Point1D = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 
                         0.2386191860831969, 0.6612093864662645, 0.9324695142031521]

       
    elif Gauss_Num == 8: # double checked, 20 digits
        Gauss_Weight1D=[0.10122853629037625915, 0.22238103445337447054, 0.31370664587788728733, 0.36268378337836198296,
                        0.36268378337836198296, 0.31370664587788728733, 0.22238103445337447054,0.10122853629037625915]
        Gauss_Point1D=[-0.960289856497536231684, -0.796666477413626739592,-0.525532409916328985818, -0.183434642495649804939,
                        0.183434642495649804939,  0.525532409916328985818, 0.796666477413626739592,  0.960289856497536231684]
        
    elif Gauss_Num == 10:
        Gauss_Weight1D=[0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        Gauss_Point1D=[-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,  
                        0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717]
        
    elif Gauss_Num == 20:
        Gauss_Weight1D=[0.017614007, 0.04060143, 0.062672048, 0.083276742,0.10193012, 0.118194532,0.131688638,
                        0.142096109, 0.149172986, 0.152753387,0.152753387,0.149172986, 0.142096109, 0.131688638,
                        0.118194532,0.10193012, 0.083276742,0.062672048,0.04060143,0.017614007]
            
        Gauss_Point1D=[-0.993128599, -0.963971927, -0.912234428, -0.839116972, -0.746331906, -0.636053681,
                        -0.510867002, -0.373706089, -0.227785851, -0.076526521, 0.076526521, 0.227785851,
                        0.373706089, 0.510867002, 0.636053681, 0.746331906, 0.839116972, 0.912234428, 0.963971927, 0.993128599]
    
    return Gauss_Weight1D, Gauss_Point1D

def uniform_mesh(d1, p, element_type, regular_mesh_bool):
      
    if element_type == 'D1LN2N': # 1D 2node linear element
    
        PD = 1 # problem dimension
        q = onp.array([0,d1], dtype=onp.double)    
        NoN = p+1 # number of nodes
        NoE = p # number of elements
        NPE = 2 # nodes per elements
        
        iffix = onp.zeros(NoN, dtype=onp.int32)
        
        ## Nodes ##
        NL = onp.zeros([NoN, PD], dtype=onp.double)
        a = (q[1]-q[0])/p # increment in the horizontal direction
        
        n = 0 # This will allow us to go through rows in NL
        for j in range(1, p+2):
            if j == 1 or j == p+1: # boundary
                NL[n,0] = q[0] + (j-1)*a # for x values
                iffix[n] = 1
            elif regular_mesh_bool: # regular mesh
                NL[n,0] = q[0] + (j-1)*a # for x values
            else: # irregular mesh
                NL[n,0] = q[0] + (j-1)*a + onp.random.normal(0,0.1,1)*a # for x values, gaussian noise; mean:0, std:0.1
            n += 1
            
        ## elements ##
        EL = onp.zeros([NoE, NPE], dtype=onp.int32)
        for j in range(1, p+1):
            
            if j == 1:
                EL[j-1, 0] = j
                EL[j-1, 1] = EL[j-1, 0] + 1
                
            else:
                EL[j-1, 0] = EL[j-2, 1]
                EL[j-1, 1] = EL[j-1, 0] + 1
            
    EL -= 1 # python indexing
    dof_global = len(iffix)
        
    return NL, EL, iffix, NoN, NoE, dof_global

def get_quad_points(Gauss_Num, dim):
    # This function is compatible with FEM and CFEM
    
    Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_Num)
    
    quad_points, quad_weights = [], []

    for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
        if dim < 2:
            quad_points.append([ipoint])
            quad_weights.append(iweight)
        else:
            for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                if dim < 3:
                    quad_points.append([ipoint, jpoint])
                    quad_weights.append(iweight * jweight)
                else:
                    for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                        quad_points.append([ipoint, jpoint, kpoint])
                        quad_weights.append(iweight * jweight * kweight)
                            
            
    quad_points = np.array(quad_points) # (quad_degree*dim, dim)
    quad_weights = np.array(quad_weights) # (quad_degree,)
    return quad_points, quad_weights


def get_shape_val_functions(elem_type):
    """Hard-coded first order shape functions in the parent domain.
    Important: f1-f8 order must match "self.cells" by gmsh file!
    """
    if elem_type == 'D1LN2N':
        f1 = lambda x: 1./2.*(1 - x[0])
        f2 = lambda x: 1./2.*(1 + x[0]) 
        shape_fun = [f1, f2]
        
    return shape_fun

def get_shape_grad_functions(elem_type):
    shape_fns = get_shape_val_functions(elem_type)
    return [jax.grad(f) for f in shape_fns]

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type'])
def get_shape_vals(Gauss_Num, dim, elem_type):
    """Pre-compute shape function values

    Returns
    -------
    shape_vals: ndarray
        (8, 8) = (num_quads, num_nodes)  
    """
    shape_val_fns = get_shape_val_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = np.array(shape_vals) # (num_quads, num_nodes)
    # assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
    return shape_vals

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # must
def get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes):
    """Pre-compute shape function gradients

    Returns
    -------
    shape_grads_physical: ndarray
        (cell, num_quads, num_nodes, dim)  
    JxW: ndarray
        (cell, num_quads)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
            # Page 147, Eq. (3.9.3)
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
 
        shape_grads.append(physical_shape_grads)

    shape_grads = np.array(shape_grads) # (num_quads, num_nodes, dim)
    # print(shape_grads.shape, shape_grads)
    # assert shape_grads.shape == (global_args['num_quads'], global_args['num_nodes'], global_args['dim'])

    physical_coos = np.take(XY, Elem_nodes, axis=0) # (num_cells, num_nodes, dim)
    # print(physical_coos)
    # physical_coos: (num_cells, none,      num_nodes, dim, none)
    # shape_grads:   (none,      num_quads, num_nodes, none, dim)
    # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, dim, dim)
    jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2) # dx/deta
    # print(jacobian_dx_deta.shape)
    jacbian_det = np.linalg.det(jacobian_dx_deta) # (num_cells, num_quads)
    # print(jacbian_det)
    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta) # (num_cells, num_quads, dim, dim) # deta/dx
    # print(jacobian_deta_dx.shape)
    # (num_cell, num_quads, num_nodes, dim) matmul (num_cells, num_quads, dim, dim) -> (num_cell, num_quads, num_nodes, dim)
    shape_grads_physical = shape_grads[None, :, :, :] @ jacobian_deta_dx
    # print(shape_grads_physical.shape, shape_grads_physical)

    # For first order FEM with 8 quad points, those quad weights are all equal to one
    # quad_weights = 1.
    JxW = jacbian_det * quad_weights[None, :]
    return shape_grads_physical, JxW # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)

# compute FEM basic stuff
def get_A_b_FEM(XY, Elem_nodes, Gauss_Num_FEM, dim, elem_type, dof_global, c_body):
    
    # decide how many blocks are we gonnna use
    quad_num_FEM = Gauss_Num_FEM**dim
    size_BTB = nelem * quad_num_FEM * elem_dof * elem_dof
    nblock = int(size_BTB // max_array_size_block + 1)
    nelem_per_block_regular = nelem // nblock
    if nelem % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem % nblock
    print(f"FEM A_sp -> {nblock} blocks")
    
    shape_vals = get_shape_vals(Gauss_Num_FEM, dim, elem_type) # (num_quads, num_nodes)  
    
    # compute shape_vals_physical and store at CPU memory
    # also, compute preconditioner
    rhs = np.zeros(dof_global, dtype=np.float64)
    # compute A_sp
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
            
        Elem_nodes_block = Elem_nodes[elem_idx_block]
        shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num_FEM, dim, elem_type, XY, Elem_nodes_block)    
        # print(shape_grads_physical_block.shape)
        
        BTB_block = np.matmul(shape_grads_physical_block, np.transpose(shape_grads_physical_block, (0,1,3,2))) # (num_cells, num_quads, num_nodes, num_nodes)
        # print(BTB_block.shape)
        V_block = np.sum(BTB_block * JxW_block[:, :, None, None], axis=(1)).reshape(-1) # (num_cells, num_nodes, num_nodes) -> (1 ,)
        I_block = np.repeat(Elem_nodes_block, nodes_per_elem, axis=1).reshape(-1)
        J_block = np.repeat(Elem_nodes_block, nodes_per_elem, axis=0).reshape(-1)
        
        if iblock == 0:
            A_sp_scipy =  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        else:
            A_sp_scipy +=  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
    
        physical_coos_block = np.take(XY, Elem_nodes_block, axis=0) # (num_cells, num_nodes, dim)
        XYs_block = np.sum(shape_vals[None, :, :, None] * physical_coos_block[:, None, :, :], axis=2) # (num_cell, num_quad, dim)
        body_force_block = np.squeeze(vv_b_fun(XYs_block, c_body))
        v_vals_block = np.repeat(shape_vals[None, :, :], nelem_per_block, axis=0) # (num_cells, num_quads, num_nodes)
        rhs_vals_block = np.sum(v_vals_block * body_force_block[:,:,None] * JxW_block[:, :, None], axis=1).reshape(-1) # (num_cells, num_nodes) -> (num_cells*num_nodes)
        rhs = rhs.at[Elem_nodes_block.reshape(-1)].add(rhs_vals_block)  # assemble 
    # A_sp_scipy = A_sp_scipy.sort_indices()
    # A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
    # print(A_sp_scipy.toarray())
    # print(rhs)
    return A_sp_scipy, rhs


############### Convolutional FEM ###############

def get_adj_mat(Elem_nodes, nnode, s_patch):
    # Sparse matrix multiplication for graph theory.
    
    # get adjacency matrix of graph theory based on nodal connectivity
    adj_rows, adj_cols = [], []
    # self 
    for inode in range(nnode):
        adj_rows += [inode]
        adj_cols += [inode]
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        for (inode, jnode) in combinations(list(elem_nodes), 2):
            adj_rows += [inode, jnode]
            adj_cols += [jnode, inode]
    adj_values = onp.ones(len(adj_rows), dtype=onp.int32)
    adj_rows = onp.array(adj_rows, dtype=onp.int32)
    adj_cols = onp.array(adj_cols, dtype=onp.int32)
    
    # build sparse matrix
    adj_sp = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    adj_s = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    # print(adj_sp.toarray())

    # compute s th power of the adjacency matrix to get s th order of connectivity
    for itr in range(s_patch-1):
        adj_s = adj_s.dot(adj_sp)
    indices = adj_s.indices
    indptr = adj_s.indptr
    return indices, indptr
    

def get_dex_max(indices, indptr, elem_type, s_patch, d_c, XY, Elem_nodes, nelem, nnode, nodes_per_elem, dim):
    edex_max = (2+2*s_patch)**dim # estimated value of edex_max
    Elemental_patch_nodes_st = (-1)*onp.ones((nelem, edex_max+50), dtype=onp.int32) # giving extra +20 spaces 
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 2: # 1D Linear element
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ]))) # node_idx 1
        
        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
        
    edex_max = onp.max(edexes)
    
    Elemental_patch_nodes_st = Elemental_patch_nodes_st[:,:edex_max]  # static, (num_elements, edex_max)
    
    # computes nodal support domain info and vmap_inputs
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (num_cells, num_nodes)
    for ielem, (elem_nodes, edex, elemental_patch_nodes_st) in enumerate(zip(
            Elem_nodes, edexes, Elemental_patch_nodes_st)):
        
        elemental_patch_nodes = elemental_patch_nodes_st[:edex]
        
        for inode_idx, inode in enumerate(elem_nodes):
            
            dist_mat = onp.absolute(XY[elemental_patch_nodes,:] - XY[inode,:])
            # if ielem == 5:
            #     print(XY[elemental_patch_nodes,:] - XY[inode,:])
            
            if elem_type == 'CPE4' or elem_type == 'CPE3' or 'D2QU4N' or 'D1LN2N':
                dist_patch = (s_patch+0.9)*d_c
            elif elem_type == 'C3D4':
                # dist_patch = onp.max(dist_mat, axis=0) * 1.1
                dist_patch = onp.max(dist_mat) * 0.8
            elif elem_type == 'C3D8':
                dist_patch = (s_patch+0.9)*d_c
            
            # print(dist_patch)
            if dim == 1:                                # check x coord               
                nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch))[0]
            
            ndex = len(nodal_patch_nodes_idx)
            ndexes[ielem, inode_idx] = ndex
            
    ndex_max = onp.max(ndexes)
    edex_min = onp.min(edexes)
    ndex_min = onp.min(ndexes)
    print(f'edex_min / ndex_min: {edex_min} / {ndex_min}')

    return edex_max, ndex_max

def get_patch_info(indices, indptr, elem_type, s_patch, d_c, edex_max, ndex_max, XY, Elem_nodes, nelem, nodes_per_elem, dim):
    
    # Elemental patch, ~s
    Elemental_patch_nodes_st = onp.zeros((nelem, edex_max), dtype=onp.int32) # edex_max should be grater than 100!
    # here, elemental_patch_nodes_st should be initialized with zeros, because of the global stiffness assembly!
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 2: # 1D Linear element
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ]))) # node_idx 1

        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
    
    Nodal_patch_nodes_st = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    Nodal_patch_nodes_bool = onp.zeros((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    Nodal_patch_nodes_idx = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (num_cells, num_nodes)
    
    # Nodal patch, ~a
    for ielem, (elem_nodes, edex, elemental_patch_nodes_st) in enumerate(zip(
            Elem_nodes, edexes, Elemental_patch_nodes_st)):
        
        elemental_patch_nodes = elemental_patch_nodes_st[:edex]
        
        for inode_idx, inode in enumerate(elem_nodes):
            
            dist_mat = onp.absolute(XY[elemental_patch_nodes,:] - XY[inode,:])
            
            if elem_type == 'D1LN2N':
                dist_patch = (s_patch+0.9)*d_c
            
            # print(dist_patch)
            if dim == 1:                                # check x coord               
                nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch))[0]
            elif dim == 2:                                # check x coord                 # check y coord
                nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch) & (dist_mat[:,1]<=dist_patch))[0]
            elif dim == 3:                              # check x coord                 # check y zoord             # check z coord
                # nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch[0]) & (dist_mat[:,1]<=dist_patch[1]) & (dist_mat[:,2]<=dist_patch[2]))[0]
                nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch) & (dist_mat[:,1]<=dist_patch) & (dist_mat[:,2]<=dist_patch))[0]
               
            nodal_patch_nodes = elemental_patch_nodes[nodal_patch_nodes_idx]
            ndex = len(nodal_patch_nodes)
            
            Nodal_patch_nodes_st[ielem, inode_idx, :ndex] = nodal_patch_nodes  # convert to global nodes
            Nodal_patch_nodes_bool[ielem, inode_idx, :ndex] = onp.where(nodal_patch_nodes>=0, 1, 0)
            Nodal_patch_nodes_idx[ielem, inode_idx, :ndex] = nodal_patch_nodes_idx
            ndexes[ielem, inode_idx] = ndex
            
    # Convert everything to device array
    Elemental_patch_nodes_st = np.array(Elemental_patch_nodes_st)
    edexes = np.array(edexes)
    Nodal_patch_nodes_st = np.array(Nodal_patch_nodes_st)
    Nodal_patch_nodes_bool = np.array(Nodal_patch_nodes_bool)
    Nodal_patch_nodes_idx = np.array(Nodal_patch_nodes_idx)
    ndexes = np.array(ndexes)
    
    return Elemental_patch_nodes_st, edexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes

@partial(jax.jit, static_argnames=['ndex_max','mbasis']) # This will slower the function
def Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis=0):
    # x, y: point of interest
    # xv: [nx - dim X ndex] support domain nodal coords.
    # ndex: number of supporting nodes
    # R: Shape parameter, R = alpha_c * d_c
    # q: shape parameter
    # nRBF: type of RBF, 1-MQ, 2-Cubic spline
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    # derivative: order of derivative
    
    dim = len(xy)
    RP = np.zeros(ndex_max + mbasis, dtype=np.double)
    
    for i in range(ndex_max):
        ndex_bool = nodal_patch_nodes_bool[i]
        zI = np.linalg.norm(xy - xv[i])/a_dil # by defining support domain, zI is bounded btw 0 and 1.
        
        bool1 = np.ceil(0.50001-zI) # returns 1 when 0 < zI <= 0.5 and 0 when 0.5 < zI <1
        bool2 = np.ceil(zI-0.50001) # returns 0 when 0 < zI <= 0.5 and 1 when 0.5 < zI <1
        bool3 = np.heaviside(1-zI, 1) # returns 1 when zI <= 1 // 0 when 1 < zI
                
        # Cubic spline
        RP = RP.at[i].add( ((2/3 - 4*zI**2 + 4*zI**3         ) * bool1 +    # phi_i
                            (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * bool2) * bool3 * ndex_bool)   # phi_i
        
    if dim == 1:    
        if mbasis > 0: # 1st
            RP = RP.at[ndex].set(1)         # N
            RP = RP.at[ndex+1].set(xy[0])   # N
        if mbasis > 2: # 2nd
            RP = RP.at[ndex+2].set(xy[0]**2)   # N
        if mbasis > 3: # 3nd
            RP = RP.at[ndex+3].set(xy[0]**3)   # N

            
    return RP

@partial(jax.jit, static_argnames=['ndex_max','mbasis']) # This will slower the function
def Compute_RadialBasis(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis=0):
    # x, y: point of interest
    # xv: [nx - dim X ndex] support domain nodal coords.
    # ndex: number of supporting nodes
    # R: Shape parameter, R = alpha_c * d_c
    # q: shape parameter
    # nRBF: type of RBF, 1-MQ, 2-Cubic spline
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    # derivative: order of derivative
    
    # if derivative == 0:
        # RP = np.zeros((ndex_max + mbasis, 1), dtype=np.double)
    # elif derivative == 1:
    dim = len(xy)
    RP = np.zeros((ndex_max + mbasis, dim+1), dtype=np.double)

    for i in range(ndex_max):
        ndex_bool = nodal_patch_nodes_bool[i]
        zI = np.linalg.norm(xy - xv[i])/a_dil # by defining support domain, zI is bounded btw 0 and 1.
        
        bool1 = np.ceil(0.50001-zI) # returns 1 when 0 < zI <= 0.5 and 0 when 0.5 < zI <1
        bool2 = np.ceil(zI-0.50001) # returns 0 when 0 < zI <= 0.5 and 1 when 0.5 < zI <1
        bool3 = np.heaviside(1-zI, 1) # returns 1 when zI <= 1 // 0 when 1 < zI
        
        # get derivatives
        dzIdx = (xy[0] - xv[i,0])/(a_dil**2*zI)
        dzIdy = (xy[1] - xv[i,1])/(a_dil**2*zI)
        dzIdz = (xy[2] - xv[i,2])/(a_dil**2*zI)
        
        
        # Cubic spline
        RP = RP.at[i,0].add( ((2/3 - 4*zI**2 + 4*zI**3         ) * bool1 +    # phi_i
                              (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * bool2) * bool3 * ndex_bool)   # phi_i
        
        RP = RP.at[i,1].add( ((    -8*zI + 12*zI**2)*dzIdx * bool1 +  # phi_i,x
                              (-4 + 8*zI -  4*zI**2)*dzIdx * bool2) * bool3 * ndex_bool) # phi_i,x
        RP = RP.at[i,2].add( ((    -8*zI + 12*zI**2)*dzIdy * bool1 + # phi_i,y
                              (-4 + 8*zI -  4*zI**2)*dzIdy * bool2) * bool3 * ndex_bool) # phi_i,y
        RP = RP.at[i,3].add( ((    -8*zI + 12*zI**2)*dzIdz * bool1 + # phi_i,y
                              (-4 + 8*zI -  4*zI**2)*dzIdz * bool2) * bool3 * ndex_bool) # phi_i,z
     
    if dim == 1:    
        if mbasis > 0: # 1st
            RP = RP.at[ndex,   0].set(1)         # N
            RP = RP.at[ndex+1, 0].set(xy[0])   # N
            
            RP = RP.at[ndex+1, 1].set(1)   # dNdx
            
        if mbasis > 2: # 2nd
            RP = RP.at[ndex+2, 0].set(xy[0]**2)         # N
            RP = RP.at[ndex+2, 1].set(2 * xy[0]   )     # dNdx
            
        if mbasis > 3: # 3nd
            RP = RP.at[ndex+3, 0].set(xy[0]**3)         # N    
            RP = RP.at[ndex+3, 1].set(3 * xy[0]**2)     # dNdx
            
    return RP

# @partial(jax.jit, static_argnames=['ndex_max','mbasis']) # unneccessary
def get_G(xv, ndex, ndex_max, nodal_patch_nodes_bool, a_dil, mbasis):
    # nodal_patch_nodes_bool: (ndex_max,)
    G = np.zeros((ndex_max + mbasis, ndex_max + mbasis), dtype=np.double)

    # Build RP
    for idx, (X, ndex_bool) in enumerate(zip(xv, nodal_patch_nodes_bool)):

        RP = Compute_RadialBasis_1D(X, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                                  a_dil, mbasis) # (edex_max,) but only 'ndex+1' nonzero terms
        G = G.at[:,idx].set(RP * ndex_bool)             
    
    # Make symmetric matrix
    G = np.tril(G) + np.triu(G.T, 1)
    # Build diagonal terms to nullify dimensions
    for idx, ndex_bool in enumerate(nodal_patch_nodes_bool):
        G = G.at[idx + mbasis, idx + mbasis].add(abs(ndex_bool-1))
    return G # G matrix

def get_Gs(vmap_input_G, 
            Nodal_patch_nodes_st, Nodal_patch_nodes_bool, ndexes, ndex_max,
            XY,  a_dil, mbasis):
    
    ielem = vmap_input_G[0]
    inode_idx = vmap_input_G[1]
    
    ndex = ndexes[ielem, inode_idx]
    nodal_patch_nodes = Nodal_patch_nodes_st[ielem, inode_idx, :] # static
    nodal_patch_nodes_bool = Nodal_patch_nodes_bool[ielem, inode_idx, :] # static
    
    xv = XY[nodal_patch_nodes,:]
    G = get_G(xv, ndex, ndex_max, nodal_patch_nodes_bool,  a_dil, mbasis)
    
    return G

@partial(jax.jit, static_argnames=['edex_max','ndex_max','mbasis'])
def get_phi(vmap_input, elem_idx, Gs, shape_vals, edex_max, # 5
            Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes, ndex_max, # 5
            XY, Elem_nodes, a_dil, mbasis): # 7
    
    ielem_idx = vmap_input[0]
    ielem = elem_idx[ielem_idx]
    iquad = vmap_input[1]
    inode_idx = vmap_input[2]
    shape_val = shape_vals[iquad,:]

    elem_nodes = Elem_nodes[ielem, :]
    xy_elem = XY[elem_nodes,:] # (num_nodes, dim)
    
    ndex = ndexes[ielem_idx, inode_idx]
    nodal_patch_nodes = Nodal_patch_nodes_st[ielem_idx, inode_idx, :] # static
    nodal_patch_nodes_bool = Nodal_patch_nodes_bool[ielem_idx, inode_idx, :] # static
    nodal_patch_nodes_idx = Nodal_patch_nodes_idx[ielem_idx, inode_idx, :] # static

    xv = XY[nodal_patch_nodes,:]
    G = Gs[ielem_idx, inode_idx, :, :]
    
    xy = np.sum(shape_val[:, None] * xy_elem, axis=0, keepdims=False)
    RP = Compute_RadialBasis(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis)
    phi_org = np.linalg.solve(G.T, RP)[:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    phi = np.zeros((edex_max + 1, 1+dim))  # trick, add dummy node at the end
    phi = phi.at[nodal_patch_nodes_idx, :].set(phi_org) 
    phi = phi[:edex_max,:] # trick, delete dummy node
    
    return phi

def get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                       XY, XY_host, Elem_nodes, Elem_nodes_host, shape_vals, Gauss_Num, dim, elem_type, nodes_per_elem,
                       indices, indptr, s_patch, d_c, edex_max, ndex_max,
                        a_dil, mbasis):
    
    # time_patch, time_G, time_Phi = 0,0,0
    quad_num = Gauss_Num**dim
    elem_idx_block_host = onp.array(elem_idx_block) # cpu
    elem_idx_block = np.array(elem_idx_block) # gpu
    Elem_nodes_block_host = Elem_nodes_host[elem_idx_block_host] # cpu
    Elem_nodes_block = Elem_nodes[elem_idx_block] # gpu
    shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes_block) # gpu
    
    # start_patch = time.time()
    (Elemental_patch_nodes_st_block, edexes_block,
     Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, 
     ndexes_block) = get_patch_info(indices, indptr, elem_type, s_patch, d_c, edex_max, ndex_max, XY_host, Elem_nodes_block_host, 
                                    nelem_per_block, nodes_per_elem, dim)                         
    # time_patch += (time.time() - start_patch)
    # print(Elemental_patch_nodes_st_block)
    
    #############  get G #############
    # start_G = time.time()
    # compute moment matrix for the meshfree shape functions
    vmap_inputs_G = np.concatenate((np.repeat(np.arange(nelem_per_block), nodes_per_elem), # ielem
                      np.tile(np.arange(nodes_per_elem), nelem_per_block))).reshape(2,-1).T # inode
    
    # print(vmap_inputs_G)
    Gs = jax.vmap(get_Gs, in_axes = (0, 
            None, None, None, None, 
            None, None, None), out_axes = 0
            )(vmap_inputs_G,
              Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, ndexes_block, ndex_max,
              XY, a_dil, mbasis) ########################################### XY_norm!!
    Gs = np.reshape(Gs, (nelem_per_block, nodes_per_elem, ndex_max+mbasis, ndex_max+mbasis))
    # Gs: (num_cells, num_nodes, ndex_max+mbasis, ndex_max+mbasis)
    # time_G += (time.time() - start_G)
    
    ############# Phi ###############
    # start_Phi = time.time()
    # compute meshfree shape functions
    # linearize vmap inputs (ielem, iquad, inode)
    vmap_inputs = np.concatenate((np.repeat(np.arange(nelem_per_block), quad_num*nodes_per_elem), # ielem_idx
                      np.tile(np.repeat(np.arange(quad_num), nodes_per_elem), nelem_per_block), # iquad
                      np.tile(np.arange(nodes_per_elem), nelem_per_block*quad_num))).reshape(3,-1).T # inode
    
    # print(vmap_inputs)
    # Vmap for CFEM shape functions for regular elements
    Phi = jax.vmap(get_phi, in_axes = (0, None, None, None, None, 
            None, None, None, None, None, 
            None, None, None, None), out_axes = 0
            )(vmap_inputs, elem_idx_block, Gs, shape_vals, edex_max,
            Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block, ndex_max,
            XY, Elem_nodes, a_dil, mbasis) ########################################### XY_norm!!

   
    # (num_cells, num_quads, num_nodes, edex_max, 1+dim)
    Phi = np.reshape(Phi, (nelem_per_block, quad_num, nodes_per_elem, edex_max, 1+dim))
    N_til_block = np.sum(shape_vals[None, :, :, None]*Phi[:,:,:,:,0], axis=2) # (num_cells, num_quads, edex_max)
    Grad_N_til_block = (np.sum(shape_grads_physical_block[:, :, :, None, :]*Phi[:,:,:,:,:1], axis=2) 
                      + np.sum(shape_vals[None, :, :, None, None]*Phi[:,:,:,:,1:], axis=2) )
    
    # Check partition of unity
    # if 0 in elem_idx_block: 
    if not ( np.allclose(np.sum(N_til_block, axis=2), np.ones((nelem_per_block, quad_num), dtype=np.double)) and
            np.allclose(np.sum(Grad_N_til_block, axis=2), np.zeros((nelem_per_block, quad_num, dim), dtype=np.double)) ):
        print(f"PoU Check failed at element {elem_idx_block[0]}~{elem_idx_block[-1]}")
        # print(np.sum(N_til_block, axis=2))
        PoU_Check_N = (np.linalg.norm(np.sum(N_til_block, axis=2) - np.ones((nelem_per_block, quad_num), dtype=np.float64))**2/(nelem_per_block*quad_num))**0.5
        PoU_Check_Grad_N = (np.linalg.norm(np.sum(Grad_N_til_block, axis=2))**2/(nelem_per_block*quad_num*dim))**0.5
        print(f'PoU check N / Grad_N: {PoU_Check_N:.4e} / {PoU_Check_Grad_N:.4e}')
        
    return N_til_block, Grad_N_til_block, JxW_block, Elemental_patch_nodes_st_block, Elem_nodes_block

def get_A_b_CFEM(XY, XY_host, Elem_nodes, Elem_nodes_host, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
                indices, indptr, s_patch, d_c, edex_max, ndex_max,
                 a_dil, mbasis):

    start_time = time.time()
    # time_vmap, time_BTB, time_sp, time_rhs = 0, 0,0,0
    # decide how many blocks are we gonnna use
    quad_num_CFEM = Gauss_Num_CFEM**dim
    nelem = len(Elem_nodes)
    size_BTB = int(nelem) * int(quad_num_CFEM) * int(edex_max*dim) * int(edex_max*dim)
    size_Phi = int(nelem) * int(quad_num_CFEM) * int(elem_dof) * int(edex_max) * int(1+dim)
    size_Gs  = int(nelem) * int(quad_num_CFEM) * int(elem_dof) * int((ndex_max+mbasis)**2)
    nblock = int(max(size_BTB, size_Phi, size_Gs) // max_array_size_block + 1) # regular blocks + 1 remainder
    # print('nelem', nelem, nblock)
    
    nelem_per_block_regular = nelem // nblock # set any value
    (quo, rem) = divmod(nelem, nelem_per_block_regular)
    if rem == 0:
        nblock = quo
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nblock = quo + 1
        nelem_per_block_remainder = rem       
    # if itr == 0: # this is for C-HiDeNN iteration
    #     print(f"CFEM A_sp -> {nblock} blocks with {nelem_per_block_regular} elems per block and remainder {nelem_per_block_remainder}")
    assert nelem == (nblock-1) * nelem_per_block_regular + nelem_per_block_remainder
    # print(f"nelem_per_block_remainder: {nelem_per_block_remainder}")
    
    shape_vals = get_shape_vals(Gauss_Num_CFEM, dim, elem_type) # (num_quads, num_nodes)     
    rhs = np.zeros(dof_global, dtype=np.double)
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
        # print(f'{iblock}th block, {nelem_per_block} elems')
        
        # Elem_nodes_block = Elem_nodes[elem_idx_block]
        
        
        # start_vmap = time.time()
        (N_til_block, Grad_N_til_block, JxW_block, 
         Elemental_patch_nodes_st_block, Elem_nodes_block) = get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                   XY, XY_host, Elem_nodes, Elem_nodes_host, shape_vals, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
                   indices, indptr, s_patch, d_c, edex_max, ndex_max,
                    a_dil, mbasis)
        if iblock == 0:
            jit_time = time.time() - start_time
            print(f"CFEM shape function jit compile took {jit_time:.4f} seconds")
        # time_vmap += (time.time() - start_vmap)
        
        # start_BTB = time.time()
        BTB_block = onp.matmul(Grad_N_til_block, np.transpose(Grad_N_til_block, (0,1,3,2))) # (num_cells, num_quads, edex_max, edex_max)
        V_block = onp.sum(BTB_block * JxW_block[:, :, None, None], axis=(1)).reshape(-1) # (num_cells, num_nodes, num_nodes) -> (1 ,)
        I_block = onp.repeat(Elemental_patch_nodes_st_block, edex_max, axis=1).reshape(-1)
        J_block = onp.repeat(Elemental_patch_nodes_st_block, edex_max, axis=0).reshape(-1)
        # time_BTB += (time.time() - start_BTB)
        
        # start_sp = time.time()
        if iblock == 0:
            A_sp_scipy =  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        else:
            A_sp_scipy +=  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        # time_sp += (time.time() - start_sp)
        
        # start_rhs = time.time()
        physical_coos_block = np.take(XY, Elem_nodes_block, axis=0) # (num_cells, num_nodes, dim)
        XYs_block = np.sum(shape_vals[None, :, :, None] * physical_coos_block[:, None, :, :], axis=2)
        body_force_block = np.squeeze(vv_b_fun(XYs_block, c_body))
        rhs_vals_block = np.sum(N_til_block * body_force_block[:,:,None] * JxW_block[:, :, None], axis=1).reshape(-1) # (num_cells, num_nodes) -> (num_cells*num_nodes)
        rhs = rhs.at[Elemental_patch_nodes_st_block.reshape(-1)].add(rhs_vals_block)  # assemble 
        # time_rhs += (time.time() - start_rhs)
        
    # print(f"vmap / BTB / sp / rhs took {time_vmap} / {time_BTB} / {time_sp} / {time_rhs} seconds")
    print(f"CFEM A_sp took {time.time() - start_time:.4f} seconds\n")
    return A_sp_scipy, rhs, nblock, jit_time


#%% ############## Error analysis ###############

@jax.jit
def u_fun_1D(x, a):
    # x is a scalar, a is a parameter, mostly being np.pi
    u = (np.exp(-a*(x-2.5)**2)-np.exp(-6.25*a)) + 2*(np.exp(-a*(x-7.5)**2)-np.exp(-56.25*a)) - ((np.exp(-6.25*a)-np.exp(-56.25*a)))/10*x
    return u

@jax.jit
def Grad_u_fun_1D(x, a):
    # x is a scalar, a is a parameter, mostly being np.pi
    dudx = -2*a*(x-2.5)*np.exp(-a*(x-2.5)**2) -4*a*(x-7.5)*np.exp(-a*(x-7.5)**2) + ((np.exp(-6.25*a)-np.exp(-56.25*a)))/10
    return dudx

@jax.jit
def b_fun_1D(x, a):
    # x is a scalar, a is a parameter, mostly being np.pi
    b = -(4*a**2*(x-2.5)**2-2*a)*np.exp(-a*(x-2.5)**2) - 2*(4*a**2*(x-7.5)**2-2*a)*np.exp(-a*(x-7.5)**2)
    return b

def get_FEM_norm(XY, Elem_nodes, u, Gauss_Num_norm, dim, elem_type):

    ''' Variables
    L2_nom: (num_cells, num_quads)
    L2_denom: (num_cells, num_quads)
    H1_nom: (num_cells, num_quads)
    H1_denom: (num_cells, num_quads)
    XY_norm: (num_cells, num_quads, dim)
    u_exact: (num_cells, num_quads)
    uh: (num_cells, num_quads)
    Grad_u_exact: (num_cells, num_quads, dim)
    Grad_uh: (num_cells, num_quads, dim) 
    '''
    
    # decide how many blocks are we gonnna use
    quad_num_norm = Gauss_Num_norm**dim
    size_shape_grads_physical = nelem * quad_num_norm * nodes_per_elem * dim
    nblock = int(size_shape_grads_physical // max_array_size_block + 1)
    nelem_per_block_regular = nelem // nblock
    if nelem % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem % nblock
    print(f"FEM norm -> {nblock} blocks")
    
    # Compute norm
    L2_nom, L2_denom, H1_nom, H1_denom = 0,0,0,0
    shape_vals = get_shape_vals(Gauss_Num_norm, dim, elem_type) # (num_quads, num_nodes)  
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
    
        Elem_nodes_block = Elem_nodes[elem_idx_block]
        shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num_norm, dim, elem_type, XY, Elem_nodes_block)    

        physical_coos = np.take(XY, Elem_nodes_block, axis=0) # (num_cells_block, num_nodes, dim)
        xy_norm = np.sum(shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        u_exact = np.squeeze(vv_u_fun(xy_norm, c_body))
        Grad_u_exact = vv_Grad_u_fun(xy_norm, c_body)

        u_coos = np.take(u, Elem_nodes_block, axis=0) # (num_cells_block, num_nodes)
        uh = np.sum(shape_vals[None, :, :] * u_coos[:, None, :], axis=2)
        Grad_uh = np.sum(shape_grads_physical_block[:, :, :, :] * u_coos[:, None, :, None], axis=2)
        
        # print(u_exact.shape, uh.shape, Grad_u_exact.shape, Grad_uh.shape)
        
        L2_nom += np.sum((u_exact-uh)**2 * JxW_block)
        L2_denom += np.sum((u_exact)**2 * JxW_block)
        H1_nom += np.sum(((u_exact-uh)**2 + np.sum((Grad_u_exact-Grad_uh)**2, axis=2)) * JxW_block)
        H1_denom += np.sum(((u_exact)**2 + np.sum((Grad_u_exact)**2, axis=2)) * JxW_block)
    
    L2_norm = (L2_nom/L2_denom)**0.5
    H1_norm = (H1_nom/H1_denom)**0.5
    print(f'L2_norm_FEM: {L2_norm:.4e}')
    print(f'H1_norm_FEM: {H1_norm:.4e}')
    return L2_norm, H1_norm

def get_CFEM_norm(u, XY, XY_host, Elem_nodes, Elem_nodes_host, 
                  Gauss_Num_norm, dim, elem_type, nodes_per_elem,
                  indices, indptr, s_patch, d_c, edex_max, ndex_max,
                   a_dil, mbasis, nblock):
    
    
    # decide how many blocks are we gonnna use
    quad_num_norm = Gauss_Num_norm**dim
    size_Phi = nelem * quad_num_norm * nodes_per_elem * edex_max * (1+dim)
    nblock = int(size_Phi // max_array_size_block + 1)
    nelem_per_block_regular = nelem // nblock
    if nelem % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem % nblock
    print(f"CFEM norm -> {nblock} blocks")
    
    # Compute norm
    L2_nom, L2_denom, H1_nom, H1_denom = 0,0,0,0
    shape_vals = get_shape_vals(Gauss_Num_norm, dim, elem_type) # (num_quads, num_nodes)  
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
            
        elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
        # Elem_nodes_block = Elem_nodes[elem_idx_block]
        (N_til_block, Grad_N_til_block, JxW_block, 
         Elemental_patch_nodes_st_block, Elem_nodes_block) = get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                   XY, XY_host, Elem_nodes, Elem_nodes_host, shape_vals, Gauss_Num_norm, dim, elem_type, nodes_per_elem,
                   indices, indptr, s_patch, d_c, edex_max, ndex_max,
                    a_dil, mbasis)
        # print(shape_grads_physical_block.shape)    
        if iblock == 0:
            print(f"CFEM shape fuction jit compile took {time.time() - start_time:.4f} seconds\n")
        
        
        
        physical_coos = np.take(XY, Elem_nodes_block, axis=0) # (num_cells_block, num_nodes, dim)
        xy_norm = np.sum(shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        u_exact = np.squeeze(vv_u_fun(xy_norm, c_body))
        Grad_u_exact = vv_Grad_u_fun(xy_norm, c_body)

        u_coos = np.take(u, Elemental_patch_nodes_st_block, axis=0) # (num_cells_block, num_nodes)
        uh = np.sum(N_til_block[:, :, :] * u_coos[:, None, :], axis=2)
        Grad_uh = np.sum(Grad_N_til_block[:, :, :, :] * u_coos[:, None, :, None], axis=2)
        
        # print(u_exact.shape, uh.shape, Grad_u_exact.shape, Grad_uh.shape)
        
        L2_nom += np.sum((u_exact-uh)**2 * JxW_block)
        L2_denom += np.sum((u_exact)**2 * JxW_block)
        H1_nom += np.sum(((u_exact-uh)**2 + np.sum((Grad_u_exact-Grad_uh)**2, axis=2)) * JxW_block)
        H1_denom += np.sum(((u_exact)**2 + np.sum((Grad_u_exact)**2, axis=2)) * JxW_block)
    
    L2_norm = (L2_nom/L2_denom)**0.5
    H1_norm = (H1_nom/H1_denom)**0.5
    print(f'L2_norm_FEM: {L2_norm:.4e}')
    print(f'H1_norm_FEM: {H1_norm:.4e}')
    return L2_norm, H1_norm


#%% Solver

@partial(jax.jit, static_argnames=['nchunk','dof_per_chunk','dof_per_chunk_remainder'])
def get_residual(sol, A_sp, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    res = b - A_sp @ sol
    res = res.at[inds_nodes_list].set(0) # disp. BC
    return res

@partial(jax.jit, static_argnames=['nchunk','dof_per_chunk','dof_per_chunk_remainder'])
def get_Ap(p, A_sp, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    Ap = A_sp @ p
    Ap = Ap.at[inds_nodes_list].set(0) # disp. BC
    return Ap

def get_residual_chunks(sol, A_sp_scipy, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    
    res = np.zeros(dof_global, dtype=np.double)  
    for ichunk in range(nchunk):
        if ichunk < nchunk-1:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_per_chunk*(ichunk+1)), dtype=np.int32)
        else:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_global), dtype=np.int32)
        A_sp_chunk = BCOO.from_scipy_sparse(A_sp_scipy[dof_idx_chunk,:])
        res_chunk = b[dof_idx_chunk] - A_sp_chunk @ sol
        res = res.at[dof_idx_chunk].set(res_chunk)
    res = res.at[inds_nodes_list].set(0) # disp. BC
    return res
    
def get_Ap_chunks(p, A_sp_scipy, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    
    Ap = np.zeros(dof_global, dtype=np.double)  
    for ichunk in range(nchunk):
        if ichunk < nchunk-1:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_per_chunk*(ichunk+1)), dtype=np.int32)
        else:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_global), dtype=np.int32)
        A_sp_chunk = BCOO.from_scipy_sparse(A_sp_scipy[dof_idx_chunk,:])
        Ap_chunk = A_sp_chunk @ p
        Ap = Ap.at[dof_idx_chunk].set(Ap_chunk)
    Ap = Ap.at[inds_nodes_list].set(0) # disp. BC
    return Ap

def CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
              nchunk, dof_per_chunk, dof_per_chunk_remainder):
    start_time = time.time()
    r = get_residual(sol, A_sp, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder)
    p = r
    rsold = np.dot(r,r)
    for step in range(dof_global):
        # print(p)
        Ap = get_Ap(p, A_sp, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder)
        alpha = rsold / np.dot(p, Ap)
        sol += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r,r)
        
        if step%1000 == 0:
            print(f"step = {step}, res l_2 = {rsnew**0.5}") 
        
        if rsnew**0.5 < tol:
            break
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    print(f"CG solver took {time.time() - start_time:.4f} seconds\n")
    return sol


#%% Main program - Cubic spline functions

############# 2D Poisson ###############


gpu_idx = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

# Problem settings
ps = [2]      # [0, 1, 2, 3]
alpha_dils = [10]       # [1.2, 1.4, 2.2, 2.4, 3.2, 3.4, 4.2, 4.4]    # dilation parameter
# alpha_dils = onp.arange(1, 6.2, 0.2)
s_patches = [2]         # elemental patch_size
nelems = [40]        # [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

max_array_size_block = 2e7  # 5e8 for Athena / 2e7 for laptop
max_array_size_chunk = 2e7  #
GPUs = GPUtil.getGPUs()
if len(GPUs) > 1:
    max_array_size_block = 5e8  # 5e8 for Athena / 2e7 for laptop
    max_array_size_chunk = 5e8  #
else:
    max_array_size_block = 2e7  # 5e8 for Athena / 2e7 for laptop
    max_array_size_chunk = 2e7  #

L = 10              # size of the domain
body_force_type = "double"  # 'single', 'double'
c_body = 5          # body force constant, 5
elem_types = ['D1LN2N'] # 
dim = 1
Gauss_Num_FEM = 4
Gauss_Num_CFEM = 6  # 6
Gauss_Num_norm = 8  # 8 or higher is enough

run_FEM = False
run_CFEM = True
radial_patch_bool = False
regular_mesh_bool = True

#%%
H1_norm = []
for elem_type in elem_types:
    nodes_per_elem = int(elem_type[4:-1])
    elem_dof = nodes_per_elem    
    # print(nodes_per_elem, elem_dof)
    if dim == 1:
        p_dict={0:0, 1:2, 2:3, 3:4, 4:5}
    if dim == 2:
        p_dict={0:0, 1:3, 2:6, 3:10, 4:15, 5:21, 6:28, 7:36, 8:45, 9:55, 10:66} # number of complete basis for polynomial order p
    elif dim == 3:
        p_dict={0:0, 1:4, 2:10, 3:20, 4:35} # number of complete basis for polynomial order p
    
    # define body force type
    vv_b_fun = jax.vmap(jax.vmap(b_fun_1D, in_axes = (0, None)), in_axes = (0, None))
    vv_u_fun = jax.vmap(jax.vmap(u_fun_1D, in_axes = (0, None)), in_axes = (0, None))
    vv_Grad_u_fun = jax.vmap(jax.vmap(Grad_u_fun_1D, in_axes = (0, None)), in_axes = (0, None))
    
    for s_patch in s_patches:
        for p in ps: # polynomial orders
            mbasis = p_dict[p]  
            for alpha_dil in alpha_dils:
                for nelem_x in nelems:
                    
                    #FEM settings
                    XY_host, Elem_nodes_host, iffix, nnode, nelem, dof_global = uniform_mesh(L, nelem_x, elem_type, regular_mesh_bool)
                    inds_nodes_list = onp.where(iffix==1)    # fixed nodes
                    
                    # host to device array
                    XY = np.array(XY_host)
                    Elem_nodes = np.array(Elem_nodes_host)
                    inds_nodes_list = np.array(inds_nodes_list)
                    shape_vals = get_shape_vals(Gauss_Num_FEM, dim, elem_type) # (num_quads, num_nodes) 
                    
                    # mem_report(1, gpu_idx)

                    if run_FEM == True:
                        print(f"\n----------------- FEM elem_tpye/ nelem_x: {elem_type} / {nelem_x} --------------------")
                        # start_time = time.time()
                        
                        # compute FEM basic stuff
                        start_time = time.time()
                        A_sp_scipy, b = get_A_b_FEM(XY, Elem_nodes, Gauss_Num_FEM, dim, elem_type, dof_global, c_body)
                        print(f"FEM A and b took {time.time() - start_time:.4f} seconds")
                        
                        # CG solver
                        # Divide A_sp matrix into nchunk
                        size_A_sp = nnode * 9 * 3 # 9 is nodal connectivity. Each node is connected to 9 surrounding nodes. 3 means sparse values & indicies 
                        # print(f'A_sp size: {size_A_sp*20/1e6:.1f} MB')
                        nchunk = int(size_A_sp // max_array_size_chunk + 1)
                        if nnode % nchunk == 0:
                            dof_per_chunk = nnode // nchunk
                            dof_per_chunk_remainder = dof_per_chunk
                        else:
                            dof_per_chunk = nnode // nchunk
                            dof_per_chunk_remainder = dof_per_chunk + nnode % nchunk
                        print(f"A_sp array -> {nchunk} chunks")
                        
                        sol = np.zeros(dof_global, dtype=np.double)              # (dof,)
                        sol = sol.at[inds_nodes_list].set(0)
                        tol = 1e-10
                        
                        if nchunk == 1: # when DOFs are small
                            A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
                            sol = CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
                                            nchunk, dof_per_chunk, dof_per_chunk_remainder)
                        else: # when DOFs are big
                            sol = CG_solver(get_residual_chunks, get_Ap_chunks, sol, A_sp_scipy, b, inds_nodes_list, dof_global, tol, 
                                            nchunk, dof_per_chunk, dof_per_chunk_remainder)
                       
                        # FEM error norm
                        L2_norm_FEM, H1_norm_FEM = get_FEM_norm(XY, Elem_nodes, sol, Gauss_Num_norm, dim, elem_type)
                        # mem_report(2, gpu_idx)

                    ########################## CFEM ######################
                    if run_CFEM == False:
                        continue
                    if elem_type != 'D2QU4N' and elem_type != 'D1LN2N':
                        continue
                    if s_patch*2 > nelem_x:
                        continue
                    if p > s_patch or p > alpha_dil:
                        continue
                    # compute adjacency matrix - Serial
                    print(f"\n- - - - - - CFEM nelem_x: {nelem_x} with s: {s_patch}, a: {alpha_dil}, p: {p} - - - - - -")  
                    
                    start_time_org = time.time()
                    # print(Elem_nodes_host)
                    # print(nnode, s_patch)
                    indices, indptr = get_adj_mat(Elem_nodes_host, nnode, s_patch)
                    print(f"CFEM adj_s matrix took {time.time() - start_time_org:.4f} seconds")
                    
                    # # patch settings
                    d_c = L/nelem_x     # characteristic length in physical coord.
                    a_dil = alpha_dil * d_c
                    # a_dil = 6
                    
                    # compute Elemental patch - Serial
                    start_time = time.time()
                    edex_max, ndex_max = get_dex_max(indices, indptr, elem_type, s_patch, d_c, XY_host, Elem_nodes_host, nelem, nnode, nodes_per_elem, dim)
                    print(f'edex_max / ndex_max: {edex_max} / {ndex_max}, took {time.time() - start_time:.4f} seconds')
                    
                    
                    A_sp_scipy, b, nblock, jit_time = get_A_b_CFEM(XY, XY_host, Elem_nodes, Elem_nodes_host, Gauss_Num_CFEM, dim, elem_type, nodes_per_elem,
                                            indices, indptr, s_patch, d_c, edex_max, ndex_max, a_dil, mbasis)
                    # mem_report(2, gpu_idx)
        
                    # CG solver
                    # Divide A_sp matrix into nchunk
                    size_A_sp = nnode * edex_max * 3 # 9 is nodal connectivity. Each node is connected to 9 surrounding nodes. 3 means sparse values & indicies 
                    # print(f'A_sp size: {size_A_sp*20/1e6:.1f} MB')
                    nchunk = int(size_A_sp // max_array_size_chunk + 1)
                    if nnode % nchunk == 0:
                        dof_per_chunk = nnode // nchunk
                        dof_per_chunk_remainder = dof_per_chunk
                    else:
                        dof_per_chunk = nnode // nchunk
                        dof_per_chunk_remainder = dof_per_chunk + nnode % nchunk
                    print(f"A_sp array -> {nchunk} chunks\n")
        
    
                    sol = np.zeros(dof_global, dtype=np.double)              # (dof,)
                    sol = sol.at[inds_nodes_list].set(0)
                    tol = 1e-10
                    
                    if nchunk == 1: # when DOFs are small
                        A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
                        sol = CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
                                        nchunk, dof_per_chunk, dof_per_chunk_remainder)
                    else: # when DOFs are big
                        sol = CG_solver(get_residual_chunks, get_Ap_chunks, sol, A_sp_scipy, b, inds_nodes_list, dof_global, tol, 
                                        nchunk, dof_per_chunk, dof_per_chunk_remainder)
                    
                    computation_time = time.time() - start_time_org - jit_time
        

                    # Error norm #
                    L2_norm_CFEM, H1_norm_CFEM = get_CFEM_norm(sol, XY, XY_host, Elem_nodes, Elem_nodes_host, 
                                                                Gauss_Num_norm, dim, elem_type, nodes_per_elem,
                                                                indices, indptr, s_patch, d_c, edex_max, ndex_max,
                                                                  a_dil, mbasis, nblock)
                    mem_report(4, gpu_idx)


