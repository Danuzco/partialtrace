import numpy as np

def single_sys_partial_trace(X, d_local, sys2btraced):
    
    dN = int(X.shape[0]) # Size of X. dN = d_local^N
    N = np.round(np.log(dN)/np.log(d_local), decimals=0) # N: Number of qudits
    N = int(N)

    d1 = d_local**(sys2btraced)
    I1 = np.identity(d1)
    d2 = d_local**(N-sys2btraced-1)
    I2 = np.identity(d2)

    v = np.identity(d_local)
    bra = np.kron(I1, np.kron(v[0],I2))
    Y = bra @ X @ bra.T

    for i in range(1,d_local):
        bra = np.kron(I1, np.kron(v[i],I2))
        Y += bra @ X @ bra.T

    return Y

        
def partialTrace(X, d_local, complement):
    """
    X: nxn matrix to be traced
    sub_systems: sub systems to trace
    d: local dimension
    """    
    complement = sorted(complement)
    
    for n,label in enumerate(complement):
        if n > 0:
            X = single_sys_partial_trace(X, d_local, label-n)
        else:
            X = single_sys_partial_trace(X, d_local, label)
    return X


def swapper(d):
    
    p = 0
    Id = np.identity(d)
    for i in range(d):
        for j in range(d):
            v = np.outer(Id[:,i],Id[:,j])
            u = np.transpose(v)
            p += np.kron(v,u)
            
    return p


def Pj( in_label, marginal, dl, num_of_qudits, swapper_d ):
    """
    dl: local dimension
    marginal: reduced system with labels given in the tuple "in_label"
    
    """
    
    label = in_label
    n = num_of_qudits - int(np.log(marginal.shape[0])/np.log(dl))
    # n = num_of_qudits - len( marginal.dims() ) 
    dims = tuple( [ dl for i in range( num_of_qudits ) ] )
    swapped_matrix = kron( marginal.data, np.identity( dl**n ) )
    
    all_labels = [ i for i in range( num_of_qudits ) ]
    right_labels = [ i for i in range( list( label )[-1] + 1, num_of_qudits ) ]
    left_labels = [ i for i in range( list( label )[0] ) ]
    
    if left_labels + list( label ) + right_labels == all_labels:
        nl  = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        return swapped_matrix/np.trace(swapped_matrix)
    else:
        nl = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        label = tuple( left_labels + list( label ) )
        
    length = len( label )
    remaining = tuple( [ i for i in range( length ) ] )
    
    while length > 0 and label != remaining:
        
        last = label[-1]
        numOfswapps = np.abs( last  - length ) 
        l1, l2 = length - 1, num_of_qudits - ( length + 1 )
        I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
        gate = kron( I1, swapper_d, I2 ) 
        swapped_matrix = gate @ swapped_matrix @ gate
        
        for i in range( numOfswapps ):
            l1, l2 = l1 + 1, l2 - 1 
            I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
            gate = kron( I1, swapper_d, I2 )
            swapped_matrix = gate @ swapped_matrix @ gate

        label = tuple( list( label[:-1] ) )
        length = len( label )
        remaining = tuple( [ i for i in range( length ) ] ) 
    
    return swapped_matrix/np.trace(swapped_matrix)


def kron(*matrices):
    
    m1, m2, *ms = matrices
    m3 = np.kron(m1, m2)
    
    for m in ms:
        m3 = np.kron(m3, m)
    
    return m3


def compute_marginals_distance(rho0, prescribed_marginals,d,num_of_qudits):
    
    all_systems = set( list( range(num_of_qudits)) )
    marginal_hsd = 0
    projected_marginals = {}
    
    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set(l) )
        projected_marginals[l] = partialTrace(rho0 , d, list( antisys ) )
        marginal_hsd += np.linalg.norm( projected_marginals[l] - prescribed_marginals[l])**2
    
    norm = len( list( prescribed_marginals.keys() ) )
    marginal_hsd = np.sqrt(marginal_hsd/norm)
    
    return projected_marginals, marginal_hsd


def get_marginals(rho_in, d, num_of_qudits, labels_marginals):

    dn = d**num_of_qudits

    marginals = {}
    all_systems = set( list( range(num_of_qudits)) )

    for s in labels_marginals:
        tracedSystems = tuple( all_systems - set( s ) )
        if len(tracedSystems) > 0:
            marginals[s] = partialTrace(rho_in,d,list(tracedSystems))
        else:
            marginals[s] = partialTrace(rho_in,d,list(s))
            
    return marginals



def masking( in_label, k, dl,num_of_qudits ):
    """
    dl: local dimension
    marginal: reduced system with labels given in the tuple "in_label"
    
    """
    
    swapper_d = swapper(dl)

    marginal = np.ones((dl**k,dl**k))
    label = in_label
    n = num_of_qudits - k
    # n = num_of_qudits - len( marginal.dims() ) 
    dims = tuple( [ dl for i in range( num_of_qudits ) ] )
    swapped_matrix = kron( marginal.data, np.identity( dl**n ) )
    
    all_labels = [ i for i in range( num_of_qudits ) ]
    right_labels = [ i for i in range( list( label )[-1] + 1, num_of_qudits ) ]
    left_labels = [ i for i in range( list( label )[0] ) ]
    
    if left_labels + list( label ) + right_labels == all_labels:
        nl  = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        return swapped_matrix
    else:
        nl = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        label = tuple( left_labels + list( label ) )
        
    length = len( label )
    remaining = tuple( [ i for i in range( length ) ] )
    
    while length > 0 and label != remaining:
        
        last = label[-1]
        numOfswapps = np.abs( last  - length ) 
        l1, l2 = length - 1, num_of_qudits - ( length + 1 )
        I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
        gate = kron( I1, swapper_d, I2 ) 
        swapped_matrix = gate @ swapped_matrix @ gate
        
        for i in range( numOfswapps ):
            l1, l2 = l1 + 1, l2 - 1 
            I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
            gate = kron( I1, swapper_d, I2 )
            swapped_matrix = gate @ swapped_matrix @ gate

        label = tuple( list( label[:-1] ) )
        length = len( label )
        remaining = tuple( [ i for i in range( length ) ] ) 
    
    return swapped_matrix
