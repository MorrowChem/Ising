import numpy as np

def calcEnergy(int [:,::1] config, float j0, float j1, float sav, float h):
    '''Returns the energy of a configuration.
    Parameters:
    config      : NxN numpy array describing the spin states
    floats       : j0,j1,sav
    h           : magnetic field, optional not currently implemented
    Depends on:
    j0,j1 coupling constants
    '''
    cdef:
        float s,nb,energy
        int a,b,i,j,N
    energy = 0.
    N = len(config)
    for a in range(N):
        for b in range(N):
            s = config[a,b] # config generator gives 1s and -1s
            nb = j1*(config[(a+1)%N,b]+config[(a-1)%N,b])\
               + j0*(config[a,(b+1)%N]+config[a,(b-1)%N])
            energy += (-0.5*nb + h)*s*sav #-0.5 in here to avoid double-counting the interactions and make like spins favourable, s2 as only product of spins is relevant
    return energy

def calcMag(config, float s1, float s2): 
    '''Magnetization of a given configuration (in atomic units)''' 
    mag = s1*np.sum(config.flatten()[0:-1:2]) + s2*np.sum(config.flatten()[1:-1:2])
    return mag

def mcmoves(int [:,::1] config, float beta, int steps, int N, float j0, float j1, float sav, float h=0):
    '''Monte Carlo moves using Metropolis algorithm, perfomed as a block for efficiency
    Sytem-scaled, so performs N**2 moves per step.
    Parameters:
    config      : NxN numpy array describing the spin states
    beta        : 1/T, couplings constants are defined in temperature units (K), so give beta in K too 
    steps       : Number of MC moves to perform
    N           : System size (N**2 is number of sites)
    j0          : coupling along horizontal
    j1          : coupling along vertical
    sav         : product of spin1 and spin2
    h           : magnetic field, optional not currently implemented
    
    Returns:
    None'''
    cdef: 
        int s,nb,a,b,i,j,k
        int [:,::1] rands
        float [:,::1] randfloat 
        float [::1] cost_lookup
    rg = np.random.Generator(np.random.PCG64())
    ### Lookup table for the exponentials #####
    cost_lookup = 2.7182818**(-2*beta*sav*(np.array([-(-2*j0-2*j1),
                                                  -(-2*j0+0*j1),
                                                  -(-2*j0+2*j1),
                                                  -( 0*j0-2*j1),
                                                  -( 0*j0+0*j1),
                                                  -( 0*j0+2*j1),
                                                  -( 2*j0-2*j1),
                                                  -( 2*j0+0*j1),
                                                  -( 2*j0+2*j1),
                                                  #
                                                  (-2*j0-2*j1),
                                                  (-2*j0+0*j1),
                                                  (-2*j0+2*j1),
                                                  ( 0*j0-2*j1),
                                                  ( 0*j0+0*j1),
                                                  ( 0*j0+2*j1),
                                                  ( 2*j0-2*j1),
                                                  ( 2*j0+0*j1),
                                                  ( 2*j0+2*j1),
                                                  ]))).astype('float32')
    for k in range(steps):
        rands = rg.integers(0,N, size=(N,2*N), dtype='int32')
        randfloat = rg.random((N,N), dtype='float32')
        for i in range(N): 
            for j in range(N):
                a = rands[i,j] 
                b = rands[i,-j]
                s = config[a, b]
                nb = ((config[(a+1)%N,b]+config[(a-1)%N,b]\
                   + 3*(config[a,(b+1)%N]+config[a,(b-1)%N]))\
                   //2 + (s//2+1)*9 + 4)
                if randfloat[i,j] < cost_lookup[nb]:  
                    config[a, b] *= -1

def ind_mcmoves(int [:,::1] config, float beta, int steps, int N, float j0, float j1, float sav, float h=0):
    '''Monte Carlo moves using Metropolis algorithm, perfomed as a block for efficiency
    Parameters:
    config      : NxN numpy array describing the spin states
    beta        : 1/T, couplings constants are defined in temperature units (K), so give beta in K too 
    steps       : Number of MC moves to perform
    N           : System size (N**2 is number of sites)
    j0          : coupling along horizontal
    j1          : coupling along vertical
    sav         : product of spin1 and spin2
    h           : magnetic field, optional not currently implemented
    
    Returns:
    None'''
    cdef: 
        int a,b,k,nb,s
        int [:,::1] rands 
        float [::1] randfloat
        float [::1] cost_lookup
    rg = np.random.Generator(np.random.PCG64())
    ### Lookup table for the exponentials #####
    cost_lookup = 2.7182818**(-2*beta*sav*(np.array([-(-2*j0-2*j1),
                                                  -(-2*j0+0*j1),
                                                  -(-2*j0+2*j1),
                                                  -( 0*j0-2*j1),
                                                  -( 0*j0+0*j1),
                                                  -( 0*j0+2*j1),
                                                  -( 2*j0-2*j1),
                                                  -( 2*j0+0*j1),
                                                  -( 2*j0+2*j1),
                                                  #
                                                  (-2*j0-2*j1),
                                                  (-2*j0+0*j1),
                                                  (-2*j0+2*j1),
                                                  ( 0*j0-2*j1),
                                                  ( 0*j0+0*j1),
                                                  ( 0*j0+2*j1),
                                                  ( 2*j0-2*j1),
                                                  ( 2*j0+0*j1),
                                                  ( 2*j0+2*j1),
                                                  ]))).astype('float32')
    rands = rg.random(0,N, size=(2,2*steps), dtype='int32')
    randfloat = rg.random(steps,dtype='float32')
    for k in range(steps):
        a = rands[0,k] 
        b = rands[1,-k-1] 
        s = config[a, b]
        nb = ((config[(a+1)%N,b]+config[(a-1)%N,b]\
                   + 3*(config[a,(b+1)%N]+config[a,(b-1)%N]))\
                   //2 + (s//2+1)*9 + 4)
        if randfloat[k] < cost_lookup[nb]:  
                    config[a, b] *= -1

def get_NN(p, int [:,::1] config, int N):
    '''returns list of nearest neighbour points'''
    cdef:
        int [4][3] NN
        int [2] np
        int k
    for k in (-1,1):
        np = [(p[1]+k)%N,(p[2]+k)%N]
        NN[k+1] = [config[np[0],p[2]], np[0], p[2]]
        NN[k+2] = [config[p[1],np[1]], p[1], np[1]]
    return NN
