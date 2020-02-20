import numpy as np
from core import *
from matplotlib import pyplot as plt
import os
import random


def initialstate(N):
    '''Generates a random spin configuration
    Parameters:
    N : system size'''
    state = 2*np.random.randint(2, size=(N,N))-1 # either 1 or -1
    state = state.astype('int32')
    return state

def orderedstate(N):
    '''Generates an initial ordered state
    Parameters:
    N : system size'''
    state = np.ones([N,N]).astype('int32')
    return state

def Ae(Ektav,Eavk,E2avk):
    '''Autocorrelation function:
    Parameters:
    Ektav  :
    Eavk   :
    E2avk  : '''
    return (Ektav - Eavk**2)/(E2avk-Eavk**2)

def quick_config(config,N):
    '''Quickly plots a picture of the current config
    Parameters:
    N : system size'''
    f = plt.figure(figsize=(5,5))
    X, Y = np.meshgrid(range(N), range(N))
    sp =  f.add_subplot(1,1,1)
    plt.setp(sp.get_yticklabels(), visible=False)
    plt.setp(sp.get_xticklabels(), visible=False)
    plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu);
    plt.axis('tight')
    plt.clim(-2,2)
    plt.show()
    return(f)

class Simulation_Average():
    '''Averages a list of simulations performed at the same T points
    Parameters:
    sim : list of instances of Simulations'''
    def __init__(self, sim):
        self.nt = sim[0].nt
        self.N = sim[0].N
        self.j0 = sim[0].j0
        self.j1 = sim[0].j1
        self.s1 = sim[0].s1
        self.s2 = sim[0].s2
        self.eqSteps = sim[0].eqSteps
        self.interSteps = sim[0].interSteps 
        self.ncalcs = sim[0].ncalcs
        self.algo = sim[0].algo
        self.sim = sim
        self.T = sim[0].T
        self.avE,self.avM,self.avC,self.avX =\
        np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)
        self.avE2,self.avM2,self.avC2,self.avX2 =\
        np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)

        d = 1/len(sim)

        for i in range(len(sim)):
            self.avE += sim[i].E*d
            self.avM += abs(sim[i].M*d)
            self.avC += sim[i].C*d
            self.avX += sim[i].X*d
            self.avE2 += sim[i].E**2*d
            self.avM2 += abs(sim[i].M**2*d)
            self.avC2 += sim[i].C**2*d
            self.avX2 += sim[i].X**2*d
        self.sigE = (self.avE2 - self.avE**2)**0.5
        self.sigC = (self.avC2 - self.avC**2)**0.5
        print('average energies are ',self.avE,'\n')
        print('std deviations are ',self.sig)

class Autocorrelation_Average():
    '''averages a list of Autocorrelation simulations performed at the same temperature points
    Parameters:
    sim : a list of Autocorrelation_simulation objects'''
    def __init__(self, sim):
        self.nt = sim[0].nt
        self.sim = sim
        self.steps_test = sim[0].steps_test
        self.steps = np.array(range(*self.steps_test))
        self.Aes = np.zeros((self.nt,len(range(*self.steps_test))))
        self.cavs = np.zeros(self.nt)
        self.avE,self.avE2,self.avM,self.avM2,self.avC,self.avX =\
        np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test))))

        d = 1/len(sim)
        print(sim[0].cavs)
        for k in range(len(sim)): # evaluate the Ae for each run for each T for each value of interSteps
            self.cavs += np.array(sim[k].cavs)*d
            for j in range(sim[k].nt):
                for i in range(len(range(*self.steps_test))):
                    self.Aes[j,i] += Ae(sim[k].Ektav[j,i],sim[k].Eavk[j,i],
                            sim[k].E2avk[j,i])*d 
                    self.avE[j,i] += sim[k].Eavk[j,i]*d
                    self.avE2[j,i] += sim[k].E2avk[j,i]*d
                    self.avM[j,i] += sim[k].Mavk[j,i]*d
                    self.avM2[j,i] += sim[k].M2avk[j,i]*d
                    self.avC[j,i] += sim[k].Cf[j,i]*d
                    self.avX[j,i] += sim[k].Xf[j,i]*d
                    # add the weighted value of the Autocorrelation function the the list of Aes for
                    # a given T and number of intervening steps
        print(self.Aes)

def decay_function(t,m,c): # model function for the decay
    return m*t+c

def auto_fit(auto_av,data):
    """Fits autocorrelation vs time measurements to a simple exponential decay
    Parameters:
    auto_av : instance from Autocorrelation_Average class
    data = slice object,
    choose the data you want to fit too 
    (often the large steps limit is dominated by statistical errors)

    Returns:
    None

    Modifies:
    adds (1D list) x and (2D list) y to auto_av instance"""
    from scipy import optimize
    auto_av.params = np.zeros([len(auto_av.Aes),2])

    Aes = auto_av.Aes
    params = auto_av.params

    for i in range(0,len(auto_av.sim[0].T)):
        params[i], params_cov = optimize.curve_fit(decay_function,\
                                  auto_av.steps[data],\
                                  np.log(np.ma.masked_less(Aes[i][data],0)\
                                 .filled(fill_value=1e-5)),\
                                  p0=[-0.015,-0.5])
    print(params)
    print(1/params.T[0])

    auto_av.x = auto_av.steps
    auto_av.y = []

    for i in range(len(params)):
        auto_av.y.append(decay_function(auto_av.x,params[i][0],params[i][1]))


def post_fit(T,steps,aes,data):
    """Fits autocorrelation vs time measurements to a simple exponential decay
    Parameters:
    auto_av : instance from Autocorrelation_Average class
    data = slice object,
    choose the data you want to fit too (often the large steps limit is dominated by statistical errors)

    Returns:
    None

    Modifies:
    adds (1D list) x and (2D list) y to auto_av instance"""
    from scipy import optimize

    params = np.zeros([len(T),2])

    for i in range(0,len(T)):
        params[i], params_cov = optimize.curve_fit(decay_function,\
                              steps[0][data],\
                              np.log(np.ma.masked_less(aes[i][data],0).\
                                     filled(fill_value=1e-5)),\
                              p0=[-0.015,-0.5])
    print(params)
    print(1/params.T[0])
    x = np.array(steps[0])
    y = []

    for i in range(len(params)):
        y.append(decay_function(x,params[i][0],params[i][1]))
    return(x,y)

def Wolff_old(config,beta,j0,j1,sav):
    '''Simple implementation of Wolff cluster algorithm
    Parameters:
    config : conifguration to be updated
    beta   : 1/T, designed to be done in K^-1 units
    j0 : horizontal coupling constant
    j1 : vertical coupling constant
    sav : product of the spins on different sites'''
    
    N = len(config)
    E0 = 1 - np.exp(-2*beta*j0*sav) # pre-calculate expensive functions
    E1 = 1 - np.exp(-2*beta*j1*sav)
    r1 = random.randint(0,N-1) # darts to start
    r2 = random.randint(0,N-1)
    cluster = [[config[r1,r2], r1, r2]]
    config[r1,r2] *= -1 # must flip the very first one here, doesn't affect value in cluster
    n = 0
    while cluster:
        for counter,j in enumerate(get_NN(cluster[0],config,N)):
            if counter < 2: # two separate acceptance criteria needed for inequivalent directions
                if j[0]*cluster[0][0] > 0 and E0 > random.random():
                    cluster.append((config[j[1],j[2]],j[1],j[2]))
                    config[j[1],j[2]] *= -1 # trick: immediately flip spin once added to the cluster, removes need to check if spin is in cluster and invert at the end 
            else:
                if j[0]*cluster[0][0] > 0 and E1 > random.random():
                    cluster.append((config[j[1],j[2]],j[1],j[2])) # key to append to cluster first, then flip the spin (otherwise breaks the next loop spin-equivalence check)
                    config[j[1],j[2]] *= -1
        cluster.pop(0) # remove cluster item once considered, so not done twice      
        n += 1
        
def Wolff(config,beta,j0,j1,sav,rg):
    '''Simple implementation of Wolff cluster algorithm
    Parameters:
    config : conifguration to be updated
    beta   : 1/T, designed to be done in K^-1 units
    j0 : horizontal coupling constant
    j1 : vertical coupling constant
    sav : product of the spins on different sites
    rs : random number generator state
    '''
    N = len(config)
    E0 = 1 - np.exp(-2*beta*j0*sav) # pre-calculate expensive functions
    E1 = 1 - np.exp(-2*beta*j1*sav)
    r1 = rg.integers(0,N) # darts to start
    r2 = rg.integers(0,N)
    rand = rg.random(4*N**2)
    cluster = [[config[r1,r2], r1, r2]]
    config[r1,r2] *= -1 # must flip the very first one here, doesn't affect value in cluster
    n = 0
    c = 0
    while cluster:
        for counter,j in enumerate(get_NN(cluster[0],config,N)):
            if counter < 2: # two separate acceptance criteria needed for inequivalent directions
                if j[0]*cluster[0][0] > 0 and E0 > rand[n]:
                    cluster.append((config[j[1],j[2]],j[1],j[2]))
                    config[j[1],j[2]] *= -1 # trick: immediately flip spin once added to the cluster, removes need to check if spin is in cluster and invert at the end 
            else:
                if j[0]*cluster[0][0] > 0 and E1 > rand[n]:
                    cluster.append((config[j[1],j[2]],j[1],j[2])) # key to append to cluster first, then flip the spin (otherwise breaks the next loop spin-equivalence check)
                    config[j[1],j[2]] *= -1
            n += 1
        cluster.pop(0) # remove cluster item once considered, so not done twice      
        c += 1
    return(c)

def Td_plot(simulation):
    '''Quickly plots thermodynamic properties from a Simulation object
    Parameters:
    simulation : instance of simulation class
    Note that this must include T,E,M,C,X attributes, which we plot'''
    f = plt.figure(figsize=(18, 10)); # plot the calculated values    

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(simulation.T, simulation.E, s=10, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2 );
    plt.scatter(simulation.T, abs(simulation.M), s=10, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(simulation.T, simulation.C, s=10, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(simulation.T, simulation.X, s=10, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20); 
    plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
    return(f)


def Td_plot_read(T,E,M,C,X):
    '''Quickly plots thermodynamic properties from raw data read from file
    Parameters:
    T
    E
    M
    C
    X (all list objects)'''
    M = np.array(M) # so it can be passed to abs as an array
    f = plt.figure(figsize=(18, 10)); # plot the calculated values    

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T, E, s=10, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2 );
    plt.scatter(T, abs(M), s=10, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.scatter(T, C, s=10, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');

    sp =  f.add_subplot(2, 2, 4 );
    plt.scatter(T, X, s=10, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20); 
    plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
    return(f)

def aes_plot(T,steps,aes,cavs,data):
    '''Quickly plots aes lines from write aes output
    Parameters:
    T : list of Temperature points
    steps: list for the time axis (in units of simulation steps)
    aes: list of values of the autocorrelation function
    data : slice object choosing the data that you want to fit to
    n.b. ignore the start as non-exponential and the end as statistically
    errored'''
    x,y = post_fit(T,steps,aes,data)
    f = plt.figure(figsize=(6, 6))
    f.suptitle('Autocorrelation vs. time')
    sp =  f.add_subplot(1, 2, 1 );
    for i in range(len(T)):
        x[i] = x[i]*cavs[i] # scale the time coordinate
        plt.scatter(steps[0], aes[i], marker='x',s=2)
        plt.plot(x,np.e**y[i])
        plt.xlabel("Steps", fontsize=20);
        plt.ylabel("Autocorrelation function", fontsize=20);  plt.axis('tight');

    sp =  f.add_subplot(1, 2, 2 );
    for i in range(len(T)):
        plt.scatter(steps[0], aes[i], marker='x', s=2,\
                    label='{0:4.2f}'.format(T[i]))
        plt.plot(x,np.e**y[i])
    plt.xlabel("Steps", fontsize=16);
    plt.yscale('log')
    plt.legend(title='Temperature / K')

    return(f)

def write_aes(path,a):
    """write Autocorrelation stuff to file
    Parameters:
    path (str) : path that you want to write to
    a          : autocorrelation_average object"""
    print('writing...')
    f = open(path,'a+')
    f.writelines('Autocorrelation simulation %s\n' % str(a))
    f.writelines('Key: Time/steps Aes E E2 M M2 <C>\n')

    for i in range(a.sim[0].nt):
        f.writelines('\n')
        f.writelines('T: {0:7.3f}\n'.format(a.sim[0].T[i]))
        for j in range(len(range(*a.steps_test))):
            f.writelines('{0:>5d} {1:>7.3f} {2:> 12.3e} {3:> 12.3e}'
                         '{4: >8.3f} {5: >8.3f} {6:> 12.3e} {7:> 12.3e} {8:> 8.3f}\n'
                         .format(a.steps[j],a.Aes[i,j],a.avE[i,j],a.avE2[i,j],
                         a.avM[i,j],a.avM2[i,j],a.avC[i,j],a.avX[i,j],a.cavs[i]))
    f.writelines('END')
    f.close()
    print('done')

def write_sim(path,a):
    """write Simulation  stuff to file
    Parameters:
    path (str) : path that you want to write to
    a          : autocorrelation_average object"""
    f = open(path,'a+')
    f.writelines('Simulation %s\n' % str(a))
    p = vars(a)
    attrs = [p['nt'],p['N'],p['j0'],p['j1'],p['s1'],p['s2']]
    for i in range(len(attrs)):
        attrs[i] = str(attrs[i])
    f.writelines('A Monte-Carlo Simulation of the Ising model with the following\
                 parameters\n')
    f.writelines('Simulation inputs:\n'+'nt,N,j0,j1,s1,s2\n')
    f.writelines(', '.join(attrs)+'\n\n')
    f.writelines('Key: {0:>7s} {1:>7s} {2:>12s} {3:>7s} {4:>7s}\n'.format(\
                 'T', 'M', 'E',  'C', 'X'))
    for i in range(a.nt):
        f.writelines('     {0:7.3f} {1:7.3f} {2:12.4e} {3:7.3f} {4:7.3f}\n'.format(\
                     a.T[i],a.avM[i],a.avE[i],a.avC[i],a.avX[i]))
def sim_amalg(path):
    '''Collects simulation data from a folder full of individual text files from
    parallel simulations.
    Paramters: 
        path to file (str)
    Returns:
    data - 5xN array with T,M,E,C,X lists'''
    
    
    data = [[] for i in range(5)]
    for i in os.listdir(path):
        for j in range(5):
            data[j].extend(read_sim(path+i)[j])
    return(data)

def read_aes(path):
    """Reads in data from a write_aes file into a python object ready for
    plotting with Td plot
    Parameters: path of file (str)"""
    T = []
    steps = []
    aes = []
    cavs = []
    T_ct = 0
    with open(path) as f:
        lines = f.readlines()
    for i, val in enumerate(lines):
        if val == '\n':
            T.append(float(lines[i+1].split()[1]))
            steps.append([])
            aes.append([])
            j = i+2
            while j < len(lines) and lines[j] != '\n' and lines[j] != 'END':
                steps[T_ct].append(int(lines[j].split()[0]))
                aes[T_ct].append(float(lines[j].split()[1]))
                cavs[T_ct].append(float(lines[j].split()[-1]))
                j += 1
            T_ct += 1
            print(T_ct)
    return(T,steps,aes)

def read_sim(path):
    """Reads in data from a write_sim file into a python object ready for
    plotting with Td plot
    Parameters: path of file (str)"""
    T = []
    E = []
    M = []
    C = []
    X = []
    with open(path) as f:
        lines = f.readlines()
    for i, val in enumerate(lines):
        print(val)
        if 'Key:' in val:
            j = i+1
            while j < len(lines) and lines[j].split()[0] != '\n' :
                dat = lines[j].split()
                T.append(float(dat[0]))
                M.append(float(dat[1]))
                E.append(float(dat[2]))
                C.append(float(dat[3]))
                X.append(float(dat[4]))
                j += 1
            break
    return(T,E,M,C,X)
