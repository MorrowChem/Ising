import numpy as np
from core import *
from matplotlib import pyplot as plt
from scipy import optimize
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

class Simulation_Average():
    '''Averages a list of simulations performed at the same T points
    Parameters:
    sim : list of instances of Simulations'''
    def __init__(self, sim):
        self.nt = sim[0].nt
        self.sim = sim
        self.avE,self.avM,self.avC,self.avX =\
        np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)

        d = 1/len(sim)

        for i in range(len(sim)):
            self.avE += sim[i].E*d
            self.avM += abs(sim[i].M*d)
            self.avC += sim[i].C*d
            self.avX += sim[i].X*d
        print('average energy is %f\n\n' % self.avE)

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
        self.avE,self.avE2,self.avM,self.avM2,self.avC,self.avX =\
        np.zeros((self.nt,len(range(*self.steps_test)))),np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),np.zeros((self.nt,len(range(*self.steps_test)))),\
        np.zeros((self.nt,len(range(*self.steps_test)))),np.zeros((self.nt,len(range(*self.steps_test))))

        d = 1/len(sim)

        for k in range(len(sim)): # evaluate the Ae for each run for each T for each value of interSteps
            for j in range(sim[k].nt):
                for i in range(len(range(*self.steps_test))):
                    self.Aes[j,i] += Ae(sim[k].Ektav[j,i],sim[k].Eavk[j,i],sim[k].E2avk[j,i])*d 
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
    choose the data you want to fit too (often the large steps limit is dominated by statistical errors)

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
                                          np.log(np.ma.masked_less(Aes[i][data],0).filled(fill_value=1e-5)),\
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
                                          np.log(np.ma.masked_less(aes[i][data],0).filled(fill_value=1e-5)),\
                                          p0=[-0.015,-0.5])
    print(params)
    print(1/params.T[0])
    x = np.array(steps[0])
    y = []

    for i in range(len(params)):
        y.append(decay_function(x,params[i][0],params[i][1]))
    return(x,y)

def Wolff(config,beta,j0,j1,sav):
    '''Simple implementation of Wolff cluster algorithm
    Parameters:
    config : conifguration to be updated
    beta   : 1/T, designed to be done in K^-1 units
    j0 : horizontal coupling constant
    j1 : vertical coupling constant
    sav : product of the spins on different sites
    '''
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

def aes_plot(T,steps,aes,data):
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
        plt.scatter(steps[0], aes[i], marker='x')
        plt.plot(x,np.e**y[i])
        plt.xlabel("Steps", fontsize=20);
        plt.ylabel("Autocorrelation function", fontsize=20);  plt.axis('tight');

    sp =  f.add_subplot(1, 2, 2 );
    for i in range(len(T)):
        plt.scatter(steps[0], aes[i], marker='x',\
                    label='{0:4.2f}'.format(T[i]))
        plt.plot(x,np.e**y[i])
    plt.xlabel("Steps", fontsize=16);
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1,1),title='Temperature / K')

    return(f)

def write_aes(path,a):
    """write Autocorrelation stuff to file
    Parameters:
    path (str) : path that you want to write to
    a          : autocorrelation_average object"""
    print('writing...')
    f = open(path,'a+')
    f.writelines('Autocorrelation simulation %s\n' % str(a))
    f.writelines('Key: Time/steps Aes E E2 M M2 \n')

    for i in range(a.sim[0].nt):
        f.writelines('\n')
        f.writelines('T: {0:7.3f}\n'.format(a.sim[0].T[i]))
        for j in range(len(range(*a.steps_test))):
            f.writelines('{0:<5d} {1:<7.3f} {2:< 12.3e} {3:< 12.3e} {4:< 8.3f} {5:< 8.3f} {6:< 12.3e} {7:< 12.3e}\n'.format\
                         (a.steps[j],a.Aes[i,j],a.avE[i,j],a.avE2[i,j],a.avM[i,j],a.avM2[i,j],a.avC[i,j],a.avX[i,j]))
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
    keys = list(vars(a).keys())
    vals = list(vars(a).values())
    print(keys,vals)
    attrs = []
    for i in range(5):
        attrs.append("{0}: {1:3.3f}".format(keys[i],vals[i]))
    for i in range(6,15):
        attrs.append("{0}: {1:3.3f}".format(keys[i],vals[i]))
    f.writelines('Simulation inputs:\n'+', '.join(attrs)+'\n\n')
    f.writelines('Key: T       M       E       C       X\n')

def read_aes(path):
    """Reads in data from a write_aes file into a python object ready for
    plotting with Td plot
    Parameters: path of file (str)"""
    T = []
    steps = []
    aes = []
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
                j += 1
            T_ct += 1
            print(T_ct)
    return(T,steps,aes)
