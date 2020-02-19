import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from core import *
from aux import *
import random


class Simulation():
    '''Standard MC simulation object, performs over a range of temperatures
    Parameters:
    nt : number of temperature points
    N  : system size (length of square box)
    eqSteps : number of equilibration steps to perform
    ncalcs  : number of property calculations to perform
    interSteps : number of intervening Metropolis steps between property calcs
    Ts : intial temperature
    Tf : final temperature
    h (optional) : magnetic field, not implemented yet'''
    def __init__(self, nt, N, eqSteps, ncalcs, interSteps, Ts, Tf, j0, j1, s1, s2, h=0.0):
        self.nt = nt
        self.N = N
        self.eqSteps = eqSteps
        self.ncalcs = ncalcs
        self.interSteps = interSteps
        self.T = np.linspace(Ts, Tf, nt)
        self.h = h*5.78838e-5/8.617e-5 # BM in eV/T, scale by bolztmann factor as well 
        self.k = 8.61733e-5
        self.n1 = 1.0/(self.ncalcs*self.N*self.N)
        self.n2 = 1.0/(self.ncalcs*self.ncalcs*self.N*self.N)
        self.j0 = j0/self.k
        self.j1 = j1/self.k
        self.s1 = s1
        self.s2 = s2
        self.sav = s1 * s2
        self.rg = np.random.Generator(np.random.PCG64()) # set up the RNG

    def run_Met(self,config=False):
        '''Standard Metropolis algorithm '''
        self.algo = 'Metropolis-Hastings'
        self.E,self.M,self.C,self.X = np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)
    
        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = initialstate(self.N) # Initialise a random starting config
            iT=1.0/self.T[tt]; iT2=iT*iT;  # Calculate the 1/T values                                                
            mcmoves(self.config, iT, self.eqSteps, self.N, self.j0, self.j1, self.sav, self.h) # Equilibrate
            
            if config:
                print('Config after equilibration:\n')
                quick_config(self.config,self.N)
            
            for i in range(self.ncalcs):
                mcmoves(self.config, iT, self.interSteps, self.N, self.j0, self.j1,\
                        self.sav, self.h) # Perform intervening MC steps       
                Ene = calcEnergy(self.config, self.j0, self.j1, self.sav, self.h)     # calculate the energy
                Mag = calcMag(self.config, self.s1, self.s2)        # calculate the magnetisation

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag 
                E2 = E2 + Ene*Ene 

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT
            
            print("%i T: %f   M: %f   E: %f   C: %f   X: %f met" % \
                 (tt, self.T[tt],self.M[tt],self.E[tt],self.C[tt],self.X[tt]),\
                  flush=True)
            if config:
                print('Final config:\n')
                quick_config(self.config,self.N)       
    
    def run_Wolff(self,w_rat,config=False):
        '''Wolff cluster algorithm every w_rat Metropolis steps
        Parameters:
        w_rat : the ratio of Metropolis steps to Wolff steps'''
        self.algo = 'Hybrid'
        w_int_eq = self.eqSteps//w_rat
        w_int = self.interSteps//w_rat
        print(w_int_eq,w_int)
        # Initialise the data vectors
        self.E,self.M,self.C,self.X = np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)
    
        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = initialstate(self.N) # Initialise a random starting config
            iT=1.0/self.T[tt]; iT2=iT*iT;  # Calculate the 1/T values                                                
            
            for a in range(w_int_eq): # equilibration
                Wolff(self.config,iT, self.j0, self.j1, self.sav)
                mcmoves(self.config, iT, w_rat, self.N, self.j0, self.j1, self.sav, self.h) 
            if config:
                print('config after equilibration: %i' % tt)
                quick_config(self.config,self.N)   
            
            for j in range(self.ncalcs): # heart of the run: calculate the properties
                for a in range(w_int):
                    Wolff(self.config,iT, self.j0, self.j1, self.sav)
                    mcmoves(self.config, iT, w_rat, self.N, self.j0, self.j1, self.sav, self.h)
                Ene = calcEnergy(self.config, self.j0, self.j1, self.sav, self.h)     # calculate the energy
                Mag = calcMag(self.config, self.s1, self.s2)        # calculate the magnetisation

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag 
                E2 = E2 + Ene*Ene 

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT
            
            print("%i T: %f   M: %f   E: %f   C: %f   X: %f hybrid" % \
                 (tt, self.T[tt],self.M[tt],self.E[tt],self.C[tt],self.X[tt]),\
                  flush = True)
            
            if config:
                print('Final config:\n')
                quick_config(self.config,self.N)     

    def run_pure_Wolff(self,config=False):
        '''Standard Wolff algorithm '''
        self.algo = 'Pure Wolff'
        self.E,self.M,self.C,self.X = np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt),np.zeros(self.nt)
    
        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = initialstate(self.N) # Initialise a random starting config
            iT=1.0/self.T[tt]; iT2=iT*iT;  # Calculate the 1/T values                                                
            
            for i in range(self.eqSteps):
                Wolff(self.config, iT, self.j0, self.j1, self.sav) # Equilibrate
            
            if config:
                print('Config after equilibration:\n')
                quick_config(self.config,self.N)
            
            for i in range(self.ncalcs):
                for j in range(self.interSteps):
                    Wolff(self.config, iT, self.j0, self.j1, self.sav) # Perform intervening MC steps       
                Ene = calcEnergy(self.config, self.j0, self.j1, self.sav, self.h)     # calculate the energy
                Mag = calcMag(self.config, self.s1, self.s2)        # calculate the magnetisation

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag 
                E2 = E2 + Ene*Ene 

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT
            
            print("%i T: %f   M: %f   E: %f   C: %f   X: %f Wolff"\
             % (tt, self.T[tt],self.M[tt],self.E[tt],self.C[tt],self.X[tt]),\
                flush=True)
            if config:
                print('Final config:\n')
                quick_config(self.config,self.N)

class AutoCorrelation():
    """Runs MC simulations designed to explore the autocorrelation function
    Parameters:
    nt : number of different temperature points to investigate
    N  : system size
    eqSteps : number of steps done in equilibration phase
    ncalcs  : number of evaluations of the Energy, Magnetization etc.
    steps_test (tuple) : (starting value, final value, increment)
    Ts : starting temperature
    Tf : finishing temperature
    j0 : horizontal coupling constant
    j1 : vertical coupling constant
    s1 : spin1
    s2 : spin2
    h=0 : magnetic field strenght (note gives wrong results for Wolff algorithm) 
    
    Attributes:
    During a Markov chain on a particular config:
        Es   : E
        E2s  : E**2
        Ekts : E(step k) * E(step k+step)
        Ms
        M2s
    Average over the course of the Markov chain above, these are the values you actually use
        Eavk  
        E2avk
        Ektav : key correlation value
        Mavk
        M2avk
        Cf : Heat capacity
        Xf : Magnetic susceptibility
    """
    def __init__(self, nt, N, eqSteps, ncalcs, steps_test, Ts, Tf, j0, j1, s1, s2, h=0): 
        self.nt = nt
        self.N = N
        self.k = 8.61733e-5
        self.eqSteps = eqSteps
        self.ncalcs = ncalcs
        self.steps_test = steps_test
        self.T = np.linspace(Ts, Tf, nt)
        self.h = h*5.78838e-5/8.617e-5# BM in eV/T, scale by boltzmann factor as well 
        self.n1 = 1.0/(self.ncalcs*self.N*self.N)
        self.n2 = 1.0/(self.ncalcs*self.ncalcs*self.N*self.N)
        self.kb = 8.617e-5
        self.q = len(range(self.steps_test[0],self.steps_test[1],self.steps_test[2]))
        self.j0 = j0/self.k
        self.j1 = j1/self.k
        self.s1 = s1
        self.s2 = s2
        self.sav = s1*s2
        self.rg = np.random.Generator(np.random.PCG64()) # set up the RNG
    
    def run_Met(self,config=False):
        ''' Main simulation code employing pure metropolis algorithm
        Parameters:
        config (bool) : True prints configurations after each equilibrium step and at the end of that run
        '''
        steps_list = list(range(self.steps_test[0],self.steps_test[1],self.steps_test[2])) # the values for the intervening steps
        # Initialise the lists we're going to use to store our data
        self.Es,self.E2s,self.Ms,self.M2s,self.Ekts = np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs))

        self.Eavk,self.E2avk,self.Ektav,self.Mavk,self.M2avk,self.Cf,self.Xf =\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q))
        for i in range(self.nt):
            self.config = initialstate(self.N) # new random configuration for each T
            iT = 1.0/self.T[i]
            iT2 = iT**2
            mcmoves(self.config, iT, self.eqSteps, self.N, self.j0, self.j1, self.sav, self.h) # equilibration
            if config:
                print('config after equilibration: %i' % i)
                quick_config(self.config,self.N)   

            for k in range(self.q): # perform *steps_test* iterations of mc calcs, at different
                                             # multiples of intervening steps
                steps = steps_list[k] # number of steps between point calculations

                self.Es[i,k,0]   = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h) # calculate the initial values after eqm
                self.E2s[i,k,0]  = self.Es[i,k,0]*self.Es[i,k,0]
                self.Ms[i,k,0]   = calcMag(self.config,self.s1,self.s2)
                self.M2s[i,k,0]  = self.Ms[i,k,0]*self.Ms[i,k,0]

                for j in range(1,self.ncalcs): # heart of the run: calculate the properties
                    ind_mcmoves(self.config, iT, steps, self.N, self.j0, self.j1, self.sav, self.h) # perform intervening MC steps
                    Ene = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h)
                    Mag = calcMag(self.config,self.s1,self.s2)
                    self.Es[i,k,j] = Ene      # add values and their squares to appropriate lists 
                    self.E2s[i,k,j] = Ene*Ene
                    self.Ms[i,k,j]   = Mag
                    self.M2s[i,k,0]  = Mag*Mag
                    self.Ekts[i,k,j-1] = self.Es[i,k,j-1]*self.Es[i,k,j] # key autocorrelation indicator

                self.Eavk[i,k]  = sum(self.Es[i,k])  / len(self.Es[i,k]) # Average each of the Td variables over their walk through config space
                self.E2avk[i,k] = sum(self.E2s[i,k]) / len(self.E2s[i,k])
                self.Ektav[i,k] = sum(self.Ekts[i,k])/ len(self.Ekts[i,k])
                self.Mavk[i,k]  = sum(self.Ms[i,k])  / len(self.Ms[i,k])
                self.M2avk[i,k] = sum(self.M2s[i,k]) / len(self.M2s[i,k])

                self.Cf[i,k] = (self.n1*self.E2avk[i,k] - self.n2*self.Eavk[i,k]*self.Eavk[i,k])*iT2
                self.Xf[i,k] = (self.n1*self.M2avk[i,k] - self.n2*self.Mavk[i,k]*self.Mavk[i,k])*iT
                print(steps,self.T[i],self.Eavk[i,k]*self.kb,flush=True)
            if config:
                print('Final config:')
                quick_config(self.config,self.N)
    
    def run_Wolff(self,w_rat,config=False):
        '''Main simulation code employing Wolff cluster algorithm every w_rat Metropolis steps
        Parameters:
        w_rat : the ratio of Metropolis steps to Wolff steps'''
        w_int_eq = self.eqSteps//w_rat
        steps_list = list(range(*self.steps_test)) # the values for the intervening steps
        # Initialise the lists we're going to use to store our data
        self.Es,self.E2s,self.Ms,self.M2s,self.Ekts = np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs))

        self.Eavk,self.E2avk,self.Ektav,self.Mavk,self.M2avk,self.Cf,self.Xf =\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q))
        
        for i in range(self.nt):
            self.config = initialstate(self.N) # new config for each T
            iT = 1.0/self.T[i]
            iT2 = iT**2
            
            for a in range(w_int_eq): # equilibration
                Wolff(self.config,iT, self.j0, self.j1, self.sav)
                mcmoves(self.config, iT, w_rat, self.N, self.j0, self.j1, self.sav, self.h) 
            if config:
                print('config after equilibration: %i' % i)
                quick_config(self.config,self.N)   

            for k in range(self.q): # perform *steps_test* iterations of mc calcs, at different
                                             # multiples of intervening steps
                steps = steps_list[k] # number of steps between point calculations
                w_int = steps//w_rat

                self.Es[i,k,0]   = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h) # calculate the initial values after eqm
                self.E2s[i,k,0]  = self.Es[i,k,0]*self.Es[i,k,0]
                self.Ms[i,k,0]   = calcMag(self.config,self.s1,self.s2)
                self.M2s[i,k,0]  = self.Ms[i,k,0]*self.Ms[i,k,0]

                for j in range(1,self.ncalcs): # heart of the run: calculate the properties
                    for a in range(w_int):
                        Wolff(self.config,iT, self.j0, self.j1, self.sav)
                        ind_mcmoves(self.config, iT, min(steps,w_rat), self.N, self.j0, self.j1, self.sav, self.h) # perform intervening MC steps
                    Ene = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h) # Calculate the energy
                    Mag = calcMag(self.config,self.s1,self.s2) # Calculate the magnetisation
                    self.Es[i,k,j] = Ene      # add these values and their squares to the appropriate lists 
                    self.E2s[i,k,j] = Ene*Ene
                    self.Ms[i,k,j]   = Mag
                    self.M2s[i,k,0]  = Mag*Mag
                    self.Ekts[i,k,j-1] = self.Es[i,k,j-1]*self.Es[i,k,j] # this is the key autocorrelation indicator

                self.Eavk[i,k]  = sum(self.Es[i,k])  / len(self.Es[i,k]) # Average each of the Td variables over their walk through phase space
                self.E2avk[i,k] = sum(self.E2s[i,k]) / len(self.E2s[i,k])
                self.Ektav[i,k] = sum(self.Ekts[i,k])/ len(self.Ekts[i,k])
                self.Mavk[i,k]  = sum(self.Ms[i,k])  / len(self.Ms[i,k])
                self.M2avk[i,k] = sum(self.M2s[i,k]) / len(self.M2s[i,k])

                self.Cf[i,k] = (self.n1*self.E2avk[i,k] - self.n2*self.Eavk[i,k]*self.Eavk[i,k])*iT2
                self.Xf[i,k] = (self.n1*self.M2avk[i,k] - self.n2*self.Mavk[i,k]*self.Mavk[i,k])*iT
                print(steps,self.T[i],self.Eavk[i,k]*self.kb,flush=True)
            if config:
                print('Final config:')
                quick_config(self.config,self.N)

    def run_pure_Wolff(self,config=False):
        ''' Simulation code employing pure Wolff algorithm
        Parameters:
        config (bool) : True prints configurations after each equilibrium step and at the end of that run
        '''
        steps_list = list(range(self.steps_test[0],self.steps_test[1],self.steps_test[2])) # the values for the intervening steps
        # Initialise the lists we're going to use to store our data
        self.Es,self.E2s,self.Ms,self.M2s,self.Ekts = np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs)),np.zeros((self.nt,self.q,self.ncalcs)),\
                                                      np.zeros((self.nt,self.q,self.ncalcs))

        self.Eavk,self.E2avk,self.Ektav,self.Mavk,self.M2avk,self.Cf,self.Xf =\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
                                                        np.zeros((self.nt,self.q))
        for i in range(self.nt):
            self.config = initialstate(self.N) # sets new random configuration for each temperature evaluation
            iT = 1.0/self.T[i]
            iT2 = iT**2
            for j in range(self.eqSteps):
                Wolff(self.config, iT, self.j0, self.j1, self.sav) # equilibration
            if config:
                print('config after equilibration: %i' % i)
                quick_config(self.config,self.N)   

            for k in range(self.q): # perform *steps_test* iterations of mc calcs, at different
                                             # multiples of intervening steps
                steps = steps_list[k] # number of steps between point calculations

                self.Es[i,k,0]   = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h) # calculate the initial values after eqm
                self.E2s[i,k,0]  = self.Es[i,k,0]*self.Es[i,k,0]
                self.Ms[i,k,0]   = calcMag(self.config,self.s1,self.s2)
                self.M2s[i,k,0]  = self.Ms[i,k,0]*self.Ms[i,k,0]

                for j in range(1,self.ncalcs): # heart of the run: calculate the properties
                    for l in range(steps):
                        Wolff(self.config, iT, self.j0, self.j1, self.sav) # perform intervening MC steps
                    Ene = calcEnergy(self.config,self.j0,self.j1,self.sav,self.h) # Calculate the energy
                    Mag = calcMag(self.config,self.s1,self.s2) # Calculate the magnetisation
                    self.Es[i,k,j] = Ene      # add these values and their squares to the appropriate lists 
                    self.E2s[i,k,j] = Ene*Ene
                    self.Ms[i,k,j]   = Mag
                    self.M2s[i,k,0]  = Mag*Mag
                    self.Ekts[i,k,j-1] = self.Es[i,k,j-1]*self.Es[i,k,j] # this is the key autocorrelation indicator

                self.Eavk[i,k]  = sum(self.Es[i,k])  / len(self.Es[i,k]) # Average each of the Td variables over their walk through phase space
                self.E2avk[i,k] = sum(self.E2s[i,k]) / len(self.E2s[i,k])
                self.Ektav[i,k] = sum(self.Ekts[i,k])/ len(self.Ekts[i,k])
                self.Mavk[i,k]  = sum(self.Ms[i,k])  / len(self.Ms[i,k])
                self.M2avk[i,k] = sum(self.M2s[i,k]) / len(self.M2s[i,k])

                self.Cf[i,k] = (self.n1*self.E2avk[i,k] - self.n2*self.Eavk[i,k]*self.Eavk[i,k])*iT2
                self.Xf[i,k] = (self.n1*self.M2avk[i,k] - self.n2*self.Mavk[i,k]*self.Mavk[i,k])*iT
                print(steps,self.T[i],self.Eavk[i,k]*self.kb,flush=True)
            if config:
                print('Final config:')
                quick_config(self.config,self.N)
