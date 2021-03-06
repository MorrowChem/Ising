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
    h (optional) : magnetic field, awaiting implementation
    '''


    def __init__(self, nt, N, eqSteps, ncalcs, interSteps, Ts, Tf, j0, j1, s1,\
                 s2, h=0.0):
        self.nt = nt
        self.N = N
        self.eqSteps = eqSteps
        self.ncalcs = ncalcs
        self.interSteps = interSteps
        self.T = np.linspace(Ts, Tf, nt)
        self.h = h*5.78838e-5/8.617e-5 # BM in eV/T scaled by Boltzmann factor 
        self.k = 8.61733e-5
        self.n1 = 1.0/(self.ncalcs*self.N*self.N)
        self.n2 = 1.0/(self.ncalcs*self.ncalcs*self.N*self.N)
        self.j0 = j0/self.k
        self.j1 = j1/self.k
        self.s1 = s1
        self.s2 = s2
        self.sav = s1*s2
        self.rg = np.random.Generator(np.random.PCG64()) # set up the RNG
        self.cavs = np.ones(self.nt) # average cluster sizes


    def run_Met(self,config=False):
        '''Standard Metropolis algorithm '''
        self.algo = 'Metropolis-Hastings'
        self.E, self.M, self.C, self.X =\
            np.zeros(self.nt), np.zeros(self.nt),\
            np.zeros(self.nt), np.zeros(self.nt)

        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = initialstate(self.N)  # Initialise random config
            iT=1.0/self.T[tt]; iT2=iT*iT;  # Calculate the 1/T values                                                
            mcmoves(self.config, iT, self.eqSteps, self.N,  # Equilibrate
                    self.j0, self.j1, self.sav, self.h)

            if config:
                print('Config after equilibration:\n')
                quick_config(self.config,self.N)

            for i in range(self.ncalcs):
                mcmoves(self.config, iT, self.interSteps, self.N,\
                        self.j0, self.j1, self.sav, self.h) # intervening steps       
                Ene = calcEnergy(self.config, self.j0, self.j1,\
                                 self.sav, self.h)
                Mag = calcMag(self.config, self.s1, self.s2)

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag
                E2 = E2 + Ene*Ene

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT

            print("{0:<12.3f}{1:< 12.3f}{2:< 12.4e}{3:< 12.4e}{4:< 12.4e}".\
                  format(self.T[tt],self.M[tt],self.E[tt],self.C[tt],\
                         self.X[tt]),flush=True)
            if config:
                print('Final config:\n')
                quick_config(self.config,self.N)


    def run_Wolff(self,w_rat,config=False):
        '''Wolff cluster algorithm every w_rat Metropolis steps
        Parameters:
        w_rat : the ratio of Metropolis steps to Wolff steps
        '''


        self.algo = 'Hybrid'
        w_int_eq = self.eqSteps//w_rat # ratios for Met vs Wolff steps
        w_int = self.interSteps//w_rat
        print(w_int_eq,w_int)
        # Initialise the data vectors
        self.E, self.M, self.C, self.X =\
            np.zeros(self.nt), np.zeros(self.nt),\
            np.zeros(self.nt), np.zeros(self.nt)

        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = initialstate(self.N) # Initialise random config
            iT=1.0/self.T[tt]; iT2=iT*iT;

            for a in range(w_int_eq): # equilibration
                Wolff(self.config,iT, self.j0, self.j1, self.sav, self.rg)
                mcmoves(self.config, iT, w_rat, self.N, self.j0,\
                        self.j1, self.sav, self.h)
            if config:
                print('config after equilibration: %i' % tt)
                quick_config(self.config,self.N)

            for j in range(self.ncalcs): # calculate the properties
                for a in range(w_int):
                    Wolff(self.config,iT, self.j0, self.j1, self.sav, self.rg)
                    mcmoves(self.config, iT, w_rat, self.N, self.j0, self.j1,\
                            self.sav, self.h)
                Ene = calcEnergy(self.config, self.j0, self.j1,\
                                 self.sav, self.h)
                Mag = calcMag(self.config, self.s1, self.s2)

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag
                E2 = E2 + Ene*Ene

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT

            print("{0:<12.3f}{1:< 12.3f}{2:< 12.4e}{3:< 12.4e}{4:< 12.4e}".\
                  format(self.T[tt],self.M[tt],self.E[tt],self.C[tt],\
                         self.X[tt]),flush=True)

            if config:
                print('Final config:\n')
                quick_config(self.config,self.N)


    def run_pure_Wolff(self,config=False):
        '''Standard Wolff algorithm '''
        self.algo = 'Pure Wolff'
        self.E, self.M, self.C, self.X =\
            np.zeros(self.nt), np.zeros(self.nt),\
            np.zeros(self.nt), np.zeros(self.nt)

        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            self.config = orderedstate(self.N)  # Init config - ordered more
            #                                     efficient for pure Wolff
            iT=1.0/self.T[tt]; iT2=iT*iT;

            for i in range(self.eqSteps):  # Equilibrate
                Wolff(self.config, iT, self.j0, self.j1, self.sav, self.rg)

            if config:
                print('Config after equilibration:\n')
                quick_config(self.config,self.N)

            for i in range(self.ncalcs):
                for j in range(self.interSteps):  # Perform intervening steps 
                    Wolff(self.config, iT, self.j0, self.j1, self.sav, self.rg)
                Ene = calcEnergy(self.config, self.j0, self.j1,\
                                 self.sav, self.h)
                Mag = calcMag(self.config, self.s1, self.s2)

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag
                E2 = E2 + Ene*Ene

            self.E[tt] = self.n1*E1
            self.M[tt] = self.n1*M1
            self.C[tt] = (self.n1*E2 - self.n2*E1*E1)*iT2
            self.X[tt] = (self.n1*M2 - self.n2*M1*M1)*iT

            print("{0:<12.3f}{1:< 12.3f}{2:< 12.4e}{3:< 12.4e}{4:< 12.4e}".\
                  format(self.T[tt],self.M[tt],self.E[tt],self.C[tt],\
                         self.X[tt]),flush=True)

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
    h=0 : magnetic field (note gives wrong results for Wolff algorithm)

    Attributes:
    During a Markov chain on a particular config:
        Es   : E
        E2s  : E**2
        Ekts : E(step k) * E(step k+step)
        Ms
        M2s
    Average over the Markov chain above, these are the values you actually use
        Eav
        E2avk
        Ektav : key correlation value
        Mavk
        M2avk
        Cf : Heat capacity
        Xf : Magnetic susceptibility
    """


    def __init__(self, nt, N, eqSteps, ncalcs, steps_test, Ts, Tf,\
                 j0, j1, s1, s2, h=0):
        self.nt = nt
        self.N = N
        self.k = 8.61733e-5
        self.eqSteps = eqSteps
        self.ncalcs = ncalcs
        self.steps_test = steps_test
        self.T = np.linspace(Ts, Tf, nt)
        self.h = h*5.78838e-5/8.617e-5# BM in eV/T scale by Boltzmann factor
        self.n1 = 1.0/(self.ncalcs*self.N*self.N)
        self.n2 = 1.0/(self.ncalcs*self.ncalcs*self.N*self.N)
        self.kb = 8.617e-5
        self.q = len(range(*self.steps_test))
        self.j0 = j0/self.k
        self.j1 = j1/self.k
        self.s1 = s1
        self.s2 = s2
        self.sav = s1*s2
        self.rg = np.random.Generator(np.random.PCG64()) # set up the RNG


    def run_Met(self, config=False):
        ''' Main simulation code employing pure Metropolis-Hastings algorithm
        Parameters:
        config (bool) : True prints configurations after each equilibrium step
                        and at the end of that run
        '''


        steps_list = list(range(*self.steps_test))
        # Initialise the lists we're going to use to store our data
        self.Es, self.E2s, self.Ms, self.M2s, self.Ekts = \
            np.zeros((self.nt, self.q, self.ncalcs)),\
            np.zeros((self.nt, self.q, self.ncalcs)),\
            np.zeros((self.nt, self.q, self.ncalcs)),\
            np.zeros((self.nt, self.q, self.ncalcs)),\
            np.zeros((self.nt, self.q, self.ncalcs-1)) #  1 fewer of Ek*Ek+t

        self.Eavk, self.E2avk, self.Ektav, self.Mavk,\
            self.M2avk, self.Cf, self.Xf =\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q)),\
            np.zeros((self.nt, self.q))

        self.cavs = np.ones(self.nt) # cluster sizes, 1 by definition for met
        print("Steps       T           Eav^2       Ek*Ek+t     E^2av       Mav"\
              , flush=True)
        for i in range(self.nt):
            self.config = initialstate(self.N) # new random config for each T
            iT = 1.0/self.T[i]; iT2 = iT**2
            mcmoves(self.config, iT, self.eqSteps, self.N,  # equilibration
                    self.j0, self.j1, self.sav, self.h)

            if config:
                print('config after equilibration: %i' % i)
                quick_config(self.config,self.N)

            for k in range(self.q):  # do *steps_test* iterations of prop calcs
                                     # at q multiples of intervening steps
                steps = steps_list[k] # steps between prop calculations

                self.Es[i,k,0]  = calcEnergy(self.config, self.j0, self.j1,\
                       self.sav,self.h) # calculate initial values after eqm
                self.E2s[i,k,0] = self.Es[i,k,0] * self.Es[i,k,0]
                self.Ms[i,k,0]  = calcMag(self.config, self.s1, self.s2)
                self.M2s[i,k,0] = self.Ms[i,k,0] * self.Ms[i,k,0]

                for j in range(1,self.ncalcs): # calculate the properties
                    ind_mcmoves(self.config, iT, steps, self.N,  # intervening
                                self.j0, self.j1, self.sav, self.h)
                    Ene = calcEnergy(self.config, self.j0, self.j1,\
                                     self.sav, self.h)
                    Mag = calcMag(self.config, self.s1, self.s2)
                    self.Es[i,k,j] = Ene
                    self.E2s[i,k,j] = Ene*Ene
                    self.Ms[i,k,j]   = Mag
                    self.M2s[i,k,0]  = Mag*Mag
                    # key autocorrelation value:
                    self.Ekts[i,k,j-1] = self.Es[i,k,j-1]*self.Es[i,k,j]

                # Average each of the Td vars over their walk
                self.Eavk[i,k]  = sum(self.Es[i,k])  / len(self.Es[i,k])
                self.E2avk[i,k] = sum(self.E2s[i,k]) / len(self.E2s[i,k])
                self.Ektav[i,k] = sum(self.Ekts[i,k])/ len(self.Ekts[i,k])
                self.Mavk[i,k]  = sum(self.Ms[i,k])  / len(self.Ms[i,k])
                self.M2avk[i,k] = sum(self.M2s[i,k]) / len(self.M2s[i,k])

                self.Cf[i,k] = (self.n1*self.E2avk[i,k] - \
                       self.n2*self.Eavk[i,k]*self.Eavk[i,k])*iT2

                self.Xf[i,k] = (self.n1*self.M2avk[i,k] - \
                       self.n2*self.Mavk[i,k]*self.Mavk[i,k])*iT

                print("{0:^12d}{1:<12.3f}{2:<12.4e}{3:<12.4e}"
                      "{4:<12.4e}{5:< 12.4e}".\
                      format(steps, self.T[i], (self.Eavk[i,k]*self.kb)**2,\
                             self.Ektav[i,k]*self.kb**2,\
                             self.E2avk[i,k]*self.kb**2,\
                             self.Mavk[i,k], flush=True))
            if config:
                print('Final config:')
                quick_config(self.config,self.N)


    def run_Wolff(self,w_rat,config=False):
        '''Main simulation code employing Wolff cluster algorithm
        every w_rat Metropolis steps
        Parameters:
        w_rat : the ratio of Metropolis steps to Wolff steps
        '''


        w_int_eq = self.eqSteps//w_rat
        steps_list = list(range(*self.steps_test))
        # Initialise the lists we're going to use to store data
        self.Es,self.E2s,self.Ms,self.M2s,self.Ekts = \
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs-1)) #  1 fewer of Ek*Ek+t

        self.Eavk,self.E2avk,self.Ektav,self.Mavk,self.M2avk,self.Cf,self.Xf =\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q))

        self.cavs = np.zeros(self.nt) # average cluster sizes
        print("Steps       T           Eav^2       Ek*Ek+t     E^2av       Mav"\
              , flush=True)

        for i in range(self.nt):
            self.config = initialstate(self.N) # new config for each T
            iT = 1.0/self.T[i]
            iT2 = iT**2
            cs = [] # list of cluster sizes
            mcmoves(self.config, iT, self.eqSteps, self.N, self.j0,
                                   self.j1, self.sav, self.h)
            for a in range(w_int_eq): # equilibration
                Wolff(self.config,iT, self.j0, self.j1, self.sav, self.rg)
                mcmoves(self.config, iT, w_rat, self.N, self.j0,
                                   self.j1, self.sav, self.h)
            if config:
                print('config after equilibration: %i' % i)
                quick_config(self.config, self.N)

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
                        cs.append(Wolff(self.config,iT, self.j0, self.j1, self.sav, self.rg))
                        cs.extend([1 for i in range(min(steps,w_rat))])
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

                print("{0:^12d}{1:<12.3f}{2:<12.4e}{3:<12.4e}"
                      "{4:<12.4e}{5:< 12.4e}".\
                      format(steps, self.T[i], (self.Eavk[i,k]*self.kb)**2,\
                             self.Ektav[i,k]*self.kb**2,\
                             self.E2avk[i,k]*self.kb**2,\
                             self.Mavk[i,k], flush=True))

            self.cavs[i] = sum(cs)/len(cs) # calculate average cluster size
            if config:
                print('Final config:')
                quick_config(self.config,self.N)

    def run_pure_Wolff(self,config=False):
        ''' Simulation code employing pure Wolff algorithm
        Parameters:
        config (bool) : True prints configurations after each equilibrium step and at the end of that run
        '''
        steps_list = list(range(*self.steps_test)) # the values for the intervening steps
        # Initialise the lists we're going to use to store our data
        self.Es,self.E2s,self.Ms,self.M2s,self.Ekts = \
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs)),\
        np.zeros((self.nt,self.q,self.ncalcs-1))

        self.Eavk,self.E2avk,self.Ektav,self.Mavk,self.M2avk,self.Cf,self.Xf =\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q)),np.zeros((self.nt,self.q)),\
        np.zeros((self.nt,self.q))

        self.cavs = np.zeros(self.nt) # average cluster sizes
        print("Steps       T           Eav^2       Ek*Ek+t     E^2av       Mav"\
              , flush=True)
        for i in range(self.nt):
            self.config = orderedstate(self.N) # more efficient to start with ordered configuration
            iT = 1.0/self.T[i]
            iT2 = iT**2
            cs = [] # list of cluster sizes
            for j in range(self.eqSteps):
                Wolff(self.config, iT, self.j0, self.j1, self.sav, self.rg) # equilibration
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

                for j in range(self.ncalcs): # heart of the run: calculate the properties
                    for l in range(steps):
                        cs.append(Wolff(self.config, iT, self.j0, self.j1, self.sav, self.rg)) # perform intervening MC steps
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
                print("{0:^12d}{1:<12.3f}{2:<12.4e}{3:<12.4e}{4:<12.4e}{5:< 12.4e}".\
                      format(steps,self.T[i],(self.Eavk[i,k]*self.kb)**2,self.Ektav[i,k]*self.kb**2,\
                      self.E2avk[i,k]*self.kb**2,self.Mavk[i,k],flush=True))
                if config:
                    print('Final config:')
                    quick_config(self.config,self.N)

            self.cavs[i] = sum(cs)/len(cs) # calculate average cluster size
            if config:
                print('Final config:')
                quick_config(self.config,self.N)
