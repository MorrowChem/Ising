from aux import *
from core import *
from Simulations import *
import multiprocessing

def met():
  autos_met_test = []
  for i in range(5):
      autos_met_test.append(AutoCorrelation(nt=10, N=24, eqSteps=10000, \
                                            ncalcs=2000, \
                                            steps_test=(50,1000,40), Ts =\
                                            2.2, Tf= 5, j0 = 1*8.61733e-5,\
                                            j1 = 1*8.61733e-5, s1 = 1, s2 =\
                                            1, h=0.))
      autos_met_test[i].run_Met()
  autos_met_av = Autocorrelation_Average(autos_met_test)
  write_aes('autos_met_test.txt',autos_met_av)

def wolff_20():
  autos_wolff_test = []
  for i in range(5):
      autos_wolff_test.append(AutoCorrelation(nt=10, N=24,\
                                              eqSteps=10000,ncalcs=2000,\
                                                      steps_test=(50,1000,40),\
                                              Ts = 2.2, Tf= 5, j0 =\
                                              1*8.61733e-5, j1 =\
                                              1*8.61733e-5, s1 = 1, s2 = 1,\
                                              h=0.))
      autos_wolff_test[i].run_Wolff(20)
  autos_wolff_av = Autocorrelation_Average(autos_wolff_test)
  write_aes('autos_wolff_test.txt',autos_wolff_av)

def wolff_60():
  autos_wolff60_test = []
  for i in range(5):
      autos_wolff60_test.append(AutoCorrelation(nt=10, N=24, eqSteps=10000,\
                                                        ncalcs=2000,\
                                                steps_test=(50,1000,40), Ts\
                                                = 2.2, Tf= 5, j0 =\
                                                1*8.61733e-5,\
                                                        j1 = 1*8.61733e-5,\
                                                s1 = 1, s2 = 1, h=0.))
      autos_wolff60_test[i].run_Wolff(60)
  autos_wolff60_av = Autocorrelation_Average(autos_wolff60_test)
  write_aes('autos_wolff60_av_test.txt',autos_wolff60_av)

def wolff_pure():
    autos_pure_test = []
    for i in range(20):
        autos_pure_test.append(AutoCorrelation(nt=10, N=24, eqSteps=10000,\
                                                       ncalcs=2000,\
                                               steps_test=(50,1000,40), Ts\
                                               = 2.2, Tf= 5, j0 =\
                                               1*8.61733e-5,\
                                                       j1 = 1*8.61733e-5,\
                                               s1 = 1, s2 = 1, h=0.))
        autos_pure_test[i].run_pure_Wolff()
    autos_pure_av = Autocorrelation_Average(autos_pure_test)
    write_aes('autos_pure_test.txt',autos_pure_av)

if __name__ == '__main__':
    p = []
    p.append(multiprocessing.Process(target=met))
    p.append(multiprocessing.Process(target=wolff_20))
    p.append(multiprocessing.Process(target=wolff_60))
    #p.append(multiprocessing.Process(target=wolff_pure))
    for i in p:
        i.start()
