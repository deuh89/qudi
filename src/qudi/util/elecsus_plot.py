# -*- coding: utf-8 -*-

from elecsus.elecsus_methods import calculate
import elecsus.elecsus_methods as EM
import numpy as np


class Elecsus(object):

    def plotTheory(self, p_dict, detuning_range = np.linspace(-10,10,1000)*1e3, E_in = np.array([1,0,0]), outputs=['S0']):
        return calculate(detuning_range,E_in,p_dict,outputs)
    
    def defineEfield(self):
        '''
        need to express E_in as Ex, Ey and phase difference for fitting
        '''
        E_in = np.sqrt(1./2) * np.array([1.,1.,0])
        E_in_angle = [E_in[0].real,[abs(E_in[1]),np.angle(E_in[1])]]

        return E_in, E_in_angle
    
    def binData(self,x, y, blength):
        '''
        Takes 2 arrays x and y and bins them into groups of blength
        '''
        if blength % 2 == 0: 
            blength -= 1
        nobins = int(len(x)/blength)
        xmid = (blength-1)/2
        xbinmax = nobins*blength - xmid
        a=0
        binned = np.zeros((nobins,3))
        xout,yout,yerrout = np.array([]), np.array([]), np.array([])
        for i in range(int(xmid),int(xbinmax),int(blength)):
            xmin = i-int(xmid)
            xmax = i+int(xmid)
            xout = np.append(xout,sum(x[xmin:xmax+1])/blength)
            yout = np.append(yout,sum(y[xmin:xmax+1])/blength)
            yerrout = np.append(yerrout,np.std(y[xmin:xmax+1]))

        return xout,yout,yerrout
    
    def fitSpectrum(self,x, y, p_dict, p_dict_bools, p_dict_bounds, specType, fitAlgorithm):
        ''' 
        takes the spectrum and fits it using Elecsus. Then it calculates the residuals
        So far, ML works fastest and dnly RR and ML are implemented
        '''
        E_in, E_in_angle = self.defineEfield()

        data = [x,y]
        if fitAlgorithm == 'ML':
            print('Fit started, ML')
            best_params, RMS, result = EM.fit_data(data, p_dict, p_dict_bools, E_in_angle, data_type=specType, fit_algorithm=fitAlgorithm)
        if fitAlgorithm == 'RR':
            print('Fit started, RR')
            best_params, RMS, result = EM.fit_data(data, p_dict, p_dict_bools, E_in=E_in_angle, p_dict_bounds=p_dict_bounds, data_type=specType, fit_algorithm=fitAlgorithm)
        report = result.fit_report()
        fit = result.best_fit

        residuals = 100 * (y - fit)
        print(report)
        return fit, report, RMS, residuals

