# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:53:24 2021

@author: bener807
"""

import pickle
import numpy as np
import sys
sys.path.insert(0, 'C:/python/definitions/')
import useful_defs as udfs
import definitions_heimdall as dfs
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import scipy.optimize as optimize

class Adq14():
    def __init__(self, tof_edges, fuel_config):
        # Set charge bins
        self.s1_charge_edges = np.arange(0, 1E6, .2E4)
        self.s1_charge_centres = get_bin_centres(self.s1_charge_edges)
        
        self.s2_charge_edges = np.arange(0, 2E6, 0.2E4)
        self.s2_charge_centres = get_bin_centres(self.s2_charge_edges)
        
        # Set TOF axis
        self.tof_edges = tof_edges
        self.tof_centres = get_bin_centres(self.tof_edges)
        
        # Set fuel config
        self.fuel_config = fuel_config
        self.fuel_ratios = None
        
        # Dictionaries to store histograms
        s1_fill = np.zeros([len(self.tof_centres), len(self.s1_charge_centres)])
        s2_fill = np.zeros([len(self.tof_centres), len(self.s2_charge_centres)])
        self.s1_q = dfs.get_dictionaries('nested', fill = s1_fill)
        self.s2_q = dfs.get_dictionaries('nested', fill = s2_fill)
        self.analysed_shots = []
        
    def parse_data(self, file_names):
        # Read data
        for i, file_name in enumerate(file_names):
            shot_number = int(file_name[-12:-7])
            if not self.get_fuel_config(shot_number) == self.fuel_config: 
                print(f'{i + 1}/{len(file_names)} {shot_number} skipped.')
                continue
            
            info = udfs.unpickle(file_name)
            charge = info['energies']
            times = info['times_of_flight']
            
            # Loop over s1-s2 combinations
            for s1 in times.keys():
                for s2 in times[s1].keys():
                    # Slice flight times
                    self.slice_tof(times[s1][s2], charge[s1][s2], s1, s2)
            self.analysed_shots.append(shot_number)
            print(f'{i + 1}/{len(file_names)} {shot_number} complete.')
            
    def get_fuel_config(self, shot_number):
        if self.fuel_ratios == None:
            # Load T fraction data
            t_fractions = np.loadtxt('t_gas_fraction.dat')
            self.fuel_ratios = {int(key):value for key, value in t_fractions}
    
        t_ratio = self.fuel_ratios[shot_number]                        
        # Check fuel ratios
        if t_ratio < 0.02: return 'DD'
        elif t_ratio > 0.02 and t_ratio < 0.7: return 'DT'
        elif t_ratio > 0.8: return 'TT'
        else: return 'Unknown'
        

        
    def slice_tof(self, times, charge, s1, s2):
        # sliced_charges  = np.zeros(len(tof_edges) - 1)
        for i, (low_edge, high_edge) in enumerate(zip(self.tof_edges[:-1], self.tof_edges[1:])):
            # Find charges within tof slice
            mask = ((times > low_edge) & (times < high_edge))
            try: s1_sliced_charges = charge[0][mask]
            except: continue
            try: s2_sliced_charges = charge[1][mask]
            except: continue
            
            # Histogram
            s1_vals, _ = np.histogram(s1_sliced_charges, 
                                      bins = self.s1_charge_edges)
            s2_vals, _ = np.histogram(s2_sliced_charges, 
                                      bins = self.s2_charge_edges)
            
            # Add to dictionary
            self.s1_q[s1][s2][i] += s1_vals
            self.s2_q[s1][s2][i] += s2_vals
            
    def combined_plot(self, s1, s2):
        fig, (s1_ax, s2_ax) = plt.subplots(2)
        for row in self.s1_q[s1][s2]:
            s1_ax.semilogy(self.s1_charge_centres, row)
            s1_ax.set_ylabel('Counts')
        for row in self.s2_q[s1][s2]:
            s2_ax.semilogy(self.s2_charge_centres, row)
            s2_ax.set_xlabel('Integrated charge [a.u.]')
            s2_ax.set_ylabel('Counts')
        
        plt.subplots_adjust(hspace = 0.3)
        s1_ax.text(0.9, 0.9, f'{s1}', transform = s1_ax.transAxes)
        s2_ax.text(0.9, 0.9, f'{s2}', transform = s2_ax.transAxes)
    
    
    def individual_plot(self, s1, s2, t_start, t_end, log = False):
        start = np.where(self.tof_centres == t_start)[0][0]
        end = np.where(self.tof_centres == t_end)[0][0]
        for i in range(start, end + 1):
            # Create figure title
            fig, (s1_ax, s2_ax) = plt.subplots(2, num = f'{self.tof_centres[i]}')
            fig.suptitle(f'{self.tof_centres[i]} ns')
            
            # S1 plot
            plot_sx(charge_centres = self.s1_charge_centres, 
                    q = self.s1_q[s1][s2][i], 
                    ax = s1_ax, 
                    log = log,
                    sx = s1)
            
            # S2 plot
            plot_sx(charge_centres = self.s2_charge_centres, 
                    q = self.s2_q[s1][s2][i], 
                    ax = s2_ax, 
                    log = log,
                    sx = s2)
            
            plt.subplots_adjust(hspace = 0.3)
    
    def save_data(self, file_name):
        to_pickle = {'s1 charge':self.s1_q,
                     's1 charge edges':self.s1_charge_edges,
                     's1 charge centres':self.s1_charge_centres,
                     's2 charge':self.s2_q,
                     's2 charge edges':self.s2_charge_edges,
                     's2 charge centres':self.s2_charge_centres,
                     'tof edges':self.tof_edges,
                     'tof centres':self.tof_centres,
                     'analysed shots':self.analysed_shots,
                     }
        
        udfs.pickler(file_name, to_pickle)
         
    def fit_s1_dt(self, s1, s2):
        start = np.where(self.tof_centres == 25)[0][0]
        end = np.where(self.tof_centres == 28)[0][0]
        fit_ranges = np.array([[2E5, 4.2E5], [1E5, 4.2E5], [1.4E5, 4.2E5], [1.15E5, 4.2E5]])
        params = np.array([])
        for i, time in enumerate(range(start, end + 1)):
            y = self.s1_q[s1][s2][time] 
            x = self.s1_charge_centres
            
            mu_0 = 1.7E5
            sigma_0 = 100000
            skew_0 = 10
            bgr_0 = 1
            amplitude_0 = np.max(y)
            parameters = optimize.minimize(fun = optimization, 
                                           x0 = [mu_0, sigma_0, skew_0, amplitude_0, bgr_0], 
                                           args = (x, y, fit_ranges[i]),
                                           bounds = ((None, None), (None, None), (None, None), (None, None), (0, 2)))
            fig, ax = plt.subplots()
            plot_sx(x, y, s1, ax, log = False)
            mu, sigma, skew, amplitude, background = return_parameters(parameters.x)
            
            ax.plot(x, skew_normal(mu, sigma, skew, amplitude, background, x))
            halfway = halfway_point(parameters)
            ax.axvline(halfway, linestyle = '--', color = 'k')
            
            print()
            print('----------')
            print(parameters)
            print(f'  halfway: {halfway:.2f}')
            print()
            params = np.append(params, parameters)
        return params

    def fit_s1_dd(self, s1, s2):
        start = np.where(self.tof_centres == 63)[0][0]
        end = np.where(self.tof_centres == 67)[0][0]
        fit_ranges = np.array([[2000, 6E4], [6500, 8E4], [5000, 8E4], [2500, 8E4], [7E3, 6.6E4]])
        params = np.array([])
        for i, time in enumerate(range(start, end + 1)):
            y = self.s1_q[s1][s2][time] 
            x = self.s1_charge_centres
            
            mu_0 = 1.7E4
            sigma_0 = 1000
            skew_0 = 7
            bgr_0 = 1
            amplitude_0 = np.max(y)
            parameters = optimize.minimize(fun = optimization, 
                                           x0 = [mu_0, sigma_0, skew_0, amplitude_0, bgr_0], 
                                           args = (x, y, fit_ranges[i]),
                                           bounds = ((None, None), (None, None), (0, None), (0, None), (0, 50)))
            fig, ax = plt.subplots()
            plot_sx(x, y, s1, ax, log = False)
            mu, sigma, skew, amplitude, background = return_parameters(parameters.x)
            
            ax.plot(x, skew_normal(mu, sigma, skew, amplitude, background, x))
            halfway = halfway_point(parameters)
            ax.axvline(halfway, linestyle = '--', color = 'k')
            
            print(f'i = {time}, nit = {parameters.nit}')
            print('----------')
            print(f'chi2:    {parameters.fun}')
            print(f'mu:      {parameters.x[0]}')
            print(f'sigma:   {parameters.x[1]}')
            print(f'skew:    {parameters.x[2]}')
            print(f'ampl.:   {parameters.x[3]}')
            print(f'backgr.: {parameters.x[4]}')
            
            print(f'  halfway: {halfway:.2f}')
            print()
            params = np.append(params, parameters)
        return params

    def fit_2500(self, s1, s2):
        '''
        Find the 2.5 MeV edge in the DD spectrum
        '''
        start = np.where(self.tof_centres == 35)[0][0]
        end = np.where(self.tof_centres == 48)[0][0]
        params = np.array([])
        y = self.s1_q[s1][s2][start]
        for time in range(start + 1, end + 1): 
            y += self.s1_q[s1][s2][time] 
        x = self.s1_charge_centres  
        
        fig, ax = plt.subplots()
        plot_sx(x, y, s1, ax, log = True)
        
        # Edge chosen by eye
        ax.axvline(143000, linestyle = '--', color = 'k')
        
        return None


def return_parameters(parameters):
    mu          = parameters[0]
    sigma       = parameters[1]
    skew        = parameters[2]
    amplitude   = parameters[3]
    background  = parameters[4]
    
    return mu, sigma, skew, amplitude, background
        

def get_bin_centres(bin_edges):
    return bin_edges[1:] - np.diff(bin_edges)[0]/2    

def optimization(parameters, x, y, fit_range):
    # Grab initial guesses
    mu, sigma, skew, amplitude, background = return_parameters(parameters)
    skewed_norm = skew_normal(mu, sigma, skew, amplitude, background, x, fit_range)
    
    # Set fit range
    if (fit_range != [0, 0]).any():
        y = y[np.searchsorted(x, fit_range[0]):np.searchsorted(x, fit_range[1])]
    
    # Calculate chi2
    chi2 = np.sum((y - skewed_norm)**2/skewed_norm) / len(y)
    return chi2
    
def skew_normal(mu, sigma, skew, amplitude, background, x, fit_range = np.array([0, 0])):
    # Calculate skewed normal distribution
    pdf = amplitude/(np.sqrt(2*np.pi)) * np.exp(-((x - mu)/sigma)**2/2)
    cdf = (1 + erf(skew*(x - mu)/(sigma * np.sqrt(2)))) / 2
    skew_norm = 2*pdf*cdf + background
    
    if (fit_range != [0, 0]).any():
        skew_norm = skew_norm[np.searchsorted(x, fit_range[0]):np.searchsorted(x, fit_range[1])]
    
    return skew_norm


def plot_sx(charge_centres, q, sx, ax, log = False):
        cap_size = 1.5
        line_width = 1
        marker = '.'
        marker_size = 1.5
        color = 'k'
        ax.plot(charge_centres, 
                q,
                color = color,
                marker = marker,
                markersize = marker_size,
                linestyle = 'None')
            
        ax.errorbar(charge_centres,
                    q, 
                    np.sqrt(q),
                    elinewidth = line_width,
                    capsize = cap_size,
                    linestyle = 'None',
                    color = color)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Integrated charge (a.u.)')
        ax.text(0.9, 0.9, f'{sx}', transform = ax.transAxes)
        if log: ax.set_yscale('log')

def halfway_point(parameters):
    mu, sigma, skew, amplitude, background = return_parameters(parameters.x)

    # Create x, y
    x = np.linspace(1000, 8E5, 100000)
    y = skew_normal(mu, sigma, skew, amplitude, background, x)
     
    # Select right side of peak
    x = x[np.argmax(y):]
    y = y[np.argmax(y):]
    
    # Find halfway point
    arg_half = np.argmin(np.abs(y - np.max(y)/2))

    return x[arg_half]


if __name__ == '__main__':
    
    # Create adq_14 object with given bins and fuel configuration
    adq_14 = Adq14(tof_edges = np.arange(9.5, 69.5, 1), 
                    fuel_config = 'TT')
    
    file_name = f'../data/parsed_data/adq14_{adq_14.fuel_config}.pickle'
    if os.path.isfile(file_name):
        data = udfs.unpickle(file_name)
        # S1
        adq_14.s1_q              = data['s1 charge']
        adq_14.s1_charge_centres = data['s1 charge centres']
        adq_14.s1_charge_edges   = data['s1 charge edges']
        
        # S2
        adq_14.s2_q              = data['s2 charge']   
        adq_14.s2_charge_centres = data['s2 charge centres']
        adq_14.s2_charge_edges   = data['s2 charge edges']
        
        # TOF
        adq_14.tof_centres = data['tof centres']
        adq_14.tof_edges   = data['tof edges']
        
    else:
        # Get all file names
        path = '../data/adq14/'
        ans = input('Are bin widths set correctly? [y/n]')
        if ans not in ['y', 'Y']: sys.exit()
        files = os.listdir(path)
        adq14_files = [f'{path}{shot}' for shot in files]
   
        # Parse data
        adq_14.parse_data(adq14_files)
        
        # Save data
        adq_14.save_data(file_name)
    
    # Plot
    adq_14.combined_plot('S1_05', 'S2_01')
    # adq_14.individual_plot('S1_05', 'S2_06', 25, 30, log = True)
    # params = adq_14.fit_s1_dt('S1_05', 'S2_06')
    # params = adq_14.fit_s1_dd('S1_05', 'S2_06')    
    # params = adq_14.fit_2500('S1_05', 'S2_06')



