'''
Testing class for the second simulation model for the cost of the visits
05/22/2023
Hongyang YE
'''

from paramsdict import ParamsDict
import numpy as np
ax = np.newaxis
from pandas import DataFrame
from scipy.stats import norm
import matplotlib.pyplot as plt

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _0to1(a):
    return 0 <= a <= 1
def _dteq0(a):
    return a <= 0

# Define default parameter dictionary
sim_params = ParamsDict({
    'p_alpha_law': ((-2, -1),  'type', (tuple),
        '''
            (default=(-2,-1) Specify the bounds of the uniform law for alpha (patient effect).
        ''', None),
    'p_psi_law': ((-2, -1),  'type', (tuple),
    '''
        (default=(-2,-1) Specify the bounds of the uniform law for psi (doctor effect).
    ''', None),
    'beta_1': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned coefficient of the patient_age
        ''', '>= 0'),
    'beta_2': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned coefficient of the doctor_age
        ''', '>= 0'),
    'beta_3': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned coefficient of the patient_sex
        ''', '>= 0'),
    'beta_4': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned coefficient of the doctor_sex
        ''', '>= 0'),
    'beta_5': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned coefficient of the distance
        ''', '>= 0'),    
})

class CostVisitSim:
    '''
    Class of CostVisitSim, which simulates the cost of visits from existing relations
    between doctors and patients
    
    Arguments:
        params (ParamsDict or None): dictionary of parameters for simulating data. 
        relations (None): the existing relations between doctors and patients
        from the first simulation model.
    '''

    def __init__(self, visits, D, params = None):
        if params is None:
            params = sim_params()

        # Store parameters
        self.params = params
        self.relations = visits.reset_index().drop(columns = 'index', axis = 1)
        self.D = D


    def simulate(self, rng = None):
        '''
        Simulate data corresponding to the first simulated bipartite model
        Columns are as follows: 
        z = cost of visiting;
        p_alpha = prime patient effect; 
        p_psi = prime doctor effect.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): simulated cost of visit
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        print('Extracting parameters...')
        p_alpha_law, p_psi_law, beta_1, beta_2, beta_3, beta_4, beta_5 = self.params.get_multiple(('p_alpha_law', 'p_psi_law', 'beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5'))

        # Generate prime fixed effects
        print('Generating prime fixed effects...')

        p_alpha = {}
        if type(p_alpha_law) == tuple:
            for i in self.relations['i'].unique(): 
                
                p_a = np.random.uniform(p_alpha_law[0], p_alpha_law[1])
                p_alpha[i] = p_a
                
        else:
            # for i in range(len(self.relations['i'].unique())):
            for i in self.relations['i'].unique():
                
                p_a = np.random.uniform(p_alpha_law[i][0], p_alpha_law[i][1])
                p_alpha[i] = p_a
                
        p_psi = {}
        if type(p_psi_law) == tuple:
            for j in self.relations['j'].unique():
                
                p_p = np.random.uniform(p_psi_law[0], p_psi_law[1]) 
                p_psi[j] = p_p
                
        else:
            for j in self.relations['j'].unique():
                
                p_p = np.random.uniform(p_psi_law[j][0], p_psi_law[j][1]) 
                p_psi[j] = p_p
                
        # Add prime fixed effects to the visits
        
        self.relations['p_alpha'] = self.relations['i'].map(p_alpha)
        self.relations['p_alpha'] = self.relations['p_alpha'].astype(float)
        self.relations['p_psi'] = self.relations['j'].map(p_psi)
        self.relations['p_psi'] = self.relations['p_psi'].astype(float)

        # Add distance between patients and doctors
        self.relations['distance'] = self.D[self.relations['i'], self.relations['j']]

        # Simulate and add the cost of visits
        cost = []
        for i in range(0, len(self.relations)):
            z = self.relations.loc[i]['p_alpha'] + self.relations.loc[i]['p_psi'] + beta_1 * self.relations.loc[i]['age_p'] + beta_2 * self.relations.loc[i]['age_d'] + beta_3 * self.relations.loc[i]['sex_p'] + beta_4 * self.relations.loc[i]['sex_d'] + beta_5 * self.relations.loc[i]['distance']
            cost.append(z)
        self.relations['cost'] = cost

        


        return self.relations
    