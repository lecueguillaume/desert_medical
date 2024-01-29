'''
Testing class for simulating bipartite networks.
03/08/2023
Hongyang YE
Override from bipartitepandas
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
    'n_patients': (10000, 'type_constrained', (int, _gteq1),
        '''
            (default=10000) Number of patients.
        ''', '>= 1'),
    'n_doctors': (500, 'type_constrained', (int, _gteq1),
        '''
            (default=500) Number of doctors.
        ''', '> 0'),
    'age_patient': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Range of the patient's age.
        ''', '>= 1'),
    'age_doctor': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Range of the doctor's age.
        ''', '>= 1'),
    'alpha_law': ((0, 1),  'type', (tuple, list),
        '''
            (default=(0, 1) Specify the bounds of the uniform law for alpha (patient effect).
        ''', None),
    'psi_law': ((0, 1),  'type', (tuple, list),
    '''
        (default=(0, 1) Specify the bounds of the uniform law for psi (doctor effect).
    ''', None),
    'alpha_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of individual fixed effect (volatility of patient effects).
        ''', '>= 0'),
    'psi_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of doctor fixed effect (volatility of doctor effects).
        ''', '>= 0'),
    'w_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of residual in AKM wage equation (volatility of wage shocks).
        ''', '>= 0'),
    'c_sort': (1, 'type', (float, int),
        '''
            (default=1) Sorting effect.
        ''', None),
    'c_netw': (1, 'type', (float, int),
        '''
            (default=1) Network effect.
        ''', None),
    'c_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of sorting/network effects.
        ''', '>= 0'),
    'z': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) The assigned value to evaluate the relation between doctors and patients
        ''', '>= 0'),
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
    'beta_5': (-0.5, 'type_constrained', ((float, int), _dteq0),
        '''
            (default=-0.5) The assigned coefficient of the distance
        ''', '<= 0'),

    
})

class SimBipartiteTest:
    '''
    Class of SimBipartite, where SimBipartite simulates a bipartite network of doctors and patients.

    Arguments:
        params (ParamsDict or None): dictionary of parameters for simulating data. 
        A (None): the identifier matrix of relations between doctors and patients.
    '''

    def __init__(self, params=None, A = None, D = None, data = None):
        if params is None:
            params = sim_params()

        # Store parameters
        self.params = params
        self.A = A
        self.D = D
        self.data = data


    def simulate(self, rng=None):
        '''
        Simulate data corresponding to the calibrated model.
        Columns are as follows: 
        i = patient id;
        j = doctor id; 
        age_p = age of patient i; 
        age_d = age of doctor j; 
        alpha = patient effect; 
        psi = doctor effect.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): simulated network
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        print('Extracting parameters...')
        n_patients, n_doctors, age_patient, age_doctor, alpha_law, psi_law, z, beta_1, beta_2, beta_3, beta_4, beta_5 = self.params.get_multiple(('n_patients', 'n_doctors', 'age_patient', 'age_doctor', 'alpha_law', 'psi_law', 'z', 'beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5'))

        # Generate fixed effects
        print('Generating fixed effects...')
        alpha = []
        if type(alpha_law) == tuple:
            for i in range(0, n_patients): 
                a = np.random.uniform(alpha_law[0], alpha_law[1])
                alpha.append(a)
        else:
            for i in range(0, n_patients): 
                a = np.random.uniform(alpha_law[i][0], alpha_law[i][1])
                alpha.append(a)
        psi = []
        if type(psi_law) == tuple:
            for i in range(0, n_doctors + 1):
                p = np.random.uniform(psi_law[0], psi_law[1])
                psi.append(p)
        if type(psi_law) == list:
            for j in range(0, n_doctors + 1):
                if j == 0:
                    p = np.random.uniform(0, 1)
                    psi.append(p)
                else:
                    p = np.random.uniform(psi_law[j-1][0], psi_law[j-1][1])
                    psi.append(p)
        # Generate the coordinates of the doctors and the patients
        coor_patients = []
        coor_doctors = []
        for i in range(0, n_patients):
            p = np.random.uniform(0, 1, 2)
            coor_patients.append(p)
        for i in range(0, n_doctors):
            d = np.random.uniform(0, 1, 2)
            coor_doctors.append(d)

        print('Generating distance matrix...')
        # Plot the locations of doctors and patients
        for (x, y) in coor_patients:
            plt.scatter(x, y, color = (0.5, 0.5, 0.5), marker = 's', s = 5)
        for (x, y) in coor_doctors:
            plt.scatter(x, y, color = (0.3, 0.3, 0.8))
        plt.title('Location of doctors and patients')
        plt.show()
        # Generate distance matrix
        D = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)
        self.D = D
        for i in range(0, n_patients):
            for j in range(0, n_doctors + 1):
                if j == 0: # We associate the indice 0 to the "ghost doctor"
                    self.D[i][0] = 0
                else: # we take the j-1 index of coor_doctors as we added the ghost doctor, j = 1 corresponds to j = 0 in coord_doctors
                    d = np.sqrt(np.power((coor_patients[i][0] - coor_doctors[j-1][0]), 2) + np.power((coor_patients[i][1] - coor_doctors[j-1][1]), 2))
                    self.D[i][j] = d   

        # Random draws of ages for patients and doctors
        print('Assigning ages randomly...')
        sim_patient_age = rng.integers(low = 1, high = age_patient, size = n_patients)
        sim_doctor_age = rng.integers(low = 26, high = age_doctor, size = n_doctors + 1)

        # Random draws of genders of patients and dcotors
        # 1 for male and 0 for female
        print('Assigning gender randomly...')
        sim_patient_gender = rng.integers(low = 0, high = 2, size = n_patients)
        sim_doctor_gender = rng.integers(low = 0, high = 2, size = n_doctors + 1)

        # Generate the identifier matrix A based on the distance
        # if the distance between patient_i and doctor_j is higher than z, there would be no relation
        # otherwise, the relation between patient _i and doctor_j conforms to binomial distribution
        # where p_ij equals to exp(alpha* + psi*) / (1 + exp(alpha* + psi*))
        print('Generating identifier matrix...')
        A = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)
        self.A = A
        for i in range(0, n_patients):
            for j in range(0, n_doctors + 1):
                if j == 0:
                    self.A[i][0] = 1
                elif D[i][j] > z:
                    continue
                else:
                    T = alpha[i] + psi[j] + beta_1 * sim_patient_age[i] + beta_2 * sim_doctor_age[j] + beta_3 * sim_patient_gender[i] + beta_4 * sim_doctor_gender[j] + beta_5 * self.D[i][j]
                    p = np.exp(T) / (1 + np.exp(T))
                    # print(p)
                    y = np.random.binomial(1, p, 1)
                    y = y.tolist()
                    self.A[i][j] = y[0]
        
        # Compile ids
        print('Compiling ids...')
        id_p = np.repeat(range(n_patients), n_doctors + 1)
        id_d = np.tile(range(n_doctors + 1), n_patients)

        # Compile fixed effects
        print('Compiling fixed effects...')
        alp_data = np.repeat(alpha, n_doctors + 1)

        psi = psi * n_patients
        psi_data = []
        for i in psi:
            psi_data.append(i)

        # Compile observed features
        print('Compiling ages...')
        age_p_data = np.repeat(sim_patient_age, n_doctors + 1)
        age_d_data = np.tile(sim_doctor_age, n_patients)
        print('Compiling genders...')
        sex_p_data = np.repeat(sim_patient_gender, n_doctors + 1)
        sex_d_data = np.tile(sim_doctor_gender, n_patients)

        # Compile relations between doctors and patients
        print('Compiling relations...')
        relation = self.A.flatten()

        # Merge all columns into a dataframe
        data = DataFrame(data={'i': id_p, 'j': id_d, 'r' : relation, 'age_p': age_p_data, 'age_d': age_d_data, 
                               'sex_p': sex_p_data, 'sex_d': sex_d_data,
                               'alpha': alp_data, 'psi': psi_data
                                })
        
        # drop the rows if there is no relation between patient_i and doctor_j
        
        data = data.drop(data[data['r'] == 0].index)
        data = data.drop('r', axis = 1)
        # data = data.reset_index().drop(['index'], axis = 1)
        
        # cancel connections between patient i and ghost doctor if patient i isn't only connected to the ghost doctor
        number_of_connections = data['i'].value_counts(sort=None)
        for i in range(n_patients):
    
            if number_of_connections[i] > 1: # if patient i isn't only connected to the ghost doctor, we remove its connection with the ghost doctor.
        
                index_to_drop = data[data['i'] == i].index[0] # we get the index of the row to drop
                data = data.drop(index_to_drop)
        data = data.reset_index().drop(['index'], axis = 1)
        
        self.data = data
        return self.data
    