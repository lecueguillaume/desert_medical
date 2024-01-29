import fixed_effects.simbipartiteTest as simTest
import matplotlib.pyplot as plt
import fixed_effects.CostVisitSimTest as CostSim
import pandas as pd
import pytwoway as tw
import bipartitepandas as bpd
import numpy as np
# import PyChest
import ruptures as rpt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
# import scipy
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# Ignore warnings below
simplefilter("ignore", category=ConvergenceWarning) # Useful for logistic regression
pd.options.mode.chained_assignment = None  # default='warn' # Remove copy on slice warning

def temporal_simulation(nb_of_periods,
                           n_patients,
                           n_doctors,
                           z=np.sqrt(2),
                           type_distance='default',
                           alpha_law_graph=(0, 0.5),
                           psi_law_graph=(0, 0.5),
                           alpha_law_cost=(0, 0.5),
                           psi_law_cost=(0, 0.5),
                           preconditioner = 'ichol',
                           beta_age_p_graph=0.01,
                           beta_age_d_graph=0.01,
                           beta_sex_p_graph=0.5,
                           beta_sex_d_graph=0.5,
                           beta_distance_graph=0.5,
                           beta_age_p_cost=0.5,
                           beta_age_d_cost=0.5,
                           beta_sex_p_cost=0.5,
                           beta_sex_d_cost=0.5,
                           beta_distance_cost=0.5):
    """
    dataframe has to be the dataframe of connections between patients and doctors.
    """
    # We set up the parameters to estimate the FE.

    if type_distance not in ['REG', 'DEP', 'CODGEO', 'default']:
        raise Exception("type_distance must be 'REG', 'DEP', 'CODGEO' or 'default'")
    
    if preconditioner not in ['ichol', 'jacobi']:
        raise Exception("preconditioner has to be 'ichol' or 'jacobi'. Prefer 'jacobi' for large datasets.")
        
    
    fecontrol_params = tw.fecontrol_params(
    {
        'ho': True,
        'he': False,
        'feonly': True,
        'continuous_controls': ['distance', 'age_d', 'age_p'],
        'categorical_controls': ['sex_p', 'sex_d'],
        'attach_fe_estimates': True,
        'ncore': 8,
        'preconditioner': preconditioner # It looks like it gives better results (especially for large datasets ?)
    }
    )

    clean_params = bpd.clean_params(
    {
        'connectedness': 'leave_out_spell',
        'collapse_at_connectedness_measure': True,
        'drop_single_stayers': True,
        'drop_returns': 'returners',
        'copy': False
    }
    )

    rng = np.random.default_rng(None)
    alpha_graph = []
    psi_graph = []
    alpha_cost = {} # These are dicts to use the function map later
    psi_cost = {}
    changepoint_patient = np.zeros(n_patients)
    changepoint_doctor = np.zeros(n_doctors + 1)
    coor_patients = []
    coor_doctors = []
    D = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)
    log = LogisticRegression()

    for i in range(n_patients):
        
        # We generate the FE for the graph formation model
        alpha_graph.append( np.random.uniform(alpha_law_graph[0], alpha_law_graph[1]) )
        
        # We generate the FE for the cost model
        alpha_cost[i] = np.random.uniform(alpha_law_cost[0], alpha_law_cost[1])

        # We generate the periods when there's a changepoint for each patient
        changepoint_patient[i] = np.random.randint(0, nb_of_periods)

        # Generate the coordinates of the patients
        coor_patients.append( np.random.uniform(0, 1, 2) )
                               
    for j in range(n_doctors + 1):

        # We generate the FE for the graph formation model
        psi_graph.append( np.random.uniform(psi_law_graph[0], psi_law_graph[1]) )

        # We generate the FE for the cost model
        psi_cost[j] = np.random.uniform(psi_law_cost[0], psi_law_cost[1])

        # We generate the periods when there's a changepoint for each doctor
        changepoint_doctor[j] = np.random.randint(0, nb_of_periods)
        
        if j != 0:
            
            # Generate the coordinates of the doctors
            coor_doctors.append( np.random.uniform(0, 1, 2) )

    # Generate distance matrix
    if type_distance == 'default':
    
        for i in range(n_patients):
            for j in range(0, n_doctors + 1):
                if j == 0: # We associate the indice 0 to the "ghost doctor"
                    D[i][0] = 0
                else: # we take the j-1 index of coor_doctors as we added the ghost doctor, j = 1 corresponds to j = 0 in coord_doctors
                    d = np.sqrt(np.power((coor_patients[i][0] - coor_doctors[j-1][0]), 2) + np.power((coor_patients[i][1] - coor_doctors[j-1][1]), 2))
                    D[i][j] = d

    else:

        # Assign randomly a CODGEO, DEP, or REG to patients and doctors
        dist_matrix = pd.read_csv('data/' + type_distance + '.csv')
        
        del dist_matrix[dist_matrix.columns[0]]
        dist_matrix.index = dist_matrix.columns
        for i in range(len(dist_matrix)):
            dist_matrix.iloc[i, i] = 0
        arr = dist_matrix.columns.values
        for i,col in enumerate(arr):
            arr[i] = int(float(arr[i]))
        dist_matrix.columns = arr
        dist_matrix.index = arr

        # Generate code for patient and doctor
        code_patient = []
        code_doctor = []
        for i in range(n_patients):
            random_code = np.random.choice(dist_matrix.columns.values)
            code_patient.append( random_code )
        for j in range(n_doctors + 1):
            random_code = np.random.choice(dist_matrix.columns.values)
            code_doctor.append( random_code )
        for i in range(n_patients):
            for j in range(n_doctors):
                D[i, j + 1] = dist_matrix.loc[code_patient[i], code_doctor[j+1]]
                
        
        


    # Random draws of ages for patients and doctors
    sim_patient_age = rng.integers(low = 1, high = 99, size = n_patients)
    sim_doctor_age = rng.integers(low = 26, high = 99, size = n_doctors + 1)

    # Random draws of genders of patients and doctors
    sim_patient_gender = rng.integers(low = 0, high = 2, size = n_patients)
    sim_doctor_gender = rng.integers(low = 0, high = 2, size = n_doctors + 1)

    # Compile ids
    id_p = np.repeat(range(n_patients), n_doctors + 1)
    id_d = np.tile(range(n_doctors + 1), n_patients)

    # Compile fixed effects
    # alp_data = np.repeat(alpha_cost, n_doctors + 1)
    # psi_data = psi_graph * n_patients

    # Compile observed features
    age_p_data = np.repeat(sim_patient_age, n_doctors + 1)
    age_d_data = np.tile(sim_doctor_age, n_patients)
    sex_p_data = np.repeat(sim_patient_gender, n_doctors + 1)
    sex_d_data = np.tile(sim_doctor_gender, n_patients)
    if type_distance != 'default':
        code_patient_data = np.repeat(code_patient, n_doctors + 1)
        code_doctor_data = np.tile(code_doctor, n_patients)

    estimates = []
                               
    # At each period, determine connections                           
    for t in range(nb_of_periods):
    
        # Generate the identifier matrix A based on the distance
        A = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)
        for i in range(0, n_patients):
            for j in range(0, n_doctors + 1):
                if j == 0:
                    A[i][0] = 1
                elif D[i][j] > z: # if patient i and doctor j are too far away, there is no relation
                    continue
                else:
                    T = alpha_graph[i] + psi_graph[j] + beta_age_p_graph * sim_patient_age[i] + beta_age_d_graph * sim_doctor_age[j] + beta_sex_p_graph * sim_patient_gender[i] + beta_sex_d_graph * sim_doctor_gender[j] + beta_distance_graph * D[i][j]
                    p = 1 / (1 + np.exp(-T))
                    A[i][j] = np.random.binomial(1, p)

        # Compile relations between doctors and patients
        relation = A.flatten()

        # Merge all columns into a dataframe
        dataframe = pd.DataFrame(data={'i': id_p, 'j': id_d, 'y' : relation, 'age_p': age_p_data, 'age_d': age_d_data, 
                               'sex_p': sex_p_data, 'sex_d': sex_d_data
                                })
        if type_distance != 'default':
            dataframe['code_patient'] = code_patient_data
            dataframe['code_doctor'] = code_doctor_data

        dataframe['distance'] = D[dataframe['i'], dataframe['j']].astype(float)
            
        # Logistic regression for graph formation

        # Add dummy variables
        e_i = pd.DataFrame(np.zeros((n_patients*(n_doctors + 1), n_patients), dtype=int))
        for col in e_i.columns:
            e_i.rename(columns = {col :f'p_{col}'}, inplace = True)
            
        e_j = pd.DataFrame(np.zeros((n_patients*(n_doctors + 1), n_doctors + 1), dtype=int))
        for col in e_j.columns:
            e_j.rename(columns = {col :f'd_{col}'}, inplace = True)
        
        df = pd.concat([dataframe, e_i, e_j], axis = 1)
        
        for i in range(n_patients):
            indexes = df[df['i'] == i].index
            df[f'p_{i}'][indexes] = [1 for i in range(len(indexes))]
        
        for j in range(n_doctors + 1):
            indexes = df[df['j'] == j].index
            df[f'd_{j}'][indexes] = [1 for i in range(len(indexes))]
        
        y = df['y'].astype(int)
        if type_distance != 'default':
            X = df.drop(['i', 'j', 'y', 'code_patient', 'code_doctor'], axis = 1)
        else:
            X = df.drop(['i', 'j', 'y'], axis = 1)
        
        reg = log.fit(X, y)
        coeffs = reg.coef_[0]

        # drop the rows if there is no relation between patient_i and doctor_j
        dataframe = dataframe.drop(dataframe[dataframe['y'] == 0].index)
        dataframe = dataframe.drop('y', axis = 1)
        dataframe = dataframe.reset_index().drop(['index'], axis = 1)



        # We update the laws (if needed) of the patients/doctors
        list_of_indexes_patient = np.where(changepoint_patient == t)[0]
        list_of_indexes_doctor = np.where(changepoint_doctor == t)[0]
        for index_patient in list_of_indexes_patient: 
            
            alpha_cost[index_patient] = np.random.uniform( np.random.uniform(alpha_law_graph[0] + 5, alpha_law_graph[1] + 5) )
    
        for index_doctor in list_of_indexes_doctor:
            
            psi_cost[index_doctor] = np.random.uniform( np.random.uniform(psi_law_graph[0] + 5, psi_law_graph[1] + 5) )

        dataframe['alpha'] = dataframe['i'].map(alpha_cost).astype(float)
        dataframe['psi'] = dataframe['j'].map(psi_cost).astype(float)

        # Compute the cost
        dataframe['y'] = dataframe['alpha'] + dataframe['psi'] + beta_age_p_cost * dataframe['age_p'] + beta_age_d_cost * dataframe['age_d'] + beta_sex_p_cost * dataframe['sex_p'] + beta_sex_d_cost * dataframe['sex_d'] + beta_distance_cost * dataframe['distance']

        # Change dtype of categorical variables
        dataframe['sex_p'] = dataframe['sex_p'].astype("category")
        dataframe['sex_d'] = dataframe['sex_d'].astype("category")
        
        # We create a BipartiteDataFrame in order to estimate the FE
        if type_distance == 'default':
            
            bdf = bpd.BipartiteDataFrame(dataframe.drop(['alpha', 'psi'] , axis = 1),
                                     custom_categorical_dict = {'sex_p': True,
                                                                'sex_d': True},
                                     custom_dtype_dict = {'sex_p': 'categorical',
                                                          'sex_d': 'categorical'},
                                     custom_how_collapse_dict = {'sex_p': 'first',
                                                                 'sex_d': 'first'}) # We transform the dataframe as BipartitePandas dataframe to Estimate the FE.
        else:
            
            bdf = bpd.BipartiteDataFrame(dataframe.drop(['alpha', 'psi', 'code_patient', 'code_doctor'] , axis = 1),
                                     custom_categorical_dict = {'sex_p': True,
                                                                'sex_d': True},
                                     custom_dtype_dict = {'sex_p': 'categorical',
                                                          'sex_d': 'categorical'},
                                     custom_how_collapse_dict = {'sex_p': 'first',
                                                                 'sex_d': 'first'}) # We transform the dataframe as BipartitePandas dataframe to Estimate the FE.

    
        bdf.clean(clean_params)
        fe_estimator = tw.FEControlEstimator(bdf, fecontrol_params)
        print(f"Estimating FE for period {t}")
        fe_estimator.fit()
        d = {}
        d['estimates'] = fe_estimator.gamma_hat_dict # Estimates of the EF, Beta for the cost model
        d['true_value'] = dataframe # True values of the features, the initial dataframe.
        d['graph'] = {}
        d['graph']['coeffs'] = coeffs
        d['graph']['alpha'] = alpha_graph
        d['graph']['psi'] = psi_graph
        estimates.append(d)

    return estimates

def extract(temporal_simulation):
    """
    temporal_simulation[t]['true_value'] is a DataFrame of data at time t
    temporal_simulation[t]['estimates'] is a dict containing the estimates from FEControlEstimator (pytwoway module) of FE at time t
    We only extract the real doctors (ghost doctor isn't taken)
    """

    
    estimates = {}
    estimates['estimates'] = {}
    estimates['estimates']['cost'] = {}
    estimates['estimates']['graph'] = {}
    estimates['estimates']['cost']['alpha'] = {}
    estimates['estimates']['cost']['psi'] = {}
    estimates['estimates']['graph']['alpha'] = {}
    estimates['estimates']['graph']['psi'] = {}
    estimates['true_value'] = {}
    estimates['true_value']['cost'] = {}
    estimates['true_value']['graph'] = {}
    estimates['true_value']['cost']['alpha'] = {}
    estimates['true_value']['cost']['psi'] = {}
    estimates['true_value']['graph']['alpha'] = {}
    estimates['true_value']['graph']['psi'] = {}
    
    nb_of_periods = len(temporal_simulation)
    n_patients = len(temporal_simulation[0]['estimates']['alpha'])
    n_doctors = len(temporal_simulation[0]['estimates']['psi']) # contient le docteur fantôme car on ne le supprime pas

    for i in range(n_patients):

        estimates['estimates']['cost']['alpha'][i] = []
        estimates['estimates']['graph']['alpha'][i] = []
        estimates['true_value']['cost']['alpha'][i] = []
        estimates['true_value']['graph']['alpha'][i] = []
        
    for j in range(n_doctors - 1):
            
        estimates['estimates']['cost']['psi'][j] = []
        estimates['estimates']['graph']['psi'][j] = []
        estimates['true_value']['cost']['psi'][j] = []
        estimates['true_value']['graph']['psi'][j] = []
    

    for t in range(nb_of_periods):
        df = temporal_simulation[t]['true_value']
        for i in temporal_simulation[t]['true_value']['i'].unique():

            estimates['estimates']['cost']['alpha'][i].append( temporal_simulation[t]['estimates']['alpha'][i] )
            estimates['estimates']['graph']['alpha'][i].append( temporal_simulation[t]['graph']['coeffs'][5 + i] )
            estimates['true_value']['cost']['alpha'][i].append( df[df['i'] == i]['alpha'].iloc[0] )
            estimates['true_value']['graph']['alpha'][i].append( temporal_simulation[t]['graph']['alpha'][i] )


        # for j in np.delete(simulation[t]['true_value']['j'].unique(), np.where(simulation[t]['true_value']['j'].unique() == 0)) :
        for j in range(n_doctors - 1): # We dodge the ghost doctor
    
            estimates['estimates']['cost']['psi'][j].append( temporal_simulation[t]['estimates']['psi'][j+1] )
            estimates['estimates']['graph']['psi'][j].append( temporal_simulation[t]['graph']['coeffs'][5 + n_patients + j + 1] )
            estimates['true_value']['cost']['psi'][j].append( df[df['j'] == j+1]['psi'].iloc[0] )
            estimates['true_value']['graph']['psi'][j].append( temporal_simulation[t]['graph']['psi'][j + 1] )
            

            
    return estimates

def extract_from_csv(temporal_simulation):
    """
    Adapted version for simulations registered .
    """

    
    estimates = {}
    estimates['estimates'] = {}
    estimates['estimates']['cost'] = {}
    estimates['estimates']['graph'] = {}
    estimates['estimates']['cost']['alpha'] = {}
    estimates['estimates']['cost']['psi'] = {}
    estimates['estimates']['graph']['alpha'] = {}
    estimates['estimates']['graph']['psi'] = {}
    estimates['true_value'] = {}
    estimates['true_value']['cost'] = {}
    estimates['true_value']['graph'] = {}
    estimates['true_value']['cost']['alpha'] = {}
    estimates['true_value']['cost']['psi'] = {}
    estimates['true_value']['graph']['alpha'] = {}
    estimates['true_value']['graph']['psi'] = {}
    
    nb_of_periods = len(temporal_simulation)
    n_patients = len(temporal_simulation['estimates'][0]['alpha'])
    n_doctors = len(temporal_simulation['estimates'][0]['psi']) # contient le docteur fantôme car on ne le supprime pas

    for i in range(n_patients):

        estimates['estimates']['cost']['alpha'][i] = []
        estimates['estimates']['graph']['alpha'][i] = []
        estimates['true_value']['cost']['alpha'][i] = []
        estimates['true_value']['graph']['alpha'][i] = []
        
    for j in range(n_doctors - 1):
            
        estimates['estimates']['cost']['psi'][j] = []
        estimates['estimates']['graph']['psi'][j] = []
        estimates['true_value']['cost']['psi'][j] = []
        estimates['true_value']['graph']['psi'][j] = []
    

    for t in range(nb_of_periods):
        df = temporal_simulation['true_value'][t]
        for i in temporal_simulation['true_value'][t]['i'].unique():

            estimates['estimates']['cost']['alpha'][i].append( temporal_simulation['estimates'][t]['alpha'][i] )
            estimates['estimates']['graph']['alpha'][i].append( temporal_simulation['graph'][t]['coeffs'][5 + i] )
            estimates['true_value']['cost']['alpha'][i].append( df[df['i'] == i]['alpha'].iloc[0] )
            estimates['true_value']['graph']['alpha'][i].append( temporal_simulation['graph'][t]['alpha'][i] )


        # for j in np.delete(simulation[t]['true_value']['j'].unique(), np.where(simulation[t]['true_value']['j'].unique() == 0)) :
        for j in range(n_doctors - 1): # We dodge the ghost doctor
    
            estimates['estimates']['cost']['psi'][j].append( temporal_simulation['estimates'][t]['psi'][j+1] )
            estimates['estimates']['graph']['psi'][j].append( temporal_simulation['graph'][t]['coeffs'][5 + n_patients + j + 1] )
            estimates['true_value']['cost']['psi'][j].append( df[df['j'] == j+1]['psi'].iloc[0] )
            estimates['true_value']['graph']['psi'][j].append( temporal_simulation['graph'][t]['psi'][j + 1] )
            

            
    return estimates

def changepoint(estimates, process_count, cost="l2", windows_width = 20):
    """
    All the models are: "l1", "rbf", "linear", "normal", "ar"
    """
    n_patients = len(estimates['estimates']['cost']['alpha'])
    n_doctors = len(estimates['estimates']['cost']['psi'])
    changepoint_estimates = {}
    changepoint_estimates['estimates'] = {}
    changepoint_estimates['estimates']['alpha'] = {}
    changepoint_estimates['estimates']['psi'] = {}
    changepoint_estimates['true_value'] = {}
    changepoint_estimates['true_value']['alpha'] = {}
    changepoint_estimates['true_value']['psi'] = {}

    # Two best models
    # algo = rpt.Dynp(model=cost) # "l1", "l2", "rbf", "linear", "normal", "ar" 
    algo = rpt.Window(width=windows_width, model=cost)

    # Two worst models
    # algo = rpt.Binseg(model=cost)
    # algo = rpt.BottomUp(model=cost)
    
    for i in range(n_patients):

        patient_true_signal = np.array(estimates['true_value']['cost']['alpha'][i])
        patient_estimates_signal = np.array(estimates['estimates']['cost']['alpha'][i])

        
        changepoint_estimates['true_value']['alpha'][i] = algo.fit_predict(patient_true_signal, n_bkps=process_count - 1)[0]
        changepoint_estimates['estimates']['alpha'][i] = algo.fit_predict(patient_estimates_signal, n_bkps=process_count - 1)[0]

        
    for j in range(n_doctors):

        doctor_true_signal = np.array(estimates['true_value']['cost']['psi'][j])
        doctor_estimates_signal = np.array(estimates['estimates']['cost']['psi'][j]) 
        
        changepoint_estimates['true_value']['psi'][j] = algo.fit_predict(doctor_true_signal, n_bkps=process_count - 1)[0]
        changepoint_estimates['estimates']['psi'][j] = algo.fit_predict(doctor_estimates_signal, n_bkps=process_count - 1)[0]

    
    return changepoint_estimates

def changepoint_accuracy(changepoint_estimates):

    accuracy = {}
    accuracy['exact'] = {}
    accuracy['almost'] = {}
    
    n_patients = len(changepoint_estimates['estimates']['alpha'])
    n_doctors = len(changepoint_estimates['estimates']['psi'])
    patient_exact_estimation = 0
    doctor_exact_estimation = 0
    patient_almost_estimation = 0
    doctor_almost_estimation = 0
    
    for i in range(n_patients):
        
        true_value_patient = changepoint_estimates['true_value']['alpha'][i]
        estimates_value_patient = changepoint_estimates['estimates']['alpha'][i]
        
        if estimates_value_patient == true_value_patient: # Accuracy exacte
            patient_exact_estimation += 1
        
        if estimates_value_patient in np.arange( true_value_patient - 5, true_value_patient + 5): # Intervalle de confiance
            patient_almost_estimation += 1
            
    for j in range(n_doctors):

        true_value_doctor = changepoint_estimates['true_value']['psi'][j]
        estimates_value_doctor = changepoint_estimates['estimates']['psi'][j]
        
        if estimates_value_doctor == true_value_doctor: # Accuracy exacte
            doctor_exact_estimation += 1
            
        
        if estimates_value_doctor in np.arange( true_value_doctor - 5, true_value_doctor + 5): # Intervalle de confiance
            doctor_almost_estimation += 1

    accuracy['exact']['patient'] = patient_exact_estimation / n_patients
    accuracy['exact']['doctor'] = doctor_exact_estimation / n_doctors
    accuracy['almost']['patient'] = patient_almost_estimation / n_patients
    accuracy['almost']['doctor'] = doctor_almost_estimation / n_doctors
    
    return accuracy


def rmse(simulation,
         beta_agep_graph=0.01,
         beta_aged_graph=0.01,
         beta_sexp_graph=0.5,
         beta_sexd_graph=0.5,
         beta_dist_graph=-0.5,
         beta_agep_cost=0.01,
         beta_aged_cost=0.01,
         beta_sexp_cost=0.5,
         beta_sexd_cost=0.5,
         beta_dist_cost=0.5,
        ):
    """
    returns the mean of RMSE of a simulation for patients and doctors
    """
    results = {}
    results['graph'] = {}
    results['graph']['patients'] = []
    results['graph']['doctors'] = []
    results['graph']['beta'] = {}
    results['graph']['beta']['age_p'] = []
    results['graph']['beta']['age_d'] = []
    results['graph']['beta']['sex_p'] = []
    results['graph']['beta']['sex_d'] = []
    results['graph']['beta']['distance'] = []
    results['cost'] = {}
    results['cost']['patients'] = []
    results['cost']['doctors'] = []
    results['cost']['beta'] = {}
    results['cost']['beta']['age_p'] = []
    results['cost']['beta']['age_d'] = []
    results['cost']['beta']['sex_p'] = []
    results['cost']['beta']['sex_d'] = []
    results['cost']['beta']['distance'] = []

    nb_of_periods = len(simulation)
    nb_patients = len(simulation[0]['estimates']['alpha'])
    nb_doctors = len(simulation[0]['estimates']['psi']) # It counts the ghost doctor

    # order_features = "age_p", "age_d", "sex_p", "sex_d", "distance"
    # simulation[t]['estimates']['distance']
    for t in range(nb_of_periods):
        
        s_patients = 0
        s_doctors = 0
        df = simulation[t]['true_value']

        # Compute RMSE for patients at time t
        for p in range(nb_patients):
            
            s_patients += (simulation[t]['estimates']['alpha'][p] - df[df['i'] == p]['alpha'].iloc[0]) ** 2

        # Compute RMSE for doctors at time t
        for d in range(nb_doctors - 1):

            s_doctors += (simulation[t]['estimates']['psi'][d+1] - df[df['j'] == d+1]['psi'].iloc[0]) ** 2

        # Compute RMSE for the betas of the graph formation model
        results['graph']['beta']['age_p'].append( np.sqrt( (simulation[t]['graph']['coeffs'][0] - beta_agep_graph ) ** 2 ) )
        results['graph']['beta']['age_d'].append( np.sqrt( (simulation[t]['graph']['coeffs'][1] - beta_aged_graph ) ** 2 ) )
        results['graph']['beta']['sex_p'].append( np.sqrt( (simulation[t]['graph']['coeffs'][2] - beta_sexp_graph ) ** 2 ) )
        results['graph']['beta']['sex_d'].append( np.sqrt( (simulation[t]['graph']['coeffs'][3] - beta_sexd_graph ) ** 2 ) )
        results['graph']['beta']['distance'].append( np.sqrt( (simulation[t]['graph']['coeffs'][4] - beta_dist_graph ) ** 2 ) )

        #Compute RMSE for the betas of the cost model
        results['cost']['beta']['age_p'].append( np.sqrt( (simulation[t]['estimates']['age_p'] - beta_agep_cost ) ** 2 ) )
        results['cost']['beta']['age_d'].append( np.sqrt( (simulation[t]['estimates']['age_d'] - beta_aged_cost ) ** 2 ) )
        results['cost']['beta']['sex_p'].append( np.sqrt( (simulation[t]['estimates']['sex_p'] - beta_sexp_cost ) ** 2 ) )
        results['cost']['beta']['sex_d'].append( np.sqrt( (simulation[t]['estimates']['sex_d'] - beta_sexd_cost ) ** 2 ) )
        results['cost']['beta']['distance'].append( np.sqrt( (simulation[t]['estimates']['distance'] - beta_dist_cost ) ** 2 ) )

        # Append RMSE of the FE of the graph/cost models    
        results['cost']['patients'].append(np.sqrt(s_patients / nb_patients))
        results['cost']['doctors'].append(np.sqrt(s_doctors / nb_doctors))
        results['graph']['patients'].append ( mean_squared_error(simulation[t]['graph']['alpha'], simulation[t]['graph']['coeffs'][5: 5 + nb_patients], squared=True) )
        results['graph']['doctors'].append ( mean_squared_error(simulation[t]['graph']['psi'][1:], simulation[t]['graph']['coeffs'][5 + nb_patients + 1:], squared=True) ) #We "dodge" the ghost doctor

    mean_rmse_graph_age_p = np.array( results['graph']['beta']['age_p'] ).mean()
    mean_rmse_graph_age_d = np.array( results['graph']['beta']['age_d'] ).mean()
    mean_rmse_graph_sex_p = np.array( results['graph']['beta']['sex_p'] ).mean()
    mean_rmse_graph_sex_d = np.array( results['graph']['beta']['sex_d'] ).mean()
    mean_rmse_graph_distance = np.array( results['graph']['beta']['distance'] ).mean()
    mean_rmse_cost_age_p = np.array( results['cost']['beta']['age_p'] ).mean()
    mean_rmse_cost_age_d = np.array( results['cost']['beta']['age_d'] ).mean()
    mean_rmse_cost_sex_p = np.array( results['cost']['beta']['sex_p'] ).mean()
    mean_rmse_cost_sex_d = np.array( results['cost']['beta']['sex_d'] ).mean()
    mean_rmse_cost_distance = np.array( results['cost']['beta']['distance'] ).mean()

    # For each period t, we have computed the RMSE. So we take the mean of the RMSE based on all periods.
    # FE
    mean_rmse_cost_patients = np.array(results['cost']['patients']).mean()
    mean_rmse_cost_doctors = np.array(results['cost']['doctors']).mean()
    mean_rmse_graph_patients = np.array(results['graph']['patients']).mean()
    mean_rmse_graph_doctors = np.array(results['graph']['doctors']).mean()

    # Betas
    mean_rmse_graph_age_p = np.array( results['graph']['beta']['age_p'] ).mean()
    mean_rmse_graph_age_d = np.array( results['graph']['beta']['age_d'] ).mean()
    mean_rmse_graph_sex_p = np.array( results['graph']['beta']['sex_p'] ).mean()
    mean_rmse_graph_sex_d = np.array( results['graph']['beta']['sex_d'] ).mean()
    mean_rmse_graph_distance = np.array( results['graph']['beta']['distance'] ).mean()
    mean_rmse_cost_age_p = np.array( results['cost']['beta']['age_p'] ).mean()
    mean_rmse_cost_age_d = np.array( results['cost']['beta']['age_d'] ).mean()
    mean_rmse_cost_sex_p = np.array( results['cost']['beta']['sex_p'] ).mean()
    mean_rmse_cost_sex_d = np.array( results['cost']['beta']['sex_d'] ).mean()
    mean_rmse_cost_distance = np.array( results['cost']['beta']['distance'] ).mean()


    # For more clarity, we finally return a dictionary.
    d = {}
    d['cost_patients'] = mean_rmse_cost_patients
    d['cost_doctors'] = mean_rmse_cost_doctors
    d['graph_patients'] = mean_rmse_graph_patients
    d['graph_doctors'] = mean_rmse_graph_doctors
    d['graph_age_p'] = mean_rmse_graph_age_p
    d['graph_age_d'] = mean_rmse_graph_age_d
    d['graph_sex_p'] = mean_rmse_graph_sex_p
    d['graph_sex_d'] = mean_rmse_graph_sex_d
    d['graph_distance'] = mean_rmse_graph_distance
    d['cost_age_p'] = mean_rmse_cost_age_p
    d['cost_age_d'] = mean_rmse_cost_age_d
    d['cost_sex_p'] = mean_rmse_cost_sex_p
    d['cost_sex_d'] = mean_rmse_cost_sex_d
    d['cost_distance'] = mean_rmse_cost_distance
            
    # return [mean_rmse_cost_patients, mean_rmse_cost_doctors, mean_rmse_graph_patients, mean_rmse_graph_doctors]
    return d


def aggreg_rmse(nb_of_simulations,
                nb_periods=200,
                nb_patients=200,
                nb_doctors=200,
                dist=np.sqrt(2),
                beta_agep_graph=0.01,
                beta_aged_graph=0.01,
                beta_sexp_graph=0.5,
                beta_sexd_graph=0.5,
                beta_dist_graph=-0.5,
                beta_agep_cost=0.01,
                beta_aged_cost=0.01,
                beta_sexp_cost=0.5,
                beta_sexd_cost=0.5,
                beta_dist_cost=0.5,
               ):
    
    
    rmse_cost_patients = []
    rmse_cost_doctors = []
    rmse_graph_patients = []
    rmse_graph_doctors = []
    rmse_graph_age_p = []
    rmse_graph_age_d = []
    rmse_graph_sex_p = []
    rmse_graph_sex_d = []
    rmse_graph_distance = []
    rmse_cost_age_p = []
    rmse_cost_age_d = []
    rmse_cost_sex_p = []
    rmse_cost_sex_d = []
    rmse_cost_distance = []
    
    for n in range(nb_of_simulations):

        simulation = temporal_simulation(nb_of_periods=nb_periods,
                                            n_patients=nb_patients,
                                            n_doctors=nb_doctors,
                                            z=dist,
                                            alpha_law_graph=(-1, 0),
                                            psi_law_graph=(-1, 0),
                                            alpha_law_cost=(-1, 0),
                                            psi_law_cost=(-1, 0),
                                            preconditioner = 'ichol',
                                            beta_age_p_graph=beta_agep_graph,
                                            beta_age_d_graph=beta_aged_graph,
                                            beta_sex_p_graph=beta_sexp_graph,
                                            beta_sex_d_graph=beta_sexd_graph,
                                            beta_distance_graph=beta_dist_graph,
                                            beta_age_p_cost=beta_agep_cost,
                                            beta_age_d_cost=beta_aged_cost,
                                            beta_sex_p_cost=beta_sexp_cost,
                                            beta_sex_d_cost=beta_sexd_cost,
                                            beta_distance_cost=beta_dist_cost)
        
        rm = rmse(simulation,
                 beta_agep_graph=beta_agep_graph,
                 beta_aged_graph=beta_aged_graph,
                 beta_sexp_graph=beta_sexp_graph,
                 beta_sexd_graph=beta_sexd_graph,
                 beta_dist_graph=beta_dist_graph,
                 beta_agep_cost=beta_agep_cost,
                 beta_aged_cost=beta_aged_cost,
                 beta_sexp_cost=beta_sexp_cost,
                 beta_sexd_cost=beta_sexd_cost,
                 beta_dist_cost=beta_dist_cost,
                )
        rmse_cost_patients.append( rm['cost_patients'] )
        rmse_cost_doctors.append( rm['cost_doctors'] )
        rmse_graph_patients.append ( rm['graph_patients'] )
        rmse_graph_doctors.append( rm['graph_doctors'] )
        rmse_graph_age_p.append( rm['graph_age_p'] )
        rmse_graph_age_d.append( rm['graph_age_d'] )
        rmse_graph_sex_p.append( rm['graph_sex_p'] )
        rmse_graph_sex_d.append( rm['graph_sex_d'] )
        rmse_graph_distance.append( rm['graph_distance'] )
        rmse_cost_age_p.append( rm['cost_age_p'] )
        rmse_cost_age_d.append( rm['cost_age_d'] )
        rmse_cost_sex_p.append( rm['cost_sex_p'] )
        rmse_cost_sex_d.append( rm['cost_sex_d'] )
        rmse_cost_distance.append( rm['cost_distance'] )

    d = {}
    d['cost_patients'] = rmse_cost_patients
    d['cost_doctors'] = rmse_cost_doctors
    d['graph_patients'] = rmse_graph_patients
    d['graph_doctors'] = rmse_graph_doctors
    d['graph_age_p'] = rmse_graph_age_p
    d['graph_age_d'] = rmse_graph_age_d
    d['graph_sex_p'] = rmse_graph_sex_p
    d['graph_sex_d'] = rmse_graph_sex_d
    d['graph_distance'] = rmse_graph_distance
    d['cost_age_p'] = rmse_cost_age_p
    d['cost_age_d'] = rmse_cost_age_d
    d['cost_sex_p'] = rmse_cost_sex_p
    d['cost_sex_d'] = rmse_cost_sex_d
    d['cost_distance'] = rmse_cost_distance

    # return [rmse_cost_patients, rmse_cost_doctors, rmse_graph_patients, rmse_graph_doctors]
    return d

def graph_formation(n_patients,
                    n_doctors,
                    z=1.4,
                    beta_age_p_graph=0.01,
                    beta_age_d_graph=0.01,
                    beta_sex_p_graph=0.5,
                    beta_sex_d_graph=0.5,
                    beta_distance_graph=-0.5,
                    alpha_law_graph=(-1, 0),
                    psi_law_graph=(-1, 0)
                   ):
    """
    Crée seulement la partie formation de graphe et retourne le dataframe / valeurs des EF (utile pour passer à la régression logistique et estimation des EF / Beta du graph formation seulement)
    """
    coor_patients = []
    coor_doctors = []
    alpha_graph = []
    psi_graph = []
    rng = np.random.default_rng(None)
    D = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)

    for i in range(n_patients):
        
        # We generate the FE for the graph formation model
        alpha_graph.append( np.random.uniform(alpha_law_graph[0], alpha_law_graph[1]) )

        # Generate the coordinates of the patients
        coor_patients.append( np.random.uniform(0, 1, 2) )
                               
    for j in range(n_doctors + 1):

        # We generate the FE for the graph formation model
        psi_graph.append( np.random.uniform(psi_law_graph[0], psi_law_graph[1]) )
        
        if j != 0:
            
            # Generate the coordinates of the doctors
            coor_doctors.append( np.random.uniform(0, 1, 2) )

    # Generate distance matrix
    for i in range(n_patients):
        for j in range(0, n_doctors + 1):
            if j == 0: # We associate the indice 0 to the "ghost doctor"
                D[i][0] = 0
            else: # we take the j-1 index of coor_doctors as we added the ghost doctor, j = 1 corresponds to j = 0 in coord_doctors
                d = np.sqrt(np.power((coor_patients[i][0] - coor_doctors[j-1][0]), 2) + np.power((coor_patients[i][1] - coor_doctors[j-1][1]), 2))
                D[i][j] = d

    # Random draws of ages for patients and doctors
    sim_patient_age = rng.integers(low = 1, high = 99, size = n_patients)
    sim_doctor_age = rng.integers(low = 26, high = 99, size = n_doctors + 1)

    # Random draws of genders of patients and doctors
    sim_patient_gender = rng.integers(low = 0, high = 2, size = n_patients)
    sim_doctor_gender = rng.integers(low = 0, high = 2, size = n_doctors + 1)

    # Compile ids
    id_p = np.repeat(range(n_patients), n_doctors + 1)
    id_d = np.tile(range(n_doctors + 1), n_patients)

    # Compile fixed effects
    # alp_data = np.repeat(alpha_cost, n_doctors + 1)
    # psi_data = psi_graph * n_patients

    # Compile observed features
    age_p_data = np.repeat(sim_patient_age, n_doctors + 1)
    age_d_data = np.tile(sim_doctor_age, n_patients)
    sex_p_data = np.repeat(sim_patient_gender, n_doctors + 1)
    sex_d_data = np.tile(sim_doctor_gender, n_patients)

    estimates = []
                               
    # At each period, determine connections                           
    # for t in range(nb_of_periods):
    
    # Generate the identifier matrix A based on the distance
    A = np.zeros([n_patients, n_doctors + 1], dtype = np.ndarray)
    for i in range(0, n_patients):
        for j in range(0, n_doctors + 1):
            if j == 0:
                A[i][0] = 1
            elif D[i][j] > z: # if patient i and doctor j are too far away, there is no relation
                continue
            else:
                T = alpha_graph[i] + psi_graph[j] + beta_age_p_graph * sim_patient_age[i] + beta_age_d_graph * sim_doctor_age[j] + beta_sex_p_graph * sim_patient_gender[i] + beta_sex_d_graph * sim_doctor_gender[j] + beta_distance_graph * D[i][j]
                p = 1 / (1 + np.exp(-T))
                A[i][j] = np.random.binomial(1, p)

    # Compile relations between doctors and patients
    relation = A.flatten()

    # Merge all columns into a dataframe
    dataframe = pd.DataFrame(data={'i': id_p, 'j': id_d, 'y' : relation, 'age_p': age_p_data, 'age_d': age_d_data, 
                           'sex_p': sex_p_data, 'sex_d': sex_d_data
                            })
    dataframe['distance'] = D[dataframe['i'], dataframe['j']].astype(float)

    # cancel connections between patient i and ghost doctor if patient i isn't only connected to the ghost doctor
    # number_of_connections = dataframe['i'].value_counts(sort=None)
    # for i in range(n_patients):

    #     if number_of_connections[i] > 1: # if patient i isn't only connected to the ghost doctor, we remove its connection with the ghost doctor.
    
    #         index_to_drop = dataframe[dataframe['i'] == i].index[0] # we get the index of the row to drop
    #         dataframe = dataframe.drop(index_to_drop)

    dataframe = dataframe.reset_index().drop(['index'], axis = 1)
    return (dataframe, alpha_graph, psi_graph)