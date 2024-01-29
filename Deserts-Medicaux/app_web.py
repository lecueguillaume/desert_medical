import numpy as np
from scipy import optimize
import pandas as pd
from scipy.optimize import root
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from fixed_effects.simulation import *
from pandas.api.types import is_numeric_dtype
import warnings


def Prob_eval_SFCA2(d):
    """
    Calcule la probabilité qu'un patient d'une commune i visite un médecin d'une commune j
    en utilisant le modèle SFCA2.

    Args:
        d (numpy.array): Une matrice de distances. L'élément [i, j] représente la distance
            entre la commune i et la commune j.

    Returns:
        Prob (numpy.array): Une matrice de probabilités de même dimension que d. L'élément [i, j] 
            représente la probabilité qu'un patient de la commune i visite un médecin de la commune j.
    """
    W = 1.0 / d
    sum_W = np.sum(W, axis=1, keepdims=True)
    sum_W [sum_W <= 1e-7]+= 1e-4
    
    Prob = W /sum_W
    
    return Prob


def Prob_eval_SFCA3(d, S):
    """
    Calcule la matrice de probabilité qu'un patient d'une commune i visite un médecin 
    d'une commune j en utilisant le modèle SFCA3.
    La fonction prend comme entrée une matrice de distances, un vecteur de demandes d'offres 
    par commune, et un vecteur d'offres de soins par commune. Elle retourne une matrice 
    de probabilités correspondante.

    Args:
        d (numpy.array): Une matrice de distances. L'élément [i, j] représente 
            la distance entre la commune i et la commune j.
        S (numpy.array): Un vecteur de l'offre de soins pour chaque commune j.

    Returns:
        Prob (numpy.array): Une matrice de probabilités de même dimension que d. L'élément [i, j] 
            représente la probabilité qu'un patient de la commune i visite un médecin de la commune j.
    """
    W = 1.0 / d
    WS = W * S
    WS_sum = np.sum(WS, axis=1, keepdims=True)
    WS_sum [WS_sum <= 1e-7]+= 1e-4
    Prob = WS / WS_sum
    return Prob

def calc_Rj(P, Prob, S):
    """
    Calcule le vecteur R où chaque élément R[j] est donné par la formule :
    R_j = S_j / (sum_i P_i * Prob_{i,j}).


    Args:
        P (numpy.array): Un vecteur de la demande d'offre pour chaque commune.
        Prob (numpy.array): Une matrice de probabilités. L'élément [i, j] représente 
            la probabilité qu'un patient de la commune i visite un médecin de la commune j.
        S (numpy.array): Un vecteur de l'offre de soins pour chaque commune.

    Returns:
        numpy.array: Un vecteur R. Chaque élément R[j] est donné par la formule : 
            R_j = S_j / (sum_i P_i * Prob_{i,j}).
    """
    sum_prob_p = np.sum(P[:, np.newaxis] * Prob, axis=0)
    sum_prob_p [sum_prob_p==0]+= 1e-4
    R = S / sum_prob_p
    return R


def calc_Prob_ij(R, W):
    """
    Calcule la matrice de probabilités Prob où chaque élément Prob[i, j] selon le modele du point fixe donné par la formule :
    Prob[i, j] = R[j] * W[i, j] / (sum_k R[k] * W[i, k]).

    Args:
        R (numpy.array): Un vecteur des offres totales par patients potentiels pour chaque commune.
        W (numpy.array): Une matrice de perméabilité. L'élément [i, j] représente la perméabilité 
            entre la commune i et la commune j.

    Returns:
        numpy.array: Une matrice de probabilités de même dimension que W. L'élément [i, j] 
            représente la probabilité.
    """
    RW = R * W
    Prob = RW / np.sum(RW, axis=1, keepdims=True)
    return Prob

def F(R, Prob, W, P, S):
    """
    Calcule le nouveau R et Prob en utilisant les fonctions calc_Rj et calc_Prob_ij.

    Args:
        R (numpy.array): Un vecteur des offres totales par patients potentiels pour chaque commune.
        Prob (numpy.array): Une matrice de probabilités.
        W (numpy.array): Une matrice des coefficients de perméabilité.
        P (numpy.array): Un vecteur des demandes (nombre de patients dans la région i).
        S (numpy.array): Un vecteur des offres (par exemple, le nombre d'heures de travail des médecins dans la région j).

    Returns:
        R_new (numpy.array): Le nouveau vecteur R calculé par la fonction calc_Rj.
        Prob_new (numpy.array): La nouvelle matrice de probabilités calculée par la fonction calc_Prob_ij.
    """
    Prob_new = calc_Prob_ij(R, W)
    R_new = calc_Rj(P, Prob, S)
    return R_new, Prob_new


def Point_fixe_SFCA(W, S, P, maxiter = 10000):
    """
    Implémentation de l'algorithme pour calculer le vecteur R et la matrice Prob avec la méthode du point fixe.

    Args:
        W (numpy.array): Une matrice des coefficients de perméabilité.
        S (numpy.array): Un vecteur des offres (par exemple, le nombre d'heures de travail des médecins dans la région j).
        P (numpy.array): Un vecteur des demandes (nombre de patients dans la région i).
        maxiter (int): Le nombre l'itération maximale.

    Returns:
        R (numpy.array): Un vecteur des offres totales par patients potentiels.
        Prob (numpy.array): Une matrice de probabilité de connexions.
        errors (numpy.array): Un tableau des erreurs de convergence à chaque itération.
    """

    R = np.ones_like(S)
    errors = np.zeros(maxiter)

    for i in range(maxiter):
        
        P_prime = R * W
        P_prime_row_sum = np.sum(P_prime, axis=1, keepdims=True)
        P_prime_row_sum [P_prime_row_sum <= 1e-7]+= 1e-4
        Prob = P_prime / P_prime_row_sum

        PT_Prob = np.dot(P.T, Prob)
        PT_Prob [PT_Prob <= 1e-7]+= 1e-4
        R_new = S / PT_Prob
        error = np.linalg.norm(R_new - R)
        errors[i] = error
        R = R_new

    return R, Prob, errors

def find_fixed_point(F, W, P, S, max_iter=10000):
    """
    Trouve le point fixe de la fonction F.

    Args:
        F (function): La fonction pour laquelle trouver le point fixe.
        W (numpy.array): Une matrice des coefficients de perméabilité.
        P (numpy.array): Un vecteur des demandes (nombre de patients dans la région i).
        S (numpy.array): Un vecteur des offres (par exemple, le nombre d'heures de travail des médecins dans la région j).
        tol (float): La tolérance pour la convergence.
        max_iter (int): Le nombre maximal d'itérations.

    Returns:
        R (numpy.array): Le vecteur R au point fixe.
        Prob (numpy.array): La matrice de probabilités au point fixe.
    """

    R = np.ones_like(S)
    Prob = np.random.rand(len(P), len(S))

    for _ in range(max_iter) :
        R_new, Prob_new = F(R, Prob, W, P, S)
        R = R_new
        Prob = Prob_new


    return R, Prob

def F_diff(X, W, P, S):
    R, Prob = X[:len(S)], X[len(S):].reshape(W.shape)
    R_new, Prob_new = F(R, Prob, W, P, S)
    return np.concatenate([R_new - R, (Prob_new - Prob).ravel()])

def find_fixed_point_2(W, P, S, tol=1e-10, max_iter=1000):
    R_init = np.ones_like(S)
    Prob_init = np.random.rand(len(P), len(S))
    X_init = np.concatenate([R_init, Prob_init.ravel()])
    sol = root(F_diff, X_init, args=(W, P, S))
    if sol.success:
        X_fixed = sol.x
        R_fixed, Prob_fixed = X_fixed[:len(S)], X_fixed[len(S):].reshape(W.shape)
        return R_fixed, Prob_fixed
    else:
        print("La méthode n'a pas convergé.")



def deserts_medicaux_FCA(d, communes, S, P, model="SFCA3", seuil  = 0.1, error = False):
    """
    Détermine si chaque commune est un désert médical ou non en utilisant l'un des trois modèles.

    Args:
        d (numpy.array): Matrice de distances entre les communes.
        S (numpy.array): Vecteur des offres de soins pour chaque commune.
        P (numpy.array): Vecteur des demandes de chaque commune.
        model (str): Modèle à utiliser pour calculer les probabilités. Par défaut, "SFCA3".
        communes : les codes des communes (ou de départements)
        Seuil : Float, le sueil de décision
        Error (bool): Indique si les erreurs pendant l'algorithme du point fixe doivent ou pas être renvoyées. Par défaut, False.

    Returns:
        dataframe : contenant le code de chaque commune, le nb de medecins, le nb de population, son indicateur FCA, et si oui ou non c'est un desert medical .
    """
    R_calcule = False
    # Calcul des probabilités selon le modèle spécifié
    if model == "SFCA2":
        Prob = Prob_eval_SFCA2(d)
    elif model == "SFCA3":
        Prob = Prob_eval_SFCA3(d, S)
    elif model =="point fixe" :
        R_calcule = True
        R, Prob, errors = Point_fixe_SFCA(1.0/d, S, P, maxiter= 1000)
        #R, Prob = find_fixed_point_2(1.0/d, P, S)
    else:
        raise ValueError("Modèle non valide. Veuillez choisir parmi 'point fixe', 'SFCA2', ou 'SFCA3'.")

    if R_calcule == False :
        R = calc_Rj(P, Prob, S)
    # Calcul de la somme Pij * Rj pour chaque commune i
    sum_PR = np.sum(Prob * R, axis=1)
    sort_indices = np.argsort(sum_PR)
    FCA = np.zeros_like(sum_PR, dtype=bool)
    seuil_index = int(len(sort_indices) * seuil)
    FCA[sort_indices[:seuil_index + 1]] = True
    df = pd.DataFrame({
    'CODGEO\DEP': communes,
    'medecins': S,
    'population': P,
    'FCA' : sum_PR,
    'desert_medical' : FCA
    })
    
    if (model =="point fixe") and (error) :
        return df, errors
    else :
        return df




#! pip install openpyxl
@st.cache_data
def load():
    '''
    Fonction pour importer nos données
    '''
    #! pip install openpyxl
    distancier_reg = pd.read_csv("data/distancier_sur_reg.csv", sep=";")
    distancier_dep = pd.read_csv("data/distancier_sur_dep.csv", sep=";")
    distancier_com = pd.read_csv("data/distancier_sur_com.csv", sep=";")
    population = pd.read_excel("data/Medecins.xlsx")
    medecins = pd.read_excel("data/Population.xlsx")
    sf_dep = gpd.read_file('data/departements-version-simplifiee.geojson')
    sf_com = gpd.read_file('data/communes.geojson')
    sf_reg = gpd.read_file('data/regions-version-simplifiee.geojson')

    return distancier_reg, distancier_dep, distancier_com, population, medecins, sf_reg, sf_dep, sf_com


def negdist_clean(distancier):
    """
    Supprime les lignes du distancier contenant des distances négatives basées sur le nombre d'occurrences
    de ces distances pour chaque idSrc et idDst.

    Args:
        distancier (pandas.DataFrame): Le DataFrame contenant les colonnes idSrc, idDst et distance.

    Returns:
        pandas.DataFrame: Le DataFrame distancier filtré sans les lignes contenant des distances négatives
        basées sur le seuil défini.

    """
    # Calculer le seuil
    seuil = np.sqrt(len(distancier)) / 2

    # Compter les occurrences de distances négatives pour idSrc
    occurrences_idSrc = distancier[distancier['distance'] < 0]['idSrc'].value_counts()
    idsrc_a_supprimer = occurrences_idSrc[occurrences_idSrc > seuil].index

    # Compter les occurrences de distances négatives pour idDst
    occurrences_idDst = distancier[distancier['distance'] < 0]['idDst'].value_counts()
    iddst_a_supprimer = occurrences_idDst[occurrences_idDst > seuil].index

    # Supprimer les lignes correspondantes
    distancier = distancier[~distancier['idSrc'].isin(idsrc_a_supprimer) & ~distancier['idDst'].isin(iddst_a_supprimer)].dropna()
    return distancier


def clean_distancier(distancier):
    """
    Nettoie le DataFrame distancier en effectuant les opérations suivantes :
    - Sélectionne les colonnes idSrc, idDst et distance.
    - Remplace les virgules par des points dans la colonne distance et la convertit en type float.
    - Remplace les distances nulles par 1.
    - Convertit les colonnes idSrc et idDst en format numérique.
    - Applique la fonction negdist_clean() pour supprimer les lignes contenant des distances négatives.
    
    Args:
        distancier (pandas.DataFrame): Le DataFrame contenant les colonnes idSrc, idDst et distance.

    Returns:
        pandas.DataFrame: Le DataFrame distancier nettoyé.

    """

    distancier = distancier[["idSrc", "idDst", "distance"]]
    if not(is_numeric_dtype(distancier["distance"])) :
        distancier["distance"] = distancier["distance"].str.replace(',', '.').astype(float)
    # remplacer les distances nulles par 1
    distancier.loc[distancier['distance'] == 0, 'distance'] = 1
    #le codgeo ou code departement doit etre sous format numerique
    distancier['idSrc'] = pd.to_numeric(distancier['idSrc'], errors='coerce')
    distancier['idDst'] = pd.to_numeric(distancier['idDst'], errors='coerce')
    distancier = negdist_clean(distancier)
    return distancier


def clean_population(population):
    """
    Nettoie le DataFrame population en effectuant les opérations suivantes :
    - Renomme les colonnes en utilisant la ligne 4 comme noms de colonnes.
    - Sélectionne les colonnes correspondantes aux informations souhaitées.
    - Supprime les premières lignes non pertinentes.
    - Renomme la colonne PMUN20 en 'population'.
    - Convertit la colonne CODGEO en format numérique.
    - Supprime les lignes contenant des valeurs manquantes pour la colonne CODGEO.
    - Convertit la colonne population en type float.

    Args:
        population (pandas.DataFrame): Le DataFrame contenant les informations sur la population.

    Returns:
        pandas.DataFrame: Le DataFrame population nettoyé.

    """
    columns = population.iloc[4]
    population.columns = columns
    population = population[columns[:5]]
    population = population.iloc[5:]
    population = population.rename(columns={'PMUN20': 'population'})
    population['CODGEO'] = pd.to_numeric(population['CODGEO'], errors='coerce')
    population = population.dropna(subset=['CODGEO'])
    population["population"] = population["population"].astype(float)

    return population


def clean_medecins(medecins, spe = "Médecin généraliste") :
    """
    Nettoie le DataFrame medecins en effectuant les opérations suivantes :
    - Renomme les colonnes en utilisant la ligne 3 comme noms de colonnes.
    - Sélectionne les colonnes correspondantes aux informations souhaitées.
    - Supprime les premières lignes non pertinentes.
    - Renomme la colonne spécifiée par 'spe' en 'medecins'.
    - Sélectionne les colonnes spécifiées pour le filtrage (nombre de medecins de la specialité, département, codgeo...).
    - Convertit la colonne CODGEO en format numérique.
    - Supprime les lignes contenant des valeurs manquantes pour la colonne CODGEO.
    - Convertit la colonne medecins en type float.

    Args:
        medecins (pandas.DataFrame): Le DataFrame contenant les informations sur les médecins.
        spe (str): La spécialité de médecin à considérer. Par défaut, "Médecin généraliste".

    Returns:
        pandas.DataFrame: Le DataFrame medecins nettoyé.

    """
    columns = medecins.iloc[3]
    columns[:4] = medecins.iloc[4][:4]
    medecins.columns = columns
    medecins = medecins.iloc[5:]
    medecins = medecins.rename(columns={spe: 'medecins'})
    filtre = list(columns[:4])
    filtre.append("medecins")
    medecins = medecins[filtre]
    medecins['CODGEO'] = pd.to_numeric(medecins['CODGEO'], errors='coerce')
    medecins = medecins.dropna(subset=['CODGEO'])
    medecins["medecins"] = medecins["medecins"].astype(float)

    return medecins

def replace_codgeo(code):
    """
    Remplace les valeurs de la colonne 'CODGEO' dans le DataFrame 'population' 
    Cette fonction prend en argument un code géographique (int) et vérifie si le code divisé par 1000 est égal à 75 (ie code postale d'un arrondissement de Paris).
    Si la condition est vraie, la fonction renvoie la valeur 75056 (le codgeo de Paris), sinon elle renvoie le code d'origine.

    Args:
        code (int): Le code géographique à remplacer.

    Returns:
        int: Le nouveau code géographique après remplacement selon la condition.

    """
    if int(code / 1000) == 75:
        return 75056
    return code

def extract_simulation(simulation):
    data = simulation[0]['true_value'][["i", "j", "code_patient", "code_doctor", "alpha", "psi"]]
    population = data[[ "i", "code_patient", "alpha"]]
    medecins = data[[ "j", "code_doctor", "psi"]]
    population.drop_duplicates(inplace = True)
    medecins.drop_duplicates(inplace = True)
    medecins.rename(columns = {"code_doctor": "CODGEO", "psi" : "medecins"}, inplace = True)
    population.rename(columns = {"code_patient": "CODGEO", "alpha" : "population"}, inplace = True)
    medecins = medecins[["CODGEO", "medecins"]].groupby('CODGEO', as_index= False).sum()
    population = population [["CODGEO", "population"]].groupby('CODGEO', as_index= False).sum()

    return medecins, population

@st.cache_data
def deserts_medicaux(distancier, population, medecins, clean = False, spe = "Médecin généraliste", scale = "DEP", model = "SFCA2", approche="Naive") : 

    """
    Calcule les déserts médicaux en utilisant les données du distancier, de la population et des médecins en fonction du model choisi.
    
    Args:
        distancier (pandas.DataFrame): Le DataFrame contenant les informations sur les distances entre les codgeo.
        population (pandas.DataFrame): Le DataFrame contenant les informations sur la population par codgeo.
        medecins (pandas.DataFrame): Le DataFrame contenant les informations sur les médecins par codgeo.
        clean (bool): Indique si les données doivent être nettoyées avant le calcul. Par défaut, False.
        spe (str): La spécialité de médecin à considérer lors du nettoyage des données. Par défaut, "Médecin généraliste".
        scale (str): L'échelle de regroupement des données (département, commune, etc.). Par défaut, "DEP".
        model (str): Le modèle à utiliser pour le calcul des déserts médicaux. Par défaut, "SFCA2".
        approche (str) : l'approche utilisée pour la quantité de soins servie et demandée. Par default "Naive" qui utilise le nombre de medecins et de population
    
    Returns:
        pandas.dataframe: Un dataframe indiquant les déserts médicaux ou non avec son indicateur FCA pour chaque échelle spécifiée.
    
    """
    if not(clean) :
        distancier = clean_distancier(distancier)
        population = clean_population(population)
        if approche == "Naive" :
            medecins = clean_medecins(medecins, spe = spe) 
        else :
            #relier les codes departement, codes regions et codes communes (codgeo)
            reg_dep_com = population[["REG", "DEP", "CODGEO", "LIBGEO"]]
            simulation = temporal_simulation(nb_of_periods=1,
                                    n_patients=600,
                                    n_doctors=60,
                                    z=1.1,
                                    alpha_law_graph=(-1, 0),
                                    psi_law_graph=(-1, 0),
                                    alpha_law_cost=(-1, 0),
                                    psi_law_cost=(-1, 0),
                                    preconditioner = 'ichol',
                                    beta_age_p_graph=0.01,
                                    beta_age_d_graph=0.01,
                                    beta_sex_p_graph=0.5,
                                    beta_sex_d_graph=0.5,
                                    beta_distance_graph=-0.5,
                                    beta_age_p_cost=0.01,
                                    beta_age_d_cost=0.01,
                                    beta_sex_p_cost=0.5,
                                    beta_sex_d_cost=0.5,
                                    beta_distance_cost=0.5,
                                    type_distance = scale)
            medecins, population = extract_simulation(simulation)
            medecins = medecins.merge(reg_dep_com, on="CODGEO", how='inner')
            population = population.merge(reg_dep_com, on="CODGEO", how='inner')
            print('okkkkkkkkkk!!!?')



    # Remplacer les code postale de paris par un code 
    population['CODGEO'] = population['CODGEO'].apply(replace_codgeo)
    ## Assembler la population et les médecins selon les départements
    medecins_scale = medecins.groupby(scale)['medecins'].sum().reset_index()
    population_scale = population.groupby(scale)['population'].sum().reset_index()

    # Jointure entre df_medecins et df_population
    df_merge1 = medecins_scale.merge(population_scale, on=scale, how='inner')

    #relier chaque codgeo de chaque lieu à son numero de scales pour représenter ce scale
    scale_CODGEO = medecins.merge(distancier, left_on='CODGEO', right_on='idSrc', how='inner')[[scale,'CODGEO']].drop_duplicates()
    if scale == "CODGEO" :
        scale_CODGEO.columns=["CODGEO", "A"]
    codgeo_list = scale_CODGEO["CODGEO"]
    df_filtered = distancier[distancier['idSrc'].isin(codgeo_list)]
    df_final = df_filtered[df_filtered['idDst'].isin(codgeo_list)]

    # On va remplacer les codgeo des communes chef lieux des scale(departement ou communeou region) par les numéro des scales respectifs 
    # Remplacer les valeurs des colonnes 'idsrc' et 'iddist' par les numéros de scale correspondants
    if not(scale == "CODGEO") : 
        codgeo_scale_dict = scale_CODGEO.set_index('CODGEO')[scale].to_dict()
        df_final['idSrc'] = df_final['idSrc'].map(codgeo_scale_dict)
        df_final['idDst'] = df_final['idDst'].map(codgeo_scale_dict)

    # Créer notre matrice de distance 
    # De plus on remarque que cela est possible car la taille de notre dataframe doit etre un carré
    matrix = df_final.pivot(index='idSrc', columns='idDst', values='distance')
    common_labels = matrix.columns.intersection(matrix.index)
    d_df = matrix.reindex(index=common_labels, columns=common_labels)
    d = d_df.values

    ## Extraire nos vecteurs P = (P_i)_i et S = (S_j)_j 
    filter = df_final['idSrc'].unique()
    S_P = df_merge1[df_merge1[scale].isin(filter)]
    S = S_P["medecins"].values
    P = S_P["population"].values
    # Appliquer notre modele
    is_desert = deserts_medicaux_FCA(d, S_P[scale].values , S, P, model=model)

    if scale == "CODGEO":
        is_desert["CODGEO\DEP"] = (is_desert["CODGEO\DEP"].astype(int)).astype(str)

    return is_desert





def visualiser_deserts_medicaux_carte(is_desert, sf):
    """
    Affiche une carte représentant les déserts médicaux à partir du tableau is_desert et du fichier shape sf.

    Args:
        is_desert (pandas.dataframe): Tableau indiquant les déserts médicaux pour chaque échelle spécifiée.
        sf (shapefile.Reader): Fichier shape contenant les informations géographiques des régions.

    """

    jf = sf.merge(is_desert, left_on='code', right_on='CODGEO\DEP', suffixes=('','_y'))

    fig = px.choropleth(jf, geojson=jf.geometry, locations=jf.index, color='FCA',
                        color_continuous_scale='RdYlBu', range_color=(jf['FCA'].max(), jf['FCA'].min()),
                        hover_data=['nom', 'code', 'medecins', 'population', 'desert_medical'])

    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>' +
                                    'Code: %{customdata[1]}<br>' +
                                    'Médecins: %{customdata[2]}<br>' +
                                    'Population: %{customdata[3]}<br>' +
                                    'Desert Medical: %{customdata[4]}<extra></extra>')

    st.plotly_chart(fig)


@st.cache_data
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_nw19osms.json")
st.title("Application de déserts médicaux")
left_column, right_column = st.columns(2)
with left_column:
    st.markdown("""
Cette application utilise des modèles d'évaluation pour identifier les zones où l'accès aux soins médicaux est limité en raison de la pénurie de médecins.

Cette application nous permettra d'évaluer les déserts médicaux en utilisant différentes approches basées sur les données de population, les distances géographiques et le nombre de médecins disponibles.
                
Pour plus de détails sur les méthodes utilisées et les données analysées, vous pouvez consulter le résumé des méthodes et données.
""")
with open("Rapport.pdf", "rb") as f:
    data = f.read()
st.download_button(label="Télécharger le résumé des méthodes et données", data=data, file_name="Rapport.pdf", mime="application/pdf")
with right_column:
    st_lottie(lottie_coding, height=300, key="coding")



distancier_reg, distancier_dep, distancier_com, population, medecins, sf_reg, sf_dep, sf_com = load()
# Sélection des configurations
spe = st.sidebar.selectbox("Spécialité :", list(medecins.iloc[3])[4:])  # Ajoutez les autres spécialités disponibles
scale = st.sidebar.selectbox("Échelle :", ["DEP", "REG", "CODGEO"])  # Ajoutez les autres échelles disponibles
model = st.sidebar.selectbox("Modèle :", ["SFCA2", "SFCA3", "point fixe"])  # Ajoutez les autres modèles disponibles
approche = st.sidebar.selectbox("Approche :", ["Naive", "Effets fixes"])  # Ajoutez les autres modèles disponibles

if scale == "DEP":
    distancier = distancier_dep
    sf = sf_dep
if scale == "REG":
    distancier = distancier_reg
    sf = sf_reg
if scale =="CODGEO":
    distancier = distancier_com
    sf = sf_com

# Bouton pour exécuter l'analyse
if st.sidebar.button("Exécuter"):
    # Appeler les fonctions correspondantes avec les configurations sélectionnées
    st.subheader(f"Carte des déserts médicaux pour {spe} :")
    is_desert = deserts_medicaux(distancier, population, medecins, model=model, spe=spe, clean=False, scale=scale, approche= approche)
    visualiser_deserts_medicaux_carte(is_desert, sf)
   

