"""ETGEMs_function.py

The code in this file reflects the Pyomo Concretemodel construction method of constrainted model. On the basis of this file, with a little modification, you can realize the constraints and object switching of various constrainted models mentioned in our manuscript.

"""

# IMPORTS
# External modules
import cobra
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from equilibrator_api import ComponentContribution, Q_
from cobra.core import Reaction
from cobra.util.solver import set_objective
import re
from equilibrator_api import ComponentContribution, Q_
from scipy.stats import pearsonr,linregress
from typing import Dict,Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime


def get_reaction_g0(model, p_h, p_mg, ionic_strength, temperature):
    """get ΔG'° use equilibrator_api.
    Arguments
    ----------
    * model: the dictionary of model
    * p_h: the pH of the environment
    * p_mg: the pMg of the environment
    * ionic_strength: the ionic strength of the environment
    * temperature: the temperature of the environment
    """
    columns = ['reaction_id', 'equation', 'gpr', 'g0']
    result_g0_df = pd.DataFrame(columns=columns)
    cc = None
    while cc is None:
        try:
            cc = ComponentContribution()
        except JSONDecodeError:
            logger.warning('Waiting for zenodo.org... Retrying in 5s')
            sleep(5)

    #cc = ComponentContribution()
    cc.p_h = Q_(p_h)
    cc.p_mg = Q_(p_mg)
    cc.ionic_strength = Q_(ionic_strength)
    cc.temperature = Q_(temperature)
    # reaction_g0={}
    for eachr in model.reactions:
        if not eachr.id.startswith("EX_") and not eachr.id.startswith("DM_"):
            reaction_left=[]
            reaction_right=[]
            for k, v in eachr.metabolites.items():
                if str(k.id).endswith('_c'):
                    k_new = "_c".join(str(k.id).split('_c')[:-1])
                elif str(k.id).endswith('_p'):
                    k_new = "_p".join(str(k.id).split('_p')[:-1])
                elif str(k.id).endswith('_e'):
                    k_new = "_e".join(str(k.id).split('_e')[:-1]) 
                else:
                    k_new = str(k.id)[:-2]
                #kegg,chebi,metanetx;kegg:C00002 + CHEBI:15377 = metanetx.chemical:MNXM7 + bigg.metabolite:pi
                if v<0:                    
                    reaction_left.append(str(-v)+' bigg.metabolite:'+k_new)
                else:
                    reaction_right.append(str(v)+' bigg.metabolite:'+k_new)
            reaction_equ=(' + ').join(reaction_left)+' -> '+(' + ').join(reaction_right)
            #get ΔG'° use equilibrator_api
            try:
                equilibrator_api_reaction = cc.parse_reaction_formula(reaction_equ)
                #print("The reaction is " + ("" if equilibrator_api_reaction.is_balanced() else "not ") + "balanced")
                dG0_prime = cc.standard_dg_prime(equilibrator_api_reaction)
            except:
                pass
                # print('error reaction: '+eachr.id)
                # print(reaction_equ)
            else:  
                # reaction_g0[eachr.id]={}
                # reaction_g0[eachr.id]['reaction']=eachr.id
                # reaction_g0[eachr.id]['equ']=eachr.reaction
                # #dG0 (kilojoule/mole)
                # reaction_g0[eachr.id]['g0']=str(dG0_prime).split(') ')[0].split('(')[1].split(' +/- ')[0]
                dG0_prime = float(str(dG0_prime).split(') ')[0].split('(')[1].split(' +/- ')[0])
            result_row = {'reaction_id' : eachr.id, 'equation' : eachr.build_reaction_string()
                        ,'g0' : dG0_prime, 'gpr' : eachr.gene_reaction_rule}       
            result_g0_df = pd.concat([result_g0_df, pd.DataFrame(result_row, index=[0])], ignore_index=True)
    return result_g0_df

#Extracting information from GEM (iML1515 model)
def Get_Model_Data_old(model):
    """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
    Notes: 
    ----------
    *model： is in SBML format (.xml).
    """
    reaction_list=[]
    metabolite_list=[]
    lb_list={}
    ub_list={}
    coef_matrix={}
    for rea in model.reactions:
        reaction_list.append(rea.id)
        lb_list[rea.id]=rea.lower_bound
        ub_list[rea.id]=rea.upper_bound
        for met in model.metabolites:
            metabolite_list.append(met.id)
            try:
                rea.get_coefficient(met.id)  
            except:
                pass
            else:
                coef_matrix[met.id,rea.id]=rea.get_coefficient(met.id)
    # print(ub_list)
    reaction_list=list(set(reaction_list))
    metabolite_list=list(set(metabolite_list))
    return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)
def Get_Model_Data(model):
    """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
    Notes: 
    ----------
    *model： is in SBML format (.xml).
    """
    reaction_list=[]
    metabolite_list=[]
    lb_list={}
    ub_list={}
    coef_matrix={}
    for rea in model.reactions:
        reaction_list.append(rea.id)
        
        if rea.lower_bound+rea.upper_bound<0:
            # print(str(rea.id) + ' has a negative flux range')
            # print('ub is : ' + str(rea.upper_bound))
            # print('lb is : ' + str(rea.lower_bound))
            lb_list[rea.id]=rea.lower_bound+1000
            ub_list[rea.id]=rea.upper_bound+1000
            for met in model.metabolites:
                metabolite_list.append(met.id)
                try:
                    rea.get_coefficient(met.id)
                    # print(rea.get_coefficient(met.id))
                except:
                    pass
                else:
                    coef_matrix[met.id,rea.id]=-rea.get_coefficient(met.id)
                    # print(-rea.get_coefficient(met.id))
        else:
            lb_list[rea.id]=rea.lower_bound
            ub_list[rea.id]=rea.upper_bound                
            for met in model.metabolites:
                metabolite_list.append(met.id)
                try:
                    rea.get_coefficient(met.id)  
                except:
                    pass
                else:
                    coef_matrix[met.id,rea.id]=rea.get_coefficient(met.id)
    reaction_list=list(set(reaction_list))
    metabolite_list=list(set(metabolite_list))
    return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)

def convert_to_irreversible(model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object which will be modified in place.

    """
    #warn("deprecated, not applicable for optlang solvers", DeprecationWarning)
    reactions_to_add = []
    coefficients = {}
    for reaction in model.reactions:
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0 and reaction.upper_bound > 0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.name=reaction.name
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            # Make the directions aware of each other
            reverse_reaction.notes = reaction.notes
            reverse_reaction.annotation = reaction.annotation
            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {k: v * -1
                             for k, v in reaction._metabolites.items()}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            reactions_to_add.append(reverse_reaction)
            
    model.add_reactions(reactions_to_add)
    set_objective(model, coefficients, additive=True)
    
#Encapsulating parameters used in Concretemodel
def Get_Concretemodel_Need_Data(reaction_g0_file,metabolites_lnC_file,model_file,reaction_kcat_MW_file):
    Concretemodel_Need_Data={}
    reaction_g0=pd.read_csv(reaction_g0_file,index_col=0)
    # reaction_g0=pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
    Concretemodel_Need_Data['reaction_g0']=reaction_g0
    # metabolites_lnC = pd.read_csv(metabolites_lnC_file, index_col=0, sep='\t')
    metabolites_lnC = pd.read_csv(metabolites_lnC_file, index_col=0, sep='\t')
    Concretemodel_Need_Data['metabolites_lnC']=metabolites_lnC
    if re.search('\.xml',model_file):
        model = cobra.io.read_sbml_model(model_file)
    elif re.search('\.json',model_file):
        model = cobra.io.json.load_json_model(model_file)
    #convert_to_irreversible(model)
    reaction_kcat_MW=pd.read_csv(reaction_kcat_MW_file,index_col=0)
    Concretemodel_Need_Data['model']=model
    Concretemodel_Need_Data['reaction_kcat_MW']=reaction_kcat_MW
    [reaction_list,metabolite_list,lb_list,ub_list,coef_matrix]=Get_Model_Data(model)
    Concretemodel_Need_Data['reaction_list']=reaction_list
    Concretemodel_Need_Data['metabolite_list']=metabolite_list
    Concretemodel_Need_Data['lb_list']=lb_list
    Concretemodel_Need_Data['ub_list']=ub_list
    Concretemodel_Need_Data['coef_matrix']=coef_matrix
    return (Concretemodel_Need_Data)


#set_obj_value,set_metabolite,set_Df: only 'True' and 'False'
def Template_Concretemodel(reaction_list=None,metabolite_list=None,coef_matrix=None,metabolites_lnC=None,reaction_g0=None,reaction_kcat_MW=None,lb_list=None,\
    ub_list=None,obj_name=None,K_value=None,obj_target=None,set_obj_value=False,set_substrate_ini=False,substrate_name=None,substrate_value=None,\
    set_biomass_ini=False,biomass_value=None,biomass_id=None,set_metabolite=False,set_Df=False,set_obj_B_value=False,set_stoi_matrix=False,\
    set_bound=False,set_enzyme_constraint=False,set_integer=False,set_metabolite_ratio=False,set_thermodynamics=False,B_value=None,\
    set_obj_E_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,set_obj_single_E_value=False,E_total=None,\
    Bottleneck_reaction_list=None,set_Bottleneck_reaction=False):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of enzymes (0.114).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)
    Concretemodel.Df = pyo.Var(reaction_list,  within=Reals)
    Concretemodel.B = pyo.Var()
    Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals)
    Concretemodel.z = pyo.Var(reaction_list,  within=pyo.Binary)
    
    #Set upper and lower bounds of metabolite concentration
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(metabolite_list,rule=set_metabolite)        

    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in metabolite_list if (i,j) in coef_matrix.keys())
        rg0=list(set(list(reaction_g0.index)).intersection(set(reaction_list))) 
        Concretemodel.set_Df = Constraint(rg0,rule=set_Df)
    
    #Set the maximum flux as the object function
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)
            
    #Set the value of maximizing the minimum thermodynamic driving force as the object function
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  

    #Set the minimum enzyme cost of a pathway as the object function
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat_MW']) for j in reaction_kcat_MW.index)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)  

    #Minimizing the flux sum of pathway (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #To calculate the variability of enzyme usage of single reaction.
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,'kcat_MW'])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)

    #To calculate the concentration variability of metabolites.
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)

    #Adding flux balance constraints （FBA）
    if set_stoi_matrix:
        def set_stoi_matrix(m,i):
            return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
        Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix)

    #Adding the upper and lower bound constraints of reaction flux
    if set_bound:
        def set_bound(m,j):
            return inequality(lb_list[j],m.reaction[j],ub_list[j])
        Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound) 

    #Set the upper bound for substrate input reaction flux
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)   
        
    #Set the lower bound for biomass synthesis reaction flux
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)  

    #Adding enzymamic constraints
    if set_enzyme_constraint:
        def set_enzyme_constraint(m):
            return sum( m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat_MW']) for j in reaction_kcat_MW.index)<= E_total
        Concretemodel.set_enzyme_constraint = Constraint(rule=set_enzyme_constraint)

    #Adding thermodynamic MDF(B) object function
    if set_obj_B_value:
        def set_obj_B_value(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_list, rule=set_obj_B_value)

    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_list, rule=set_thermodynamics)
        
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_list,rule=set_integer)    

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel

def Get_Max_Min_Df(Concretemodel_Need_Data,obj_name,obj_target,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,without metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    """ 
    max_min_Df_list=pd.DataFrame()  
    opt=Model_Solve(Concretemodel,solver)

    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

def Get_Max_Min_Df_Ratio(Concretemodel_Need_Data,obj_name,obj_target,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,with metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).   
    """
    max_min_Df_list=pd.DataFrame()
    opt = Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Solving the MDF (B value)
def MDF_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
        biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)
    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    return B_value

#Constructing a GEM (iML1515 model) using Pyomo Concretemodel framework
def EcoGEM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
            substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    return EcoGEM

#Constructing a enzymatic constraints model (EcoECM) using Pyomo Concretemodel framework
def EcoECM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoECM

#Constructing a thermodynamic constraints model (EcoTCM) using Pyomo Concretemodel framework
def EcoTCM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    return EcoTCM

#Constructing a enzymatic and thermodynamic constraints model (EcoETM) using Pyomo Concretemodel framework
def EcoETM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoETM

#Solving programming problems
def Model_Solve(model,solver):
    opt = pyo.SolverFactory(solver)
    opt.solve(model)
    return model

#Maximum growth rate calculation
def Max_Growth_Rate_Calculation(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,K_value=K_value,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value)
    opt=Model_Solve(Concretemodel,solver)
    return opt.obj()

#Minimum enzyme cost calculation
def Min_Enzyme_Cost_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_E_value=True)
    opt=Model_Solve(Concretemodel,solver)
    min_E=opt.obj()
    return min_E

#Minimum flux sum calculation（pFBA）
def Min_Flux_Sum_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_V_value=True)
    opt=Model_Solve(Concretemodel,solver)

    min_V=opt.obj()
    return [min_V,Concretemodel]

#Determination of bottleneck reactions by analysing the variability of thermodynamic driving force
def Get_Max_Min_Df_Complete(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions in a special list.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list[substrate_name]: substrate_value (the upper bound for substrate input reaction flux)
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize). 
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)    
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Determination of limiting metabolites by analysing the concentration variability
def Get_Max_Min_Met_Concentration(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,Bottleneck_reaction_list,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_Met_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total,\
        Bottleneck_reaction_list=Bottleneck_reaction_list,set_Bottleneck_reaction=True)
    """Calculation of the maximum and minimum concentrations for metabolites in a specific list.

    Notes：
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).   
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False)
    * E_total: Total amount constraint of enzymes (0.114).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """  
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Determination of key enzymes by analysing the enzyme cost variability
def Get_Max_Min_E(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_single_E_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation of the maximum and minimum enzyme cost for reactions in a specific list.

    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).  
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """  

    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Solving maximum growth by different models
def Max_OBJ_By_Four_Model(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    biomass_list=pd.DataFrame()
    ub_list[substrate_name]=substrate_value
    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    opt=Model_Solve(EcoGEM,solver)
    biomass_list.loc[substrate_value,'GEM']=opt.obj()

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoECM,solver)
    biomass_list.loc[substrate_value,'ECM']=opt.obj()

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    opt=Model_Solve(EcoTCM,solver)
    biomass_list.loc[substrate_value,'TCM']=opt.obj()

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoETM,solver)
    biomass_list.loc[substrate_value,'ETM']=opt.obj()

    return biomass_list

#Solving MDF value under preset growth rate
def Max_MDF_By_model(Concretemodel_Need_Data,substrate_name,substrate_value,biomass_value,biomass_id,K_value,E_total,obj_enz_constraint,obj_no_enz_constraint,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    MDF_list=pd.DataFrame()

    if biomass_value<=obj_no_enz_constraint:
        EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
            biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoTCM,solver)
        MDF_list.loc[biomass_value,'EcoTCM']=opt.obj()
    else:
        MDF_list.loc[biomass_value,'EcoTCM']=None

    if biomass_value<=obj_enz_constraint:
        EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
            biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoETM,solver)
        MDF_list.loc[biomass_value,'EcoETM']=opt.obj()
    else:
        MDF_list.loc[biomass_value,'EcoETM']=None
        
    return MDF_list

def Get_Results_Thermodynamics(model,Concretemodel,reaction_kcat_MW,reaction_g0,coef_matrix,metabolite_list):
    """The formatting of the calculated results, includes the metabolic flux, binary variable values, thermodynamic driving force of reactions, the enzyme amount and the metabolite concentrations. The value of "-9999" means that the missing of kinetic (kcat) or thermodynamickcat (drG'°) parameters.
    
    Notes:
    ----------
    * model: is in SBML format (.xml).
    * Concretemodel: Pyomo Concretemodel.
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    """
    result_dataframe = pd.DataFrame()
    for eachreaction in Concretemodel.reaction:
        flux=Concretemodel.reaction[eachreaction].value
        z=Concretemodel.z[eachreaction].value
        result_dataframe.loc[eachreaction,'flux']=flux
        result_dataframe.loc[eachreaction,'z']=z  
        if eachreaction in reaction_g0.index:
            result_dataframe.loc[eachreaction,'f']=-reaction_g0.loc[eachreaction,'g0']-2.579*sum(coef_matrix[i,eachreaction]*Concretemodel.metabolite[i].value  for i in metabolite_list if (i,eachreaction) in coef_matrix.keys())
        else:
            result_dataframe.loc[eachreaction,'f']=-9999
        if eachreaction in reaction_kcat_MW.index:
            result_dataframe.loc[eachreaction,'enz']= flux/(reaction_kcat_MW.loc[eachreaction,'kcat_MW'])
        else:
            result_dataframe.loc[eachreaction,'enz']= -9999 
            
        tmp=model.reactions.get_by_id(eachreaction)
        met_list=''
        for met in tmp.metabolites:    
            met_list=met_list+';'+str(met.id)+' : '+str(np.exp(Concretemodel.metabolite[met.id].value))
        result_dataframe.loc[eachreaction,'met_concentration']= met_list  
        
    return(result_dataframe)

#Visualization of calculation results
def Draw_Biomass_By_Glucose_rate(Biomass_list,save_file):
    plt.figure(figsize=(15, 10), dpi=300)

    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[0]], color="black", linewidth=3.0, linestyle="--", label=Biomass_list.columns[0])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[1]], color="red", linewidth=3.0, linestyle="-", label=Biomass_list.columns[1])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[2]], color="cyan", linewidth=3.0, linestyle="-", label=Biomass_list.columns[2])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[3]], color="darkorange", linewidth=3.0, linestyle="-", label=Biomass_list.columns[3])

    font1 = {
    'weight' : 'normal',
    'size'   : 20,
    }

    plt.legend(loc="upper left",prop=font1)

    plt.xlim(0, 15)
    plt.ylim(0, 1.4)

    plt.tick_params(labelsize=23)
    plt.xticks([0, 1, 2,3, 4,5, 6, 7, 8, 9, 10, 11, 12, 13,14,15])
    plt.yticks([0.2, 0.4, 0.6,0.8, 1.0,1.2, 1.4])

    font2 = {
    'weight' : 'normal',
    'size'   : 25,
    }
    plt.xlabel("Glucose uptake rate (mmol/gDW/h)",font2)
    plt.ylabel("Growth rate ($\mathregular{h^-1}$)",font2)
    plt.savefig(save_file)
    plt.show()

def Draw_MDF_By_Growth_rate(MDF_list,save_file):
    plt.figure(figsize=(15, 10), dpi=300)
    MDF_list=MDF_list.sort_index(ascending=True) 
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[0]], color="cyan", linewidth=3.0, linestyle="-", label=MDF_list.columns[0])
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[1]], color="darkorange", linewidth=3.0, linestyle="-", label=MDF_list.columns[1])
    font1 = {
    'weight' : 'normal',
    'size'   : 23,
    }

    font2 = {
    'weight' : 'normal',
    'size'   : 30,
    }

    plt.ylabel("MDF of pathways (kJ/mol)",font2)
    plt.xlabel("Growth rate ($\mathregular{h^-1}$)",font2)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data', 0.3))
    ax.spines['bottom'].set_position(('data', 0))
    plt.legend(loc="lower left",prop=font1)
    plt.xlim(0.3, 0.9)
    plt.ylim(-26, 3)

    plt.tick_params(labelsize=23)

    plt.xticks([0.3, 0.4, 0.5,0.6, 0.7,0.8, 0.9])
    plt.yticks([-26, -22, -18,-14, -10,-6, -2,2])

    #plt.scatter([0.633], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.6756], [-0.48186643213771774], s=80, color="red")
    #plt.scatter([0.7068], [-9.486379882991386], s=80, color="red")
    #plt.scatter([0.852], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.855], [1.4290141211096987], s=80, color="red")
    #plt.scatter([0.867], [0.06949515162540898], s=80, color="red")
    #plt.scatter([0.872], [-0.8364187795859692], s=80, color="red")
    #plt.scatter([0.876], [-9.486379882991372], s=80, color="red")

    plt.savefig(save_file)
    plt.show()

def get_recation_g0(model,p_h,p_mg,ionic_strength,temperature):
    #get ΔG'° use equilibrator_api
    cc = None
    while cc is None:
        try:
            cc = ComponentContribution()
        except JSONDecodeError:
            logger.warning('Waiting for zenodo.org... Retrying in 5s')
            sleep(5)

    #cc = ComponentContribution()
    cc.p_h = Q_(p_h)
    cc.p_mg = Q_(p_mg)
    cc.ionic_strength = Q_(ionic_strength)
    cc.temperature = Q_(temperature)
    
    reaction_g0={}
    for eachr in model.reactions:
        if 'EX_' not in eachr.id:
            reaction_left=[]
            reaction_right=[]
            for k, v in eachr.metabolites.items():
                if str(k).endswith('_c'):
                    k_new = "_c".join(str(k).split('_c')[:-1])
                elif str(k).endswith('_p'):
                    k_new = "_p".join(str(k).split('_p')[:-1])
                elif str(k).endswith('_e'):
                    k_new = "_e".join(str(k).split('_e')[:-1]) 
                    
                #kegg,chebi,metanetx;kegg:C00002 + CHEBI:15377 = metanetx.chemical:MNXM7 + bigg.metabolite:pi
                if v<0:                    
                    reaction_left.append(str(-v)+' bigg.metabolite:'+k_new)
                else:
                    reaction_right.append(str(v)+' bigg.metabolite:'+k_new)
            reaction_equ=(' + ').join(reaction_left)+' -> '+(' + ').join(reaction_right)
            #print(reaction_equ)
            #get ΔG'° use equilibrator_api
            try:
                equilibrator_api_reaction = cc.parse_reaction_formula(reaction_equ)
                #print(reaction_equ)
                #print("The reaction is " + ("" if equilibrator_api_reaction.is_balanced() else "not ") + "balanced")
                dG0_prime = cc.standard_dg_prime(equilibrator_api_reaction)
            except:
                #pass
                print('error reaction: '+eachr.id)
                print(reaction_equ)
            else:  
                reaction_g0[eachr.id]={}
                reaction_g0[eachr.id]['reaction']=eachr.id
                reaction_g0[eachr.id]['equ']=eachr.reaction
                #dG0 (kilojoule/mole)
                reaction_g0[eachr.id]['g0']=str(dG0_prime).split(') ')[0].split('(')[1].split(' +/- ')[0]
              
    return reaction_g0

def get_recation_g0_local(model,p_h,p_mg,ionic_strength,temperature):
    #get ΔG'° use equilibrator_api
    lc = LocalCompoundCache()
    cc = ComponentContribution(ccache = lc.ccache)
    cc.p_h = Q_(p_h)
    cc.p_mg = Q_(p_mg)
    cc.ionic_strength = Q_(ionic_strength)
    cc.temperature = Q_(temperature)
    
    reaction_g0={}
    for eachr in model.reactions:
        if 'EX_' not in eachr.id:
            reaction_left=[]
            reaction_right=[]
            for k, v in eachr.metabolites.items():
                if str(k).endswith('_c'):
                    k_new = "_c".join(str(k).split('_c')[:-1])
                elif str(k).endswith('_p'):
                    k_new = "_p".join(str(k).split('_p')[:-1])
                elif str(k).endswith('_e'):
                    k_new = "_e".join(str(k).split('_e')[:-1]) 
                    
                #kegg,chebi,metanetx;kegg:C00002 + CHEBI:15377 = metanetx.chemical:MNXM7 + bigg.metabolite:pi
                if v<0:                    
                    reaction_left.append(str(-v)+' bigg.metabolite:'+k_new)
                else:
                    reaction_right.append(str(v)+' bigg.metabolite:'+k_new)
            if (' + ').join(reaction_left) !=(' + ').join(reaction_right):
                reaction_equ=(' + ').join(reaction_left)+' -> '+(' + ').join(reaction_right)
                #print(reaction_equ)
                #get ΔG'° use equilibrator_api
                try:
                    equilibrator_api_reaction = cc.parse_reaction_formula(reaction_equ)
                    #print("The reaction is " + ("" if equilibrator_api_reaction.is_balanced() else "not ") + "balanced")
                    dG0_prime = cc.standard_dg_prime(equilibrator_api_reaction)
                except:
                    #pass
                    print('error reaction: '+eachr.id)
                else:  
                    reaction_g0[eachr.id]={}
                    reaction_g0[eachr.id]['reaction']=eachr.id
                    reaction_g0[eachr.id]['equ']=reaction_equ
                    reaction_g0[eachr.id]['dG0 (kilojoule/mole)']=str(dG0_prime).split(') ')[0].split('(')[1]
                    #dG0 (kilojoule/mole)
                    reaction_g0[eachr.id]['g0']=str(dG0_prime).split(') ')[0].split('(')[1].split(' +/- ')[0]
    return reaction_g0

def Get_Fusion_Protein(reaction_g0_file,model_file,opt):
    # get the reaction with positive flux
    reaction_positive_value_list = {}
    reaction_gpr = {}
    result = []
    final_result = []
    reaction_g0 = pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
    for reaction in reaction_g0.index:
        reaction_positive_value_list[reaction] = opt.reaction[reaction].value
    cobra_model = cobra.io.load_json_model(model_file)
    # gpr of the reactions
    for reaction in reaction_positive_value_list.keys():
        reaction_gpr[reaction] = cobra_model.reactions.get_by_id(reaction).gene_reaction_rule.split(" and ")
    reaction_gpr_filtered = {key : value for key,value in reaction_gpr.items() if value != ['']}
    gene_reaction_dict = {}
    for key, values in reaction_gpr_filtered.items():
        for value in values:
            gene_reaction_dict.setdefault(value, []).append(key)
    # g0 of the reactions with the same gene
    for key,values in gene_reaction_dict.items():
        for i in range(len(values)):
            for j in range(i+1,len(values)):
                if reaction_g0.loc[values[i],"g0"] * reaction_g0.loc[values[j],"g0"] < 0:
                    if values[i].split("_")[0] != values[j].split("_")[0]:
                        result.append([values[i],values[j]])
    for item in result:
        if item not in final_result:
            final_result.append(item)
    return final_result

def Get_Fusion_Protein2(reaction_g0_file,opt):
    reaction_positive_g0_list = []
    reaction_positive_value_list = []
    result = []
    final_result = []
    metabolites_standard = ["h_c","h2o_c","co2_c","atp_c","adp_c","pi_c","nadph_c","nadp_c","o2_c","nad_c","nadh_c","so4_c","coa_c","accoa_c"]
    reaction_g0 = pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
    # get the reaction with positive flux and positive g0
    for reaction in reaction_g0.index:
        reaction_positive_value_list.append(reaction)
        if reaction_g0.loc[reaction,"g0"] >0:
            reaction_positive_g0_list.append(reaction)
    # get the coefficient of main metabolites
    for reaction in reaction_positive_g0_list:
        reaction_metabolites_positive_g0 = {}
        for keys,value in Concretemodel_Need_Data['coef_matrix'].items():
            if reaction in keys and keys[0] not in metabolites_standard:
                reaction_metabolites_positive_g0[keys[0]] = value
        for key,value in reaction_metabolites_positive_g0.items():
            for keys in Concretemodel_Need_Data['coef_matrix'].keys():
                if key in keys and keys[1] in reaction_positive_value_list and Concretemodel_Need_Data['coef_matrix'][keys]*value<0 and keys[1] not in reaction_positive_g0_list:
                    if reaction.split("_")[0] != keys[1].split("_")[0]:
                        result.append([key,reaction,keys[1]])
    for item in result:
        if item not in final_result:
            final_result.append(item)

def optimize_enzyme_parameters(json_path, Concretemodel_Need_Data, obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value, expected_value, rounds_num, kcat_max_json):
    model = cobra.io.load_json_model(json_path)
    rounds = 1
    opt_etm = Model_Solve(EcoETM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value), "cplex_direct")
    with open(kcat_max_json) as f:
        kcat_max = json.load(f)
    ignore_reaction_list = []
    changed_reaction_list = []

    def check_ec_codes_not_in_dict(ec_codes, dictionary):
        if ec_codes in dictionary:
            return False
        return True
    
    def find_max_kcat_mw(ec_codes, max_kcat_dict):
        max_kcat = None  
        if ec_codes in max_kcat_dict.keys():
            if max_kcat is None or max_kcat_dict[ec_codes]["kcat_max"] > max_kcat:
                max_kcat = max_kcat_dict[ec_codes]["kcat_max"]
        return max_kcat

    for reaction in model.reactions:
        if not hasattr(reaction,"annotation"):
            ignore_reaction_list.append(reaction.id)
        elif "ec-code" not in reaction.annotation.keys():
            ignore_reaction_list.append(reaction.id)
        else:
            ec_codes = reaction.annotation["ec-code"]
            if check_ec_codes_not_in_dict(ec_codes, kcat_max):
                ignore_reaction_list.append(reaction.id)
    print("------ Start checking enzyme parameters ------")
    while(opt_etm.obj() < expected_value and rounds < rounds_num and abs(opt_etm.obj() - expected_value)/expected_value > 0.01):
        enzyme_usage_dict = {}
        for reaction_id in Concretemodel_Need_Data["reaction_kcat_MW"].index:
            enzyme_usage_dict[reaction_id] = opt_etm.reaction[reaction_id].value*Concretemodel_Need_Data["reaction_kcat_MW"].loc[reaction_id,"MW"]/Concretemodel_Need_Data["reaction_kcat_MW"].loc[reaction_id,"kcat"]
        enzyme_usage_filtered_dict = {k: v for k, v in enzyme_usage_dict.items() if k not in ignore_reaction_list}
        max_enzyme_usage_reaction_id = max(enzyme_usage_filtered_dict, key=enzyme_usage_filtered_dict.get)
        max_enzyme_usage_reaction = model.reactions.get_by_id(max_enzyme_usage_reaction_id)
        ec_code = max_enzyme_usage_reaction.annotation["ec-code"]
        Concretemodel_Need_Data["reaction_kcat_MW"].loc[max_enzyme_usage_reaction_id,"kcat"] = find_max_kcat_mw(ec_code, kcat_max)
        print(max_enzyme_usage_reaction_id, " kcat -> ", find_max_kcat_mw(ec_code, kcat_max))
        Concretemodel_Need_Data["reaction_kcat_MW"].loc[max_enzyme_usage_reaction_id,"kcat_MW"] = Concretemodel_Need_Data["reaction_kcat_MW"].loc[max_enzyme_usage_reaction_id,"kcat"]*3600000/Concretemodel_Need_Data["reaction_kcat_MW"].loc[max_enzyme_usage_reaction_id,"MW"]
        changed_reaction_list.append(max_enzyme_usage_reaction_id)
        ignore_reaction_list.append(max_enzyme_usage_reaction_id)
        etm_model = EcoETM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value)    
        opt_etm = Model_Solve(etm_model, "cplex_direct")
        print("rounds:", rounds)
        rounds += 1
        print("changed_reaction_list:", changed_reaction_list)
        print("solution:", opt_etm.obj())
        print("-------------------------------------------")

def calculate_MDF_list(Concretemodel_Need_Data, biomass_id, substrate_name, substrate_value, K_value, E_total, output_path):
    obj_min = 0.1
    opt_max = EcoECM(Concretemodel_Need_Data,biomass_id,"maximize",substrate_name,substrate_value,E_total)
    obj_max = Model_Solve(opt_max, "cplex_direct").obj()
    obj_value = obj_min
    MDF_list = []
    starttime = datetime.datetime.now()
    MDF_list.append([obj_value,MDF_Calculation(Concretemodel_Need_Data,obj_value,biomass_id,substrate_name,substrate_value,K_value,E_total,"cplex_direct")])
    while(obj_value <= obj_max):
        obj_value += 0.01
        if obj_value <= obj_max:
            B_value = MDF_Calculation(Concretemodel_Need_Data,obj_value,biomass_id,substrate_name,substrate_value,K_value,E_total,"cplex_direct")
            MDF_list.append([obj_value,B_value])
            opt_value = EcoETM(Concretemodel_Need_Data,biomass_id,"maximize",substrate_name,substrate_value,E_total,K_value,B_value)
            obj_value = Model_Solve(opt_value, "cplex_direct").obj()
            MDF_list.append([obj_value,B_value])
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    MDF_list = [[round(num, 3) for num in inner_list] for inner_list in MDF_list]
    MDF_df = pd.DataFrame()
    for item in MDF_list:
        MDF_df.loc[item[0],"MDF"] = item[1]
    MDF_df.to_csv(output_path)
    return [MDF_list, MDF_df]

def Draw_MDF_By_Product_rate(MDF_dict, xmin, xlab, xmax, step, ytick, save_file):
    """Draw the MDF curves with different growth rates.
    Arguments:
    ----------
    * MDF_dict: A dictionary where keys are IDs and values are lists of points.
    * save_file: The file name of the saved figure.
    """
    plt.figure(figsize=(15, 10), dpi=300)

    font1 = {
        'weight': 'normal',
        'size': 23,
    }

    font2 = {
        'weight': 'normal',
        'size': 30,
    }

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data', xmin))
    ax.spines['bottom'].set_position(('data', 0))

    for id, points in MDF_dict.items():
        x, y = zip(*points)
        plt.plot(x, y, label=id, linewidth=3.0)  # 使用不同的颜色

    plt.legend(loc="lower left", prop=font1)
    plt.ylabel("MDF of pathways (kJ/mol)", font2)
    plt.xlabel(xlab, font2)
    plt.xlim(xmin, xmax)
    plt.ylim(np.min(ytick), np.max(ytick))

    plt.tick_params(labelsize=23)
    plt.xticks(generate_list(xmin, xmax, step))
    plt.yticks(ytick)

    plt.savefig(save_file)
    plt.show()
    
def generate_list(min_value, max_value, step):
    result = []
    current_value = min_value
    while current_value <= max_value:
        result.append(current_value)
        current_value += step
    return result

def BeReTa_use(model_file, Concretemodel_Need_Data, reg_net, exp_comp, biomass_id, product_rxn, substrate_name,substrate_value,E_total,K_value,B_value, pval_cut, n_target_cut, f_target_cut):
    #product_rxn = "DM_" + product
    #if product_rxn in model.reactions.keys():
    #    model.remove_reactions(product_rxn)
    #model.add_demand_reaction(product)
    model = cobra.io.load_json_model(model_file)
    RegStr, RegStr_gene = generateRegStrMatrix(reg_net, exp_comp, model)
    #print(len(RegStr['Reg_TF']))
    q_slope = calculateFluxSlope(model, Concretemodel_Need_Data, biomass_id, product_rxn, substrate_name,substrate_value,E_total,K_value,B_value)##这步求解有问题
    # Trimming
    reg_list = RegStr['Reg_TF'] 
    Reg_RS = RegStr['Reg_RS']
    
    #print(RegStr,q_slope,RegStr.shape,q_slope.shape)
    # Calculate beneficial scores for all transcriptional regulators (TRs)
    beneficial_score = np.dot(Reg_RS, q_slope)
    #print(type(beneficial_score),beneficial_score, q_slope)
    # Perform permutation test
    beneficial_score_pval = permutationTestBeReTa(model, Reg_RS, q_slope, beneficial_score, reg_list)
    # Apply target criteria to select BeReTa targets
    BeReTaSolution = selectBeReTaTargets(model, RegStr, RegStr_gene, q_slope, beneficial_score, beneficial_score_pval, reg_list, pval_cut, n_target_cut, f_target_cut)
    return BeReTaSolution

def BeReTa_v1(model_file, Concretemodel_Need_Data, reg_net, exp_comp, biomass_id, product_rxn, pval_cut, n_target_cut, f_target_cut):
    # INPUTS
    # model: COBRA model structure
    # reg_net: Transcriptional regulatory network structure
    #   regulator: Cell containing regulator names
    #   target: Cell containing target names
    #   sign: Vector containing sign of regulation (activator, 1; repressor, -1; unknown, 0) (optional)
    #   regulator, target, and sign should have same length
    # exp_comp: Gene expression compendium data structure
    #   genes: Cell containing gene names
    #   expression: Gene expression levels
    #   genes and expression should have same length
    # product_rxn: Exchange reaction for target product
    #
    # OUTPUTS
    # BeReTaSolution: BeReTa solution structure (target regulator, beneficial score, p-value)

    # Generate regulatory strength matrix (RegStr) and flux slope vector (q_slope)
    tem_obj = model.obj_name
    product_rxn = "DM_" + product
    if product_rxn in model.reactions.keys():
        model.remove_reactions(product_rxn)
    model.add_demand_reaction(product)
    model.obj_name = biomass_id
    RegStr, RegStr_gene = generateRegStrMatrix(reg_net, exp_comp, model)
    #print(len(RegStr['Reg_TF']))
    q_slope = calculateFluxSlope(model, Concretemodel_Need_Data, biomass_id, product_rxn, substrate_name,substrate_value,E_total,K_value,B_value)##这步求解有问题
    # Trimming
    reg_list = RegStr['Reg_TF'] 
    Reg_RS = RegStr['Reg_RS']
    
    #print(RegStr,q_slope,RegStr.shape,q_slope.shape)
    # Calculate beneficial scores for all transcriptional regulators (TRs)
    beneficial_score = np.dot(Reg_RS, q_slope)
    #print(type(beneficial_score),beneficial_score, q_slope)
    # Perform permutation test
    beneficial_score_pval = permutationTestBeReTa(model, Reg_RS, q_slope, beneficial_score, reg_list)
    # Apply target criteria to select BeReTa targets
    BeReTaSolution = selectBeReTaTargets(model, RegStr, RegStr_gene, q_slope, beneficial_score, beneficial_score_pval, reg_list, pval_cut, n_target_cut, f_target_cut)
    model.set_obj_name(tem_obj)
    return BeReTaSolution

def generateRegStrMatrix(reg_net, exp_comp, model):
    # Generate regulatory strength matrix (RS)
    regulator = reg_net['regulator']
    target = reg_net['target']
    expression = exp_comp['expression']
    expressionid = exp_comp['genes']
    model_genes = [gene.id for gene in model.genes]
    model_rxns = [reaction.id for reaction in model.reactions]
    corr_mat1_col = list(model_genes)

    if reg_net.shape[1] == 3:
        sign = reg_net['sign']
    else:
        sign = np.zeros((reg_net.shape[0], reg_net.shape[1]))
    # Discard regulatory interactions for which correlation cannot be computed.    
    regulator_temp = []
    target_temp = []
    sign_temp = []
    for i in range(len(regulator)):
        regulator_id = np.where(expressionid == regulator[i])[0]
        target_id = np.where(expressionid == target[i])[0]
        if (len(regulator_id) == 1) and (len(target_id) == 1) and target[i] in corr_mat1_col:#有表达值，并且在模型中
            regulator_temp.append(regulator[i])
            target_temp.append(target[i])
            sign_temp.append(sign[i])
    regulator = regulator_temp
    target = target_temp
    sign = sign_temp

    reg_list = np.unique(regulator)
    corr_mat1 = np.zeros((len(reg_list), len(model_genes)))
    corr_mat1_row = list(reg_list)

    # Generate regulatory strength (RS) matrix for TR-gene interactions
    for i in range(len(regulator)):
        row_id = corr_mat1_row.index(regulator[i])
        col_id = corr_mat1_col.index(target[i])
        #print(row_id,col_id)
        x1 = expression[expressionid==regulator[i]].values[0]
        x2 = expression[expressionid==target[i]].values[0]
        ## Mr.Mao
        corr_val = pearsonr(np.array(x1).flatten(), np.array(x2).flatten())[0]

        #print(corr_val)
        if sign[i] == 0:
            corr_mat1[row_id, col_id] = corr_val
        elif sign[i] == 1:
            #print(row_id,col_id,corr_val)
            corr_mat1[row_id, col_id] = abs(corr_val)
        elif sign[i] == -1:
            corr_mat1[row_id, col_id] = -abs(corr_val)
    #print(corr_mat1.shape)
    #Remove zero columns in the matrix
    row_index = np.where(np.sum(np.abs(corr_mat1), axis=1) != 0)[0]
    col_index = np.where(np.sum(np.abs(corr_mat1), axis=0) != 0)[0]
    #print(row_index,col_index)
    corr_mat2 = corr_mat1[row_index, :][:, col_index]
    corr_mat2_row = [corr_mat1_row[i] for i in row_index]
    corr_mat2_col = [corr_mat1_col[i] for i in col_index]
    RegStr_gene = {}
    RegStr_gene['RSmat_gene'] = corr_mat2_col
    RegStr_gene['RSmat_TF'] = corr_mat2_row
    RegStr_gene['RSmat_mat'] = corr_mat2
    #RegStr_gene = [['RSmat_gene'] + list(corr_mat2_col)] + list(zip(corr_mat2_row, corr_mat2.tolist()))

    # Generate regulatory strength (RS) matrix for TR-rxn interactions
    # GPR mapping using average function
    corr_mat3 = np.zeros((len(corr_mat2_row), len(model_rxns)))
    corr_mat3_row = corr_mat2_row
    corr_mat3_col = np.array(model_rxns)

    for i in range(corr_mat2.shape[0]):
        gpr_genes = np.array(corr_mat2_col)
        gpr_levels = np.array(corr_mat2[i, :]).reshape(-1, 1)
        gpr_genes2 = gpr_genes[np.nonzero(gpr_levels)[0]]
        gpr_levels2 = gpr_levels[np.nonzero(gpr_levels)[0]]
        levels = gene_to_reaction_levels(model, list(gpr_genes2), gpr_levels2, lambda x, y: (x + y) / 2, lambda x, y: (x + y) / 2)
        corr_mat3[i, np.nonzero(~np.isnan(levels))] = levels[np.nonzero(~np.isnan(levels))]
    
    RegStr = {}
    RegStr['Reg_rxn'] = corr_mat3_col
    RegStr['Reg_TF'] = corr_mat3_row
    RegStr['Reg_RS'] = corr_mat3

    return [RegStr, RegStr_gene]

def gene_to_reaction_levels(model, genes, levels, f_and, f_or):
    # Original authors: Daniel Machado and Markus Herrgard, PLoS Computational Biology, 2014
    # Code obtained from https://github.com/cdanielmachado/transcript2flux
    #
    # Convert gene expression levels to reaction levels using GPR associations.
    # Level is NaN if there is no GPR for the reaction or no measured genes.
    #
    # INPUTS
    #       model - cobra model
    #       genes - gene names
    #       levels - gene expression levels
    #       f_and - function to replace AND
    #       f_or - function to replace OR
    #
    # OUTPUTS
    #       reaction_levels - reaction expression levels
    #
    # Author: Daniel Machado, 2013
    reaction_levels = np.zeros(len(model.reactions))
    i = 0 
    for reaction in model.reactions:
        #print(reaction.gene_reaction_rule)
        [result, level] = eval_gpr(reaction.gene_reaction_rule, genes, levels, f_and, f_or)
        #print(result)
        reaction_levels[i] = result
        i=i+1
    return reaction_levels

def eval_gpr(rule, genes, levels, f_and, f_or):
    # Original authors: Daniel Machado and Markus Herrgard, PLoS Computational Biology, 2014
    # Code obtained from https://github.com/cdanielmachado/transcript2flux
    #
    # Evaluate the expression level for a single reaction using the GPRs.
    # Note: Computes the expression level even if there are missing measured
    # values for the given rule. This implementation is a modified version of
    # an implementation provided in [Lee et al, BMC Sys Biol, 2012]
    EVAL_OK = 1
    PARTIAL_MEASUREMENTS = 0
    NO_GPR_ERROR = -1
    NO_MEASUREMENTS = -2
    MAX_EVALS_EXCEEDED = -3
    MAX_EVALS = 1000
    NONETYPE = 'NaN'
    NUMBER = r'[0-9\.\-e]+'
    MAYBE_NUMBER = f"{NUMBER}|{NONETYPE}"
    expression = rule
    result = np.nan
    status = EVAL_OK
    if not expression:
        status = NO_GPR_ERROR
    else:
        rule_genes = list(set(re.findall(r'\b(\w+|-)+\b', expression)) - {'and', 'or'})
        total_measured = 0
        for gene in rule_genes:
            #print(gene,rule_genes,genes)
            if gene in genes:
                j = genes.index(gene)
                level = str(levels[j])
                total_measured += 1
            else:
                level = NONETYPE
            expression = re.sub(r'\b{}\b'.format(gene), level, expression)
        if total_measured == 0:
            status = NO_MEASUREMENTS
        else:
            if total_measured < len(rule_genes):
                status = PARTIAL_MEASUREMENTS
            maybe_and = lambda a, b: maybe_functor(f_and, a, b)
            maybe_or = lambda a, b: maybe_functor(f_or, a, b)
            str_wrapper = lambda f, a, b: str(f(float(a), float(b)))
            counter = 0
            while np.isnan(result):
                counter += 1
                if counter > MAX_EVALS:
                    status = MAX_EVALS_EXCEEDED
                    break
                try:
                    result = eval(expression)[0]
                except Exception as e:
                    paren_expr = r'\(\s*({})\s*\)'.format(MAYBE_NUMBER)
                    and_expr = r'({})\s+and\s+({})'.format(MAYBE_NUMBER, MAYBE_NUMBER)
                    or_expr = r'({})\s+or\s+({})'.format(MAYBE_NUMBER, MAYBE_NUMBER)
                    expression = re.sub(paren_expr, r'\1', expression)
                    expression = re.sub(and_expr, lambda m: str_wrapper(maybe_and, m.group(1), m.group(2)), expression)
                    expression = re.sub(or_expr, lambda m: str_wrapper(maybe_or, m.group(1), m.group(2)), expression)
    return result, status

def maybe_functor(f, a, b):
    # Original authors: Daniel Machado and Markus Herrgard, PLoS Computational Biology, 2014
    # Code obtained from https://github.com/cdanielmachado/transcript2flux
    if np.isnan(a) and np.isnan(b):
        c = np.nan
    elif not np.isnan(a) and np.isnan(b):
        c = a
    elif np.isnan(a) and not np.isnan(b):
        c = b
    else:
        c = f(a, b)
    return c

def calculate(model, model_type, obj_target, B_value):
    if model_type == "GEM":
        initial_sol = calculate_GEM(model,obj_target)
        final_sol = Min_Flux_Sum_Calculation_for_BeReTa(model,model_type,initial_sol.reaction[model.obj_name].value,model.obj_name,B_value)
    elif model_type == "ecGEM":
        initial_sol = calculate_ECM(model,obj_target)
        final_sol = Min_Flux_Sum_Calculation_for_BeReTa(model,model_type,initial_sol.reaction[model.obj_name].value,model.obj_name,B_value)
    elif model_type == "tcGEM":
        initial_sol = calculate_TCM(model,obj_target,B_value)
        final_sol = Min_Flux_Sum_Calculation_for_BeReTa(model,model_type,initial_sol.reaction[model.obj_name].value,model.obj_name,B_value)
    elif model_type == "etGEM":
        initial_sol = calculate_ETM(model,obj_target,B_value)
        final_sol = Min_Flux_Sum_Calculation_for_BeReTa(model,model_type,initial_sol.reaction[model.obj_name].value,model.obj_name,B_value)
    return final_sol

#Minimum flux sum calculation（pFBA）
def Min_Flux_Sum_Calculation_for_BeReTa(model,model_type,biomass_value,biomass_id,B_value):
    """Calculation of minimum flux sum for a model.
    Arguments:
    ----------
    * model: The model to be calculated.
    * biomass_value: Biomass value.
    * biomass_id: Biomass reaction ID.
    * substrate_name: Substrate name.
    * substrate_value: Substrate value.
    * K_value: The value of MDF.
    * E_total: The total enzyme mass.
    * B_value: The value of MDF(B).
    * solver: The solver used to solve the model.
    Return:
    ----------
    * min_V: The value of minimum flux sum.
    """
    if model_type == "GEM":
        Concretemodel = Template_Concretemodel(model,
            set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,
            set_stoi_matrix=True,set_bound=True,
            set_integer=True,set_obj_V_value=True)
    elif model_type == "ecGEM":
        Concretemodel = Template_Concretemodel(model,
            set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,
            set_stoi_matrix=True,set_bound=True,
            set_enzyme_constraint=True,set_integer=True,set_obj_V_value=True)
    elif model_type == "tcGEM":
        Concretemodel = Template_Concretemodel(model,
            set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,
            set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_stoi_matrix=True,
            set_bound=True,set_thermodynamics=True,B_value=B_value,set_integer=True,set_obj_V_value=True)
    elif model_type == "etGEM":
        Concretemodel = Template_Concretemodel(model=model,\
            set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,\
            set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
            set_bound=True,set_enzyme_constraint=True,set_integer=True,set_metabolite_ratio=True,\
            set_thermodynamics=True,B_value=B_value,set_obj_V_value=True)
    else:
        raise ValueError(
                f"{model_type} is not a model type"
        )
    opt=Model_Solve(Concretemodel,model.solver)
    return opt

def calculateFluxSlope(model, Concretemodel_Need_Data, biomass_id, product_rxn, substrate_name,substrate_value,E_total,K_value,B_value):
    ## Calculate flux slope vector (q_slope)
    # Calculate initial and maximum flux of product Rxn.
    reaction_ids = [reaction.id for reaction in model.reactions]
    product_rxn_id =  reaction_ids.index(product_rxn)
    initial_sol = Model_Solve(EcoETM(Concretemodel_Need_Data,biomass_id,"maximize",substrate_name,substrate_value,E_total,K_value,B_value),"cplex_direct")
    initial_product_flux = initial_sol.reaction[product_rxn].value
    maximum_sol = Model_Solve(EcoETM(Concretemodel_Need_Data,product_rxn,"maximize",substrate_name,substrate_value,E_total,K_value,B_value),"cplex_direct")
    maximum_product_flux = maximum_sol.reaction[product_rxn].value
    # Flux scanning while increasing the product flux
    num_steps = 20
    flux_values = np.zeros((len(model.reactions), num_steps))
    for i in range(num_steps):
        product_flux = initial_product_flux + ((maximum_product_flux - initial_product_flux) * ((i) / num_steps))
        Concretemodel_Need_Data["ub_list"][product_rxn] = product_flux
        Concretemodel_Need_Data["lb_list"][product_rxn] = product_flux
        enforced_sol = Model_Solve(EcoETM(Concretemodel_Need_Data,biomass_id,"maximize",substrate_name,substrate_value,E_total,K_value,B_value),"cplex_direct")
        #enforced_sol = optimizeCbModel(model_enforced, 'max', 'one')
        for j in range(len(model.reactions)):
            flux_values[j, i] = np.abs(enforced_sol.reaction[model.reactions[j].id].value)
    # Use linear regression to estimate flux slopes
    flux_values_product = flux_values[product_rxn_id, :]
    q_slope = np.zeros((len(model.reactions), 1))
    for i in range(len(model.reactions)):
        linear_regression = linregress(flux_values_product, flux_values[i, :])
        q_slope[i, 0] = linear_regression.slope
    q_slope[q_slope < 0] = 0
    return q_slope

def exchanges(model):
    """Exchange reactions in model.
    Reactions that exchange mass with the exterior. Uses annotations
    and heuristics to exclude non-exchanges such as sink reactions.
    """
    exchange_list=[]
    for reaction in model.reactions:
        metabolites = reaction.metabolites
        if all(value > 0 for value in metabolites.values()) or all(value > 0 for value in metabolites.values()):
            exchange_list.append(reaction.id)
    return exchange_list

def findorphanreaction(model):
    orphan_reactions = []
    for reaction in model.reactions:
        if not reaction.gene_reaction_rule:
            orphan_reactions.append(reaction.id)
    return orphan_reactions

def permutationTestBeReTa(model, RegStr, q_slope, beneficial_score, reg_list):
    ## Calculate p-values for beneficial scores
    # Permute only for gene-associated rxns
    selExc = exchanges(model)
    orphans = findorphanreaction(model)
    reaction_ids = [reaction.id for reaction in model.reactions]
    gene_associated = np.setdiff1d(reaction_ids, selExc)
    gene_associated = np.setdiff1d(gene_associated, orphans)
    rand_rxn_ids = np.where(np.isin(reaction_ids, gene_associated))[0]
    # Calculate permuted beneficial scores
    rand_num = 10000
    beneficial_score_rand = np.zeros((len(reg_list), rand_num))
    for i in range(rand_num):
        rand_order = np.argsort(np.random.rand(len(rand_rxn_ids)))
        q_slope_rand = q_slope.copy()
        q_slope_rand[rand_rxn_ids[rand_order]] = q_slope[rand_rxn_ids]
        #print(RegStr.shape,q_slope_rand.shape,np.dot(RegStr, q_slope_rand))
        tmp=np.dot(RegStr, q_slope_rand)
        j=0
        for value in np.nditer(tmp):
            beneficial_score_rand[j, i] = value
            j=j+1

    # Calculate p-values
    beneficial_score_pval = np.ones(len(reg_list))
    for i in range(len(reg_list)):
        #print(beneficial_score[i])
        if beneficial_score[i][0] > 0:
            beneficial_score_pval[i] = len(np.where(beneficial_score_rand[i, :] > beneficial_score[i][0])[0]) / rand_num
        elif beneficial_score[i][0] < 0:
            beneficial_score_pval[i] = len(np.where(beneficial_score_rand[i, :] < beneficial_score[i][0])[0]) / rand_num
    return beneficial_score_pval

def findGenesFromRxns(model, effective_rxns):
    gene_ids_list=[]
    for reaction in model.reactions:
        if reaction.id in effective_rxns:
            gene_list = reaction.gene_reaction_rule.split(" and ")
            gene_ids_list.append(gene_list)
    return gene_ids_list

def selectBeReTaTargets(model, RegStr, RegStr_gene, q_slope, beneficial_score, beneficial_score_pval, reg_list, pval_cut, n_target_cut, f_target_cut):
    ## Apply target criteria to select BeReTa targets
    # Define target criteria   (1) TR should have non-zero beneficial score.
    #pval_cut = 0.05 #(2) The p-value of the beneficial score should be less than 0.05.
    #n_target_cut = 1 #(3) TR should have two or more effective gene/reaction targets.
    #f_target_cut = 0.1 #(4) At least 10% of target metabolic genes of TR should be beneficial, i.e. have positive flux slopes.
    # Calculate the metrics
    n_effective_genes = np.zeros(len(reg_list))
    n_effective_rxns = np.zeros(len(reg_list))
    f_effective_genes = np.zeros(len(reg_list))
    Reg_rxn = RegStr['Reg_rxn']
    #print(Reg_rxn)
    Reg_RS = RegStr['Reg_RS']
    #print(type(q_slope),type(Reg_RS[1,:]))
    for i in range(len(reg_list)):
        regulated_genes = RegStr_gene['RSmat_gene']
        regulated_genes_num = np.count_nonzero(RegStr['Reg_RS'][i])
        j=0
        Reg_RStmp=np.zeros((Reg_RS.shape[1],1))
        #print(Reg_RS[i,:].shape)
        for value in np.nditer(Reg_RS[i,:]):
            Reg_RStmp[j, 0] = value
            j=j+1
        tmp = q_slope * Reg_RStmp
        #print(q_slope.shape,Reg_RStmp.shape,tmp.shape)
        # 创建布尔类型的掩码数组，判断元素是否为0
        mask = (tmp != 0) 
        #print(mask)
        # 使用 np.where() 获取满足条件的位置索引
        indices = np.where(mask)[0]
        #print(indices)
        #print(q_slope.shape,Reg_RS.shape)
        effective_rxns = Reg_rxn[indices]
        n_effective_rxns[i] = len(effective_rxns)
        if len(effective_rxns) > 0:
            effective_genes_temp = findGenesFromRxns(model, effective_rxns)
            effective_genes = []
            for j in range(len(effective_genes_temp)):
                effective_genes.extend(effective_genes_temp[j])
            effective_genes = np.unique(effective_genes)
            effective_genes = np.intersect1d(effective_genes, regulated_genes)
            n_effective_genes[i] = len(effective_genes)
            f_effective_genes[i] = len(effective_genes) / regulated_genes_num
        else:
            n_effective_genes[i] = 0
            f_effective_genes[i] = 0
    #Select BeReTa targets
    BeReTa_metrics = np.column_stack((beneficial_score, beneficial_score_pval, n_effective_genes, n_effective_rxns, f_effective_genes))
    
    beneficial_scoretmp=np.zeros((beneficial_score.shape[0]))
    #print(beneficial_scoretmp.shape)
    j=0
    for value in np.nditer(beneficial_score):
        #print(value)
        beneficial_scoretmp[j] = abs(value)
        j=j+1
    #print(beneficial_scoretmp)
    score_order = np.argsort(-beneficial_scoretmp)
    #print(score_order)
    # 将索引数组转换为整数类型
    score_order = score_order.astype(int)
    # 将转换后的整数类型索引数组转换为普通的Python列表
    score_order = score_order.tolist()
    reg_list2 = [reg_list[i] for i in score_order]
    BeReTa_metrics = BeReTa_metrics[score_order, :]
    
    BeReTaSolution_reg_list = []
    BeReTaSolution_metrics = []
    for i in range(len(reg_list2)):
        if (BeReTa_metrics[i, 1] < pval_cut) and (BeReTa_metrics[i, 2] >= n_target_cut) and (BeReTa_metrics[i, 3] >= n_target_cut) and (BeReTa_metrics[i, 4] >= f_target_cut):
            BeReTaSolution_reg_list.append(reg_list2[i])
            BeReTaSolution_metrics.append(BeReTa_metrics[i, :])
    #print(BeReTaSolution_reg_list)
    #print(BeReTaSolution_metrics)
    #BeReTaSolution={}
    #BeReTaSolution['reglist']=BeReTaSolution_reg_list
    #BeReTaSolution['metrics']=BeReTaSolution_metrics
    #BeReTaSolution = np.column_stack((BeReTaSolution_reg_list, BeReTaSolution_metrics[:, :2])).tolist()
    # 定义列名
    columns = ['beneficial_score', 'beneficial_score_pval', 'n_effective_genes', 'n_effective_rxns', 'f_effective_genes']
    # 定义索引
    index_values = BeReTaSolution_reg_list
    # 使用pandas的DataFrame构造函数
    BeReTaSolution = pd.DataFrame(BeReTaSolution_metrics, columns=columns)
    # 设置DataFrame的索引
    BeReTaSolution.index = index_values
    return BeReTaSolution

def ETGEMs_analysis(Concretemodel_Need_Data_combine,obj_id, obj_target, substrate_name,substrate_value,E_total,K_value,cobra_model_path):
    etm = EcoETM(Concretemodel_Need_Data_combine,obj_id,obj_target,substrate_name,substrate_value,E_total,K_value,-1000)
    cobra_model = cobra.io.load_json_model(cobra_model_path)
    opt_etm = Model_Solve(etm, "cplex_direct")
    model=Concretemodel_Need_Data_combine['model']
    reaction_kcat_MW=Concretemodel_Need_Data_combine['reaction_kcat_MW']
    reaction_g0=Concretemodel_Need_Data_combine['reaction_g0']
    coef_matrix=Concretemodel_Need_Data_combine['coef_matrix']
    reaction_list=Concretemodel_Need_Data_combine['reaction_list']
    metabolite_list=Concretemodel_Need_Data_combine['metabolite_list']
    max_flux = opt_etm.obj()
    set_MDF_substrate = 0.1
    num = 1
    columns_bottleneck_reaction = ['reaction_id', 'equation', 'g0', 'gpr', 'inflection','product flux','MDF']
    columns_bottleneck_metabolite = ['metabolite_id', 'formula', 'inflection','product flux','MDF']
    columns_bottleneck_enzyme = ['reaction_id', 'equation', 'gpr', 'inflection','product flux','MDF','enzyme usage']
    bottleneck_reaction_df = pd.DataFrame(columns=columns_bottleneck_reaction)
    bottleneck_metabolite_df = pd.DataFrame(columns=columns_bottleneck_metabolite)
    bottleneck_enzyme_df = pd.DataFrame(columns=columns_bottleneck_enzyme)
    while(set_MDF_substrate < max_flux):
        B_value = MDF_Calculation(Concretemodel_Need_Data_combine,set_MDF_substrate,obj_id,substrate_name,substrate_value,K_value,E_total,'cplex_direct')
        etm = EcoETM(Concretemodel_Need_Data_combine,obj_id,obj_target,substrate_name,substrate_value,E_total,K_value,B_value)
        opt_etm = Model_Solve(etm, "cplex_direct")
        obj_enz_constraint = opt_etm.obj()
        [min_V,Concretemodel]=Min_Flux_Sum_Calculation(Concretemodel_Need_Data_combine,opt_etm.obj(),obj_id,substrate_name,substrate_value,K_value,E_total,B_value,'cplex_direct')
        use_result = Get_Results_Thermodynamics(model,Concretemodel,reaction_kcat_MW,reaction_g0,coef_matrix,metabolite_list)
        use_result = use_result[use_result['flux'] > 1e-10] 
        use_result = use_result.sort_values(by = 'flux',axis = 0,ascending = False)
        use_result["reaction"] = use_result.apply(lambda row: cobra_model.reactions.get_by_id(row.name).reaction, axis = 1)
        use_result["gpr"] = use_result.apply(lambda row: cobra_model.reactions.get_by_id(row.name).gene_reaction_rule, axis = 1)
        use_result_tmp=use_result[use_result['f']>-2765]
        use_result_select=use_result_tmp[abs(use_result_tmp['f']-B_value)<=1e-05]
        max_min_Df_list_fixed=pd.DataFrame()
        path_reac_list=list(use_result_select.index)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(Get_Max_Min_Df_Complete,Concretemodel_Need_Data_combine,eachreaction,'maximize',K_value,B_value,0.1,\
                                    obj_id,E_total,substrate_name,substrate_value,'cplex_direct'): eachreaction for eachreaction in path_reac_list}
            for future in as_completed(futures):
                tmp = future.result()
                for eachindex in tmp.index:
                    #print(eachindex,tmp.loc[eachindex,'max_value'])
                    max_min_Df_list_fixed.loc[eachindex,'max_Df_complete']=tmp.loc[eachindex,'max_value']

        max_min_Df_list_fixed=max_min_Df_list_fixed.sort_values(by='max_Df_complete',ascending = True)
        Bottleneck_reaction=max_min_Df_list_fixed[(max_min_Df_list_fixed['max_Df_complete']-B_value)<=0.01]
        Bottleneck_reaction_list = list(Bottleneck_reaction.index)
        if len(Bottleneck_reaction_list)>0:
            for reaction in Bottleneck_reaction_list:
                result_row = {'reaction_id' : reaction, 'equation' : reaction_g0.loc[reaction,"equation"]
                            ,'g0' : reaction_g0.loc[reaction,"g0"], 'gpr' : reaction_g0.loc[reaction,"gpr"], 'inflection' : num, 'product flux' : obj_enz_constraint, "MDF" : B_value}       
                bottleneck_reaction_df = pd.concat([bottleneck_reaction_df, pd.DataFrame(result_row, index=[0])], ignore_index=True)

            Bottleneck_reaction_met=[]
            for rea in reaction_list:
                if rea in Bottleneck_reaction_list:
                    #print(rea)
                    cobra_rea = cobra_model.reactions.get_by_id(rea)
                    for met in cobra_rea.metabolites.keys():
                        if met.id !='h_c' and met.id !='h2o_c':
                            Bottleneck_reaction_met.append(met.id)
            Bottleneck_reaction_met=list(set(Bottleneck_reaction_met))
            max_min_concentration_list_fixed = pd.DataFrame()
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(Get_Max_Min_Met_Concentration,Concretemodel_Need_Data_combine,eachmet,'maximize',K_value,B_value,\
                    obj_enz_constraint,obj_id,E_total,substrate_name,substrate_value,list(Bottleneck_reaction.index),'cplex_direct'): eachmet for eachmet in Bottleneck_reaction_met}
                for future in as_completed(futures):
                    tmp = future.result()
                    for eachindex in tmp.index:
                        #print(eachindex,tmp.loc[eachindex,'max_value'])
                        max_min_concentration_list_fixed.loc[eachindex,'max_concentration'] = tmp.loc[eachindex,'max_value']

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(Get_Max_Min_Met_Concentration,Concretemodel_Need_Data_combine,eachmet,'minimize',K_value,B_value,\
                    obj_enz_constraint,obj_id,E_total,substrate_name,substrate_value,list(Bottleneck_reaction.index),'cplex_direct'): eachmet for eachmet in Bottleneck_reaction_met}
                for future in as_completed(futures):
                    tmp = future.result()
                    for eachindex in tmp.index:
                        #print(eachindex,tmp.loc[eachindex,'max_value'])
                        max_min_concentration_list_fixed.loc[eachindex,'min_concentration'] = tmp.loc[eachindex,'min_value']
            Limiting_metabolite = max_min_concentration_list_fixed[(max_min_concentration_list_fixed['max_concentration'] - max_min_concentration_list_fixed['min_concentration']) <= 0.001]
            Limiting_metabolite_list = list(Limiting_metabolite.index)
        # metabolite
            for metabolite in Limiting_metabolite_list:
                Bottleneck_metabolite = cobra_model.metabolites.get_by_id(metabolite)
                result_row = {'metabolite_id' : Bottleneck_metabolite.id, 'formula' : Bottleneck_metabolite.formula
                            ,'inflection' : num, 'product flux' : obj_enz_constraint, "MDF" : B_value}       
                bottleneck_metabolite_df = pd.concat([bottleneck_metabolite_df, pd.DataFrame(result_row, index=[0])], ignore_index=True)
            use_result_sort = use_result.sort_values(by='enz',ascending = False)
            e_threshold = E_total*0.01
            enz_use_reaction_list = list(use_result_sort[use_result_sort['enz'] > e_threshold].index)
        max_min_E_list_fixed = pd.DataFrame()
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(Get_Max_Min_E,Concretemodel_Need_Data_combine,eachreaction,'maximize',K_value,B_value,\
                            obj_enz_constraint,obj_id,E_total,substrate_name,substrate_value,'cplex_direct'): eachreaction for eachreaction in enz_use_reaction_list}
            for future in as_completed(futures):
                tmp = future.result()
                for eachindex in tmp.index:
                    #print(eachindex,tmp.loc[eachindex,'max_value'])
                    max_min_E_list_fixed.loc[eachindex,'max_E']=tmp.loc[eachindex,'max_value']

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(Get_Max_Min_E,Concretemodel_Need_Data_combine,eachreaction,'minimize',K_value,B_value,\
                            obj_enz_constraint,obj_id,E_total,substrate_name,substrate_value,'cplex_direct'): eachreaction for eachreaction in enz_use_reaction_list}
            for future in as_completed(futures):
                tmp = future.result()
                for eachindex in tmp.index:
                    #print(eachindex,tmp.loc[eachindex,'max_value'])
                    max_min_E_list_fixed.loc[eachindex,'min_E']=tmp.loc[eachindex,'min_value']
        max_min_E_list_fixed.sort_values(by='max_E',ascending = False)        
        max_min_E_list_fixed = max_min_E_list_fixed.sort_values(by=['min_E'],ascending=False)
        Limiting_enzyme = max_min_E_list_fixed[(max_min_E_list_fixed['max_E'] - max_min_E_list_fixed['min_E']) <= 0.001]
        Limiting_enzyme_list = list(Limiting_enzyme.index)
        if len(Limiting_enzyme_list) > 0:
            for reaction in Limiting_enzyme_list:
                Bottleneck_enzyme = cobra_model.reactions.get_by_id(reaction)
                result_row = {'reaction_id' : Bottleneck_enzyme.id, 'equation' : Bottleneck_enzyme.build_reaction_string()
                            , 'gpr' : Bottleneck_enzyme.gene_reaction_rule, 'inflection' : num, 'product flux' : obj_enz_constraint, "MDF" : B_value, "enzyme usage" : use_result_sort.loc[Bottleneck_enzyme.id,"enz"]}       
                bottleneck_enzyme_df = pd.concat([bottleneck_enzyme_df, pd.DataFrame(result_row, index=[0])], ignore_index=True)
        set_MDF_substrate = obj_enz_constraint + 0.01
        print("inflection_points : ", num)
        num += 1
    return [bottleneck_reaction_df, bottleneck_metabolite_df, bottleneck_enzyme_df]