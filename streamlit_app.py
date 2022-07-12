import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import math
import streamlit as st

# # # # # # # # # # # # # # # # # # # # # #
# # # # # Marsh Model & Functions # # # # #
# # # # # # # # # # # # # # # # # # # # # #

def calc_model_variables_Marsh(weight, refresh_rate = 1):
  # Configure Temporal Variables
  steps_per_min = 60 / refresh_rate #number of steps in one minute

  k10 = 0.119 / steps_per_min
  k12 = 0.112 / steps_per_min
  k13 = 0.042 / steps_per_min
  k21 = 0.055 / steps_per_min
  k31 = 0.0033 / steps_per_min
  ke0 = 1.2 / steps_per_min
  # t_half_keo = np.log(2)/(ke0 * steps_per_min) # deprecated

  V1 = 0.228 * weight
  V2 = 0.463 * weight
  V3 = 2.893 * weight

  Cl1 = k10*V1
  Cl2 = k21*V2
  Cl3 = k31*V3

  a0 = k10 * k21 * k31
  a1 = k10 * k31 + k21 * k31 + k21 * k13 + k10 * k21 + k31 * k12
  a2 = k10 + k12 + k13 + k21 + k31

  p = a1 - (a2 * a2/3)
  q = (2 * a2 * a2 * a2 / 27) - (a1 * a2/3) + a0

  r1 = math.sqrt(-(p * p * p) / 27)
  phi = math.acos((-q/2) / r1)/3
  r2 = 2 * math.e**(np.log(r1) / 3)

  root1 = -(math.cos(phi) * r2- a2 / 3)
  root2 = -(math.cos(phi+ 2*math.pi/3) * r2 - a2 / 3)
  root3 = -(math.cos(phi + 4*math.pi/3)* r2 - a2 / 3)

  arr = np.sort(np.array([root1, root2, root3]))
  lambda1 = l1 = arr[2]
  lambda2 = l2 = arr[1]
  lambda3 = l3 = arr[0]

  A1 = C1 =(k21 - l1) * (k31 - l1) / (l1 - l2) / (l1 - l3)/V1
  A2 = C2 =(k21 - l2) * (k31 - l2) / (l2 - l1) / (l2 - l3)/V1
  A3 = C3 =(k21 - l3) * (k31 - l3) / (l3 - l2) / (l3 - l1)/V1

  CoefCe1 = (-ke0*A1)/(l1-ke0)
  CoefCe2 = (-ke0*A2)/(l2-ke0)
  CoefCe3 = (-ke0*A3)/(l3-ke0)
  CoefCe4 = - (CoefCe1 + CoefCe2 + CoefCe3)

  udf = A1+A2+A3

  model_variables = {
      'V1': V1,'V2': V2,'V3': V3,
      'k10': k10,'k12': k12,'k13': k13,'k21': k21,'k31': k31,'ke0': ke0,
      'a0': a0,'a1': a1,'a2': a2,
      'p': p,'q': q,
      'r1': r1,'r2': r2,
      'l1': l1,'l2': l2,'l3': l3,
      'A1': A1,'A2': A2,'A3': A3,
      'CoefCe1': CoefCe1,'CoefCe2': CoefCe2,'CoefCe3': CoefCe3,'CoefCe4': CoefCe4,
      'Cl1': Cl1,'Cl2': Cl2,'Cl3': Cl3,
      'udf': udf
  }

  return model_variables

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Schnider Model & Functions  # # # #
# # # # # # # # # # # # # # # # # # # # # # #
def calc_LBM (weight, height, gender):
  if gender == 0:
    return 1.07*weight-148*(weight/height)**2
  elif gender == 1:
    return 1.1*weight-128*(weight/height)**2
  else:
    return 0

def calc_model_variables_Schnider(weight, height, age, gender, refresh_rate = 1):
  # Configure Temporal Variables
  steps_per_min = 60 / refresh_rate #number of steps in one minute

  lbm = calc_LBM(weight, height, gender)

  # Calculate Model Variables
  V1 = 4.27 #litre
  V2 = 18.9-0.391*(age-53) #litre
  V3 = 238 #litre

  k10 = (0.443+0.0107*(weight-77)-0.0159*(lbm-59)+0.0062*(height-177)) / steps_per_min
  k12 = (0.302-0.0056*(age-53)) / steps_per_min
  k13 = 0.196 / steps_per_min
  k21 = (1.29-0.024*(age-53))/(18.9-0.391*(age-53)) / steps_per_min
  k31 = 0.0035 / steps_per_min
  ke0 = 0.456 / steps_per_min
  # t_half_keo = np.log(2)/(ke0 * steps_per_min) #deprecated

  Cl1 = k10*V1 #litre / steps per min
  Cl2 = k21*V2 #litre / steps per min
  Cl3 = k31*V3 #litre / steps per min

  a0 = k10 * k21 * k31
  a1 = k10 * k31 + k21 * k31 + k21 * k13 + k10 * k21 + k31 * k12
  a2 = k10 + k12 + k13 + k21 + k31

  p = a1 - (a2 * a2/3)
  q = (2 * a2 * a2 * a2 / 27) - (a1 * a2/3) + a0

  r1 = math.sqrt(-(p * p * p) / 27)
  r2 = 2 * math.e**(np.log(r1) / 3)
  phi = math.acos((-q/2) / r1)/3

  root1 = -(math.cos(phi) * r2- a2 / 3)
  root2 = -(math.cos(phi+ 2*math.pi/3) * r2 - a2 / 3)
  root3 = -(math.cos(phi + 4*math.pi/3)* r2 - a2 / 3)

  arr = np.sort(np.array([root1, root2, root3]))
  lambda1 = l1 = arr[2]
  lambda2 = l2 = arr[1]
  lambda3 = l3 = arr[0]

  A1 = C1 = (k21 - l1) * (k31 - l1) / (l1 - l2) / (l1 - l3) / V1
  A2 = C2 = (k21 - l2) * (k31 - l2) / (l2 - l1) / (l2 - l3) / V1
  A3 = C3 = (k21 - l3) * (k31 - l3) / (l3 - l2) / (l3 - l1) / V1

  CoefCe1 = (-ke0*A1)/(l1-ke0)
  CoefCe2 = (-ke0*A2)/(l2-ke0)
  CoefCe3 = (-ke0*A3)/(l3-ke0)
  CoefCe4 = - (CoefCe1 + CoefCe2 + CoefCe3)

  udf = A1+A2+A3

  model_variables = {
      'V1': V1,'V2': V2,'V3': V3,
      'k10': k10,'k12': k12,'k13': k13,'k21': k21,'k31': k31,'ke0': ke0,
      'a0': a0,'a1': a1,'a2': a2,
      'p': p,'q': q,
      'r1': r1,'r2': r2,
      'l1': l1,'l2': l2,'l3': l3,
      'A1': A1,'A2': A2,'A3': A3,
      'CoefCe1': CoefCe1,'CoefCe2': CoefCe2,'CoefCe3': CoefCe3,'CoefCe4': CoefCe4,
      'Cl1': Cl1,'Cl2': Cl2,'Cl3': Cl3,
      'udf': udf
  }

  return model_variables


# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Eleveld Model & Functions   # # # #
# # # # # # # # # # # # # # # # # # # # # # #
def sigmoid(x, y, z):
  return (x**z) / (x**z + y**z)

def central(x):
  return sigmoid(x, 33.6, 1)

def ageing(x, age):
  return math.exp(x*(age-35))

def clmaturation(x):
  return sigmoid(x, 42.3, 9.06)

def q3maturation(x): #Note from simTIVA: age already converted to weeks
  return sigmoid(x+40, 68.3, 1)

def bmi(weight, height):
  return (weight / (height/100)**2)

def ffm(weight, height, age, gender): #fat-free mass
  b = bmi(weight, height)
  if gender == 1:
    return (0.88 + (1-0.88)/(1 + (age/13.4)**-12.7)) * ((9270 * weight)/(6680+216*b))
  else:
    return (1.11 + (1-1.11)/(1 + (age/7.1)**-1.1)) * ((9270 * weight)/(8780+244*b))

def calc_model_variables_Eleveld(weight, height, age, gender, refresh_rate = 1):
  # Configure Temporal Variables
  steps_per_min = 60 / refresh_rate #number of steps in one minute

  # Calculate Model Variables
  opioid = 1 # arbitralily set YES to intraop opioids
  year_to_weeks = 52.1429 # year to weeks constant
  PMA = age*year_to_weeks+40 #arbitrarily set PMA 40 weeks +age

  V1 = 6.28 * central(weight)/central(70)
  V2 = 25.5 * weight/70 * ageing(-0.0156, age)
  V2ref = 25.5
  ffmref = (0.88 + (1-0.88)/(1 + (35/13.4)**-12.7)) * ((9270 * 70)/(6680+216*24.22145))
  V3 = 0
  if (opioid == 1):
    V3 = 273 * ffm(weight = weight, height = height, age = age, gender = gender) / ffmref * math.exp(-0.0138*age)
  else:
    V3 = 273 * ffm(weight = weight, height = height, age = age, gender = gender) / ffmref
  V3ref = 273 #Note from simTIVAjust use this from the table

  Cl1 = 0
  if (gender == 1):
    Cl1 = 1.79 * (weight/70)**0.75 * (clmaturation(PMA)/clmaturation(35*year_to_weeks+40))* math.exp(-0.00286*age) / steps_per_min
  else:
    Cl1 = 2.1 * (weight/70)**0.75 * (clmaturation(PMA)/clmaturation(35*year_to_weeks+40))* math.exp(-0.00286*age) / steps_per_min
  Cl2 = 1.75 * (V2/V2ref)**0.75* (1 + 1.3*(1-q3maturation(age*year_to_weeks))) / steps_per_min
  Cl3 = 1.11 * (V3/V3ref)**0.75*(q3maturation(age*year_to_weeks)/q3maturation(35*year_to_weeks)) / steps_per_min

  k10 = Cl1 / V1
  k12 = Cl2 / V1
  k13 = Cl3 / V1
  k21 = Cl2 / V2
  k31 = Cl3 / V3
  ke0 = 0.146 * (weight/70)**-0.25

  a0 = k10 * k21 * k31
  a1 = k10 * k31 + k21 * k31 + k21 * k13 + k10 * k21 + k31 * k12
  a2 = k10 + k12 + k13 + k21 + k31

  p = a1 - (a2 * a2/3)
  q = (2 * a2 * a2 * a2 / 27) - (a1 * a2/3) + a0

  r1 = math.sqrt(-(p * p * p) / 27)
  r2 = 2 * math.e**(np.log(r1) / 3)
  phi = math.acos((-q/2) / r1)/3

  root1 = -(math.cos(phi) * r2- a2 / 3)
  root2 = -(math.cos(phi+ 2*math.pi/3) * r2 - a2 / 3)
  root3 = -(math.cos(phi + 4*math.pi/3)* r2 - a2 / 3)

  arr = np.sort(np.array([root1, root2, root3]))
  lambda1 = l1 = arr[2]
  lambda2 = l2 = arr[1]
  lambda3 = l3 = arr[0]

  A1 = C1 = (k21 - l1) * (k31 - l1) / (l1 - l2) / (l1 - l3) / V1
  A2 = C2 = (k21 - l2) * (k31 - l2) / (l2 - l1) / (l2 - l3) / V1
  A3 = C3 = (k21 - l3) * (k31 - l3) / (l3 - l2) / (l3 - l1) / V1

  CoefCe1 = (-ke0*A1)/(l1-ke0)
  CoefCe2 = (-ke0*A2)/(l2-ke0)
  CoefCe3 = (-ke0*A3)/(l3-ke0)
  CoefCe4 = - (CoefCe1 + CoefCe2 + CoefCe3)

  udf = A1+A2+A3

  model_variables = {
      'V1': V1,'V2': V2,'V3': V3,
      'k10': k10,'k12': k12,'k13': k13,'k21': k21,'k31': k31,'ke0': ke0,
      'a0': a0,'a1': a1,'a2': a2,
      'p': p,'q': q,
      'r1': r1,'r2': r2,
      'l1': l1,'l2': l2,'l3': l3,
      'A1': A1,'A2': A2,'A3': A3,
      'CoefCe1': CoefCe1,'CoefCe2': CoefCe2,'CoefCe3': CoefCe3,'CoefCe4': CoefCe4,
      'Cl1': Cl1,'Cl2': Cl2,'Cl3': Cl3,
      'udf': udf
  }

  return model_variables

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Simuate Model: Marsh, Schnider, Eleveld
# weight in kg
# height in cm
# age in years old
# gender, 0 = Female , 1 = Male, follows number of Y chromosome
# depth as in CeT (mcg/mL)
# duration in mins (integer)

# Default Temporal Configurations
# duration_in_secs = duration * 60 in seconds
# refresh_rate = 10 second

# Defualt Propofol Configuration
# propofol_density = 10 mg/ml
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def simulate_model(weight, height, age, gender, depth, duration, model, refresh_rate = 10, propofol_density = 10):

  # Configure Temporal Variables
  duration_in_secs = duration * 60 #in seconds
  steps_per_min = 60 / refresh_rate #number of steps in one minute

  # #Calculate & unpack model variables
  model_variables = {}
  if(model == 'Schnider'):
    model_variables = calc_model_variables_Schnider(weight = weight, height = height, age = age, gender = gender, refresh_rate = refresh_rate)
  elif(model == 'Eleveld'):
    model_variables = calc_model_variables_Eleveld(weight = weight, height = height, age = age, gender = gender, refresh_rate = refresh_rate)
  elif (model == 'Marsh'):
    model_variables = calc_model_variables_Marsh(weight = weight, refresh_rate = refresh_rate)
  CoefCe1 = model_variables['CoefCe1']
  CoefCe2 = model_variables['CoefCe2']
  CoefCe3 = model_variables['CoefCe3']
  CoefCe4 = model_variables['CoefCe4']
  l1 = model_variables['l1']
  l2 = model_variables['l2']
  l3 = model_variables['l3']
  ke0 = model_variables['ke0']
  udf = model_variables['udf']
  A1 = model_variables['A1']
  A2 = model_variables['A2']
  A3 = model_variables['A3']

  # Initiate Empty Variables for the Iterative Process
  bolus = 0
  infusion = 0

  Cp_target = depth
  Ce_states = [0.0, 0.0, 0.0, 0.0]
  Cp_comp = [0.0, 0.0, 0.0]
  Cp_trail = [0.0, 0.0, 0.0]

  Cp = 0
  Ce = 0
  udf_delta = 0
  step_delta = 0
  df = pd.DataFrame(columns=['Step','Dr','Time','Step Delta', 'Bolus', 'Infusion', 'CP', 'CE'])

  # Commence Iterative Calculation, step_current = 0 is when the initial bolus is applied
  for d in np.arange(0, duration_in_secs + refresh_rate, refresh_rate):
    step_current = int(d/refresh_rate)
    duration_current = d/60
    time = timedelta(seconds = int(d))

    if(step_current == 0):
      if (model == 'Marsh'):
        bolus = (0.2197*Cp_target**2 + 15.693*Cp_target) / 70 * weight
      elif((model == 'Schnider') | (model == 'Eleveld')): #determinal bolus via peak Ce
        peak_Ce, peak_TTPE = cacl_peak_Ce(weight = weight, height = height, age = age, gender = gender, model = model)
        bolus = depth / peak_Ce
      step_delta = 0
    else:
      bolus = 0
      step_delta = 1

    Cp_trail[0] = Cp_comp[0]*math.e**(-l1*step_delta)
    Cp_trail[1] = Cp_comp[1]*math.e**(-l2*step_delta)
    Cp_trail[2] = Cp_comp[2]*math.e**(-l3*step_delta)

    udf_delta = max([(Cp_target-sum(Cp_trail))/udf,0.0])
    if(step_current == 0):
      infusion = 0.0
    else:
      infusion = udf_delta

    Cp_comp[0] = bolus*A1+Cp_comp[0]*math.e**(-l1*step_delta)+(infusion * (A1/l1) * (1 - math.e**(-l1 * step_delta)))
    Cp_comp[1] = bolus*A2+Cp_comp[1]*math.e**(-l2*step_delta)+(infusion * (A2/l2) * (1 - math.e**(-l2 * step_delta)))
    Cp_comp[2] = bolus*A3+Cp_comp[2]*math.e**(-l3*step_delta)+(infusion * (A3/l3) * (1 - math.e**(-l3 * step_delta)))
    Cp = sum(Cp_comp)

    #Calcuate Ce for non-Marsh model
    if(model != 'Marsh'):
      Ce_states[0] = bolus*CoefCe1+Ce_states[0]*math.e**(-l1*step_delta) + infusion * (CoefCe1/l1) * (1 - math.e**(-l1 * step_delta))
      Ce_states[1] = bolus*CoefCe2+Ce_states[1]*math.e**(-l2*step_delta) + infusion * (CoefCe2/l2) * (1 - math.e**(-l2 * step_delta))
      Ce_states[2] = bolus*CoefCe3+Ce_states[2]*math.e**(-l3*step_delta) + infusion * (CoefCe3/l3) * (1 - math.e**(-l3 * step_delta))
      Ce_states[3] = bolus*CoefCe4+Ce_states[3]*math.e**(-ke0*step_delta) + infusion * (CoefCe4/ke0) * (1 - math.e**(-ke0 * step_delta))
      Ce = sum(Ce_states)

    df=df.append(pd.DataFrame({'Step':[step_current], 'Dr':[duration_current], 'Time': [time], 'Step Delta':[step_delta], 'Bolus':[bolus], 'Infusion':[infusion], 'CP': [Cp], 'CE': [Ce] }), ignore_index=True)

  df['Infusion Accumulated'] = df['Infusion'].cumsum() # in mcg
  df['Bolus Accumulated'] = df['Bolus'].cumsum() # in mcg
  df['Volume'] = (df['Bolus Accumulated'] + df['Infusion Accumulated']) / propofol_density #in mL

  #Drop Ce column for Marsh model
  if(model == 'Marsh'):
    df.drop('CE', axis=1, inplace=True)

  return df

#cacl_peak_Ce requires refresh rate of 1 second, therefore all the configurations are different than the main function.
def cacl_peak_Ce(weight, height, age, gender, model):

  # Configure Temporal Variables
  refresh_rate = 1 #second, 1 second refresh rate for calculating Peak Ce to achieve the most accurate result
  step = 60 / refresh_rate #number of steps in one minute

  #Calculate & unpack model variables
  model_variables = {}
  if(model == 'Schnider'):
    model_variables = calc_model_variables_Schnider(weight = weight, height = height, age = age, gender = gender, refresh_rate = refresh_rate)
  elif(model =='Eleveld'):
    model_variables = calc_model_variables_Eleveld(weight = weight, height = height, age = age, gender = gender, refresh_rate = refresh_rate)
  CoefCe1 = model_variables['CoefCe1']
  CoefCe2 = model_variables['CoefCe2']
  CoefCe3 = model_variables['CoefCe3']
  CoefCe4 = model_variables['CoefCe4']
  l1 = model_variables['l1']
  l2 = model_variables['l2']
  l3 = model_variables['l3']
  ke0 = model_variables['ke0']

  #Calculate Peak Ce with 1 mcg bolus
  bolus = 1 #mcg
  infusion = 0 #mcg
  step_delta = 1
  Ce_states = [bolus * CoefCe1, bolus * CoefCe2, bolus * CoefCe3, bolus * CoefCe4]
  Ce = [0]

  while ((len(Ce) <= 2) | (Ce[len(Ce)-2] < Ce[len(Ce)-1])):
    bolus = 0
    Ce_states[0] = bolus*CoefCe1+Ce_states[0]*math.e**(-l1*step_delta) + infusion * (CoefCe1/l1) * (1 - math.e**(-l1 * step_delta))
    Ce_states[1] = bolus*CoefCe2+Ce_states[1]*math.e**(-l2*step_delta) + infusion * (CoefCe2/l2) * (1 - math.e**(-l2 * step_delta))
    Ce_states[2] = bolus*CoefCe3+Ce_states[2]*math.e**(-l3*step_delta) + infusion * (CoefCe3/l3) * (1 - math.e**(-l3 * step_delta))
    Ce_states[3] = bolus*CoefCe4+Ce_states[3]*math.e**(-ke0*step_delta) + infusion * (CoefCe4/ke0) * (1 - math.e**(-ke0 * step_delta))
    Ce.append(sum(Ce_states))
    # print(Ce_prev)
    # print(Ce_states)

  peak_Ce = Ce[len(Ce)-2]
  peak_TTPE = (len(Ce)-2) * refresh_rate
  return peak_Ce, peak_TTPE


st.title('Propofol Dream Online App')

is_disabled=False

weight = 0.0
height = 0.0
age = 0
gender = 0

model = st.selectbox(
     'Please select your model',
     ('Marsh', 'Schnider'))

st.write('Model selected:', model)

if model == 'Marsh':
    weight = st.number_input('Please enter Weight',min_value=0.1, step = 1.0, value = 40.0)
    st.write('''Patient's Weight is ''', weight,' kg')

if model == 'Schnider':
    weight = st.number_input('Please enter Weight',min_value=0.1, step = 1.0, value = 40.0)
    st.write('''Patient's Weight is ''', weight,' kg')

    height = st.number_input('Please enter Height',min_value=10, step = 10, value = 140)
    st.write('''Patient's Height is ''', height,' cm')

    age = st.number_input('Please enter Age',min_value=0, step = 1, value = 30)
    st.write('''Patient's Age is ''', age,' years-old')

    gender_selected = st.selectbox(
         'Please select Gender',
         ('Female', 'Male'))

    st.write('Gender selected:', gender_selected)

    if gender_selected == 'Female':
        gender = 0
    else:
        gender = 1

duration = st.number_input('Please enter Duration',min_value=0.0, step = 1.0, value = 20.0)
st.write('Operation Duration is ', duration,' mins')

depth = st.number_input('Please enter Depth',min_value=0.1, step = 0.5, value = 3.0)
st.write('Operation Depth is ', depth,' CeT (mcg/mL)')

df_sim = 0

if model == 'Marsh':
    df_sim = simulate_model(age = 0, weight = weight, height = 0, gender = 0, duration = duration, depth = depth, model = model)
    # df_sim.drop('Duration', axis = 1, inplace = True)
    # result = df_sim['Volume'].iloc[-1]
    st.write(df_sim.dtypes.astype(str))
    st.dataframe(df_sim)
    st.download_button(
         label="Download data as CSV",
         data= df_sim.to_csv().encode('utf-8'),
         file_name='df_sim.csv',
         mime='text/csv')


elif model == 'Schnider':
    df_sim = simulate_model(age = age, weight = weight, height = height, gender = gender, duration = duration, depth = depth, model = model)
    # df_sim.drop('Duration', axis = 1, inplace = True)
    # result = df_sim['Volume'].iloc[-1]
    st.dataframe(df_sim)
    st.download_button(
         label="Download data as CSV",
         data= df_sim.to_csv().encode('utf-8'),
         file_name='df_sim.csv',
         mime='text/csv')
