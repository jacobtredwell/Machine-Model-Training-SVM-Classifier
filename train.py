import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle



#================================================================================================================ 
#Get Columns from INSULINData CSV file
#================================================================================================================

Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_in = pd.read_csv('InsulinData.csv', usecols=Insulin_col_list)
df_in['DateTime']= pd.to_datetime(df_in.pop("Date")) + pd.to_timedelta(df_in.pop("Time"))

#================================================================================================================ 
#removes all rows without carb intake
#================================================================================================================

df_in.dropna(how='any', inplace=True)  
df_in.reset_index(drop=True, inplace=True)

#print(df_in.head(50))

datetime_insulin_series = df_in["DateTime"]
carb_series = df_in["BWZ Carb Input (grams)"]

CGM_col_list = ["Date", "Time", "Sensor Glucose (mg/dL)"]
df_cgm = pd.read_csv('CGMData.csv', usecols=CGM_col_list)
df_cgm.dropna()

df_cgm['DateTime'] = pd.to_datetime(df_cgm.pop("Date")) + pd.to_timedelta(df_cgm.pop("Time"))


datetime_glucose_series = df_cgm["DateTime"]
glucose_series = df_cgm["Sensor Glucose (mg/dL)"]

#--------------------------------------------------------------------------------------------------------------------
# if carbs exists at a time between 2 hours, then get the time of the start of the meal

#get all the carbs that are not NaN
#--------------------------------------------------------------------------------------------------------------------


#print(df_in.head(100))

insulin_length = df_in.index

in_size = len(insulin_length)

total_meal_count = 0

meal_start_times = []

#================================================================================================================ 
#Find Meal Start times - patient 1
#================================================================================================================ 


for i in range(0, in_size - 1, 1):
    
    if (datetime_insulin_series[i] - datetime_insulin_series[i+1]) >= pd.Timedelta(hours= 2, minutes=30) and carb_series[i+1] != 0:
        #print(datetime_insulin_series[i])
    
        meal_start_time = (datetime_insulin_series[i+1] - pd.Timedelta("30 minutes"))
        #print(meal_start_time)
        total_meal_count += 1
        meal_start_times.append(meal_start_time)
       


#print(total_meal_count)
#print(meal_start_times)
#print(len(meal_start_times))




#================================================================================================================ 
#FINDING CORRESPONDING TIME IN OTHER FILE
#================================================================================================================ 

index = df_cgm.index
cgm_total_rows = len(index)

cgm_meal_start_indices = []



loopstart=0

for meal_time in meal_start_times:
    #print(meal_time)
    
    for i in range(loopstart, cgm_total_rows, 1):
    
        if pd.Timedelta("0 minutes") <= (datetime_glucose_series[i] - meal_time) < pd.Timedelta("5 minutes"):
            #print(datetime_glucose_series[i])
            cgm_meal_start_indices.append(i)
            loopstart = i
            break
    


#print(cgm_meal_start_indices)

#print(len(cgm_meal_start_indices))

#print(cgm_meal_start_indices[1])
#print(glucose_series[cgm_meal_start_indices[1]])

#================================================================================================================ 
#GET MEAL DATA USING INDICES OF MEAL DATA STARTS
#================================================================================================================ 


meal_array =[0]*30
meal_data_df = pd.DataFrame()

for index in cgm_meal_start_indices:
    for i in range(0,30,1):
        meal_array[i] = glucose_series[index-i]
    
    meal_series = pd.Series(meal_array)
    meal_data_df = meal_data_df.append(meal_series, ignore_index=True) 
    
#print (meal_data_df.head(500))   

#meal_data_df.to_csv('Meal_data_Results.csv', header=False, index=False)


#================================================================================================================ 
#IMPORT PATIENT 2
#================================================================================================================ 
Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_in_2 = pd.read_csv('Insulin_patient2.csv', usecols=Insulin_col_list)
df_in_2['DateTime']= pd.to_datetime(df_in_2.pop("Date")) + pd.to_timedelta(df_in_2.pop("Time"))

#================================================================================================================ 
#removes all rows without carb intake
#================================================================================================================ 

print("Patient 2 .....................................")
df_in_2.dropna(how='any', inplace=True)  
df_in_2.reset_index(drop=True, inplace=True)

#print(df_in_2.head(50))

datetime_insulin_series_2 = df_in_2["DateTime"]
carb_series_2 = df_in_2["BWZ Carb Input (grams)"]


CGM_col_list = ["Date", "Time", "Sensor Glucose (mg/dL)"]
df_cgm_2 = pd.read_csv('CGM_patient2.csv', usecols=CGM_col_list)
df_cgm_2.dropna()
df_cgm_2['DateTime'] = pd.to_datetime(df_cgm_2.pop("Date")) + pd.to_timedelta(df_cgm_2.pop("Time"))


datetime_glucose_series_2 = df_cgm_2["DateTime"]
glucose_series_2 = df_cgm_2["Sensor Glucose (mg/dL)"]





#================================================================================================================ 
#Find Meal Start times - patient 2
#================================================================================================================ 

insulin_length_2 = df_in_2.index

in_2_size = len(insulin_length_2)

total_meal_count_2 = 0

meal_start_times_2 = []

for i in range(0, in_2_size - 1, 1):
    
    if (datetime_insulin_series_2[i] - datetime_insulin_series_2[i+1]) >= pd.Timedelta(hours= 2, minutes=30) and carb_series_2[i+1] != 0:
        #print(datetime_insulin_series[i])
    
        meal_start_time_2 = (datetime_insulin_series_2[i+1] - pd.Timedelta("30 minutes"))
        #print(meal_start_time_2)
        total_meal_count_2 += 1
        meal_start_times_2.append(meal_start_time_2)

        
#print(meal_start_times_2)
#print(total_meal_count_2)

#================================================================================================================ 
#FINDING CORRESPONDING TIME IN OTHER FILE
#================================================================================================================ 

index_2 = df_cgm_2.index
cgm_total_rows_2 = len(index_2)

#print(cgm_total_rows_2)
cgm_meal_start_indices_2 = []

loopstart_2 = 0

for meal_time_2 in meal_start_times_2:
    
    
    for j in range(loopstart_2, cgm_total_rows_2, 1):
        #print(meal_time_2)
        #print(datetime_glucose_series_2[i])
        #print(datetime_glucose_series_2[i] - meal_time_2)
        
        
        if pd.Timedelta("0 minutes") <= (datetime_glucose_series_2[j] - meal_time_2) < pd.Timedelta("10 minutes"):
            #print("Success!")
            cgm_meal_start_indices_2.append(j)
            #print(j)
            loopstart_2 = j
            break
    


#print(cgm_meal_start_indices_2)

#print(len(cgm_meal_start_indices_2))

#print(cgm_meal_start_indices_2[1])
#print(glucose_series[cgm_meal_start_indices_2[1]])

#================================================================================================================ 
#Get MEAL DATA - patient 2
#================================================================================================================ 

#print("now on last step")

meal_array_2 =[0]*30
#meal_data_df = pd.DataFrame()  - ALREADY EXISTS

for index_2 in cgm_meal_start_indices_2:
    for i in range(0,30,1):
        meal_array_2[i] = glucose_series_2[index_2 - i]
    
    meal_series_2 = pd.Series(meal_array_2)
#APPEND to meal_data_df
    meal_data_df = meal_data_df.append(meal_series_2, ignore_index=True) 
    
#print (meal_data_df.head(500))   

#meal_data_df.to_csv('Meal_data_Results.csv', header=False, index=False)




#================================================================================================================ 
#GET NO MEAL DATA - patient 1
#================================================================================================================ 

Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_in_NM = pd.read_csv('InsulinData.csv', usecols=Insulin_col_list)
df_in_NM['DateTime']= pd.to_datetime(df_in_NM.pop("Date")) + pd.to_timedelta(df_in_NM.pop("Time"))

NM_datetime_insulin_series = df_in_NM["DateTime"]
NM_carb_series = df_in_NM["BWZ Carb Input (grams)"]

NM_insulin_length = df_in_NM.index
NM_in_size = len(NM_insulin_length)
#print(NM_in_size)
total_no_meal_count = 0

df_cgm = pd.read_csv('CGMData.csv', usecols=CGM_col_list)
df_cgm.dropna()

df_cgm['DateTime'] = pd.to_datetime(df_cgm.pop("Date")) + pd.to_timedelta(df_cgm.pop("Time"))


datetime_glucose_series = df_cgm["DateTime"]
glucose_series_2 = df_cgm["Sensor Glucose (mg/dL)"]

no_meal_start_times = []

prev_meal = NM_datetime_insulin_series[0] + pd.Timedelta("35 minutes")

for meal_time in meal_start_times:
    
    for i in range(0, NM_in_size, 1):
        
        if (prev_meal - NM_datetime_insulin_series[i]) > pd.Timedelta("30 minutes") and (
            NM_datetime_insulin_series[i] - pd.Timedelta("2 hours")) > meal_time:
            
            no_meal_start_times.append(NM_datetime_insulin_series[i] - pd.Timedelta("2 hours"))
            total_no_meal_count += 1
            prev_meal = meal_time
            break
                                       
       
    #else:
        #continue  # only runs if inner loop DID NOT break
        
    #break   #runs if inner loop DID break

                                       
#print("no_meal_start_times........")                                       
#print(no_meal_start_times)
#print("total_no_meal_count = ")
#print(total_no_meal_count)

#================================================================================================================ 
#FINDING CORRESPONDING TIME IN OTHER FILE
#================================================================================================================
#print('moving on................')
index = df_cgm.index
cgm_total_rows = len(index)

no_meal_start_indices = []

#print("finding times in other file")

loopstart=0

for no_meal_time in no_meal_start_times:
    #print(meal_time)
    
    for i in range(loopstart, cgm_total_rows, 1):
    
        if pd.Timedelta("0 minutes") <= (datetime_glucose_series[i] - no_meal_time) < pd.Timedelta("5 minutes"):
            #print(datetime_glucose_series[i])
            no_meal_start_indices.append(i)
            loopstart = i
            break
    
#print("done with finding no meal corresponding time")
#================================================================================================================ 
#GET MEAL DATA USING INDICES OF NO MEAL DATA -1- STARTS
#================================================================================================================ 

no_meal_array =[0]*24
no_meal_data_df = pd.DataFrame()

for index in no_meal_start_indices:
    for i in range(0,24,1):
        no_meal_array[i] = glucose_series[index-i]
        #print(no_meal_array[i])
    
    no_meal_series = pd.Series(no_meal_array)
    no_meal_data_df = no_meal_data_df.append(no_meal_series, ignore_index=True) 
    
#print (no_meal_data_df.head(500))   

#meal_data_df.to_csv('No_Meal_data_Results.csv', header=False, index=False)

#================================================================================================================ 


#================================================================================================================ 
#Handle Missing Data
#====================================================================================
# removes row if there are more than 3 Nan values
#print(len(meal_data_df))
#print(meal_data_df)
      
#print(len(no_meal_data_df))
#print(no_meal_data_df)

meal_data_df = meal_data_df[meal_data_df.isnull().sum(axis=1) < 3]

no_meal_data_df = no_meal_data_df[no_meal_data_df.isnull().sum(axis=1) < 3]

meal_data_df.to_csv('Refined_Meal_data_Results.csv', header=False, index=False)
no_meal_data_df.to_csv('Refined_No_Meal_data_Results.csv', header=False, index=False)

meal_data_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                       '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
#print(meal_data_df)

no_meal_data_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                       '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
#print(no_meal_data_df)

#================================================================================================================ 
#FEATURE EXTRACTOR
#================================================================================================================ 

#meal_data_features
total_meal_data_rows = meal_data_df.count
total_no_meal_data_rows = no_meal_data_df.count

column_names = ["Max-Min %", "Time", "Gradient", "Label"]

meal_feature_DF = pd.DataFrame(columns = column_names)
no_meal_feature_DF = pd.DataFrame(columns = column_names)


# MIN MAX PERCENTAGE %  --------------------------------------------------


Meal_Max_Min_Percentage = (meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14',
                                         '15', ]].max(axis=1) - meal_data_df[['1', '2','3']].min(axis=1)) / meal_data_df[['1', '2']].min(axis=1)

No_Meal_Max_Min_Percentage = (no_meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14',
                                         '15']].max(axis=1) - no_meal_data_df[['1', '2','3']].min(axis=1)) / no_meal_data_df[['1', '2', '3']].min(axis=1)

#print(max_meal_CGM)
#print(max_nomeal_CGM)

#print(min_meal_CGM)
#print(min_nomeal_CGM)

#print(Meal_Max_Min_Percentage)
#print(No_Meal_Max_Min_Percentage)

meal_feature_DF["Max-Min %"] = Meal_Max_Min_Percentage
no_meal_feature_DF["Max-Min %"] = No_Meal_Max_Min_Percentage

#print(meal_feature_DF["Max-Min %"].mean())
#print(no_meal_feature_DF["Max-Min %"].mean())


# TIME until MAX glucose -------------------------------------------------

time_max_G_meal = meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14',
                                         '15', '16', '17', '18']].idxmax(axis=1).astype(int)*5
time_max_G_no_meal = no_meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14',
                                         '15', '16', '17', '18']].idxmax(axis=1).astype(int)*5

#print(time_max_G_meal)
##print(time_max_G_no_meal)

meal_feature_DF["Time"] = time_max_G_meal
no_meal_feature_DF["Time"] = time_max_G_no_meal

#print(meal_feature_DF["Time"].mean())
#print(no_meal_feature_DF["Time"].mean())


# FFT FAST FOURIER TRANSFORM ----------------------------------------------

fft_meal = np.fft.fft(meal_data_df)
fft_nomeal = np.fft.fft(no_meal_data_df)

#meal_feature_DF["FFT"] = fft_meal
#no_meal_feature_DF["FFT"] = fft_nomeal

#print (fft_meal)

# GRADIENT - dCGM/dt ------------------------------------------------------------------

meal_gradient = np.gradient(meal_data_df)
no_meal_gradient = np.gradient(no_meal_data_df)

meal_gradient_df = pd.DataFrame()
no_meal_gradient_df = pd.DataFrame()

#print(meal_gradient)

size = len(meal_data_df)

max_gradients = [0]*size

for array in meal_gradient:
    
    for i in range(len(array)):
        
        x = max(array[i])
        #print(x)
        #meal_gradient_df.append(pd.Series(x), ignore_index=True)
        max_gradients[i] = x



meal_gradient_df = meal_gradient_df.append(max_gradients)
#print (max_gradients)

no_size = len(no_meal_data_df)

nm_max_gradients = [0]*no_size

for array in no_meal_gradient:
    
    for i in range(len(array)):
        
        x = max(array[i])
        
        nm_max_gradients[i] = x

no_meal_gradient_df = no_meal_gradient_df.append(nm_max_gradients)
#print (meal_gradient_df)

#print("Number of meal max gradients: ", len(meal_gradient_df))
#print("Number of no meal max gradients: ", len(no_meal_gradient_df))
#print(no_meal_gradient_df)



meal_feature_DF["Gradient"] = meal_gradient_df
no_meal_feature_DF["Gradient"] = no_meal_gradient_df

#print(meal_feature_DF["Gradient"].mean())
#print(no_meal_feature_DF["Gradient"].mean())



#print("--- %s seconds ---" % (time.time() - start_time))

#====================================================================================================
#TRAIN MODEL ----------------------------------------------------------------------------------------
#====================================================================================================

# print the names of the 3 features
print("Features: Max-Min%, Time, Gradient")

# print the label type of cancer('malignant' 'benign')
print("Labels: 'Meal' 'No Meal'")

meal_feature_DF["Label"] = 1
no_meal_feature_DF["Label"] = 0

meal_feature_DF.to_csv('Meal_Features_DF.csv', header=False, index=False)

no_meal_feature_DF.to_csv('No_Meal_Features_DF.csv', header=False, index=False)

all_feature_df = pd.DataFrame()
all_feature_df = all_feature_df.append(meal_feature_DF)
all_feature_df = all_feature_df.append(no_meal_feature_DF)

all_feature_df =  all_feature_df.dropna()

all_feature_df.to_csv('All_Feature_DF.csv', header=False, index=False)

X_train, X_test, y_train, y_test = train_test_split(all_feature_df.drop(["Label"], axis=1), all_feature_df["Label"], test_size=0.2, random_state = 109)

#new Classifier

clf = SVC(kernel='linear', gamma='auto')

clf.fit(X_train, y_train)  #train the model using training sets

#predict response for test dataset

y_pred = clf.predict(X_test)

from sklearn import metrics

acc = metrics.accuracy_score(y_test, y_pred)

#print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

prec = metrics.precision_score(y_test, y_pred)

#print("Precision: ", metrics.precision_score(y_test, y_pred))

recall = metrics.recall_score(y_test, y_pred)

#print ("Recall: ", metrics.recall_score(y_test, y_pred))

f1 = 2*recall*prec/(recall + prec)

#print("F1 Score: ", f1)

#print(y_pred)

#print(len(y_pred))

filename = 'finalized_model.pickle'
pickle.dump(clf, open(filename, 'wb'))
