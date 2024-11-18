import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model (ensure you have saved your model using joblib or pickle)
model = joblib.load('expresso_churn_model.pkl')

# Streamlit UI
st.title('Expresso Churn Prediction')

# Input fields for each feature (match these with your dataset columns)
Montant = st.number_input('MONTANT', min_value=0)
Frequence_rech = st.number_input('FREQUENCE_RECH')
Revenue = st.number_input('REVENUE')
Frequence = st.number_input('FREQUENCE')
Data_volume = st.number_input('DATA_VOLUME')
On_net = st.number_input('ON_NET')
Orange = st.number_input('ORANGE')
Regularity=  st.number_input('REGULARITY')
Freq_Top_Pack = st.number_input('FREQ_TOP_PACK')
regions = ['FATICK', 'DAKAR', 'LOUGA', 'TAMBACOUNDA', 'KAOLACK', 'THIES',
 'SAINT-LOUIS', 'KOLDA,' 'KAFFRINE', 'DIOURBEL', 'ZIGUINCHOR', 'MATAM',
 'SEDHIOU', 'KEDOUGOU']
Region = st.selectbox("Region",regions)
Tenure_e = ['K > 24 month', 'I 18-21 month', 'G 12-15 month', 'H 15-18 month',
 'J 21-24 month', 'F 9-12 month', 'D 3-6 month', 'E 6-9 month']
Tenure = st.selectbox("Tenure",Tenure_e)
Mrg = ['NO']
MRG = st.selectbox("MRG",Mrg)
TOP_PAC = ['On net 200F=Unlimited _call24H', 'On-net 1000F=10MilF;10d',
 'Data:1000F=5GB,7d', 'Mixt 250F=Unlimited_call24H',
 'MIXT:500F= 2500F on net _2500F off net;2d', 'All-net 500F=2000F;5d',
 'On-net 500F_FNF;3d', 'Data: 100 F=40MB,24H',
 'MIXT: 200mnoff net _unl on net _5Go;30d', 'Jokko_Daily',
 'Data: 200 F=100MB,24H', 'Data:490F=1GB,7d', 'Twter_U2opia_Daily',
 'On-net 500=4000,10d', 'Data:1000F=2GB,30d', 'IVR Echat_Daily_50F',
 'Pilot_Youth4_490', 'All-net 500F =2000F_AllNet_Unlimited',
 'Twter_U2opia_Weekly', 'Data:200F=Unlimited,24H', 'On-net 200F=60mn;1d',
 'All-net 600F= 3000F ;5d', 'Pilot_Youth1_290',
 'All-net 1000F=(3000F On+3000F Off);5d', 'VAS(IVR_Radio_Daily)',
 'Data:3000F=10GB,30d', 'All-net 1000=5000;5d', 'Twter_U2opia_Monthly',
 'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h\t', 'FNF2 ( JAPPANTE)',
 'Yewouleen_PKG', 'Data:150F=SPPackage1,24H', 'WIFI_Family_2MBPS',
 'Data:500F=2GB,24H', 'MROMO_TIMWES_RENEW', 'New_YAKALMA_4_ALL',
 'Data:1500F=3GB,30D', 'All-net 500F=4000F ; 5d' 'Jokko_promo',
 'All-net 300=600;2d', 'Data:300F=100MB,2d',
 'MIXT: 590F=02H_On-net_200SMS_200 Mo;24h\t\t',
 'All-net 500F=1250F_AllNet_1250_Onnet;48h', 'Facebook_MIX_2D',
 '500=Unlimited3Day', 'On net 200F= 3000F_10Mo ;24H', '200=Unlimited1Day',
 'YMGX 100=1 hour FNF, 24H/1 month', 'SUPERMAGIK_5000',
 'Data:DailyCycle_Pilot_1.5GB', 'Staff_CPE_Rent',
 'MIXT:1000F=4250 Off net _ 4250F On net _100Mo; 5d', 'Data:50F=30MB_24H',
 'Data:700F=SPPackage1,7d', 'Data: 490F=Night,00H-08H', 'Data:700F=1.5GB,7d',
 'Data:1500F=SPPackage1,30d', 'Data:30Go_V 30_Days', 'MROMO_TIMWES_OneDAY',
 'On-net 300F=1800F;3d', 'All-net 5000= 20000off+20000on;30d',
 'WIFI_ Family _4MBPS', 'CVM_on-net bundle 500=5000',
 'Internat: 1000F_Zone_3;24h\t\t', 'DataPack_Incoming', 'Jokko_Monthly',
 'EVC_500=2000F' 'On-net 2000f_One_Month_100H; 30d',
 'MIXT:10000F=10hAllnet_3Go_1h_Zone3;30d\t\t', 'EVC_Jokko_Weekly',
 '200F=10mnOnNetValid1H', 'IVR Echat_Weekly_200F', 'WIFI_ Family _10MBPS',
 'Internat: 1000F_Zone_1;24H\t\t', 'Jokko_Weekly', 'SUPERMAGIK_1000',
 'MIXT: 500F=75(SMS, ONNET, Mo)_1000FAllNet;24h\t\t',
 'VAS(IVR_Radio_Monthly)', 'MIXT: 5000F=80Konnet_20Koffnet_250Mo;30d\t\t',
 'Data: 200F=1GB,24H', 'EVC_JOKKO30', 'NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE',
 'TelmunCRBT_daily', 'FIFA_TS_weekly', 'VAS(IVR_Radio_Weekly)',
 'Internat: 2000F_Zone_2;24H\t\t', 'APANews_weekly', 'EVC_100Mo',
 'pack_chinguitel_24h', 'Data_EVC_2Go24H',
 'Mixt : 500F=2500Fonnet_2500Foffnet ;5d', 'FIFA_TS_daily',
 'MIXT: 4900F= 10H on net_1,5Go ;30d', 'CVM_200f=400MB',
 'IVR Echat_Monthly_500F', 'All-net 500= 4000off+4000on;24H',
 'FNF_Youth_ESN', 'Data:1000F=700MB,7d', '1000=Unlimited7Day',
 'Incoming_Bonus_woma', 'CVM_100f=200 MB', 'CVM_100F_unlimited',
 'pilot_offer6', '305155009', 'Postpaid FORFAIT 10H Package', 'EVC_1Go',
 'GPRS_3000Equal10GPORTAL', 'NEW_CLIR_PERMANENT_LIBERTE_MOBILE',
 'Data_Mifi_10Go_Monthly', '1500=Unlimited7Day', 'EVC_700Mo',
 'CVM_100f=500 onNet', 'CVM_On-net 1300f=12500', 'pilot_offer5',
 'EVC_4900=12000F', 'CVM_On-net 400f=2200F', 'YMGX on-net 100=700F, 24H',
 'CVM_150F_unlimited', 'EVC_MEGA10000F', 'pilot_offer7', 'CVM_500f=2GB',
 'SMS Max', '301765007', '150=unlimited pilot auto',
 'MegaChrono_3000F=12500F TOUS RESEAUX', 'pilot_offer4', 'Go-NetPro-4 Go',
 '200=unlimited pilot auto', 'ESN_POSTPAID_CLASSIC_RENT', 'Data_Mifi_10Go',
 'Data:New-GPRS_PKG_1500F', 'GPRS_BKG_1000F MIFI',
 'Data:OneTime_Pilot_1.5GB', 'FIFA_TS_monthly', 'GPRS_PKG_5GO_ILLIMITE',
 'Data_Mifi_20Go', 'APANews_monthly',
 'NEW_CLIR_TEMPRESTRICTED_LIBERTE_MOBILE', 'GPRS_5Go_7D_PORTAL',
 'Package3_Monthly']
TOP_PACK = st.selectbox("TOP_PACK",TOP_PAC)
Arpus= ['Low', 'Medium', 'High']
Arpu_segment = st.selectbox('ARPU_SEGMENT', Arpus )  


loaded_le = joblib.load('le.pkl')
Region = loaded_le.transform([Region])
loaded_le.fit(['FATICK', 'DAKAR', 'LOUGA', 'TAMBACOUNDA', 'KAOLACK', 'THIES',
 'SAINT-LOUIS', 'KOLDA,' 'KAFFRINE', 'DIOURBEL', 'ZIGUINCHOR', 'MATAM',
 'SEDHIOU', 'KEDOUGOU'])

loaded_le1 = joblib.load('le1.pkl')
Tenure = loaded_le1.transform(Tenure_e)
loaded_le1.fit(['K > 24 month', 'I 18-21 month', 'G 12-15 month', 'H 15-18 month',
 'J 21-24 month', 'F 9-12 month', 'D 3-6 month', 'E 6-9 month'])

loaded_le2 = joblib.load('le2.pkl')
MRG = loaded_le2.transform(Mrg)
loaded_le2.fit(['NO'])


# Load the pre-trained LabelEncoder
loaded_le3 = joblib.load('le3.pkl')

# Existing TOP_PAC labels (new input data with possible unseen labels)
TOP_PAC = ['On net 200F=Unlimited _call24H', 'nan', 'On-net 1000F=10MilF;10d',
 'Data:1000F=5GB,7d', 'Mixt 250F=Unlimited_call24H',
 'MIXT:500F= 2500F on net _2500F off net;2d', 'All-net 500F=2000F;5d',
 'On-net 500F_FNF;3d', 'Data: 100 F=40MB,24H',
 'MIXT: 200mnoff net _unl on net _5Go;30d', 'Jokko_Daily',
 'Data: 200 F=100MB,24H', 'Data:490F=1GB,7d', 'Twter_U2opia_Daily',
 'On-net 500=4000,10d', 'Data:1000F=2GB,30d', 'IVR Echat_Daily_50F',
 'Pilot_Youth4_490', 'All-net 500F =2000F_AllNet_Unlimited',
 'Twter_U2opia_Weekly', 'Data:200F=Unlimited,24H', 'On-net 200F=60mn;1d',
 'All-net 600F= 3000F ;5d', 'Pilot_Youth1_290',
 'All-net 1000F=(3000F On+3000F Off);5d', 'VAS(IVR_Radio_Daily)',
 'Data:3000F=10GB,30d', 'All-net 1000=5000;5d', 'Twter_U2opia_Monthly',
 'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h\t', 'FNF2 ( JAPPANTE)',
 'Yewouleen_PKG', 'Data:150F=SPPackage1,24H', 'WIFI_Family_2MBPS',
 'Data:500F=2GB,24H', 'MROMO_TIMWES_RENEW', 'New_YAKALMA_4_ALL',
 'Data:1500F=3GB,30D', 'All-net 500F=4000F ; 5d' 'Jokko_promo',
 'All-net 300=600;2d', 'Data:300F=100MB,2d',
 'MIXT: 590F=02H_On-net_200SMS_200 Mo;24h\t\t',
 'All-net 500F=1250F_AllNet_1250_Onnet;48h', 'Facebook_MIX_2D',
 '500=Unlimited3Day', 'On net 200F= 3000F_10Mo ;24H', '200=Unlimited1Day',
 'YMGX 100=1 hour FNF, 24H/1 month', 'SUPERMAGIK_5000',
 'Data:DailyCycle_Pilot_1.5GB', 'Staff_CPE_Rent',
 'MIXT:1000F=4250 Off net _ 4250F On net _100Mo; 5d', 'Data:50F=30MB_24H',
 'Data:700F=SPPackage1,7d', 'Data: 490F=Night,00H-08H', 'Data:700F=1.5GB,7d',
 'Data:1500F=SPPackage1,30d', 'Data:30Go_V 30_Days', 'MROMO_TIMWES_OneDAY',
 'On-net 300F=1800F;3d', 'All-net 5000= 20000off+20000on;30d',
 'WIFI_ Family _4MBPS', 'CVM_on-net bundle 500=5000',
 'Internat: 1000F_Zone_3;24h\t\t', 'DataPack_Incoming', 'Jokko_Monthly',
 'EVC_500=2000F' 'On-net 2000f_One_Month_100H; 30d',
 'MIXT:10000F=10hAllnet_3Go_1h_Zone3;30d\t\t', 'EVC_Jokko_Weekly',
 '200F=10mnOnNetValid1H', 'IVR Echat_Weekly_200F', 'WIFI_ Family _10MBPS',
 'Internat: 1000F_Zone_1;24H\t\t', 'Jokko_Weekly', 'SUPERMAGIK_1000',
 'MIXT: 500F=75(SMS, ONNET, Mo)_1000FAllNet;24h\t\t',
 'VAS(IVR_Radio_Monthly)', 'MIXT: 5000F=80Konnet_20Koffnet_250Mo;30d\t\t',
 'Data: 200F=1GB,24H', 'EVC_JOKKO30', 'NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE',
 'TelmunCRBT_daily', 'FIFA_TS_weekly', 'VAS(IVR_Radio_Weekly)',
 'Internat: 2000F_Zone_2;24H\t\t', 'APANews_weekly', 'EVC_100Mo',
 'pack_chinguitel_24h', 'Data_EVC_2Go24H',
 'Mixt : 500F=2500Fonnet_2500Foffnet ;5d', 'FIFA_TS_daily',
 'MIXT: 4900F= 10H on net_1,5Go ;30d', 'CVM_200f=400MB',
 'IVR Echat_Monthly_500F', 'All-net 500= 4000off+4000on;24H',
 'FNF_Youth_ESN', 'Data:1000F=700MB,7d', '1000=Unlimited7Day',
 'Incoming_Bonus_woma', 'CVM_100f=200 MB', 'CVM_100F_unlimited',
 'pilot_offer6', '305155009', 'Postpaid FORFAIT 10H Package', 'EVC_1Go',
 'GPRS_3000Equal10GPORTAL', 'NEW_CLIR_PERMANENT_LIBERTE_MOBILE',
 'Data_Mifi_10Go_Monthly', '1500=Unlimited7Day', 'EVC_700Mo',
 'CVM_100f=500 onNet', 'CVM_On-net 1300f=12500', 'pilot_offer5',
 'EVC_4900=12000F', 'CVM_On-net 400f=2200F', 'YMGX on-net 100=700F, 24H',
 'CVM_150F_unlimited', 'EVC_MEGA10000F', 'pilot_offer7', 'CVM_500f=2GB',
 'SMS Max', '301765007', '150=unlimited pilot auto',
 'MegaChrono_3000F=12500F TOUS RESEAUX', 'pilot_offer4', 'Go-NetPro-4 Go',
 '200=unlimited pilot auto', 'ESN_POSTPAID_CLASSIC_RENT', 'Data_Mifi_10Go',
 'Data:New-GPRS_PKG_1500F', 'GPRS_BKG_1000F MIFI',
 'Data:OneTime_Pilot_1.5GB', 'FIFA_TS_monthly', 'GPRS_PKG_5GO_ILLIMITE',
 'Data_Mifi_20Go', 'APANews_monthly',
 'NEW_CLIR_TEMPRESTRICTED_LIBERTE_MOBILE', 'GPRS_5Go_7D_PORTAL',
 'Package3_Monthly']

# Find new labels not seen during original fitting
new_labels = [label for label in TOP_PAC if label not in loaded_le3.classes_]

# If there are any new labels, refit the encoder with both old and new labels
if new_labels:
    print(f"New labels encountered: {new_labels}")
    # Refit with the combination of original classes and new labels
    loaded_le3.fit(list(loaded_le3.classes_) + new_labels)

# Now transform the TOP_PAC data
TOP_PACK = loaded_le3.transform(TOP_PAC)


# Assuming you have the 'le4.pkl' saved encoder
loaded_le4 = joblib.load('le4.pkl')

# Check if the input is correctly formatted
Arpus = ['Low']  # Example input, replace with your actual data

# Ensure all items are strings and clean any special characters or whitespaces
Arpus_cleaned = [str(val).strip().replace(' ', '') for val in Arpus]

# Refit the encoder if necessary (on the full set of categories)
categories = ['Low', 'Medium', 'High']  # Replace with actual categories seen during training
loaded_le4.fit(categories)

# Now transform the cleaned input data
Arpu_segment = loaded_le4.transform(Arpus_cleaned)
print(f"Transformed labels: {Arpu_segment}")

# Load the scaler
scaler = joblib.load('scaler.pkl')

input_data = None
prediction = None 


# Prepare your input data (make sure this matches the scaler's expected number of features)
data_to_scale = pd.DataFrame(['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'FREQUENCE',
       'DATA_VOLUME', 'ON_NET', 'ORANGE', 'REGULARITY', 'FREQ_TOP_PACK'])

data_to_scale1 = data_to_scale.apply(pd.to_numeric, errors='coerce')

input_data_scaled = None 
# Now you can scale it
input_data_scaled = scaler.fit(data_to_scale1)

input_vector_reshaped = np.array(input_data_scaled)
input_vector_reshaped = input_vector_reshaped.reshape(-1, 1)
Tenure = Tenure.reshape(-1, 1) 
MRG = MRG.reshape(-1, 1) 
TOP_PACK = TOP_PACK.reshape(-1, 1) 
Arpu_segment = Arpu_segment.reshape(-1, 1) 

if input_vector_reshaped.shape[1] >= 10:
    input_vector_reshaped = np.insert(input_vector_reshaped, 10, Region, axis=1)
else:
    input_vector_reshaped = np.insert(input_vector_reshaped, input_vector_reshaped.shape[1], Region, axis=1)
    input_vector_reshaped = np.insert(input_vector_reshaped, input_vector_reshaped.shape[1], Tenure, axis=1)
    input_vector_reshaped = np.insert(input_vector_reshaped, input_vector_reshaped.shape[1], MRG, axis=1)
    input_vector_reshaped = np.insert(input_vector_reshaped, input_vector_reshaped.shape[1], TOP_PACK, axis=1)
    input_vector_reshaped = np.insert(input_vector_reshaped, input_vector_reshaped.shape[1], Arpu_segment, axis=1)


# Ensure that input_vector_reshaped is now assigned to input_data
input_data = np.array(input_vector_reshaped)

model = joblib.load('expresso_churn_model.pkl')

# Add a button to trigger the prediction
if st.button('Predict'):
    if input_data is not None:  # Check if input data is ready
        try:
            # Make the prediction using the model
            prediction = model.predict(input_data)
            
            # Show the result
            st.write(f"Prediction: {prediction[0]}")  # Assuming the model's prediction output is a single value (e.g., 0 or 1)
            
            if prediction[0] == 0:
                st.write("The model predicts the client will not churn.")
            else:
                st.write("The model predicts the client will churn.")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            prediction = None  # If an error occurs, set prediction to None
    else:
        st.error("Input data is not ready. Please check the input fields.")
    
# Handle case when prediction is not made
if prediction is None:
    st.write("Prediction was not made yet. Please fill in the form and click 'Predict'.")



    