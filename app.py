import streamlit as st
import pickle
import numpy as np

# Importing the model

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Predictor')

# brand
company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
Ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32, 64])

# Weight
weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1366X768', '1440X900', '1600X900', '1920X1080',
       '1920X1200', '2160X1440', '2256X1504', '2304X1440',
       '2400X1600', '2560X1440', '2560X1600', '2736X1824',
       '2880X1800', '3200X1800', '3840X2160'])

#cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [  0, 500,   1024,   2048,  32, 128])

ssd = st.selectbox('SSD(in GB)', [128, 0, 256, 512, 32, 64, 1024, 16, 180, 240, 8])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
       # Query
       ppi = None

       if touchscreen == 'Yes':
              touchscreen = 1
       else:
              touchscreen = 0

       if ips == 'Yes':
              ips = 1
       else:
              ips = 0

       xres = int(resolution.split('X')[0])
       yres =int(resolution.split('X')[1])
       ppi = ((xres**2) + (yres**2))**0.5/ screen_size

       query = np.array([company, type, Ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

       query = query.reshape(1,12)
       st.title('The predicted price of this configuration is: ' + str(int(np.exp(pipe.predict(query)[0]))))
