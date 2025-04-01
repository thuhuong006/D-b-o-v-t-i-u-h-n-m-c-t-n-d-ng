import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Tải mô hình đã huấn luyện từ file
xgb_model = joblib.load('xgb_model.pkl')  
scaler = joblib.load('scaler.pkl') 

# Tạo giao diện web
st.title('🔍 DỰ ĐOÁN HẠN MỨC TÍN DỤNG')
name = st.text_input("Tên khách hàng: ")
AGE = st.number_input("Tuổi: ", min_value=18, format="%d")
SEX = st.selectbox("Giới tính", ["Nam", "Nữ"])
EDUCATION = st.selectbox("Trình độ học vấn", ["Cao học", "Đại học", "Trung học phổ thông", "Khác"])
MARRIAGE = st.selectbox("Tình trạng hôn nhân", ["Đã kết hôn", "Độc thân", "Khác"])
estimated_income = st.number_input("Thu nhập hàng tháng: ", min_value=0, format="%d")
bill_balance = st.number_input("Số dư hóa đơn trung bình trong 6 tháng: ", min_value=0.0, format="%f")
CUR = st.number_input("Tỷ lệ sử dụng tín dụng:", min_value=0.0)

# Nhập số dư hóa đơn
BILL_BALANCE_Apr = st.number_input("Số dư hóa đơn tháng 1 (Apr): ", min_value=0.0)
BILL_BALANCE_May = st.number_input("Số dư hóa đơn tháng 2 (May): ", min_value=0.0)
BILL_BALANCE_Jun = st.number_input("Số dư hóa đơn tháng 3 (Jun): ", min_value=0.0)
BILL_BALANCE_Jul = st.number_input("Số dư hóa đơn tháng 4 (Jul): ", min_value=0.0)
BILL_BALANCE_Aug = st.number_input("Số dư hóa đơn tháng 5 (Aug): ", min_value=0.0)
BILL_BALANCE_Sept = st.number_input("Số dư hóa đơn tháng 6 (Sept): ", min_value=0.0)

# Nhập số tiền thanh toán cho từng tháng (tháng Apr đến tháng Sept)
PAY_AMOUNT_Apr = st.number_input("Số tiền thanh toán tháng 1 (Apr): ", min_value=0.0)
PAY_AMOUNT_May = st.number_input("Số tiền thanh toán tháng 2 (May): ", min_value=0.0)
PAY_AMOUNT_Jun = st.number_input("Số tiền thanh toán tháng 3 (Jun): ", min_value=0.0)
PAY_AMOUNT_Jul = st.number_input("Số tiền thanh toán tháng 4 (Jul): ", min_value=0.0)
PAY_AMOUNT_Aug = st.number_input("Số tiền thanh toán tháng 5 (Aug): ", min_value=0.0)
PAY_AMOUNT_Sept = st.number_input("Số tiền thanh toán tháng 6 (Sept): ", min_value=0.0)

# Nhập tình trạng thanh toán cho từng tháng (tháng Apr đến tháng Sept)
PAY_STATUS_Apr = st.selectbox("Tình trạng thanh toán tháng 1 (Apr)", [0, 1, 2, 3, 4], index=0)
PAY_STATUS_May = st.selectbox("Tình trạng thanh toán tháng 2 (May)", [0, 1, 2, 3, 4], index=0)
PAY_STATUS_Jun = st.selectbox("Tình trạng thanh toán tháng 3 (Jun)", [0, 1, 2, 3, 4], index=0)
PAY_STATUS_Jul = st.selectbox("Tình trạng thanh toán tháng 4 (Jul)", [0, 1, 2, 3, 4], index=0)
PAY_STATUS_Aug = st.selectbox("Tình trạng thanh toán tháng 5 (Aug)", [0, 1, 2, 3, 4], index=0)
PAY_STATUS_Sept = st.selectbox("Tình trạng thanh toán tháng 6 (Sept)", [0, 1, 2, 3, 4], index=0)


# Khi nhấn nút dự đoán
if st.button("Dự đoán"):
    # Chuyển đổi các giá trị nhập vào thành một vector cho mô hình
    SEX = 1 if SEX == "Nam" else 0
    EDUCATION = 1 if EDUCATION == "Cao học" else (2 if EDUCATION == "Đại học" else (3 if EDUCATION == "Trung học phổ thông" else 4))
    MARRIAGE = 1 if MARRIAGE == "Đã kết hôn" else (2 if MARRIAGE == "Độc thân" else 3)

    # Tạo feature input cho mô hình
    features = np.array([AGE, SEX, EDUCATION, estimated_income, bill_balance,
                         BILL_BALANCE_Apr, BILL_BALANCE_May, BILL_BALANCE_Jun, 
                         BILL_BALANCE_Jul, BILL_BALANCE_Aug, BILL_BALANCE_Sept, 
                         PAY_STATUS_Apr, PAY_STATUS_May, PAY_STATUS_Jun, 
                         PAY_STATUS_Jul, PAY_STATUS_Aug, PAY_STATUS_Sept, 
                         PAY_AMOUNT_Apr, PAY_AMOUNT_May, PAY_AMOUNT_Jun, 
                         PAY_AMOUNT_Jul, PAY_AMOUNT_Aug, PAY_AMOUNT_Sept, 
                         MARRIAGE, CUR]).reshape(1, -1)

    # Chuẩn hóa dữ liệu với scaler đã huấn luyện
    features_scaled = scaler.transform(features)
    
    # Dự đoán từ mô hình
    prediction = xgb_model.predict(features_scaled)

    # Hiển thị kết quả dự đoán
    st.write(f"Hạn mức tín dụng tối ưu: {prediction[0]:.2f} VND")



