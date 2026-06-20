import pandas as pd
import numpy as np
import random
import os

def generate_wafer_data():
    np.random.seed(42)
    random.seed(42)
    
    data = []
    wafer_radius = 15 # 웨이퍼 반지름 (가상의 좌표계 기준)
    
    # 10장의 웨이퍼 생성
    for w in range(1, 11):
        wafer_id = f"Wafer_{w:02d}"
        
        # 특정 웨이퍼(7번, 8번)에서 수율 하락(Yield Drop)이 발생하도록 설정
        is_bad_wafer = w in [7, 8]
        
        for x in range(-wafer_radius, wafer_radius + 1):
            for y in range(-wafer_radius, wafer_radius + 1):
                # 원형 웨이퍼 내부의 칩만 생성 (x^2 + y^2 <= r^2)
                if x**2 + y**2 <= wafer_radius**2:
                    
                    # 1. BIN Code 및 양/불량 할당
                    if is_bad_wafer:
                        # 불량 웨이퍼는 특정 설비(Etch_Chamber_B)를 거쳤을 확률이 높고 수율이 낮음
                        bin_probs = [0.60, 0.20, 0.10, 0.10] # BIN01 확률 60%
                        equipment = random.choice(["Photo_Track_A", "Etch_Chamber_B", "Etch_Chamber_B"])
                    else:
                        # 정상 웨이퍼
                        bin_probs = [0.90, 0.05, 0.03, 0.02] # BIN01 확률 90%
                        equipment = random.choice(["Photo_Track_A", "Photo_Track_B", "Etch_Chamber_A", "Etch_Chamber_B"])
                    
                    bin_code = np.random.choice(["BIN01", "BIN02", "BIN03", "BIN04"], p=bin_probs)
                    
                    # 2. Defect Type 할당 (결함)
                    if bin_code == "BIN01":
                        defect_type = np.random.choice(["None", "Particle"], p=[0.95, 0.05])
                    elif bin_code == "BIN02": # Open/Short (에칭 부족이나 폴리머 영향)
                        defect_type = np.random.choice(["Unetch", "Polymer"], p=[0.7, 0.3])
                    elif bin_code == "BIN03": # Leakage
                        defect_type = np.random.choice(["Particle", "Scratch"], p=[0.6, 0.4])
                    else: # Speed
                        defect_type = "Pattern_Deformation"
                        
                    # 3. FDC Data (설비 센서 데이터) - 온도, 압력
                    # 불량인 경우 설비 파라미터가 타겟에서 약간 벗어나도록 유도
                    if bin_code != "BIN01" and equipment == "Etch_Chamber_B":
                        fdc_temp = np.random.normal(45.0, 2.0) # 기준(40도)보다 높음
                        fdc_pressure = np.random.normal(12.0, 1.0) # 기준(10)보다 높음
                    else:
                        fdc_temp = np.random.normal(40.0, 0.5)
                        fdc_pressure = np.random.normal(10.0, 0.2)
                        
                    # 4. Critical Dimension (CD, 선폭)
                    target_cd = 50.0
                    if is_bad_wafer:
                        actual_cd = np.random.normal(51.5, 1.8) # 산포가 넓고 타겟에서 벗어남
                    else:
                        actual_cd = np.random.normal(50.2, 0.8) # 안정적인 산포
                        
                    data.append({
                        "Wafer_ID": wafer_id,
                        "X_Die": x,
                        "Y_Die": y,
                        "BIN_Code": bin_code,
                        "Defect_Type": defect_type,
                        "Equipment": equipment,
                        "FDC_Temp": round(fdc_temp, 2),
                        "FDC_Pressure": round(fdc_pressure, 2),
                        "Target_CD": target_cd,
                        "Actual_CD": round(actual_cd, 2)
                    })

    df = pd.DataFrame(data)
    df.to_csv("wafer_log_data.csv", index=False)
    print("성공적으로 'wafer_log_data.csv' 파일이 생성되었습니다.")

if __name__ == "__main__":
    generate_wafer_data()