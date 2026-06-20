import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="반도체 수율 분석 대시보드", layout="wide")
st.title("🔬 반도체 웨이퍼 수율 및 결함 분석 대시보드")

# 색상 맵핑 (BIN Code)
COLOR_MAP = {
    "BIN01": "#2ca02c", # 초록 (Good)
    "BIN02": "#d62728", # 빨강 (Open/Short)
    "BIN03": "#ff7f0e", # 주황 (Leakage) - 수정된 맵핑 반영
    "BIN04": "#ffbb78"  # 노랑 (Speed)
}

# 1. 파일 업로드 및 데이터 로드 기능
uploaded_file = st.file_uploader("CSV 형태의 Wafer Log Data를 업로드하세요.", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 1. 웨이퍼 맵 (Wafer Map)", 
            "📈 2. 수율 및 결함 분석", 
            "🔍 3. 수율 Drop 원인 추적", 
            "⚙️ 4. 공정 능력 지수 (Cp, Cpk)"
        ])
        
        # ==========================================
        # 탭 1: 웨이퍼 탐색 및 시각화 (Wafer Map)
        # ==========================================
        with tab1:
            st.subheader("Wafer Map Visualization")
            
            # 웨이퍼 선택 드롭다운
            wafer_list = sorted(df['Wafer_ID'].unique())
            selected_wafer = st.selectbox("분석할 Wafer_ID를 선택하세요:", wafer_list)
            
            # 선택된 웨이퍼 데이터 필터링
            df_wafer = df[df['Wafer_ID'] == selected_wafer].copy()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plotly Scatter로 웨이퍼 맵 구현
                fig_map = px.scatter(
                    df_wafer, x="X_Die", y="Y_Die", color="BIN_Code",
                    color_discrete_map=COLOR_MAP,
                    title=f"{selected_wafer} Wafer Map",
                    hover_data=["Defect_Type", "Equipment", "Actual_CD"]
                )
                # 원형 비율 유지를 위해 축 설정 변경
                fig_map.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig_map, use_container_width=True)
                
            with col2:
                # Center vs Edge 수율 비교 로직
                # 수학적 근거: 웨이퍼 중심점(0,0)으로부터의 거리를 피타고라스 정리(x^2 + y^2 의 제곱근)로 구함
                max_radius = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2).max()
                df_wafer['Distance'] = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2)
                
                # 거리가 최대 반지름의 절반 이하면 Center, 초과면 Edge로 분류
                df_wafer['Region'] = np.where(df_wafer['Distance'] <= (max_radius * 0.5), 'Center', 'Edge')
                
                # 수율 = (BIN01 칩 수 / 전체 칩 수) * 100
                def calc_yield(data):
                    if len(data) == 0: return 0
                    return (len(data[data['BIN_Code'] == 'BIN01']) / len(data)) * 100
                
                center_yield = calc_yield(df_wafer[df_wafer['Region'] == 'Center'])
                edge_yield = calc_yield(df_wafer[df_wafer['Region'] == 'Edge'])
                
                st.metric(label="전체 수율", value=f"{calc_yield(df_wafer):.2f}%")
                st.metric(label="Center 영역 수율", value=f"{center_yield:.2f}%")
                st.metric(label="Edge 영역 수율", value=f"{edge_yield:.2f}%", 
                          delta=f"Center 대비 {edge_yield - center_yield:.2f}%", delta_color="normal")
                st.info("💡 둥근 웨이퍼 특성상 플라즈마나 식각액의 불균일도로 인해 통상적으로 Edge의 수율이 더 낮습니다.")

        # ==========================================
        # 탭 2: 수율 및 결함 분석 대시보드
        # ==========================================
        with tab2:
            st.subheader("Yield & Defect Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. 불량 항목별 파이 차트
                st.write("**테스트 항목(BIN Code)별 비율**")
                bin_counts = df['BIN_Code'].value_counts().reset_index()
                bin_counts.columns = ['BIN_Code', 'Count']
                fig_pie = px.pie(bin_counts, values='Count', names='BIN_Code', 
                                 color='BIN_Code', color_discrete_map=COLOR_MAP, hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                # 2. YIR (Yield Impact Ratio) 분석
                # 논리적 근거: YIR = (해당 결함이 있으면서 불량인 칩) / (해당 결함이 발견된 전체 칩)
                st.write("**결함 유형별 수율 영향도 (YIR)**")
                # 'None' 결함 제외
                df_defect = df[df['Defect_Type'] != 'None'].copy()
                
                # 불량 칩 여부(BIN01이 아님) 파생 변수 생성
                df_defect['Is_Fail'] = df_defect['BIN_Code'] != 'BIN01'
                
                # 결함별 그룹화
                yir_df = df_defect.groupby('Defect_Type').agg(
                    Total_Defect=('Is_Fail', 'count'),
                    Fail_Count=('Is_Fail', 'sum')
                ).reset_index()
                
                yir_df['YIR (%)'] = (yir_df['Fail_Count'] / yir_df['Total_Defect']) * 100
                yir_df = yir_df.sort_values('YIR (%)', ascending=False)
                
                fig_yir = px.bar(yir_df, x='Defect_Type', y='YIR (%)', 
                                 text=yir_df['YIR (%)'].apply(lambda x: f"{x:.1f}%"),
                                 color='YIR (%)', color_continuous_scale='Reds')
                st.plotly_chart(fig_yir, use_container_width=True)
                st.info("💡 YIR이 높은 결함(예: Unetch)부터 우선적으로 공정 개선을 진행해야 수율 상승 효과가 가장 큽니다.")

        # ==========================================
        # 탭 3: 수율 Drop 원인 추적
        # ==========================================
        with tab3:
            st.subheader("Yield Drop Commonality Analysis")
            
            # 전체 웨이퍼 수율 트렌드 계산
            trend_df = df.groupby('Wafer_ID').apply(
                lambda x: (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100
            ).reset_index(name='Yield')
            
            fig_trend = px.line(trend_df, x='Wafer_ID', y='Yield', markers=True, title="웨이퍼별 수율 트렌드")
            fig_trend.add_hline(y=trend_df['Yield'].mean(), line_dash="dot", annotation_text="평균 수율")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                # Commonality(공통성) 분석 알고리즘
                st.write("**불량 칩 설비 공통성(Commonality) 분석**")
                analyze_wafer = st.selectbox("수율이 낮은 웨이퍼를 선택하여 분석하세요:", wafer_list, index=6) # 기본값 Wafer_07
                
                target_df = df[(df['Wafer_ID'] == analyze_wafer) & (df['BIN_Code'] != 'BIN01')]
                if not target_df.empty:
                    eq_counts = target_df['Equipment'].value_counts()
                    worst_eq = eq_counts.idxmax()
                    worst_rate = (eq_counts.max() / len(target_df)) * 100
                    
                    st.error(f"🚨 **경고!** 선택하신 {analyze_wafer}의 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다.")
                    st.dataframe(eq_counts.reset_index().rename(columns={'index':'Equipment', 'Equipment':'Fail_Chip_Count'}))
                else:
                    st.success("선택한 웨이퍼에 불량 칩이 없습니다.")
                    
            with col2:
                # FDC 데이터와 불량 간의 상관관계 Scatter Plot
                st.write("**설비 파라미터(FDC) 이상 연동 확인**")
                df['Pass/Fail'] = np.where(df['BIN_Code'] == 'BIN01', 'Pass', 'Fail')
                
                fig_fdc = px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail",
                                     color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"},
                                     opacity=0.6)
                st.plotly_chart(fig_fdc, use_container_width=True)

        # ==========================================
        # 탭 4: 공정 능력 지수 (Cp, Cpk) 모니터링
        # ==========================================
        with tab4:
            st.subheader("Process Capability Index (Cp, Cpk)")
            
            # 사양 한계 설정 (가정치)
            target = df['Target_CD'].iloc[0]
            USL = target + 4.0 # 상한치 (Upper Specification Limit)
            LSL = target - 4.0 # 하한치 (Lower Specification Limit)
            
            # 수학적 근거: 
            # Cp (공정 능력): 규격 폭(USL-LSL)을 공정의 6시그마 산포로 나눈 값. 값이 클수록 산포가 좁음.
            # Cpk (치우침 고려 공정 능력): 공정 평균(mu)이 타겟에서 얼마나 치우쳐 있는지를 고려한 값.
            mu = df['Actual_CD'].mean()
            sigma = df['Actual_CD'].std()
            
            if sigma > 0:
                cp = (USL - LSL) / (6 * sigma)
                cpk = min((USL - mu)/(3 * sigma), (mu - LSL)/(3 * sigma))
            else:
                cp, cpk = 0, 0
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Target CD", f"{target:.1f} nm")
            col2.metric("Cp (Process Capability)", f"{cp:.2f}")
            col3.metric("Cpk (Actual Capability)", f"{cpk:.2f}")
            
            if cpk < 1.0:
                st.error("🚨 Cpk 값이 1.0 미만입니다! 공정 산포가 규격을 벗어나고 있거나, 평균이 심하게 치우쳐 있습니다. Recipe 또는 Target 변경이 필요합니다.")
            elif cpk >= 1.33:
                st.success("✅ Cpk 값이 1.33 이상으로 공정 능력이 매우 우수하고 안정적입니다.")
            else:
                st.warning("⚠️ Cpk 값이 양호하나, 공정 중심점(Target)을 개선할 여지가 있습니다.")
                
            # 선폭(CD) 분포도 히스토그램 시각화
            fig_cd = px.histogram(df, x="Actual_CD", color="Pass/Fail", 
                                  nbins=40, title="Actual CD(선폭) 분포 및 규격 한계선")
            fig_cd.add_vline(x=target, line_dash="dash", line_color="black", annotation_text="Target")
            fig_cd.add_vline(x=USL, line_dash="solid", line_color="red", annotation_text="USL")
            fig_cd.add_vline(x=LSL, line_dash="solid", line_color="red", annotation_text="LSL")
            st.plotly_chart(fig_cd, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. 파일 형식을 확인해주세요. (에러 내역: {e})")
else:
    st.info("👈 먼저 파이썬 코드로 생성한 'wafer_log_data.csv' 파일을 업로드해주세요.")