import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="반도체 수율 분석 대시보드", layout="wide")
st.title("🔬 반도체 웨이퍼 수율 및 결함 분석 대시보드 (Pro Ver.)")

COLOR_MAP = {
    "BIN01": "#2ca02c", 
    "BIN02": "#d62728", 
    "BIN03": "#ff7f0e", 
    "BIN04": "#ffbb78"  
}

uploaded_file = st.file_uploader("CSV 형태의 Wafer Log Data를 업로드하세요.", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 💡 Pandas 결측치 에러 방지 (기존 안전 코드 유지)
        if 'Defect_Type' in df.columns:
            df['Defect_Type'] = df['Defect_Type'].fillna('No_Defect')
        
        # Tab 5 (Die Inspector)가 추가된 탭 리스트
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 1. 3D 웨이퍼 맵", 
            "📈 2. 수율 및 결함 분석", 
            "🔍 3. 수율 Drop 원인 추적", 
            "⚙️ 4. 공정 능력 지수 (Cp, Cpk)",
            "🔎 5. 특정 Die 상세 추적"
        ])
        
        wafer_list = sorted(df['Wafer_ID'].unique())

        # ==========================================
        # 탭 1: 3D 웨이퍼 탐색 및 시각화
        # ==========================================
        with tab1:
            st.subheader("3D Wafer Map Visualization")
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_wafer = st.selectbox("분석할 Wafer_ID를 선택하세요:", wafer_list, key="map_wafer")
            with col_sel2:
                # 💡 [핵심] Z축에 매핑할 센서/계측 데이터 선택
                # 선택한 파라미터에 따라 3D 맵의 높낮이가 달라져 공정 산포를 입체적으로 파악 가능합니다.
                z_axis_choice = st.selectbox("Z축 데이터를 선택하세요 (입체 산포 분석):", 
                                             ["Actual_CD", "FDC_Temp", "FDC_Pressure", "RF_Power(W)"])
            
            df_wafer = df[df['Wafer_ID'] == selected_wafer].copy()
            
            # 레이아웃을 넉넉하게 쓰기 위해 컬럼 비율 조정
            col1, col2 = st.columns([3, 1])
            with col1:
                # 💡 기존 px.scatter를 px.scatter_3d로 업그레이드
                fig_map = px.scatter_3d(
                    df_wafer, x="X_Die", y="Y_Die", z=z_axis_choice, 
                    color="BIN_Code", color_discrete_map=COLOR_MAP,
                    hover_data=["Defect_Type", "Equipment"]
                )
                
                # 마우스 컨트롤이 용이하도록 여백(margin)을 줄이고 높이를 대폭 확장
                fig_map.update_layout(
                    height=700, 
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate',
                        zaxis_title=z_axis_choice
                    )
                )
                st.plotly_chart(fig_map, use_container_width=True)
                
            with col2:
                # Center vs Edge 수율 계산 로직 (기존 유지)
                max_radius = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2).max()
                df_wafer['Distance'] = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2)
                
                df_wafer['Region'] = 'Edge'
                if pd.notna(max_radius):
                    df_wafer.loc[df_wafer['Distance'] <= (max_radius * 0.5), 'Region'] = 'Center'
                
                def calc_yield(data):
                    if len(data) == 0: return 0
                    return (len(data[data['BIN_Code'] == 'BIN01']) / len(data)) * 100
                
                center_yield = calc_yield(df_wafer[df_wafer['Region'] == 'Center'])
                edge_yield = calc_yield(df_wafer[df_wafer['Region'] == 'Edge'])
                
                st.metric(label="전체 수율", value=f"{calc_yield(df_wafer):.2f}%")
                st.metric(label="Center 영역 수율", value=f"{center_yield:.2f}%")
                st.metric(label="Edge 영역 수율", value=f"{edge_yield:.2f}%", 
                          delta=f"Center 대비 {edge_yield - center_yield:.2f}%", delta_color="normal")
                st.info("💡 Z축을 이리저리 돌려보세요. 불량(빨간색)이 특정 지형(높거나 낮은 곳)에 밀집되어 있다면, 설비 내 해당 파라미터의 물리적 불균일성이 원인일 수 있습니다.")

        # ==========================================
        # 탭 2: 수율 및 결함 분석 (기존 유지)
        # ==========================================
        with tab2:
            st.subheader("Yield & Defect Analysis")
            col1, col2 = st.columns(2)
            with col1:
                bin_counts = df['BIN_Code'].value_counts().reset_index()
                bin_counts.columns = ['BIN_Code', 'Count']
                fig_pie = px.pie(bin_counts, values='Count', names='BIN_Code', 
                                 color='BIN_Code', color_discrete_map=COLOR_MAP, hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                df_defect = df[~df['Defect_Type'].isin(['None', 'No_Defect'])].copy()
                if not df_defect.empty:
                    df_defect['Is_Fail'] = df_defect['BIN_Code'] != 'BIN01'
                    yir_df = df_defect.groupby('Defect_Type').agg(
                        Total_Defect=('Is_Fail', 'count'),
                        Fail_Count=('Is_Fail', 'sum')
                    ).reset_index()
                    
                    if not yir_df.empty and yir_df['Total_Defect'].sum() > 0:
                        yir_df['YIR (%)'] = (yir_df['Fail_Count'] / yir_df['Total_Defect']) * 100
                        yir_df = yir_df.sort_values('YIR (%)', ascending=False)
                        fig_yir = px.bar(yir_df, x='Defect_Type', y='YIR (%)', 
                                         text=yir_df['YIR (%)'].apply(lambda x: f"{x:.1f}%"),
                                         color='YIR (%)', color_continuous_scale='Reds')
                        st.plotly_chart(fig_yir, use_container_width=True)
                    else:
                        st.success("데이터에 불량을 유발한 결함이 없습니다.")
                else:
                    st.success("데이터에 결함 내역이 존재하지 않습니다.")

        # ==========================================
        # 탭 3: 수율 Drop 원인 추적 (기존 유지)
        # ==========================================
        with tab3:
            st.subheader("Yield Drop Commonality Analysis")
            trend_df = df.groupby('Wafer_ID').apply(
                lambda x: (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100
            ).reset_index(name='Yield')
            
            fig_trend = px.line(trend_df, x='Wafer_ID', y='Yield', markers=True, title="웨이퍼별 수율 트렌드")
            fig_trend.add_hline(y=trend_df['Yield'].mean(), line_dash="dot", annotation_text="평균 수율")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                default_idx = min(6, len(wafer_list) - 1)
                analyze_wafer = st.selectbox("수율이 낮은 웨이퍼를 선택하세요:", wafer_list, index=default_idx, key="common_wafer")
                
                target_df = df[(df['Wafer_ID'] == analyze_wafer) & (df['BIN_Code'] != 'BIN01')]
                if not target_df.empty:
                    eq_counts = target_df['Equipment'].value_counts()
                    worst_eq = eq_counts.idxmax()
                    worst_rate = (eq_counts.max() / len(target_df)) * 100
                    st.error(f"🚨 **경고!** 선택하신 {analyze_wafer}의 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다.")
                else:
                    st.success("선택한 웨이퍼에 불량 칩이 없습니다.")
                    
            with col2:
                df['Pass/Fail'] = 'Fail'
                df.loc[df['BIN_Code'] == 'BIN01', 'Pass/Fail'] = 'Pass'
                fig_fdc = px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail",
                                     color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"}, opacity=0.6)
                st.plotly_chart(fig_fdc, use_container_width=True)

        # ==========================================
        # 탭 4: 공정 능력 지수 (기존 유지)
        # ==========================================
        with tab4:
            st.subheader("Process Capability Index (Cp, Cpk)")
            target = df['Target_CD'].iloc[0]
            USL, LSL = target + 4.0, target - 4.0
            mu, sigma = df['Actual_CD'].mean(), df['Actual_CD'].std()
            
            if sigma > 0:
                cp = (USL - LSL) / (6 * sigma)
                cpk = min((USL - mu)/(3 * sigma), (mu - LSL)/(3 * sigma))
            else:
                cp, cpk = 0, 0
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Target CD", f"{target:.1f} nm")
            col2.metric("Cp (Process Capability)", f"{cp:.2f}")
            col3.metric("Cpk (Actual Capability)", f"{cpk:.2f}")
            
            fig_cd = px.histogram(df, x="Actual_CD", color="Pass/Fail", nbins=40, title="Actual CD(선폭) 분포")
            fig_cd.add_vline(x=target, line_dash="dash", line_color="black")
            st.plotly_chart(fig_cd, use_container_width=True)

        # ==========================================
        # 탭 5: 특정 Die 상세 추적 (새로운 핵심 기능)
        # ==========================================
        with tab5:
            st.subheader("🔎 특정 Die 상세 이력 조회 (Die Inspector)")
            st.markdown("문제가 발생한 웨이퍼의 특정 좌표를 입력하여, 칩(Die) 단위의 실시간 물성 데이터와 설비 이력을 추적합니다.")
            
            # 입력부 레이아웃
            insp_col1, insp_col2, insp_col3 = st.columns(3)
            with insp_col1:
                inspect_wafer = st.selectbox("웨이퍼 번호 (Wafer_ID):", wafer_list, key="inspect_wafer")
            with insp_col2:
                x_in = st.number_input("X 좌표 (X_Die):", min_value=int(df['X_Die'].min()), max_value=int(df['X_Die'].max()), value=0)
            with insp_col3:
                y_in = st.number_input("Y 좌표 (Y_Die):", min_value=int(df['Y_Die'].min()), max_value=int(df['Y_Die'].max()), value=0)
            
            st.markdown("---")
            
            # 💡 [핵심] 입력된 좌표로 데이터 필터링
            target_die = df[(df['Wafer_ID'] == inspect_wafer) & (df['X_Die'] == x_in) & (df['Y_Die'] == y_in)]
            
            if target_die.empty:
                # 데이터가 없는 빈 공간(가장자리 밖)을 찍었을 때의 예외 처리
                st.warning("⚠️ 해당 좌표에는 칩(Die) 데이터가 존재하지 않습니다. 동그란 웨이퍼 내부의 유효한 좌표를 입력해주세요.")
            else:
                # 데이터가 존재하면 첫 번째 row를 딕셔너리로 추출
                die_data = target_die.iloc[0].to_dict()
                
                # 가독성 높은 메트릭(Metric) 대시보드로 주요 항목 표시
                st.write(f"#### 📍 {inspect_wafer} [X: {x_in}, Y: {y_in}] 물성 요약")
                
                # 1열: 기본 테스트 결과
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("테스트 결과 (BIN)", die_data.get('BIN_Code', 'N/A'))
                m_col2.metric("결함 유형 (Defect)", die_data.get('Defect_Type', 'N/A'))
                m_col3.metric("진행 설비 (Equipment)", die_data.get('Equipment', 'N/A'))
                m_col4.metric("생산 로트 (Lot_ID)", die_data.get('Lot_ID', 'N/A'))
                
                st.write("") # 간격 띄우기
                
                # 2열: 설비 센서 및 계측 결과
                m_col5, m_col6, m_col7, m_col8 = st.columns(4)
                m_col5.metric("설비 온도 (FDC_Temp)", f"{die_data.get('FDC_Temp', 0)} ℃")
                m_col6.metric("설비 압력 (FDC_Pressure)", f"{die_data.get('FDC_Pressure', 0)} Torr")
                m_col7.metric("플라즈마 전력 (RF_Power)", f"{die_data.get('RF_Power(W)', 0)} W")
                m_col8.metric("계측 선폭 (Actual_CD)", f"{die_data.get('Actual_CD', 0)} nm")
                
                st.markdown("---")
                # 전체 데이터프레임 원본 보기
                st.write("**📝 Raw Log Data (전체 로우 데이터)**")
                st.dataframe(target_die, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. (에러 내역: {e})")
else:
    st.info("👈 먼저 파이썬 코드로 생성한 'wafer_log_data.csv' 파일을 업로드해주세요.")