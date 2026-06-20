import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(page_title="반도체 수율 분석 대시보드", layout="wide")
st.title("🔬 반도체 웨이퍼 수율 및 결함 분석 대시보드 (R&D Pro Ver.)")

COLOR_MAP = {
    "BIN01": "#2ca02c", # 정상 (Good)
    "BIN02": "#d62728", # Open/Short
    "BIN03": "#ff7f0e", # Leakage
    "BIN04": "#ffbb78"  # Speed
}

uploaded_file = st.file_uploader("CSV 형태의 Wafer Log Data를 업로드하세요.", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 💡 Pandas 결측치 에러 방지
        if 'Defect_Type' in df.columns:
            df['Defect_Type'] = df['Defect_Type'].fillna('No_Defect')
        
        # 7개의 탭으로 세분화 및 기능 확장
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 1. 2D 웨이퍼 맵", 
            "🔎 2. 특정 Die 상세 추적",
            "🔥 3. 누적 불량 패턴 (Composite)", 
            "📈 4. 공정 변수 상관관계 (Correlation)",
            "📉 5. 수율 및 결함 분석", 
            "🔍 6. 수율 Drop 원인 추적", 
            "⚙️ 7. 공정 능력 지수 (Cp, Cpk)"
        ])
        
        wafer_list = sorted(df['Wafer_ID'].dropna().unique())

        # ==========================================
        # 탭 1: 전문적인 2D 웨이퍼 맵 시각화 (격자 형태)
        # ==========================================
        with tab1:
            st.subheader("2D Wafer Map (Grid View)")
            selected_wafer = st.selectbox("분석할 Wafer_ID를 선택하세요:", wafer_list, key="map_wafer")
            
            df_wafer = df[df['Wafer_ID'] == selected_wafer].copy()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # 💡 [핵심] 실제 YMS 툴처럼 마커를 사각형(square)으로 설정하고 크기를 키움
                fig_map = px.scatter(
                    df_wafer, x="X_Die", y="Y_Die", color="BIN_Code",
                    color_discrete_map=COLOR_MAP,
                    title=f"{selected_wafer} Wafer Map",
                    hover_data=["Defect_Type", "Equipment", "Actual_CD"]
                )
                
                # 마커 모양 변경 (square) 및 크기(size) 조정을 통해 빈틈없는 격자 형태 모사
                fig_map.update_traces(marker=dict(symbol='square', size=14, line=dict(width=1, color='white')))
                # 원형 비율 유지를 위해 축 설정 변경 (찌그러짐 방지)
                fig_map.update_layout(
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    plot_bgcolor='rgba(240, 240, 240, 0.5)' # 배경색을 살짝 어둡게 하여 칩이 잘 보이도록 함
                )
                st.plotly_chart(fig_map, use_container_width=True)
                
            with col2:
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

        # ==========================================
        # 탭 2: 특정 Die 상세 이력 조회 (Die Inspector)
        # ==========================================
        with tab2:
            st.subheader("🔎 특정 Die 상세 이력 조회 (Die Inspector)")
            st.markdown("특정 좌표(X, Y)를 입력하여 칩 단위의 실시간 물성 데이터와 설비 이력을 추적합니다.")
            
            insp_col1, insp_col2, insp_col3 = st.columns(3)
            with insp_col1:
                inspect_wafer = st.selectbox("웨이퍼 번호:", wafer_list, key="inspect_wafer")
            with insp_col2:
                x_in = st.number_input("X 좌표:", min_value=int(df['X_Die'].min()), max_value=int(df['X_Die'].max()), value=0)
            with insp_col3:
                y_in = st.number_input("Y 좌표:", min_value=int(df['Y_Die'].min()), max_value=int(df['Y_Die'].max()), value=0)
            
            target_die = df[(df['Wafer_ID'] == inspect_wafer) & (df['X_Die'] == x_in) & (df['Y_Die'] == y_in)]
            
            if target_die.empty:
                st.warning("⚠️ 해당 좌표에는 칩(Die) 데이터가 존재하지 않습니다. 동그란 웨이퍼 내부의 유효한 좌표를 입력해주세요.")
            else:
                die_data = target_die.iloc[0].to_dict()
                st.write(f"#### 📍 {inspect_wafer} [X: {x_in}, Y: {y_in}] 물성 요약")
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("테스트 결과 (BIN)", die_data.get('BIN_Code', 'N/A'))
                m_col2.metric("결함 유형 (Defect)", die_data.get('Defect_Type', 'N/A'))
                m_col3.metric("진행 설비 (Equipment)", die_data.get('Equipment', 'N/A'))
                m_col4.metric("생산 로트 (Lot_ID)", die_data.get('Lot_ID', 'N/A'))
                
                m_col5, m_col6, m_col7, m_col8 = st.columns(4)
                m_col5.metric("설비 온도 (FDC_Temp)", f"{die_data.get('FDC_Temp', 0)} ℃")
                m_col6.metric("설비 압력 (FDC_Pressure)", f"{die_data.get('FDC_Pressure', 0)} Torr")
                m_col7.metric("플라즈마 전력 (RF_Power)", f"{die_data.get('RF_Power(W)', 0)} W")
                m_col8.metric("계측 선폭 (Actual_CD)", f"{die_data.get('Actual_CD', 0)} nm")

        # ==========================================
        # 탭 3: [신규] 누적 불량 패턴 (Composite Wafer Map)
        # ==========================================
        with tab3:
            st.subheader("🔥 Composite Wafer Map (Systematic Defect Analysis)")
            st.markdown("선택한 Lot 내의 모든 웨이퍼를 겹쳐서(Stack) 좌표별 **불량 집중도(히트맵)**를 분석합니다.")
            
            if 'Lot_ID' in df.columns:
                lot_list = sorted(df['Lot_ID'].dropna().unique())
                selected_lot = st.selectbox("분석할 Lot_ID를 선택하세요:", lot_list)
                
                df_lot = df[df['Lot_ID'] == selected_lot].copy()
                # 불량 칩 여부 식별 (BIN01이 아니면 불량(1), 맞으면 정상(0))
                df_lot['Is_Fail'] = (df_lot['BIN_Code'] != 'BIN01').astype(int)
                
                # 좌표별 누적 계산 (총 칩 수, 불량 발생 수)
                comp_df = df_lot.groupby(['X_Die', 'Y_Die'])['Is_Fail'].agg(['count', 'sum']).reset_index()
                # 해당 좌표에서의 불량률 계산
                comp_df['Fail_Rate(%)'] = (comp_df['sum'] / comp_df['count']) * 100
                
                fig_comp = px.scatter(
                    comp_df, x="X_Die", y="Y_Die", color="Fail_Rate(%)",
                    color_continuous_scale="Reds", # 붉은색 히트맵
                    title=f"[{selected_lot}] 위치별 불량 누적 히트맵",
                    hover_data={"X_Die": True, "Y_Die": True, "sum": True, "count": True, "Fail_Rate(%)": ':.1f'}
                )
                # 사각형 마커 적용
                fig_comp.update_traces(marker=dict(symbol='square', size=14, line=dict(width=0)))
                fig_comp.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor='black')
                
                st.plotly_chart(fig_comp, use_container_width=True)
                st.info("💡 붉은색이 진한 영역은 여러 웨이퍼에 걸쳐 고질적(Systematic)으로 불량이 발생하는 취약 지점입니다.")
            else:
                st.warning("데이터에 'Lot_ID' 컬럼이 없어 누적 분석을 수행할 수 없습니다.")

        # ==========================================
        # 탭 4: [신규] 공정 변수 상관관계 (Parameter Correlation)
        # ==========================================
        with tab4:
            st.subheader("📈 Parameter Correlation (공정 마진 및 상관관계 분석)")
            st.markdown("공정 변수(X)의 변화가 선폭이나 다른 변수(Y)에 미치는 영향을 추세선과 함께 분석합니다.")
            
            # 분석 가능한 수치형 변수 리스트
            param_candidates = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
            available_params = [p for p in param_candidates if p in df.columns]
            
            if len(available_params) >= 2:
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    x_param = st.selectbox("X축 변수 선택:", available_params, index=0)
                with c_col2:
                    y_param = st.selectbox("Y축 변수 선택:", available_params, index=min(3, len(available_params)-1))
                
                # 💡 [핵심] OLS(최소제곱법) 추세선을 그어 선형 상관관계 파악
                try:
                    fig_corr = px.scatter(
                        df, x=x_param, y=y_param, color="BIN_Code", 
                        color_discrete_map=COLOR_MAP, opacity=0.5,
                        trendline="ols", # statsmodels 라이브러리 필요
                        title=f"{x_param} vs {y_param} 상관도 분석"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.error("추세선을 그리기 위해 'statsmodels' 라이브러리가 필요합니다. 터미널에 `pip install statsmodels`를 입력해 설치해주세요.")
                    
            else:
                st.warning("상관관계를 분석할 수치형 데이터 컬럼이 부족합니다.")

        # ==========================================
        # 탭 5~7: 기존 대시보드 기능 (수율 분석, Commonality, Cpk)
        # ==========================================
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**테스트 항목(BIN Code)별 비율**")
                bin_counts = df['BIN_Code'].value_counts().reset_index()
                bin_counts.columns = ['BIN_Code', 'Count']
                st.plotly_chart(px.pie(bin_counts, values='Count', names='BIN_Code', color='BIN_Code', color_discrete_map=COLOR_MAP, hole=0.4), use_container_width=True)
            with col2:
                st.write("**결함 유형별 수율 영향도 (YIR)**")
                df_defect = df[~df['Defect_Type'].isin(['None', 'No_Defect'])].copy()
                if not df_defect.empty:
                    df_defect['Is_Fail'] = df_defect['BIN_Code'] != 'BIN01'
                    yir_df = df_defect.groupby('Defect_Type').agg(Total_Defect=('Is_Fail', 'count'), Fail_Count=('Is_Fail', 'sum')).reset_index()
                    if not yir_df.empty and yir_df['Total_Defect'].sum() > 0:
                        yir_df['YIR (%)'] = (yir_df['Fail_Count'] / yir_df['Total_Defect']) * 100
                        st.plotly_chart(px.bar(yir_df.sort_values('YIR (%)', ascending=False), x='Defect_Type', y='YIR (%)', color='YIR (%)', color_continuous_scale='Reds'), use_container_width=True)

        with tab6:
            trend_df = df.groupby('Wafer_ID').apply(lambda x: (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100).reset_index(name='Yield')
            st.plotly_chart(px.line(trend_df, x='Wafer_ID', y='Yield', markers=True, title="웨이퍼별 수율 트렌드").add_hline(y=trend_df['Yield'].mean(), line_dash="dot"), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                analyze_wafer = st.selectbox("수율이 낮은 웨이퍼를 선택하세요:", wafer_list, index=min(6, len(wafer_list) - 1), key="common_wafer")
                target_df = df[(df['Wafer_ID'] == analyze_wafer) & (df['BIN_Code'] != 'BIN01')]
                if not target_df.empty:
                    worst_eq = target_df['Equipment'].value_counts().idxmax()
                    worst_rate = (target_df['Equipment'].value_counts().max() / len(target_df)) * 100
                    st.error(f"🚨 **경고!** 선택하신 {analyze_wafer}의 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다.")
            with col2:
                df['Pass/Fail'] = 'Fail'
                df.loc[df['BIN_Code'] == 'BIN01', 'Pass/Fail'] = 'Pass'
                st.plotly_chart(px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail", color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"}, opacity=0.6), use_container_width=True)

        with tab7:
            target = df['Target_CD'].iloc[0] if 'Target_CD' in df.columns else 50.0
            USL, LSL = target + 4.0, target - 4.0
            mu, sigma = df['Actual_CD'].mean(), df['Actual_CD'].std()
            cp, cpk = ((USL - LSL) / (6 * sigma), min((USL - mu)/(3 * sigma), (mu - LSL)/(3 * sigma))) if sigma > 0 else (0, 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target CD", f"{target:.1f} nm")
            col2.metric("Cp (Process Capability)", f"{cp:.2f}")
            col3.metric("Cpk (Actual Capability)", f"{cpk:.2f}")
            
            fig_cd = px.histogram(df, x="Actual_CD", color="Pass/Fail", nbins=40, title="Actual CD(선폭) 분포")
            fig_cd.add_vline(x=target, line_dash="dash", line_color="black")
            st.plotly_chart(fig_cd, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. (에러 내역: {e})")
else:
    st.info("👈 먼저 파이썬 코드로 생성한 'wafer_log_data.csv' 파일을 업로드해주세요.")