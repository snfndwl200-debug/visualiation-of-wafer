import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 기본 설정 (넓은 화면 사용)
st.set_page_config(page_title="반도체 수율 분석 대시보드", layout="wide")

COLOR_MAP = {
    "BIN01": "#2ca02c", # 정상 (Good)
    "BIN02": "#d62728", # Open/Short
    "BIN03": "#ff7f0e", # Leakage
    "BIN04": "#ffbb78"  # Speed
}

# ==========================================
# 사이드바 (Sidebar) - 컨트롤 패널
# ==========================================
st.sidebar.title("⚙️ Analysis Control")
st.sidebar.markdown("데이터를 업로드하고 분석 옵션을 선택하세요.")

uploaded_file = st.sidebar.file_uploader("📁 CSV Data 업로드", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 데이터의 결측치를 안전하게 'No_Defect'로 채워 에러를 방지합니다.
        if 'Defect_Type' in df.columns:
            df['Defect_Type'] = df['Defect_Type'].fillna('No_Defect')
            
        wafer_list = sorted(df['Wafer_ID'].dropna().unique())
        
        st.sidebar.divider()
        st.sidebar.header("🎛️ 분석 옵션 설정")
        
        # [Tab 1] 웨이퍼 맵 컨트롤
        st.sidebar.subheader("📍 [Tab 1] 2D 웨이퍼 맵")
        selected_wafer = st.sidebar.selectbox("Wafer_ID 선택:", wafer_list, key="map_wafer")
        
        # [Tab 2] Die Inspector 컨트롤
        st.sidebar.subheader("🔎 [Tab 2] 특정 Die 상세 추적")
        inspect_wafer = st.sidebar.selectbox("조회할 Wafer_ID:", wafer_list, key="inspect_wafer")
        x_col, y_col = st.sidebar.columns(2)
        with x_col:
            x_in = st.number_input("X 좌표:", min_value=int(df['X_Die'].min()), max_value=int(df['X_Die'].max()), value=0)
        with y_col:
            y_in = st.number_input("Y 좌표:", min_value=int(df['Y_Die'].min()), max_value=int(df['Y_Die'].max()), value=0)
            
        # [Tab 3] Composite Map 컨트롤
        st.sidebar.subheader("🔥 [Tab 3] 누적 불량 패턴")
        if 'Lot_ID' in df.columns:
            lot_list = sorted(df['Lot_ID'].dropna().unique())
            selected_lot = st.sidebar.selectbox("Lot_ID 선택:", lot_list)
        else:
            selected_lot = None
            st.sidebar.warning("Lot_ID 컬럼이 없습니다.")
            
        # [Tab 4] Correlation 컨트롤
        st.sidebar.subheader("📈 [Tab 4] 상관관계 분석")
        param_candidates = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
        available_params = [p for p in param_candidates if p in df.columns]
        if len(available_params) >= 2:
            x_param = st.sidebar.selectbox("X축 변수:", available_params, index=0)
            y_param = st.sidebar.selectbox("Y축 변수:", available_params, index=min(3, len(available_params)-1))
        else:
            x_param, y_param = None, None
            
        # [Tab 6] Commonality 컨트롤
        st.sidebar.subheader("🚨 [Tab 6] 수율 Drop 추적")
        analyze_wafer = st.sidebar.selectbox("수율 저하 Wafer_ID:", wafer_list, index=min(6, len(wafer_list) - 1), key="common_wafer")

        # ==========================================
        # 메인 화면 (Main Content) - 7개의 탭
        # ==========================================
        st.title("🔬 반도체 웨이퍼 수율 분석 대시보드")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 1. 2D 웨이퍼 맵", 
            "🔎 2. 특정 Die 상세 추적",
            "🔥 3. 누적 불량 패턴", 
            "📈 4. 공정 변수 상관관계",
            "📉 5. 수율 및 결함 분석", 
            "🔍 6. 수율 Drop 원인 추적", 
            "⚙️ 7. 공정 능력 지수"
        ])
        
        PLOT_THEME = "plotly_white"

        with tab1:
            st.subheader("2D Wafer Map Visualization")
            df_wafer = df[df['Wafer_ID'] == selected_wafer].copy()
            
            col1, col2 = st.columns([5, 1])
            with col1:
                fig_map = px.scatter(
                    df_wafer, x="X_Die", y="Y_Die", color="BIN_Code",
                    color_discrete_map=COLOR_MAP,
                    title=f"<b>{selected_wafer} Wafer Map</b>",
                    hover_data=["Defect_Type", "Equipment", "Actual_CD"],
                    template=PLOT_THEME
                )
                
                fig_map.update_traces(
                    marker=dict(symbol='circle', size=10, opacity=0.8, line=dict(width=0.5, color='white'))
                )
                fig_map.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(t=50, b=0, l=0, r=0))
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

        with tab2:
            st.subheader("🔎 특정 Die 상세 이력 조회 (Die Inspector)")
            target_die = df[(df['Wafer_ID'] == inspect_wafer) & (df['X_Die'] == x_in) & (df['Y_Die'] == y_in)]
            
            if target_die.empty:
                st.warning("⚠️ 사이드바에서 입력하신 좌표에는 칩(Die) 데이터가 존재하지 않습니다.")
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
        # 💡 [핵심 수정] Tab 3: 그라데이션 컬러 매핑 Scatter Map 적용
        # ==========================================
        with tab3:
            st.subheader("🔥 Composite Wafer Map (Gradient Scatter)")
            if selected_lot:
                df_lot = df[df['Lot_ID'] == selected_lot].copy()
                df_lot['Is_Fail'] = (df_lot['BIN_Code'] != 'BIN01').astype(int)
                
                # 좌표별 누적 불량률 계산
                comp_df = df_lot.groupby(['X_Die', 'Y_Die'])['Is_Fail'].agg(['count', 'sum']).reset_index()
                comp_df['Fail_Rate(%)'] = (comp_df['sum'] / comp_df['count']) * 100
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    # px.density_heatmap을 버리고 px.scatter로 원복
                    fig_comp = px.scatter(
                        comp_df, x="X_Die", y="Y_Die", 
                        color="Fail_Rate(%)", # 색상을 연속형 데이터인 불량률로 설정
                        color_continuous_scale="Reds", # 하얀색/투명에서 진한 빨간색으로 이어지는 그라데이션
                        title=f"<b>[{selected_lot}] 누적 불량률 맵 (Composite)</b>",
                        hover_data={"X_Die": True, "Y_Die": True, "sum": True, "count": True, "Fail_Rate(%)": ':.1f'},
                        template=PLOT_THEME
                    )
                    
                    # 마커 디자인: 둥근 원형, 투명도 0.8, 얇은 회색 테두리로 정돈
                    fig_comp.update_traces(
                        marker=dict(symbol='circle', size=10, opacity=0.8, line=dict(width=0.5, color='gray'))
                    )
                    # 찌그러짐 방지 및 크기 극대화
                    fig_comp.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(t=50, b=0, l=0, r=0))
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                with col2:
                    st.info("💡 칩(Die) 하나하나의 위치가 정확히 유지되며, 붉은색이 짙을수록 고질적인 불량 다발 구역을 뜻합니다.")

        with tab4:
            st.subheader("📈 Parameter Correlation (공정 마진 분석)")
            if x_param and y_param:
                try:
                    fig_corr = px.scatter(
                        df, x=x_param, y=y_param, color="BIN_Code", 
                        color_discrete_map=COLOR_MAP, opacity=0.5,
                        trendline="ols", 
                        title=f"<b>{x_param} vs {y_param} 상관도 분석</b>",
                        template=PLOT_THEME
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.error("추세선을 그리기 위해 터미널에 `pip install statsmodels`를 입력해 설치해주세요.")

        with tab5:
            st.subheader("Yield & Defect Analysis")
            col1, col2 = st.columns(2)
            with col1:
                bin_counts = df['BIN_Code'].value_counts().reset_index()
                bin_counts.columns = ['BIN_Code', 'Count']
                fig_pie = px.pie(bin_counts, values='Count', names='BIN_Code', color='BIN_Code', color_discrete_map=COLOR_MAP, hole=0.4, template=PLOT_THEME, title="<b>테스트 항목별 비율</b>")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                df_defect = df[~df['Defect_Type'].isin(['None', 'No_Defect'])].copy()
                if not df_defect.empty:
                    df_defect['Is_Fail'] = df_defect['BIN_Code'] != 'BIN01'
                    yir_df = df_defect.groupby('Defect_Type').agg(Total_Defect=('Is_Fail', 'count'), Fail_Count=('Is_Fail', 'sum')).reset_index()
                    if not yir_df.empty and yir_df['Total_Defect'].sum() > 0:
                        yir_df['YIR (%)'] = (yir_df['Fail_Count'] / yir_df['Total_Defect']) * 100
                        fig_bar = px.bar(yir_df.sort_values('YIR (%)', ascending=False), x='Defect_Type', y='YIR (%)', color='YIR (%)', color_continuous_scale='Reds', template=PLOT_THEME, title="<b>결함 유형별 수율 영향도 (YIR)</b>")
                        st.plotly_chart(fig_bar, use_container_width=True)

        with tab6:
            st.subheader("Yield Drop Commonality Analysis & SPC")
            
            trend_df = df.groupby('Wafer_ID').apply(lambda x: (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100).reset_index(name='Yield')
            
            mean_yield = trend_df['Yield'].mean()
            std_yield = trend_df['Yield'].std()
            ucl = mean_yield + (3 * std_yield)
            lcl = mean_yield - (3 * std_yield)
            
            fig_trend = px.line(trend_df, x='Wafer_ID', y='Yield', markers=True, title="<b>웨이퍼별 수율 트렌드 (SPC Control Chart)</b>", template=PLOT_THEME)
            
            fig_trend.add_hline(y=mean_yield, line_dash="solid", line_color="green", annotation_text="Target (Mean)")
            fig_trend.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL (+3 Sigma)")
            fig_trend.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL (-3 Sigma)")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**[{analyze_wafer}] 설비 공통성 분석 및 리스트 추출**")
                target_df = df[(df['Wafer_ID'] == analyze_wafer) & (df['BIN_Code'] != 'BIN01')]
                
                if not target_df.empty:
                    worst_eq = target_df['Equipment'].value_counts().idxmax()
                    worst_rate = (target_df['Equipment'].value_counts().max() / len(target_df)) * 100
                    st.error(f"🚨 선택하신 웨이퍼의 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다.")
                    
                    csv_data = target_df.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📥 불량 칩 리스트 다운로드 (CSV)",
                        data=csv_data,
                        file_name=f"{analyze_wafer}_Fail_List.csv",
                        mime="text/csv",
                    )
                    st.info("현업에서 계측/품질 부서에 분석을 의뢰할 때 위 데이터를 전달합니다.")
                else:
                    st.success("선택하신 웨이퍼에 불량이 발견되지 않았습니다.")
                    
            with col2:
                df['Pass/Fail'] = 'Fail'
                df.loc[df['BIN_Code'] == 'BIN01', 'Pass/Fail'] = 'Pass'
                fig_scatter = px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail", color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"}, opacity=0.5, template=PLOT_THEME, title="<b>FDC 파라미터 이상 확인</b>")
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab7:
            st.subheader("Process Capability Index (Cp, Cpk)")
            target = df['Target_CD'].iloc[0] if 'Target_CD' in df.columns else 50.0
            USL, LSL = target + 4.0, target - 4.0
            mu, sigma = df['Actual_CD'].mean(), df['Actual_CD'].std()
            cp, cpk = ((USL - LSL) / (6 * sigma), min((USL - mu)/(3 * sigma), (mu - LSL)/(3 * sigma))) if sigma > 0 else (0, 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target CD", f"{target:.1f} nm")
            col2.metric("Cp (Process Capability)", f"{cp:.2f}")
            col3.metric("Cpk (Actual Capability)", f"{cpk:.2f}")
            
            fig_cd = px.histogram(df, x="Actual_CD", color="Pass/Fail", nbins=40, title="<b>Actual CD(선폭) 분포</b>", template=PLOT_THEME)
            fig_cd.add_vline(x=target, line_dash="dash", line_color="black")
            st.plotly_chart(fig_cd, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. (에러 내역: {e})")
else:
    st.info("👈 좌측 사이드바에서 먼저 파이썬 코드로 생성한 'wafer_log_data.csv' 파일을 업로드해주세요.")