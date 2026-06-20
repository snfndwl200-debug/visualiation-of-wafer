import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
        
        # [Tab 1, 9, 10] 웨이퍼 선택 컨트롤 공유
        st.sidebar.subheader("📍 단일 Wafer 분석")
        selected_wafer = st.sidebar.selectbox("Wafer_ID 선택:", wafer_list, key="map_wafer")
        
        # [Tab 2] Die Inspector 컨트롤
        st.sidebar.subheader("🔎 특정 Die 상세 추적")
        inspect_wafer = st.sidebar.selectbox("조회할 Wafer_ID:", wafer_list, key="inspect_wafer")
        x_col, y_col = st.sidebar.columns(2)
        with x_col:
            x_in = st.number_input("X 좌표:", min_value=int(df['X_Die'].min()), max_value=int(df['X_Die'].max()), value=0)
        with y_col:
            y_in = st.number_input("Y 좌표:", min_value=int(df['Y_Die'].min()), max_value=int(df['Y_Die'].max()), value=0)
            
        # [Tab 3] Composite Map 컨트롤
        st.sidebar.subheader("🔥 누적 불량 패턴 (Lot 단위)")
        if 'Lot_ID' in df.columns:
            lot_list = sorted(df['Lot_ID'].dropna().unique())
            selected_lot = st.sidebar.selectbox("Lot_ID 선택:", lot_list)
        else:
            selected_lot = None
            st.sidebar.warning("Lot_ID 컬럼이 없습니다.")
            
        # [Tab 4] Correlation 컨트롤
        st.sidebar.subheader("📈 상관관계 분석 변수")
        param_candidates = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
        available_params = [p for p in param_candidates if p in df.columns]
        if len(available_params) >= 2:
            x_param = st.sidebar.selectbox("X축 변수:", available_params, index=0)
            y_param = st.sidebar.selectbox("Y축 변수:", available_params, index=min(3, len(available_params)-1))
        else:
            x_param, y_param = None, None
            
        # [Tab 6] Commonality 컨트롤
        st.sidebar.subheader("🚨 수율 Drop 추적")
        analyze_wafer = st.sidebar.selectbox("수율 저하 Wafer_ID:", wafer_list, index=min(6, len(wafer_list) - 1), key="common_wafer")

        # ==========================================
        # 메인 화면 (Main Content) - 총 10개의 탭으로 확장
        # ==========================================
        st.title("🔬 반도체 웨이퍼 수율 분석 대시보드")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📊 1. 2D 웨이퍼 맵", 
            "🔎 2. 특정 Die 상세 추적",
            "🔥 3. 누적 불량 패턴", 
            "📈 4. 변수 상관관계",
            "📉 5. 결함 분석", 
            "🔍 6. 공통성 분석", 
            "⚙️ 7. 공정능력지수",
            "🤖 8. ML 변수 추출",
            "🎯 9. 패턴 자동 분류",
            "📍 10. WiW 방사형 분석"
        ])
        
        PLOT_THEME = "plotly_white"

        # 기존 Tab 1 ~ 7 로직 동일 유지
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
                fig_map.update_traces(marker=dict(symbol='square', size=10, opacity=0.8, line=dict(width=0.5, color='white')))
                fig_map.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(t=50, b=0, l=0, r=0))
                st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})
                
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

        with tab3:
            st.subheader("🔥 Composite Wafer Map (Gradient Scatter)")
            if selected_lot:
                df_lot = df[df['Lot_ID'] == selected_lot].copy()
                df_lot['Is_Fail'] = (df_lot['BIN_Code'] != 'BIN01').astype(int)
                
                comp_df = df_lot.groupby(['X_Die', 'Y_Die'])['Is_Fail'].agg(['count', 'sum']).reset_index()
                comp_df['Fail_Rate(%)'] = (comp_df['sum'] / comp_df['count']) * 100
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    fig_comp = px.scatter(
                        comp_df, x="X_Die", y="Y_Die", color="Fail_Rate(%)", color_continuous_scale="Reds",
                        title=f"<b>[{selected_lot}] 누적 불량률 맵 (Composite)</b>",
                        hover_data={"X_Die": True, "Y_Die": True, "sum": True, "count": True, "Fail_Rate(%)": ':.1f'},
                        template=PLOT_THEME
                    )
                    fig_comp.update_traces(marker=dict(symbol='square', size=10, opacity=0.8, line=dict(width=0.5, color='gray')))
                    fig_comp.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(t=50, b=0, l=0, r=0))
                    st.plotly_chart(fig_comp, use_container_width=True, config={'scrollZoom': True})
                with col2:
                    st.info("💡 칩(Die) 하나하나의 위치가 정확히 유지되며, 붉은색이 짙을수록 고질적인 불량 다발 구역을 뜻합니다.")

        with tab4:
            st.subheader("📈 Parameter Correlation (공정 마진 분석)")
            if x_param and y_param:
                df_corr = df[[x_param, y_param, 'BIN_Code']].dropna()
                fig_corr = px.scatter(
                    df_corr, x=x_param, y=y_param, color="BIN_Code", color_discrete_map=COLOR_MAP, opacity=0.5,
                    title=f"<b>{x_param} vs {y_param} 상관도 분석</b>", template=PLOT_THEME
                )
                for bin_code in df_corr['BIN_Code'].unique():
                    sub = df_corr[df_corr['BIN_Code'] == bin_code]
                    if len(sub) >= 2 and sub[x_param].std() > 0:
                        slope, intercept = np.polyfit(sub[x_param], sub[y_param], 1)
                        x_range = np.linspace(sub[x_param].min(), sub[x_param].max(), 50)
                        y_pred = slope * x_range + intercept
                        fig_corr.add_scatter(x=x_range, y=y_pred, mode='lines', line=dict(color=COLOR_MAP.get(bin_code, 'gray'), width=2), showlegend=False)
                st.plotly_chart(fig_corr, use_container_width=True, config={'scrollZoom': True})

        with tab5:
            st.subheader("Yield & Defect Analysis")
            col1, col2 = st.columns(2)
            with col1:
                bin_counts = df['BIN_Code'].value_counts().reset_index()
                bin_counts.columns = ['BIN_Code', 'Count']
                fig_pie = px.pie(bin_counts, values='Count', names='BIN_Code', color='BIN_Code', color_discrete_map=COLOR_MAP, hole=0.4, template=PLOT_THEME, title="<b>테스트 항목별 비율</b>")
                st.plotly_chart(fig_pie, use_container_width=True, config={'scrollZoom': True})
            with col2:
                df_defect = df[~df['Defect_Type'].isin(['None', 'No_Defect'])].copy()
                if not df_defect.empty:
                    df_defect['Is_Fail'] = df_defect['BIN_Code'] != 'BIN01'
                    yir_df = df_defect.groupby('Defect_Type').agg(Total_Defect=('Is_Fail', 'count'), Fail_Count=('Is_Fail', 'sum')).reset_index()
                    if not yir_df.empty and yir_df['Total_Defect'].sum() > 0:
                        yir_df['YIR (%)'] = (yir_df['Fail_Count'] / yir_df['Total_Defect']) * 100
                        fig_bar = px.bar(yir_df.sort_values('YIR (%)', ascending=False), x='Defect_Type', y='YIR (%)', color='YIR (%)', color_continuous_scale='Reds', template=PLOT_THEME, title="<b>결함 유형별 수율 영향도 (YIR)</b>")
                        st.plotly_chart(fig_bar, use_container_width=True, config={'scrollZoom': True})

        with tab6:
            st.subheader("Yield Drop Commonality Analysis & SPC")
            trend_df = df.groupby('Wafer_ID').apply(lambda x: (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100).reset_index(name='Yield')
            mean_yield, std_yield = trend_df['Yield'].mean(), trend_df['Yield'].std()
            fig_trend = px.line(trend_df, x='Wafer_ID', y='Yield', markers=True, title="<b>웨이퍼별 수율 트렌드 (SPC Control Chart)</b>", template=PLOT_THEME)
            fig_trend.add_hline(y=mean_yield, line_dash="solid", line_color="green", annotation_text="Target (Mean)")
            fig_trend.add_hline(y=mean_yield + (3 * std_yield), line_dash="dash", line_color="red", annotation_text="UCL (+3 Sigma)")
            fig_trend.add_hline(y=mean_yield - (3 * std_yield), line_dash="dash", line_color="red", annotation_text="LCL (-3 Sigma)")
            st.plotly_chart(fig_trend, use_container_width=True, config={'scrollZoom': True})
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                target_df = df[(df['Wafer_ID'] == analyze_wafer) & (df['BIN_Code'] != 'BIN01')]
                if not target_df.empty:
                    worst_eq = target_df['Equipment'].value_counts().idxmax()
                    worst_rate = (target_df['Equipment'].value_counts().max() / len(target_df)) * 100
                    st.error(f"🚨 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다.")
                else:
                    st.success("선택하신 웨이퍼에 불량이 발견되지 않았습니다.")
            with col2:
                df['Pass/Fail'] = np.where(df['BIN_Code'] == 'BIN01', 'Pass', 'Fail')
                fig_scatter = px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail", color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"}, opacity=0.5, template=PLOT_THEME, title="<b>FDC 파라미터 이상 확인</b>")
                st.plotly_chart(fig_scatter, use_container_width=True, config={'scrollZoom': True})

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
            st.plotly_chart(fig_cd, use_container_width=True, config={'scrollZoom': True})

        # ==========================================
        # 💡 [신규 추가] Tab 8: ML 기반 핵심 공정 변수 자동 추출
        # ==========================================
        with tab8:
            st.subheader("🤖 ML 기반 핵심 공정 변수 자동 추출 (Random Forest)")
            
            ml_features = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
            available_ml_features = [col for col in ml_features if col in df.columns]
            
            if len(available_ml_features) > 0:
                df_ml = df.dropna(subset=available_ml_features).copy()
                df_ml['Is_Fail'] = np.where(df_ml['BIN_Code'] != 'BIN01', 1, 0)
                
                # Random Forest 모델 학습
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(df_ml[available_ml_features], df_ml['Is_Fail'])
                
                # 변수 중요도 데이터프레임 생성
                importance_df = pd.DataFrame({
                    'Feature': available_ml_features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_rf = px.bar(
                    importance_df, x='Importance', y='Feature', orientation='h',
                    title="<b>공정 변수별 수율(불량) 기여도 (Feature Importance)</b>",
                    text_auto='.3f', color='Importance', color_continuous_scale='Blues',
                    template=PLOT_THEME
                )
                fig_rf.update_layout(height=400)
                st.plotly_chart(fig_rf, use_container_width=True, config={'scrollZoom': True})
                
                top_feature = importance_df.iloc[-1]['Feature']
                st.error(f"💡 **머신러닝(RF) 분석 결과**: **[{top_feature}]** 변수가 수율 Drop의 주요 원인일 확률이 가장 높습니다. 해당 파라미터의 설비 로그와 산포를 최우선으로 점검하세요.")
            else:
                st.warning("분석에 필요한 공정 변수(FDC) 데이터가 충분하지 않습니다.")

        # ==========================================
        # 💡 [신규 추가] Tab 9: 불량 공간 패턴(Spatial Pattern) 자동 분류
        # ==========================================
        with tab9:
            st.subheader("🎯 불량 공간 패턴(Spatial Pattern) 자동 분류 및 원인 추론")
            
            df_fail = df[(df['Wafer_ID'] == selected_wafer) & (df['BIN_Code'] != 'BIN01')].copy()
            
            if len(df_fail) < 5:
                st.success(f"현재 선택된 웨이퍼({selected_wafer})는 불량 개수가 너무 적어 명확한 패턴을 도출할 수 없습니다 (Random Pattern으로 간주).")
            else:
                # 1. 위치 기반 군집화 (DBSCAN)
                coords = df_fail[['X_Die', 'Y_Die']].values
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)
                
                # 군집화 수행
                db = DBSCAN(eps=0.5, min_samples=3).fit(coords_scaled)
                df_fail['Cluster'] = db.labels_
                
                # 2. 패턴 라벨링 알고리즘
                max_radius = np.sqrt(df['X_Die']**2 + df['Y_Die']**2).max()
                patterns = []
                
                for cluster_id in df_fail['Cluster'].unique():
                    if cluster_id == -1: # 노이즈 (군집 미포함)
                        patterns.extend(['Random'] * len(df_fail[df_fail['Cluster'] == cluster_id]))
                        continue
                        
                    cluster_data = df_fail[df_fail['Cluster'] == cluster_id]
                    mean_radius = np.mean(np.sqrt(cluster_data['X_Die']**2 + cluster_data['Y_Die']**2))
                    
                    # 스크래치(선형) 판별을 위한 종횡비 계산
                    dx = cluster_data['X_Die'].max() - cluster_data['X_Die'].min()
                    dy = cluster_data['Y_Die'].max() - cluster_data['Y_Die'].min()
                    aspect_ratio = max(dx, dy) / (min(dx, dy) + 1e-5)
                    
                    if aspect_ratio > 3.5 and len(cluster_data) >= 4:
                        pat = 'Scratch'
                    elif mean_radius > max_radius * 0.7:
                        pat = 'Edge Ring'
                    elif mean_radius < max_radius * 0.3:
                        pat = 'Center Cluster'
                    else:
                        pat = 'Random'
                        
                    patterns.extend([pat] * len(cluster_data))
                
                df_fail['Detected_Pattern'] = patterns
                
                col1, col2 = st.columns([5, 3])
                with col1:
                    fig_pattern = px.scatter(
                        df_fail, x="X_Die", y="Y_Die", color="Detected_Pattern",
                        title=f"<b>{selected_wafer} 불량 패턴 군집화 결과</b>",
                        color_discrete_map={"Edge Ring": "red", "Center Cluster": "blue", "Scratch": "orange", "Random": "gray"},
                        template=PLOT_THEME
                    )
                    fig_pattern.update_traces(marker=dict(symbol='square', size=12, line=dict(width=1, color='black')))
                    fig_pattern.update_layout(height=600, yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(t=50, b=0, l=0, r=0))
                    st.plotly_chart(fig_pattern, use_container_width=True, config={'scrollZoom': True})
                
                with col2:
                    st.markdown("### 🔍 분류된 패턴별 원인 가이드")
                    found_patterns = df_fail['Detected_Pattern'].unique()
                    
                    if 'Edge Ring' in found_patterns:
                        st.error("**[Edge Ring] 가장자리 띠 패턴**\n- **의심 원인**: 식각(Etch) 공정 플라즈마 밀도 불균일, 웨이퍼 테두리 온도 구배 문제, 포토 공정 EBR(Edge Bead Removal) 불량.")
                    if 'Center Cluster' in found_patterns:
                        st.warning("**[Center Cluster] 중심부 뭉침 패턴**\n- **의심 원인**: 박막(CVD) 증착 시 가스 분사 집중, 스핀 코팅 RPM/노즐 이상, 식각 공정 중앙부 Over-Etch.")
                    if 'Scratch' in found_patterns:
                        st.info("**[Scratch] 선형 긁힘 패턴**\n- **의심 원인**: CMP 평탄화 공정 중 슬러리 응집/패드 마모, 혹은 웨이퍼 이송 로봇 암(Robot Arm)에 의한 물리적 스크래치.")
                    if 'Random' in found_patterns:
                        st.success("**[Random] 무작위 패턴**\n- **의심 원인**: 클린룸 환경 요인, 설비 내 파티클(Particle) 낙하 등 일반적인 무작위 결함.")

        # ==========================================
        # 💡 [신규 추가] Tab 10: WiW 방사형 균일도 심층 분석
        # ==========================================
        with tab10:
            st.subheader("📍 WiW (Within-Wafer) 방사형 균일도 심층 분석")
            
            df_wiw = df[df['Wafer_ID'] == selected_wafer].copy()
            
            # 중심(0,0)으로부터의 반경(Radius) 계산
            df_wiw['Radius'] = np.sqrt(df_wiw['X_Die']**2 + df_wiw['Y_Die']**2)
            # 산포 차트를 위해 반경을 정수로 구간화
            df_wiw['Radius_Bin'] = df_wiw['Radius'].round(0)
            
            # 반경별 평균 CD와 수율 계산
            wiw_agg = df_wiw.groupby('Radius_Bin').apply(
                lambda x: pd.Series({
                    'Mean_CD': x['Actual_CD'].mean() if 'Actual_CD' in x.columns else np.nan,
                    'Yield(%)': (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100
                })
            ).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                # 산점도(투명하게) + 평균 추세선(꺾은선)
                fig_wiw_cd = px.scatter(
                    df_wiw, x='Radius', y='Actual_CD', opacity=0.2, color_discrete_sequence=['gray'],
                    title=f"<b>[{selected_wafer}] 중심부로부터 거리에 따른 CD 산포 추이</b>",
                    template=PLOT_THEME
                )
                fig_wiw_cd.add_scatter(
                    x=wiw_agg['Radius_Bin'], y=wiw_agg['Mean_CD'], mode='lines+markers',
                    line=dict(color='red', width=3), name='Mean CD Trend'
                )
                st.plotly_chart(fig_wiw_cd, use_container_width=True, config={'scrollZoom': True})
                
            with col2:
                # 거리에 따른 수율 변화를 꺾은선으로 표현
                fig_wiw_yield = px.line(
                    wiw_agg, x='Radius_Bin', y='Yield(%)', markers=True,
                    title=f"<b>[{selected_wafer}] 중심부로부터 반경별 수율(Yield) 변화</b>",
                    template=PLOT_THEME, line_shape='spline' # 부드러운 곡선
                )
                fig_wiw_yield.update_traces(line=dict(color='green', width=3), marker=dict(size=8))
                fig_wiw_yield.update_yaxes(range=[0, 105])
                st.plotly_chart(fig_wiw_yield, use_container_width=True, config={'scrollZoom': True})
                
            st.info("💡 **엔지니어링 인사이트**: 미세 공정에서는 웨이퍼 Center와 Edge 간의 챔버 온도, 플라즈마 밀도 차이로 인해 물리적 산포가 발생합니다. 위 차트에서 급격히 CD가 틀어지거나 수율이 꺾이는 지점을 확인하여 공정 Recipe(Edge Ring 온도 조정 등)를 튜닝할 수 있습니다.")

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. (에러 내역: {e})")
else:
    st.info("👈 좌측 사이드바에서 먼저 파이썬 코드로 생성한 'wafer_log_data.csv' 파일을 업로드해주세요.")