import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
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
PLOT_THEME = "plotly_white"

# ==========================================
# 0. 데이터 로드 로직 (데모 데이터 자동 로드 및 커스텀 업로드)
# ==========================================
st.sidebar.title("⚙️ Analysis Control")

# 1) 사이드바에 파일 업로더 배치
uploaded_file = st.sidebar.file_uploader("📁 새로운 CSV Data 업로드", type=['csv'])

# 2) 데이터 로드 결정 로직
df = None
if uploaded_file is not None:
    # 사용자가 직접 파일을 올린 경우
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ 업로드한 사용자 데이터가 적용되었습니다.")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 에러가 발생했습니다: {e}")
else:
    # 파일을 올리지 않은 경우, 기본 데모 파일 자동 로드
    default_file = 'wafer_log_data.csv'
    if os.path.exists(default_file):
        try:
            df = pd.read_csv(default_file)
            st.sidebar.info("💡 **포트폴리오 데모용 기본 수율 데이터가 자동 로드되었습니다.**\n\n다른 로그 파일을 분석하고 싶다면 위 업로더에 새 CSV를 넣어주세요.")
        except Exception as e:
            st.sidebar.error(f"기본 데이터를 읽는 중 에러가 발생했습니다: {e}")
    else:
        st.sidebar.warning(f"기본 데이터 파일('{default_file}')을 찾을 수 없습니다. CSV 파일을 업로드해 주세요.")

# ==========================================
# 메인 분석 로직 실행 (데이터가 정상 로드된 경우)
# ==========================================
if df is not None:
    try:
        # 데이터 전처리 (결측치 처리)
        if 'Defect_Type' in df.columns:
            df['Defect_Type'] = df['Defect_Type'].fillna('No_Defect')
            
        wafer_list = sorted(df['Wafer_ID'].dropna().unique())
        
        # ==========================================
        # 1. 사이드바 (Sidebar) - 최적화 및 공통 컨트롤 분리
        # ==========================================
        st.sidebar.divider()
        st.sidebar.header("📍 공통 분석 대상 설정")
        st.sidebar.caption("💡 여기서 선택한 웨이퍼는 [1. 2D 웨이퍼 맵], [9. 패턴 자동 분류], [10. WiW 방사형 분석] 탭의 결과에 공통으로 반영됩니다.")
        selected_wafer = st.sidebar.selectbox("Wafer_ID 선택:", wafer_list, key="map_wafer")
        st.sidebar.caption("위에서 분석할 타겟 웨이퍼를 선택해 주세요.")

        # ==========================================
        # 메인 화면 (Main Content) - 총 10개의 탭
        # ==========================================
        st.title("반도체 웨이퍼 수율 분석 대시보드")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            " 1. 2D 웨이퍼 맵", 
            " 2. 특정 Die 상세 추적",
            " 3. 누적 불량 패턴", 
            " 4. 변수 상관관계",
            " 5. 결함 분석", 
            " 6. 공통성 분석", 
            " 7. 공정능력지수",
            " 8. ML 변수 추출",
            " 9. 패턴 자동 분류",
            " 10. WiW 방사형 분석"
        ])
        
        # ------------------------------------------
        # Tab 1: 2D 웨이퍼 맵
        # ------------------------------------------
        with tab1:
            st.info("💡 **[2D 웨이퍼 맵]** 웨이퍼 전체의 칩(Die) 테스트 결과를 시각적으로 확인합니다. 특정 구역에 불량이 집중되어 있는지 직관적으로 파악할 수 있습니다.")
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
                max_r = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2).max()
                max_radius = float(max_r) if pd.notna(max_r) else 0.0
                
                df_wafer['Distance'] = np.sqrt(df_wafer['X_Die']**2 + df_wafer['Y_Die']**2)
                df_wafer['Region'] = 'Edge'
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

        # ------------------------------------------
        # Tab 2: 특정 Die 상세 추적 (전용 컨트롤러 내부 배치)
        # ------------------------------------------
        with tab2:
            st.info("💡 **[특정 Die 상세 추적]** 의심스러운 좌표의 단일 칩 이력을 추적합니다. 어떤 설비를 거쳤고 공정 센서 값(FDC)은 어땠는지 개별 원인을 분석합니다.")
            
            # 탭 전용 컨트롤러 가로 배치
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                inspect_wafer = st.selectbox("조회할 Wafer_ID:", wafer_list, key="inspect_wafer")
            with col_ctrl2:
                x_in = st.number_input("X 좌표:", min_value=int(df['X_Die'].min()), max_value=int(df['X_Die'].max()), value=0)
            with col_ctrl3:
                y_in = st.number_input("Y 좌표:", min_value=int(df['Y_Die'].min()), max_value=int(df['Y_Die'].max()), value=0)
            
            st.markdown("---")
            target_die = df[(df['Wafer_ID'] == inspect_wafer) & (df['X_Die'] == x_in) & (df['Y_Die'] == y_in)]
            
            if target_die.empty:
                st.warning("⚠️ 입력하신 좌표에는 칩(Die) 데이터가 존재하지 않습니다.")
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

        # ------------------------------------------
        # Tab 3: 누적 불량 패턴 (전용 컨트롤러 내부 배치)
        # ------------------------------------------
        with tab3:
            st.info("💡 **[누적 불량 패턴]** 특정 로트(Lot)의 웨이퍼들을 겹쳐서 불량 다발 구역을 찾습니다. 붉은색이 진할수록 고질적인 설비나 공정 문제를 의심할 수 있습니다.")
            
            if 'Lot_ID' in df.columns:
                lot_list = sorted(df['Lot_ID'].dropna().unique())
                # 탭 전용 컨트롤러 가로 배치
                selected_lot = st.selectbox("분석할 Lot_ID 선택:", lot_list)
                st.markdown("---")
                
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
            else:
                st.warning("데이터에 Lot_ID 컬럼이 없습니다.")

        # ------------------------------------------
        # Tab 4: 변수 상관관계 (전용 컨트롤러 내부 배치)
        # ------------------------------------------
        with tab4:
            st.info("💡 **[변수 상관관계]** 두 공정 변수(예: 온도 vs 압력) 간의 마진을 분석합니다. 정상 칩과 불량 칩이 분포하는 스펙 영역을 확인하여 최적의 레시피를 찾습니다.")
            
            param_candidates = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
            available_params = [p for p in param_candidates if p in df.columns]
            
            if len(available_params) >= 2:
                # 탭 전용 컨트롤러 가로 배치
                col_ctrl1, col_ctrl2 = st.columns(2)
                with col_ctrl1:
                    x_param = st.selectbox("X축 공정 변수 선택:", available_params, index=0)
                with col_ctrl2:
                    y_param = st.selectbox("Y축 공정 변수 선택:", available_params, index=min(3, len(available_params)-1))
                st.markdown("---")
                
                df_corr = df[[x_param, y_param, 'BIN_Code']].dropna()
                fig_corr = px.scatter(
                    df_corr, x=x_param, y=y_param, color="BIN_Code", color_discrete_map=COLOR_MAP, opacity=0.5,
                    title=f"<b>{x_param} vs {y_param} 상관도 분석</b>", template=PLOT_THEME
                )
                for bin_code in df_corr['BIN_Code'].unique():
                    sub = df_corr[df_corr['BIN_Code'] == bin_code]
                    sub_std = float(sub[x_param].std()) if len(sub) >= 2 else 0.0
                    if len(sub) >= 2 and sub_std > 0:
                        slope, intercept = np.polyfit(sub[x_param], sub[y_param], 1)
                        x_range = np.linspace(sub[x_param].min(), sub[x_param].max(), 50)
                        y_pred = slope * x_range + intercept
                        fig_corr.add_scatter(x=x_range, y=y_pred, mode='lines', line=dict(color=COLOR_MAP.get(bin_code, 'gray'), width=2), showlegend=False)
                st.plotly_chart(fig_corr, use_container_width=True, config={'scrollZoom': True})
            else:
                st.warning("상관관계를 분석할 공정 변수가 2개 이상 존재하지 않습니다.")

        # ------------------------------------------
        # Tab 5: 수율 및 결함 분석
        # ------------------------------------------
        with tab5:
            st.info("💡 **[수율 및 결함 분석]** 전체 테스트 통과 비율과 주요 불량 원인(결함 유형)의 수율 영향도(YIR)를 파악해, 개선 우선순위를 결정합니다.")
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

        # ------------------------------------------
        # Tab 6: 수율 Drop 공통성 분석 (전용 컨트롤러 내부 배치)
        # ------------------------------------------
        with tab6:
            st.info("💡 **[수율 Drop 추적]** 수율이 급락한 특정 웨이퍼가 어떤 설비에서 불량을 양산했는지(공통성) 역추적하여 설비 유지보수 타겟을 잡습니다.")
            
            # 탭 전용 컨트롤러 가로 배치
            col_ctrl1, col_ctrl2 = st.columns([1, 2])
            with col_ctrl1:
                analyze_wafer = st.selectbox("🚨 공통성을 분석할 수율 저하 Wafer_ID:", wafer_list, index=min(6, len(wafer_list) - 1), key="common_wafer")
            
            st.markdown("---")
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
                    worst_eq = str(target_df['Equipment'].value_counts().idxmax())
                    worst_rate = float((target_df['Equipment'].value_counts().max() / len(target_df)) * 100)
                    st.error(f"🚨 **[{analyze_wafer}]** 불량 칩 중 **{worst_rate:.1f}%**가 **[{worst_eq}]** 설비를 거쳤습니다. (해당 설비 점검 요망)")
                else:
                    st.success("선택하신 웨이퍼에 불량이 발견되지 않았습니다.")
            with col2:
                df['Pass/Fail'] = np.where(df['BIN_Code'] == 'BIN01', 'Pass', 'Fail')
                fig_scatter = px.scatter(df, x="FDC_Temp", y="FDC_Pressure", color="Pass/Fail", color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"}, opacity=0.5, template=PLOT_THEME, title="<b>FDC 파라미터 이상 산포 확인</b>")
                st.plotly_chart(fig_scatter, use_container_width=True, config={'scrollZoom': True})

        # ------------------------------------------
        # Tab 7: 공정능력지수
        # ------------------------------------------
        with tab7:
            st.info("💡 **[공정능력지수]** 핵심 치수(CD)가 타겟 스펙 안에 얼마나 잘 들어오는지(Cp, Cpk) 통계적으로 분석하여 양산 안정성을 평가합니다.")
            target = float(df['Target_CD'].iloc[0]) if 'Target_CD' in df.columns else 50.0
            USL, LSL = target + 4.0, target - 4.0
            mu = float(df['Actual_CD'].mean())
            sigma = float(df['Actual_CD'].std())
            cp, cpk = ((USL - LSL) / (6 * sigma), min((USL - mu)/(3 * sigma), (mu - LSL)/(3 * sigma))) if sigma > 0 else (0, 0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target CD", f"{target:.1f} nm")
            col2.metric("Cp (Process Capability)", f"{cp:.2f}")
            col3.metric("Cpk (Actual Capability)", f"{cpk:.2f}")
            
            fig_cd = px.histogram(df, x="Actual_CD", color="Pass/Fail", nbins=40, title="<b>Actual CD(선폭) 분포</b>", template=PLOT_THEME)
            fig_cd.add_vline(x=target, line_dash="dash", line_color="black")
            st.plotly_chart(fig_cd, use_container_width=True, config={'scrollZoom': True})

        # ------------------------------------------
        # 💡 [신규] Tab 8: ML 변수 추출
        # ------------------------------------------
        with tab8:
            st.info("💡 **[ML 변수 추출]** 수백 개의 설비 센서 데이터 중, 머신러닝이 판단한 수율 저하의 가장 큰 원인(Feature)을 자동으로 찾아내 분석 시간을 단축합니다.")
            ml_features = ['FDC_Temp', 'FDC_Pressure', 'RF_Power(W)', 'Actual_CD']
            available_ml_features = [col for col in ml_features if col in df.columns]
            
            if len(available_ml_features) > 0:
                df_ml = df.dropna(subset=available_ml_features).copy()
                df_ml['Is_Fail'] = np.where(df_ml['BIN_Code'] != 'BIN01', 1, 0)
                
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(df_ml[available_ml_features], df_ml['Is_Fail'])
                
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
                st.error(f"🚨 **머신러닝(RF) 분석 결과**: **[{top_feature}]** 변수가 수율 Drop의 주요 원인일 확률이 가장 높습니다. 해당 파라미터의 설비 로그와 산포를 최우선으로 점검하세요.")
            else:
                st.warning("분석에 필요한 공정 변수(FDC) 데이터가 충분하지 않습니다.")

        # ------------------------------------------
        # 💡 [신규] Tab 9: 패턴 자동 분류
        # ------------------------------------------
        with tab9:
            st.info("💡 **[패턴 자동 분류]** 군집화 AI가 불량의 공간적 패턴을 라벨링합니다. 패턴(Edge, Center, Scratch 등)에 따라 특정 설비나 공정 결함을 즉각적으로 추론할 수 있습니다.")
            df_fail = df[(df['Wafer_ID'] == selected_wafer) & (df['BIN_Code'] != 'BIN01')].copy()
            
            if len(df_fail) < 5:
                st.success(f"현재 선택된 웨이퍼({selected_wafer})는 불량 개수가 너무 적어 명확한 패턴을 도출할 수 없습니다 (Random Pattern으로 간주).")
            else:
                coords = df_fail[['X_Die', 'Y_Die']].values
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)
                
                db = DBSCAN(eps=0.5, min_samples=3).fit(coords_scaled)
                df_fail['Cluster'] = db.labels_
                
                max_r = np.sqrt(df['X_Die']**2 + df['Y_Die']**2).max()
                max_radius = float(max_r) if pd.notna(max_r) else 0.0
                patterns = []
                
                for cluster_id in df_fail['Cluster'].unique():
                    c_id = int(cluster_id)
                    
                    if c_id == -1: 
                        patterns.extend(['Random'] * len(df_fail[df_fail['Cluster'] == cluster_id]))
                        continue
                        
                    cluster_data = df_fail[df_fail['Cluster'] == cluster_id]
                    mean_radius = float(np.mean(np.sqrt(cluster_data['X_Die']**2 + cluster_data['Y_Die']**2)))
                    
                    dx = float(cluster_data['X_Die'].max()) - float(cluster_data['X_Die'].min())
                    dy = float(cluster_data['Y_Die'].max()) - float(cluster_data['Y_Die'].min())
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
                        st.error("**[Edge Ring] 가장자리 띠 패턴**\n- **의심 원인**: 식각(Etch) 공정 플라즈마 밀도 불균일, 웨이퍼 테두리 온도 구배 문제, 포토 공정 EBR 불량.")
                    if 'Center Cluster' in found_patterns:
                        st.warning("**[Center Cluster] 중심부 뭉침 패턴**\n- **의심 원인**: 박막(CVD) 증착 시 가스 분사 집중, 스핀 코팅 RPM 이상, 식각 공정 중앙부 Over-Etch.")
                    if 'Scratch' in found_patterns:
                        st.info("**[Scratch] 선형 긁힘 패턴**\n- **의심 원인**: CMP 평탄화 공정 중 슬러리 응집/패드 마모, 혹은 웨이퍼 이송 로봇 암(Robot Arm) 물리적 스크래치.")
                    if 'Random' in found_patterns:
                        st.success("**[Random] 무작위 패턴**\n- **의심 원인**: 클린룸 환경 요인, 설비 내 파티클(Particle) 낙하 등 일반적인 무작위 결함.")

        # ------------------------------------------
        # 💡 [신규] Tab 10: WiW 방사형 분석
        # ------------------------------------------
        with tab10:
            st.info("💡 **[WiW 방사형 분석]** 웨이퍼 중심부터 가장자리까지의 거리(Radius)에 따른 산포를 추적합니다. 미세 공정에서 Center-Edge 간의 균일도(Uniformity)를 잡는 데 필수적입니다.")
            df_wiw = df[df['Wafer_ID'] == selected_wafer].copy()
            
            if not df_wiw.empty:
                df_wiw['Radius'] = np.sqrt(df_wiw['X_Die']**2 + df_wiw['Y_Die']**2)
                df_wiw['Radius_Bin'] = df_wiw['Radius'].round(0)
                
                wiw_agg = df_wiw.groupby('Radius_Bin').apply(
                    lambda x: pd.Series({
                        'Mean_CD': x['Actual_CD'].mean() if 'Actual_CD' in x.columns else np.nan,
                        'Yield(%)': (len(x[x['BIN_Code'] == 'BIN01']) / len(x)) * 100
                    })
                ).reset_index()
                
                col1, col2 = st.columns(2)
                with col1:
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
                    fig_wiw_yield = px.line(
                        wiw_agg, x='Radius_Bin', y='Yield(%)', markers=True,
                        title=f"<b>[{selected_wafer}] 중심부로부터 반경별 수율(Yield) 변화</b>",
                        template=PLOT_THEME, line_shape='spline'
                    )
                    fig_wiw_yield.update_traces(line=dict(color='green', width=3), marker=dict(size=8))
                    fig_wiw_yield.update_yaxes(range=[0, 105])
                    st.plotly_chart(fig_wiw_yield, use_container_width=True, config={'scrollZoom': True})

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다. (에러 내역: {e})")