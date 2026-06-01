import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io

# ─── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="서울 기온 분석 · 1980년 전후",
    page_icon="🌡️",
    layout="wide",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 배경 */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* 헤더 */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 50%, #6bcb77 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero p {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* KPI 카드 */
.kpi-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.kpi-card {
    flex: 1;
    min-width: 160px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.blue::before  { background: #58a6ff; }
.kpi-card.red::before   { background: #ff6b6b; }
.kpi-card.green::before { background: #3fb950; }
.kpi-card.yellow::before{ background: #ffd93d; }

.kpi-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #e6edf3;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 0.3rem;
}

/* 섹션 타이틀 */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.7rem;
    margin: 1.8rem 0 1rem;
}

/* 인사이트 박스 */
.insight-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-size: 0.88rem;
    color: #8b949e;
    line-height: 1.7;
}
.insight-box strong { color: #e6edf3; }
.insight-box .highlight { color: #ff6b6b; font-weight: 600; }

/* Streamlit 기본 요소 스타일 덮어쓰기 */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
.stSlider label, .stCheckbox label, .stRadio label {
    color: #c9d1d9 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return None

    df.columns = df.columns.str.strip()
    df['날짜'] = df['날짜'].astype(str).str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month

    # 컬럼명 유연하게 처리
    temp_col = [c for c in df.columns if '평균기온' in c or '기온' in c]
    if temp_col:
        df = df.rename(columns={temp_col[0]: '평균기온'})
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    return df


# ─── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"],
                                help="기상청 일별 기온 데이터 (날짜, 평균기온 컬럼 필요)")

    st.markdown("---")
    split_year = st.slider("📅 분기점 연도", 1950, 2010, 1980, 1,
                           help="이 연도를 기준으로 전·후 기간을 나눕니다")

    st.markdown("---")
    show_raw = st.checkbox("원시 연간 데이터 표시", value=False)
    show_decade = st.checkbox("10년 이동평균 표시", value=True)
    show_dist = st.checkbox("분포 비교 표시", value=True)

    st.markdown("---")
    st.markdown("<small style='color:#6e7681'>데이터: 기상청 서울 관측소(108)</small>",
                unsafe_allow_html=True)


# ─── 데이터 로드 ───────────────────────────────────────────────────────────────
df = load_data(uploaded)

if df is None:
    st.markdown("""
    <div class='hero'>
      <h1>🌡️ 서울 기온 상승 분석</h1>
      <p>1980년 전후, 기온 상승 속도는 얼마나 달라졌을까?</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 왼쪽 사이드바에서 기상청 CSV 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()


# ─── 연간 평균 계산 ────────────────────────────────────────────────────────────
annual = (df.groupby('연도')['평균기온']
            .mean()
            .dropna()
            .reset_index())
annual.columns = ['연도', '평균기온']

# 분기점 기준 분리 (마지막 연도 제외 – 일부 데이터만 있을 수 있음)
max_year = annual['연도'].max()
annual_clean = annual[annual['연도'] < max_year].copy()

before = annual_clean[annual_clean['연도'] < split_year]
after  = annual_clean[annual_clean['연도'] >= split_year]

# 선형 회귀
def linreg(sub):
    x = sub['연도'].values.astype(float)
    y = sub['평균기온'].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    return slope, intercept, r**2, p

sl_b, ic_b, r2_b, p_b = linreg(before)
sl_a, ic_a, r2_a, p_a = linreg(after)

mean_b = before['평균기온'].mean()
mean_a = after['평균기온'].mean()
delta_mean = mean_a - mean_b
speed_ratio = sl_a / sl_b if sl_b != 0 else float('inf')

# 10년 이동평균
annual_clean = annual_clean.sort_values('연도')
annual_clean['이동평균_10y'] = annual_clean['평균기온'].rolling(10, center=True).mean()


# ─── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
  <h1>🌡️ 서울 기온 상승 분석</h1>
  <p>{annual_clean['연도'].min()}년 ~ {annual_clean['연도'].max()}년 · 분기점: <strong>{split_year}년</strong></p>
</div>
""", unsafe_allow_html=True)


# ─── KPI 카드 ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='kpi-row'>
  <div class='kpi-card blue'>
    <div class='kpi-label'>{split_year}년 이전 상승 속도</div>
    <div class='kpi-value'>{sl_b:+.4f}</div>
    <div class='kpi-sub'>°C / 년 &nbsp;|&nbsp; R² = {r2_b:.3f}</div>
  </div>
  <div class='kpi-card red'>
    <div class='kpi-label'>{split_year}년 이후 상승 속도</div>
    <div class='kpi-value'>{sl_a:+.4f}</div>
    <div class='kpi-sub'>°C / 년 &nbsp;|&nbsp; R² = {r2_a:.3f}</div>
  </div>
  <div class='kpi-card yellow'>
    <div class='kpi-label'>상승 속도 배율</div>
    <div class='kpi-value'>{speed_ratio:.1f}×</div>
    <div class='kpi-sub'>이후 / 이전</div>
  </div>
  <div class='kpi-card green'>
    <div class='kpi-label'>평균 기온 상승</div>
    <div class='kpi-value'>{delta_mean:+.2f}</div>
    <div class='kpi-sub'>°C (기간 평균 차이)</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── 차트 1: 연간 기온 추세 + 회귀선 ──────────────────────────────────────────
st.markdown("<div class='section-title'>📈 연간 평균기온 추세 및 회귀선</div>",
            unsafe_allow_html=True)

fig1 = go.Figure()

# 전체 원시 데이터 (회색)
if show_raw:
    fig1.add_trace(go.Scatter(
        x=annual_clean['연도'], y=annual_clean['평균기온'],
        mode='markers',
        marker=dict(size=3, color='#30363d'),
        name='연간 평균기온', hovertemplate='%{x}년: %{y:.2f}°C<extra></extra>'
    ))

# 10년 이동평균
if show_decade:
    fig1.add_trace(go.Scatter(
        x=annual_clean['연도'], y=annual_clean['이동평균_10y'],
        mode='lines', line=dict(color='#8b949e', width=1.5, dash='dot'),
        name='10년 이동평균', hovertemplate='%{x}년: %{y:.2f}°C<extra></extra>'
    ))

# 이전 구간 회귀선
x_b = np.array([before['연도'].min(), before['연도'].max()])
fig1.add_trace(go.Scatter(
    x=x_b, y=ic_b + sl_b * x_b,
    mode='lines', line=dict(color='#58a6ff', width=3),
    name=f'{split_year}년 이전 추세 ({sl_b:+.4f}°C/년)'
))

# 이후 구간 회귀선
x_a = np.array([after['연도'].min(), after['연도'].max()])
fig1.add_trace(go.Scatter(
    x=x_a, y=ic_a + sl_a * x_a,
    mode='lines', line=dict(color='#ff6b6b', width=3),
    name=f'{split_year}년 이후 추세 ({sl_a:+.4f}°C/년)'
))

# 분기점 수직선
fig1.add_vline(x=split_year, line_dash='dash', line_color='#ffd93d',
               line_width=1.5,
               annotation_text=f"  {split_year}년", annotation_font_color='#ffd93d')

# 배경 색상 영역
fig1.add_vrect(x0=annual_clean['연도'].min(), x1=split_year,
               fillcolor='rgba(88,166,255,0.04)', line_width=0)
fig1.add_vrect(x0=split_year, x1=annual_clean['연도'].max(),
               fillcolor='rgba(255,107,107,0.04)', line_width=0)

fig1.update_layout(
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    font=dict(color='#8b949e', family='Noto Sans KR'),
    legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1,
                font=dict(size=11)),
    xaxis=dict(gridcolor='#21262d', zeroline=False, title='연도'),
    yaxis=dict(gridcolor='#21262d', zeroline=False, title='평균기온 (°C)'),
    hovermode='x unified', margin=dict(l=10, r=10, t=20, b=10),
    height=420,
)

st.plotly_chart(fig1, use_container_width=True)


# ─── 차트 2+3: 10년 평균 막대 / 분포 ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>📊 10년별 평균기온</div>",
                unsafe_allow_html=True)

    annual_clean['decade'] = (annual_clean['연도'] // 10 * 10)
    decade_avg = annual_clean.groupby('decade')['평균기온'].mean().reset_index()
    decade_avg['기간'] = [f"{d}s" for d in decade_avg['decade']]
    decade_avg['색상'] = decade_avg['decade'].apply(
        lambda d: '#ff6b6b' if d >= split_year else '#58a6ff'
    )

    fig2 = go.Figure(go.Bar(
        x=decade_avg['기간'], y=decade_avg['평균기온'],
        marker_color=decade_avg['색상'],
        text=decade_avg['평균기온'].apply(lambda v: f'{v:.2f}°C'),
        textposition='outside', textfont=dict(size=11, color='#8b949e'),
        hovertemplate='%{x}: %{y:.2f}°C<extra></extra>',
    ))
    fig2.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='Noto Sans KR'),
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d', title='°C',
                   range=[decade_avg['평균기온'].min() - 0.5,
                          decade_avg['평균기온'].max() + 0.8]),
        margin=dict(l=10, r=10, t=10, b=10), height=330,
        showlegend=False,
    )
    # 이전/이후 범례 추가
    fig2.add_trace(go.Bar(x=[None], y=[None], name=f'{split_year}년 이전',
                          marker_color='#58a6ff'))
    fig2.add_trace(go.Bar(x=[None], y=[None], name=f'{split_year}년 이후',
                          marker_color='#ff6b6b'))
    fig2.update_layout(showlegend=True,
                       legend=dict(bgcolor='#161b22', bordercolor='#30363d',
                                   borderwidth=1, orientation='h',
                                   yanchor='bottom', y=1.01))
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    if show_dist:
        st.markdown("<div class='section-title'>📉 기온 분포 비교</div>",
                    unsafe_allow_html=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Violin(
            y=before['평균기온'], name=f'{split_year}년 이전',
            box_visible=True, meanline_visible=True,
            line_color='#58a6ff', fillcolor='rgba(88,166,255,0.15)',
            hovertemplate='%{y:.2f}°C<extra></extra>',
        ))
        fig3.add_trace(go.Violin(
            y=after['평균기온'], name=f'{split_year}년 이후',
            box_visible=True, meanline_visible=True,
            line_color='#ff6b6b', fillcolor='rgba(255,107,107,0.15)',
            hovertemplate='%{y:.2f}°C<extra></extra>',
        ))
        fig3.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            font=dict(color='#8b949e', family='Noto Sans KR'),
            yaxis=dict(gridcolor='#21262d', title='연간 평균기온 (°C)'),
            xaxis=dict(gridcolor='#21262d'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
            margin=dict(l=10, r=10, t=10, b=10), height=330,
            violinmode='group',
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown("<div class='section-title'>📉 기온 분포 비교</div>",
                    unsafe_allow_html=True)
        st.info("왼쪽 사이드바에서 '분포 비교 표시'를 켜세요.")


# ─── 차트 4: 월별 기온 상승 히트맵 ────────────────────────────────────────────
st.markdown("<div class='section-title'>🗓️ 월별 기온 변화 (이전 vs 이후 평균 차이)</div>",
            unsafe_allow_html=True)

monthly_before = df[df['연도'] < split_year].groupby('월')['평균기온'].mean()
monthly_after  = df[df['연도'] >= split_year].groupby('월')['평균기온'].mean()
monthly_diff   = (monthly_after - monthly_before).reindex(range(1, 13))

month_labels = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월']
colors_bar = ['#ff6b6b' if v >= 0 else '#58a6ff' for v in monthly_diff.values]

fig4 = go.Figure(go.Bar(
    x=month_labels, y=monthly_diff.values,
    marker_color=colors_bar,
    text=[f'{v:+.2f}°C' for v in monthly_diff.values],
    textposition='outside', textfont=dict(size=11, color='#8b949e'),
    hovertemplate='%{x}: %{y:+.2f}°C<extra></extra>',
))
fig4.add_hline(y=0, line_color='#30363d', line_width=1)
fig4.update_layout(
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    font=dict(color='#8b949e', family='Noto Sans KR'),
    xaxis=dict(gridcolor='#21262d'),
    yaxis=dict(gridcolor='#21262d', title='기온 차이 (°C)',
               zeroline=False),
    margin=dict(l=10, r=10, t=10, b=10), height=300,
    showlegend=False,
)
st.plotly_chart(fig4, use_container_width=True)


# ─── 통계 검정 결과 ────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🔬 통계 검정</div>",
            unsafe_allow_html=True)

t_stat, t_p = stats.ttest_ind(before['평균기온'], after['평균기온'])
ks_stat, ks_p = stats.ks_2samp(before['평균기온'], after['평균기온'])

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("t-검정 (평균 차이)", f"t = {t_stat:.3f}",
              delta=f"p = {t_p:.2e} {'✅ 유의' if t_p < 0.05 else '❌ 비유의'}")
with col_b:
    st.metric("KS 검정 (분포 차이)", f"D = {ks_stat:.3f}",
              delta=f"p = {ks_p:.2e} {'✅ 유의' if ks_p < 0.05 else '❌ 비유의'}")
with col_c:
    st.metric("회귀 p-value (이후 구간)", f"{p_a:.2e}",
              delta="✅ 유의" if p_a < 0.05 else "❌ 비유의")


# ─── 인사이트 요약 ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>💡 분석 요약</div>",
            unsafe_allow_html=True)

sig_b = "통계적으로 유의" if p_b < 0.05 else "통계적으로 유의하지 않음"
sig_a = "통계적으로 유의" if p_a < 0.05 else "통계적으로 유의하지 않음"

st.markdown(f"""
<div class='insight-box'>
  <p>
  서울 기상 관측 데이터를 <strong>{split_year}년</strong>을 기준으로 분석한 결과, 두 기간 사이의 기온 상승 속도에 
  <span class='highlight'>뚜렷한 차이</span>가 확인됩니다.
  </p>
  <ul>
    <li><strong>{split_year}년 이전</strong>: 연 평균 <strong>{sl_b:+.4f}°C</strong> 상승 ({sig_b}, R²={r2_b:.3f})</li>
    <li><strong>{split_year}년 이후</strong>: 연 평균 <strong>{sl_a:+.4f}°C</strong> 상승 ({sig_a}, R²={r2_a:.3f})</li>
    <li>이후 구간의 상승 속도는 이전보다 약 <span class='highlight'>{speed_ratio:.1f}배</span> 빠릅니다.</li>
    <li>두 기간의 <strong>평균 기온 차이</strong>는 <span class='highlight'>{delta_mean:+.2f}°C</span>이며,
        t-검정 결과 {'유의미한 차이가 확인됩니다' if t_p < 0.05 else '유의미한 차이는 확인되지 않습니다'} 
        (p = {t_p:.2e}).</li>
  </ul>
  <p style='margin-bottom:0; color:#6e7681; font-size:0.82rem;'>
  ※ 데이터: 기상청 서울 관측소(108) · 분석 단위: 연간 평균기온
  </p>
</div>
""", unsafe_allow_html=True)


# ─── 원시 데이터 테이블 ────────────────────────────────────────────────────────
if show_raw:
    st.markdown("<div class='section-title'>📋 연간 평균기온 데이터</div>",
                unsafe_allow_html=True)
    st.dataframe(
        annual_clean[['연도', '평균기온', '이동평균_10y']]
            .rename(columns={'이동평균_10y': '10년이동평균'})
            .set_index('연도')
            .style.format({'평균기온': '{:.2f}°C', '10년이동평균': '{:.2f}°C'})
            .background_gradient(subset=['평균기온'], cmap='RdYlBu_r'),
        use_container_width=True, height=300,
    )
