import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================================
# 한글 폰트 · CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""", unsafe_allow_html=True)

PLOT_FONT = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"


# ============================================================
# CSV 읽기 (인코딩 이중 시도 · 탭 구분 안전장치)
# ============================================================
@st.cache_data
def read_csv_file(file):
    """utf-8-sig → cp949 순으로 시도.
    열이 1개만 나오면 탭(tab) 구분일 가능성이 높으므로
    sep=None(자동감지) 또는 sep='\\t'로 다시 읽는다."""
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            # 구분자가 틀려서 열이 1개만 나온 경우 → 재도전
            if df.shape[1] == 1:
                first_val = str(df.iloc[0, 0])
                if "\t" in first_val:
                    df_auto = pd.read_csv(
                        io.BytesIO(raw), encoding=enc, sep=None, engine="python"
                    )
                    if df_auto.shape[1] > 1:
                        return df_auto
                    df_tab = pd.read_csv(io.BytesIO(raw), encoding=enc, sep="\t")
                    if df_tab.shape[1] > 1:
                        return df_tab
            return df
        except Exception:
            continue
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# 열 이름 표준화 (KeyError 방지)
# ============================================================
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            extracted = (
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0]
            )
            num = pd.to_numeric(extracted, errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


# ============================================================
# 환경 요약 표
# ============================================================
def env_summary(df):
    cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "측정 항목": c,
            "평균": round(s.mean(), 2),
            "최소": round(s.min(), 2),
            "최대": round(s.max(), 2),
        })
    return pd.DataFrame(rows)


# ============================================================
# 일별 시계열 꺾은선
# ============================================================
def daily_timeseries(df, col, color, title):
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].dropna()
    if len(d) == 0:
        return None
    d["day"] = d["time"].dt.date
    daily = d.groupby("day")[col].mean().dropna()
    if len(daily) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index,
        y=daily.values,
        mode="lines",
        name=col,
        line=dict(color=color, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=col,
        font=dict(family=PLOT_FONT),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
    )
    return fig


# ============================================================
# 학교명 추출
# ============================================================
def _get_filename(obj):
    if obj is None:
        return None
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


def school_name_from_fileobj(fileobj):
    fn = _get_filename(fileobj)
    if fn is None:
        return "우리 학교"
    base = os.path.splitext(fn)[0]
    name = base.replace("_", " ").replace("(", "").replace(")", "")
    return name if name else "우리 학교"


# ============================================================
# Tab 1 — 실험 개요
# ============================================================
def render_tab1(env_df, growth_df):
    with st.container():
        try:
            st.subheader("🔬 연구 배경 및 목적")
            st.markdown("""
**극지식물(Extremophile plant)**은 온도·습도·pH·염분(EC) 등
척박한 환경에서도 생존하고 성장하는 식물입니다.

이 대시보드에서는 우리 학교 극지식물 개체의 **생육 결과**와
재배 기간 동안의 **환경 데이터**를 함께 분석하여
개체 간 생육 차이의 원인을 탐색합니다.

> **통계 용어 간단 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 값 — 대표값
> - **표준편차**: 값이 평균 주위에 얼마나 흩어져 있는지 — 클수록 편차 큼
> - **CV(변동계수)**: 표준편차를 평균으로 나눈 값 — 단위가 다른 집단 비교에 유용
> - **상관계수**: 두 변수가 함께 움직이는 정도 — -1 ~ +1, 0에 가까우면 관계 약함
            """)

            st.subheader("📋 데이터 요약")
            summary_rows = []

            if not growth_df.empty:
                g_cols = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                "지상부 질량", "지하부 질량", "생중량"]
                    if c in growth_df.columns
                ]
                summary_rows.append({
                    "데이터": "생육 결과",
                    "개체 수": str(len(growth_df)),
                    "측정 기간": "측정 완료",
                    "측정 항목": ", ".join(g_cols) if g_cols else "(없음)",
                })

            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    env_period = (
                        f"{t.min().strftime('%Y-%m-%d')} ~ "
                        f"{t.max().strftime('%Y-%m-%d')}"
                    )
                else:
                    env_period = "알 수 없음"
                env_cols = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns
                ]
                summary_rows.append({
                    "데이터": "환경 데이터",
                    "개체 수": "―",
                    "측정 기간": env_period,
                    "측정 항목": ", ".join(env_cols) if env_cols else "(없음)",
                })

            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("📁 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

            st.subheader("📊 주요 지표")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    n = len(growth_df.dropna(subset=["생중량"]))
                    st.metric("총 개체 수", f"{n}개")
                elif not growth_df.empty:
                    st.metric("총 개체 수", f"{len(growth_df)}개")
                else:
                    st.metric("총 개체 수", "―")

            with col2:
                if not env_df.empty and "temperature" in env_df.columns:
                    avg_t = env_df["temperature"].dropna().mean()
                    if not np.isnan(avg_t):
                        st.metric("평균 온도", f"{avg_t:.1f}℃")
                    else:
                        st.metric("평균 온도", "―")
                else:
                    st.metric("평균 온도", "―")

            with col3:
                if not env_df.empty and "humidity" in env_df.columns:
                    avg_h = env_df["humidity"].dropna().mean()
                    if not np.isnan(avg_h):
                        st.metric("평균 습도", f"{avg_h:.1f}%")
                    else:
                        st.metric("평균 습도", "―")
                else:
                    st.metric("평균 습도", "―")

            with col4:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    avg_b = growth_df["생중량"].dropna().mean()
                    if not np.isnan(avg_b):
                        st.metric("평균 생중량", f"{avg_b:.2f}g")
                    else:
                        st.metric("평균 생중량", "―")
                else:
                    st.metric("평균 생중량", "―")

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# Tab 2 — 환경 데이터
# ============================================================
def render_tab2(env_df):
    with st.container():
        try:
            if env_df.empty or "time" not in env_df.columns:
                st.info(
                    "📁 환경 CSV를 사이드바에서 업로드하면 "
                    "환경 데이터 탭이 활성화됩니다."
                )
                return

            st.subheader("🌡️ 환경 데이터 요약")
            st.caption(
                "온도·습도·pH·EC의 측정 기간 동안 평균·최소·최대값입니다."
            )

            summ = env_summary(env_df)
            if not summ.empty:
                st.dataframe(summ, width="stretch", use_container_width=True)
            else:
                st.info("표시할 환경 측정 항목이 없습니다.")

            st.subheader("📈 날짜별 환경 변화 (일별 평균)")
            st.caption("매일 측정된 값을 일별로 평균내어 꺾은선으로 표시합니다.")
            st.info("💡 pH·EC는 해당 열을 측정한 학교만 그래프가 표시됩니다.")

            fig_t = daily_timeseries(env_df, "temperature", "#e74c3c",
                                     "일별 평균 온도 변화")
            if fig_t is not None:
                st.plotly_chart(fig_t, width="stretch")
            else:
                st.info("온도(temperature) 열이 없어 온도 그래프를 표시할 수 없습니다.")

            fig_h = daily_timeseries(env_df, "humidity", "#3498db",
                                     "일별 평균 습도 변화")
            if fig_h is not None:
                st.plotly_chart(fig_h, width="stretch")
            else:
                st.info("습도(humidity) 열이 없어 습도 그래프를 표시할 수 없습니다.")

            if "ph" in env_df.columns:
                fig_ph = daily_timeseries(env_df, "ph", "#9b59b6",
                                           "일별 평균 pH 변화")
                if fig_ph is not None:
                    st.plotly_chart(fig_ph, width="stretch")
                else:
                    st.info("pH 데이터 포인트가 부족해 그래프를 표시할 수 없습니다.")
            else:
                st.info("pH(ph) 열이 없는 학교입니다. pH 그래프는 생략됩니다.")

            if "ec" in env_df.columns:
                fig_ec = daily_timeseries(env_df, "ec", "#1abc9c",
                                           "일별 평균 EC 변화")
                if fig_ec is not None:
                    st.plotly_chart(fig_ec, width="stretch")
                else:
                    st.info("EC 데이터 포인트가 부족해 그래프를 표시할 수 없습니다.")
            else:
                st.info("EC(ec) 열이 없는 학교입니다. EC 그래프는 생략됩니다.")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=True)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name="환경데이터_download.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# Tab 3 — 생육 결과
# ============================================================
def render_tab3(growth_df):
    with st.container():
        try:
            if growth_df.empty:
                st.info(
                    "📁 생육 결과 CSV를 사이드바에서 업로드하면 "
                    "생육 결과 탭이 활성화됩니다."
                )
                return

            has_bw = "생중량" in growth_df.columns

            if not has_bw:
                st.warning(
                    "⚠️ 이 학교의 생육 데이터에는 '생중량(g)' 열이 없습니다. "
                    "지상부 질량 + 지하부 질량으로 생중량을 계산해 분석합니다."
                )

            st.subheader("🥇 핵심 결과")
            st.caption(
                "개체별 생중량(생체 무게)을 기준으로 한 주요 지표입니다."
            )
            st.info(
                "💡 생중량 = 지상부 질량 + 지하부 질량 "
                "(생중량(g) 열이 없는 학교는 자동 계산)"
            )

            id_col = "개체번호" if "개체번호" in growth_df.columns else None

            if has_bw:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 1 and not bw.empty:
                    mean_bw = bw.mean()
                    max_idx = bw.idxmax()
                    min_idx = bw.idxmin()

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("평균 생중량", f"{mean_bw:.2f}g")
                    with col_b:
                        id_max = (
                            growth_df.loc[max_idx, id_col]
                            if id_col else f"#{max_idx}"
                        )
                        st.metric(
                            "📈 생중량 최대 개체",
                            f"{growth_df.loc[max_idx, '생중량']:.2f}g",
                            f"개체: {id_max}",
                        )
                    with col_c:
                        id_min = (
                            growth_df.loc[min_idx, id_col]
                            if id_col else f"#{min_idx}"
                        )
                        st.metric(
                            "📉 생중량 최소 개체",
                            f"{growth_df.loc[min_idx, '생중량']:.2f}g",
                            f"개체: {id_min}",
                        )
                else:
                    st.info("생중량 값이 하나도 없어 지표를 계산할 수 없습니다.")
            else:
                for c in st.columns(3):
                    c.metric("―", "생중량 데이터 없음")

            st.divider()

            # ── 상·하위 25% 그룹 비교 ──
            st.subheader(
                "📊 생육 측정 항목: 상위 25% vs 하위 25% 개체 비교"
            )
            st.caption(
                "생중량이 가장 큰 상위 25% 개체들과 가장 작은 하위 25% 개체들의 "
                "평균 측정 항목을 비교합니다. 어떤 측정 항목이 생중량 차이를 만드는지 확인하세요."
            )

            if has_bw and len(growth_df.dropna(subset=["생중량"])) >= 4:
                bw_vals = growth_df["생중량"].dropna()
                q75 = bw_vals.quantile(0.75)
                q25 = bw_vals.quantile(0.25)

                top_mask = growth_df["생중량"] >= q75
                bot_mask = growth_df["생중량"] <= q25
                n_top = int(top_mask.sum())
                n_bot = int(bot_mask.sum())

                top_df = growth_df[top_mask]
                bot_df = growth_df[bot_mask]

                item_cols = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                "지상부 질량", "지하부 질량", "생중량"]
                    if c in growth_df.columns
                ]

                if item_cols:
                    top_means = [top_df[c].dropna().mean() for c in item_cols]
                    bot_means = [bot_df[c].dropna().mean() for c in item_cols]

                    x = np.arange(len(item_cols))
                    w = 0.35

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=x + w / 2,
                        y=top_means,
                        name=f"상위 25% (n={n_top})",
                        marker_color="#2ecc71",
                        text=[f"{v:.2f}" for v in top_means],
                        textposition="outside",
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=x - w / 2,
                        y=bot_means,
                        name=f"하위 25% (n={n_bot})",
                        marker_color="#e74c3c",
                        text=[f"{v:.2f}" for v in bot_means],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        xaxis=dict(
                            tickvals=x.tolist(),
                            ticktext=[
                                c.replace(" ", "<br>") for c in item_cols
                            ],
                            tickfont=dict(family=PLOT_FONT, size=11),
                        ),
                        barmode="group",
                        font=dict(family=PLOT_FONT),
                        yaxis_title="평균값",
                        legend=dict(font=dict(family=PLOT_FONT)),
                        margin=dict(l=50, r=20, t=40, b=80),
                        height=400,
                    )
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("표시할 측정 항목이 없습니다.")
            else:
                st.info(
                    "생중량 데이터가 충분하지 않아 "
                    "(상위·하위 25% 구분 불가) 그룹 비교를 건너뜁니다."
                )

            st.divider()

            # ── 생중량 분포 (박스플롯) ──
            st.subheader("📦 생중량 분포 (박스플롯)")
            st.caption(
                "상자(box) 안 선은 중앙값, 상자 위아래 경계는 25·75 percentile, "
                "수염(whisker)은 이상치를 제외한 범위, 점은 이상치입니다."
            )
            if has_bw:
                bw_col = growth_df["생중량"].dropna()
                if len(bw_col) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=bw_col.values,
                        name="생중량",
                        fillcolor="#3498db",
                        line_color="#2980b9",
                        boxpoints="outliers",
                        hovertemplate="생중량: %{y:.2f}g<extra></extra>",
                    ))
                    fig_box.update_layout(
                        yaxis_title="생중량 (g)",
                        font=dict(family=PLOT_FONT),
                        margin=dict(l=40, r=20, t=30, b=40),
                        height=300,
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info(
                        "생중량 값이 2개 미만이라 박스플롯을 그릴 수 없습니다."
                    )
            else:
                st.info("생중량 열이 없어 박스플롯을 표시할 수 없습니다.")

            st.divider()

            # ── 상관관계 분석 ──
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
            st.caption(
                "추세선은 np.polyfit(최소제곱법)으로 구한 직선입니다. "
                "상관계수 r은 -1 ~ +1이며 |r|이 클수록 두 변수가 강하게 연관됩니다. "
                "인과관계가 아니라 '함께 변하는 정도'만 나타냅니다."
            )

            if has_bw:
                if "잎 수" in growth_df.columns:
                    xy = growth_df[["잎 수", "생중량"]].dropna()
                    if len(xy) >= 2:
                        x = xy["잎 수"].values
                        y = xy["생중량"].values
                        r = np.corrcoef(x, y)[0, 1]
                        slope, intercept = np.polyfit(x, y, 1)
                        xs = np.linspace(x.min(), x.max(), 100)
                        ys = slope * xs + intercept

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(
                            x=x, y=y,
                            mode="markers",
                            name="개체",
                            marker=dict(
                                color="#3498db", size=8, opacity=0.7
                            ),
                            hovertemplate=(
                                "잎 수: %{x}<br>생중량: %{y:.2f}g<extra></extra>"
                            ),
                        ))
                        fig_a.add_trace(go.Scatter(
                            x=xs, y=ys,
                            mode="lines",
                            name=f"추세선 (r={r:.2f})",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                        ))
                        fig_a.update_layout(
                            xaxis_title="잎 수 (장)",
                            yaxis_title="생중량 (g)",
                            font=dict(family=PLOT_FONT),
                            margin=dict(l=40, r=20, t=30, b=40),
                            legend=dict(font=dict(family=PLOT_FONT)),
                            height=380,
                        )
                        st.plotly_chart(fig_a, width="stretch")
                        st.caption(
                            f"잎 수 ↔ 생중량 상관계수 r = {r:.3f}  "
                            f"(|r| ≥ 0.7: 강한 관계, 0.3~0.7: 중간, < 0.3: 약함)"
                        )
                    else:
                        st.info(
                            "잎 수와 생중량 모두 값이 있는 개체가 "
                            "2마리 미만이라 산점도를 그릴 수 없습니다."
                        )
                else:
                    st.info(
                        "잎 수(잎 수) 열이 없어 잎 수 vs 생중량 산점도를 "
                        "표시하지 않습니다."
                    )

                st.divider()

                if "지상부 길이" in growth_df.columns:
                    xy2 = growth_df[["지상부 길이", "생중량"]].dropna()
                    if len(xy2) >= 2:
                        x2 = xy2["지상부 길이"].values
                        y2 = xy2["생중량"].values
                        r2 = np.corrcoef(x2, y2)[0, 1]
                        slope2, intercept2 = np.polyfit(x2, y2, 1)
                        xs2 = np.linspace(x2.min(), x2.max(), 100)
                        ys2 = slope2 * xs2 + intercept2

                        fig_b = go.Figure()
                        fig_b.add_trace(go.Scatter(
                            x=x2, y=y2,
                            mode="markers",
                            name="개체",
                            marker=dict(
                                color="#9b59b6", size=8, opacity=0.7
                            ),
                            hovertemplate=(
                                "지상부 길이: %{x}mm<br>"
                                "생중량: %{y:.2f}g<extra></extra>"
                            ),
                        ))
                        fig_b.add_trace(go.Scatter(
                            x=xs2, y=ys2,
                            mode="lines",
                            name=f"추세선 (r={r2:.2f})",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                        ))
                        fig_b.update_layout(
                            xaxis_title="지상부 길이 (mm)",
                            yaxis_title="생중량 (g)",
                            font=dict(family=PLOT_FONT),
                            margin=dict(l=40, r=20, t=30, b=40),
                            legend=dict(font=dict(family=PLOT_FONT)),
                            height=380,
                        )
                        st.plotly_chart(fig_b, width="stretch")
                        st.caption(
                            f"지상부 길이 ↔ 생중량 상관계수 r = {r2:.3f}  "
                            f"(|r| ≥ 0.7: 강한 관계, 0.3~0.7: 중간, < 0.3: 약함)"
                        )
                    else:
                        st.info(
                            "지상부 길이와 생중량 모두 값이 있는 개체가 "
                            "2마리 미만이라 산점도를 그릴 수 없습니다."
                        )
                else:
                    st.info(
                        "지상부 길이(지상부 길이) 열이 없어 "
                        "지상부 길이 vs 생중량 산점도를 표시하지 않습니다."
                    )
            else:
                st.info("생중량 데이터가 없어 상관관계 분석을 표시할 수 없습니다.")

            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=True)
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name="생육결과_download.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("⚙️ 데이터 업로드")
st.sidebar.caption(
    "CSV 파일을 업로드하세요. (인코딩: utf-8-sig / cp949 자동 처리)"
)

env_file = st.sidebar.file_uploader("🌿 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌱 생육 결과 CSV", type="csv")

school_label = school_name_from_fileobj(
    env_file if env_file is not None else growth_file
)
st.sidebar.markdown(f"🏫 **{school_label}**")
st.sidebar.divider()
st.sidebar.caption("※ 업로드 전에는 탭 내용이 비어 있습니다.")
st.sidebar.caption("※ 한 파일만 올려도 해당 탭은 동작합니다.")


# ============================================================
# 메인
# ============================================================
st.set_page_config(
    page_title="극지식물 생육 분석 대시보드", layout="wide"
)
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption(f"데이터 출처: {school_label} 극지식물 실험 데이터 | 작성일: 2026")

with st.spinner("데이터를 읽고 표준화하는 중..."):
    env_df = read_csv_file(env_file)
    growth_df = read_csv_file(growth_file)
    env_df = std_cols(env_df) if not env_df.empty else env_df
    growth_df = std_cols(growth_df) if not growth_df.empty else growth_df

st.success("✅ 데이터 로딩 완료")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
])

with tab1:
    render_tab1(env_df, growth_df)

with tab2:
    render_tab2(env_df)

with tab3:
    render_tab3(growth_df)
