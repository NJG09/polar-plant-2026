import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
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
# 일별 시계열 꺾은선
# ============================================================
def daily_line(df, col, color, title):
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].dropna()
    if len(d) == 0:
        return None
    d = d.copy()
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
        height=260,
    )
    return fig


# ============================================================
# 산점도 + np.polyfit 추세선 + r값
#   hovertemplate은 f-string 대신 .replace()로 조립 (SyntaxError 방지)
# ============================================================
def scatter_with_trend(df, x_col, y_col, color, x_label, y_label, title):
    if x_col not in df.columns or y_col not in df.columns:
        return None, None
    d = df[[x_col, y_col]].dropna()
    if len(d) < 2:
        return None, None
    x = d[x_col].values
    y = d[y_col].values
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept

    # hovertemplate 템플릿: %{{x}}, %{{y:.2f}} 가 최종 결과물
    ht = (
        ": %{{x}}<br>"
        + y_label
        + ": %{{y:.2f}} g<extra></extra>"
    )
    ht = ht.replace(": %{{x}}", x_label + ": %{{x}}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="개체",
        marker=dict(color=color, size=7, opacity=0.7),
        hovertemplate=ht,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name="추세선 (r={:.2f})".format(r),
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family=PLOT_FONT),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(font=dict(family=PLOT_FONT)),
        height=340,
    )
    return fig, r


# ============================================================
# 학교명 추출
# ============================================================
def _get_filename(obj):
    if obj is None:
        return None
    return getattr(obj, "name", str(obj))


def school_name_from_fileobj(fileobj):
    fn = _get_filename(fileobj)
    if fn is None:
        return "우리 학교"
    base = os.path.splitext(fn)[0]
    name = base.replace("_", " ").replace("(", "").replace(")", "")
    return name if name else "우리 학교"


# ============================================================
# Tab 1 — 실험 소개
# ============================================================
def render_tab1(env_df, growth_df):
    with st.container():
        try:
            st.subheader("🔬 실험 목적")
            st.markdown("""
**극지식물(Extremophile plant)**은 온도·습도·pH·EC 등
비우호적인 환경에서도 살아남는 식물입니다.

이 실험에서는 우리 학교 극지식물 개체를 재배하면서
**환경 데이터**(온도·습도·pH·EC)와 **생육 결과 데이터**(개체별 크기·무게)를
함께 수집하여, 개체마다 생육에 어떤 차이가 있는지 분석합니다.

> **통계 용어 간단 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 값 — 대표값
> - **표준편차**: 값이 평균 주위에 얼마나 흩어져 있는지 — 클수록 편차 큼
> - **상관계수(r)**: 두 변수가 함께 움직이는 정도 — -1 ~ +1,
>   |r|이 0.7 이상이면 강한 관계, 0.3~0.7은 중간, 0.3 미만은 약한 관계로 봅니다.
>   인과관계가 아니라 '함께 변하는 정도'만 나타냅니다.
            """)

            st.divider()
            st.subheader("📊 핵심 숫자")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if not growth_df.empty:
                    n = len(growth_df)
                    st.metric("총 개체 수", str(n) + "개")
                else:
                    st.metric("총 개체 수", "― (CSV 업로드 필요)")

            with c2:
                if not env_df.empty and "temperature" in env_df.columns:
                    avg_t = env_df["temperature"].dropna().mean()
                    if not np.isnan(avg_t):
                        st.metric("평균 온도", "{:.1f}℃".format(avg_t))
                    else:
                        st.metric("평균 온도", "―")
                else:
                    st.metric("평균 온도", "― (환경 CSV 업로드 필요)")

            with c3:
                if not env_df.empty and "humidity" in env_df.columns:
                    avg_h = env_df["humidity"].dropna().mean()
                    if not np.isnan(avg_h):
                        st.metric("평균 습도", "{:.1f}%".format(avg_h))
                    else:
                        st.metric("평균 습도", "―")
                else:
                    st.metric("평균 습도", "― (환경 CSV 업로드 필요)")

            with c4:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    bw = growth_df["생중량"].dropna()
                    if len(bw) > 0 and not np.isnan(bw.mean()):
                        st.metric("평균 생중량", "{:.2f}g".format(bw.mean()))
                    else:
                        st.metric("평균 생중량", "―")
                else:
                    st.metric("평균 생중량", "― (생육 CSV 업로드 필요)")

            st.divider()

            st.subheader("🧪 데이터 요약")
            rows = []
            if not growth_df.empty:
                g_cols = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                "지상부 질량", "지하부 질량", "생중량"]
                    if c in growth_df.columns
                ]
                rows.append({
                    "데이터": "생육 결과",
                    "개체 수": str(len(growth_df)),
                    "측정 항목": ", ".join(g_cols) if g_cols else "(없음)",
                })
            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    period = "{} ~ {}".format(
                        t.min().strftime('%Y-%m-%d'),
                        t.max().strftime('%Y-%m-%d'),
                    )
                else:
                    period = "알 수 없음"
                e_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                          if c in env_df.columns]
                rows.append({
                    "데이터": "환경 데이터",
                    "개체 수": "―",
                    "측정 기간": period,
                    "측정 항목": ", ".join(e_cols) if e_cols else "(없음)",
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows), width="stretch", hide_index=True
                )
            else:
                st.info("📁 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

            st.divider()
            st.info(
                "💡 이 대시보드는 우리 학교 극지식물 개체끼리의 차이만 비교합니다. "
                "개체번호는 그래프 라벨로만 쓰이고 분석 계산에는 포함되지 않습니다."
            )

        except Exception as e:
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))


# ============================================================
# Tab 2 — 환경 데이터
# ============================================================
def render_tab2(env_df):
    with st.container():
        try:
            if env_df.empty:
                st.info(
                    "📁 환경 CSV를 사이드바에서 업로드하면 "
                    "환경 데이터 탭이 활성화됩니다."
                )
                return

            if "time" not in env_df.columns:
                st.info("환경 데이터에 시간 열이 없어 이 탭을 표시할 수 없습니다.")
                return

            st.subheader("🌡️ 환경 데이터 요약")
            st.caption(
                "측정 기간 동안 온도·습도·pH·EC의 평균·최소·최대값입니다."
            )

            cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns]
            if cols:
                rows = []
                for c in cols:
                    s = env_df[c].dropna()
                    if len(s) == 0:
                        continue
                    rows.append({
                        "측정 항목": c,
                        "평균": round(s.mean(), 2),
                        "최소": round(s.min(), 2),
                        "최대": round(s.max(), 2),
                    })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        use_container_width=True,
                    )
                else:
                    st.info("표시할 환경 측정 값이 없습니다.")
            else:
                st.info("온도·습도·pH·EC 중 어떤 측정 항목 열도 없습니다.")

            st.divider()
            st.subheader("📈 날짜별 환경 변화 (일별 평균)")
            st.caption("매일 측정한 값을 일별로 평균내어 꺾은선으로 보여줍니다.")

            fig_t = daily_line(
                env_df, "temperature", "#e74c3c", "일별 평균 온도 변화 (℃)"
            )
            if fig_t is not None:
                st.plotly_chart(fig_t, width="stretch")
            else:
                st.info("온도(temperature) 열이 없거나 데이터 포인트가 부족해 "
                        "온도 그래프를 표시할 수 없습니다.")

            fig_h = daily_line(
                env_df, "humidity", "#3498db", "일별 평균 습도 변화 (%)"
            )
            if fig_h is not None:
                st.plotly_chart(fig_h, width="stretch")
            else:
                st.info("습도(humidity) 열이 없거나 데이터 포인트가 부족해 "
                        "습도 그래프를 표시할 수 없습니다.")

            if "ph" in env_df.columns:
                fig_ph = daily_line(
                    env_df, "ph", "#9b59b6", "일별 평균 pH 변화"
                )
                if fig_ph is not None:
                    st.plotly_chart(fig_ph, width="stretch")
                else:
                    st.info("pH 데이터가 부족해 pH 그래프를 건너뜁니다.")
            if "ec" in env_df.columns:
                fig_ec = daily_line(
                    env_df, "ec", "#1abc9c", "일별 평균 EC 변화"
                )
                if fig_ec is not None:
                    st.plotly_chart(fig_ec, width="stretch")
                else:
                    st.info("EC 데이터가 부족해 EC 그래프를 건너뜁니다.")

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
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))


# ============================================================
# Tab 3 — 개체별 생장
# ============================================================
def render_tab3(growth_df):
    with st.container():
        try:
            if growth_df.empty:
                st.info(
                    "📁 생육 결과 CSV를 사이드바에서 업로드하면 "
                    "개체별 생장 탭이 활성화됩니다."
                )
                return

            has_bw = "생중량" in growth_df.columns

            if not has_bw:
                st.warning(
                    "⚠️ 이 학교의 생육 데이터에는 '생중량(g)' 열이 없습니다. "
                    "지상부 질량 + 지하부 질량으로 생중량을 계산해 분석합니다."
                )

            id_col = "개체번호" if "개체번호" in growth_df.columns else None

            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")
            st.caption("개체별 생중량(생체 무게) 기준 주요 지표입니다.")
            if has_bw:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 1:
                    mean_bw = bw.mean()
                    max_idx = bw.idxmax()
                    min_idx = bw.idxmin()
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("평균 생중량", "{:.2f}g".format(mean_bw))
                    with col_b:
                        id_max = (
                            growth_df.loc[max_idx, id_col]
                            if id_col is not None else "개체 #" + str(max_idx)
                        )
                        val_max = growth_df.loc[max_idx, "생중량"]
                        st.metric(
                            "📈 생중량 최대 개체",
                            "{:.2f}g".format(val_max),
                            "개체: " + str(id_max),
                        )
                    with col_c:
                        id_min = (
                            growth_df.loc[min_idx, id_col]
                            if id_col is not None else "개체 #" + str(min_idx)
                        )
                        val_min = growth_df.loc[min_idx, "생중량"]
                        st.metric(
                            "📉 생중량 최소 개체",
                            "{:.2f}g".format(val_min),
                            "개체: " + str(id_min),
                        )
                else:
                    st.info("생중량 값이 하나도 없어 지표를 계산할 수 없습니다.")
            else:
                for c in st.columns(3):
                    c.metric("―", "생중량 데이터 없음")

            st.divider()

            # ── 개체별 생중량 막대 그래프 ──
            st.subheader("📊 개체별 생중량 분포 (막대 그래프)")
            st.caption(
                "각 개체의 생중량을 막대로 보여줍니다. "
                "초록색은 상위 25%, 빨간색은 하위 25% 개체입니다."
            )
            if has_bw:
                bw_vals = growth_df["생중량"].dropna()
                if len(bw_vals) >= 4:
                    q75 = bw_vals.quantile(0.75)
                    q25 = bw_vals.quantile(0.25)
                    top_mask = growth_df["생중량"] >= q75
                    bot_mask = growth_df["생중량"] <= q25

                    plot_df = growth_df.copy()
                    plot_df["_color"] = "보통"
                    plot_df.loc[top_mask, "_color"] = "상위 25%"
                    plot_df.loc[bot_mask, "_color"] = "하위 25%"
                    plot_df["_y"] = pd.to_numeric(
                        plot_df["생중량"], errors="coerce"
                    )
                    plot_df = plot_df.dropna(subset=["_y"]).sort_values("_y")

                    if id_col is not None:
                        plot_df["label"] = plot_df[id_col].astype(str)
                    else:
                        plot_df["label"] = plot_df.index.astype(str)

                    n = len(plot_df)
                    if n <= 20:
                        fig_bar = go.Figure()
                        for i, (_, row) in enumerate(plot_df.iterrows()):
                            c = "#2ecc71" if row["_color"] == "상위 25%" else (
                                "#e74c3c" if row["_color"] == "하위 25%" else "#95a5a6"
                            )
                            show_legend = (
                                i == 0 and row["_color"] != "보통"
                            )
                            fig_bar.add_trace(go.Bar(
                                x=[row["label"]],
                                y=[row["_y"]],
                                name=row["_color"],
                                marker_color=c,
                                showlegend=show_legend,
                            ))
                        fig_bar.update_layout(
                            barmode="group",
                            xaxis_title="개체",
                            yaxis_title="생중량 (g)",
                            font=dict(family=PLOT_FONT),
                            margin=dict(l=40, r=20, t=30, b=60),
                            height=360,
                        )
                        st.plotly_chart(fig_bar, width="stretch")
                    else:
                        fig_bar = go.Figure()
                        colors = [
                            "#2ecc71" if r == "상위 25%" else
                            "#e74c3c" if r == "하위 25%" else "#95a5a6"
                            for r in plot_df["_color"]
                        ]
                        fig_bar.add_trace(go.Bar(
                            x=plot_df["label"].values,
                            y=plot_df["_y"].values,
                            marker_color=colors,
                            hovertemplate=(
                                "개체: %{x}<br>생중량: %{y:.2f} g<extra></extra>"
                            ),
                        ))
                        fig_bar.update_layout(
                            xaxis_title="개체",
                            yaxis_title="생중량 (g)",
                            font=dict(family=PLOT_FONT),
                            margin=dict(l=40, r=20, t=30, b=60),
                            height=360,
                        )
                        st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("생중량 데이터가 4개 미만이라 상위·하위 25% 구분이 "
                            "불가능합니다.")
            else:
                st.info("생중량 열이 없어 막대 그래프를 표시할 수 없습니다.")

            st.divider()

            # ── [핵심분석 1] 잎 수 vs 생중량 ──
            st.subheader("[핵심분석 1] 잎 수 vs 생중량")
            st.caption(
                "잎 수가 많을수록 생중량이 무거운지 확인하는 산점도입니다. "
                "추세선은 np.polyfit(최소제곱법)으로 구한 직선이고, "
                "r값이 1에 가까울수록 더 강한 관계입니다."
            )
            fig_leaf, r_leaf = scatter_with_trend(
                growth_df, "잎 수", "생중량",
                "#3498db", "잎 수 (장)", "생중량", "잎 수 ↔ 생중량"
            )
            if fig_leaf is not None:
                st.plotly_chart(fig_leaf, width="stretch")
                st.caption("🌿 잎 수 ↔ 생중량 상관계수 r = {:.3f}".format(r_leaf))
            else:
                st.info("잎 수 또는 생중량 데이터가 2개 미만이라 산점도를 "
                        "그릴 수 없습니다.")

            # ── [핵심분석 1] 지상부 길이 vs 생중량 ──
            st.divider()
            st.subheader("[핵심분석 1] 지상부 길이 vs 생중량")
            st.caption(
                "지상부(잎+줄기) 길이가 길수록 생중량이 무거운지 확인하는 산점도입니다."
            )
            fig_shoot, r_shoot = scatter_with_trend(
                growth_df, "지상부 길이", "생중량",
                "#9b59b6", "지상부 길이 (mm)", "생중량", "지상부 길이 ↔ 생중량"
            )
            if fig_shoot is not None:
                st.plotly_chart(fig_shoot, width="stretch")
                st.caption("🌱 지상부 길이 ↔ 생중량 상관계수 r = {:.3f}".format(r_shoot))
            else:
                st.info("지상부 길이 또는 생중량 데이터가 2개 미만이라 "
                        "산점도를 그릴 수 없습니다.")

            # ── r값 비교 ──
            if fig_leaf is not None and fig_shoot is not None:
                st.divider()
                bigger = "잎 수" if abs(r_leaf) >= abs(r_shoot) else "지상부 길이"
                st.info(
                    "🔍 r값 비교: 잎 수 r = {:.3f} vs 지상부 길이 r = {:.3f} → "
                    "|r|이 더 큰 {}이(가) 생중량과 더 밀접합니다.".format(
                        r_leaf, r_shoot, bigger
                    )
                )

            # ── [핵심분석 3] 지하부 길이 vs 지상부 길이 ──
            st.divider()
            st.subheader("[핵심분석 3] 지하부 길이 vs 지상부 길이")
            st.caption(
                "지하부(뿌리) 길이와 지상부(잎+줄기) 길이가 함께 자라는지 "
                "확인하는 산점도입니다. r이 양수면 둘 다 커지는 방향이 같습니다."
            )
            fig_root_shoot, r_root_shoot = scatter_with_trend(
                growth_df, "지하부 길이", "지상부 길이",
                "#16a085", "지하부 길이 (mm)", "지상부 길이 (mm)",
                "지하부 길이 ↔ 지상부 길이"
            )
            if fig_root_shoot is not None:
                st.plotly_chart(fig_root_shoot, width="stretch")
                st.caption(
                    "🌿 지하부 길이 ↔ 지상부 길이 상관계수 r = {:.3f}".format(
                        r_root_shoot
                    )
                )
                if r_root_shoot > 0:
                    st.info(
                        "r = {:.3f} (양수) → 지하부가 길수록 "
                        "지상부도 길어지는 경향이 있습니다.".format(r_root_shoot)
                    )
                else:
                    st.info(
                        "r = {:.3f} (음수) → 지하부와 지상부 길이 사이에 "
                        "뚜렷한 동행 관계는 보이지 않습니다.".format(r_root_shoot)
                    )
            else:
                st.info("지하부 길이 또는 지상부 길이 데이터가 2개 미만이라 "
                        "산점도를 그릴 수 없습니다.")

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
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))


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
st.sidebar.markdown("🏫 **" + school_label + "**")
st.sidebar.divider()
st.sidebar.caption("※ 업로드 전에는 탭 내용이 비어 있습니다.")
st.sidebar.caption("※ 한 파일만 올려도 해당 탭은 동작합니다.")


# ============================================================
# 메인
# ============================================================
st.set_page_config(
    page_title="극지식물 생육 분석 대시보드", layout="wide"
)
st.title("송천고에서는 극지식물이 어떻게 자랄까????")
st.caption("데이터 출처: " + school_label + " 극지식물 실험 데이터 | 작성일: 2026")

with st.spinner("데이터를 읽고 표준화하는 중..."):
    env_df = read_csv_file(env_file)
    growth_df = read_csv_file(growth_file)
    env_df = std_cols(env_df) if not env_df.empty else env_df
    growth_df = std_cols(growth_df) if not growth_df.empty else growth_df

st.success("✅ 데이터 로딩 완료")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🧪 실험 소개",
    "🌡️ 환경 데이터",
    "📊 개체별 생장",
])

with tab1:
    render_tab1(env_df, growth_df)

with tab2:
    render_tab2(env_df)

with tab3:
    render_tab3(growth_df)
