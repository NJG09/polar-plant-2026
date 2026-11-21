# main.py
import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="극지 식물 생장 탐구", page_icon="🌱", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, h1, h2, h3 {
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""", unsafe_allow_html=True)

ID = "개체번호"
MASS = "생중량"
LEAF = "잎 수"
SHOOT = "지상부 길이"
ROOT = "지하부 길이"

GROW = {
    MASS: "생중량 (g)", LEAF: "잎 수 (장)", SHOOT: "지상부 길이 (mm)",
    ROOT: "지하부 길이 (mm)", "지상부 질량": "지상부 질량 (g)",
    "지하부 질량": "지하부 질량 (g)",
}
ENV = {
    "temperature": "온도 (°C)", "humidity": "습도 (%)", "ec": "EC",
    "ph": "pH", "water_temp": "수온 (°C)", "co2": "CO₂",
}
COLORS = {"상위": "#147451", "하위": "#bd6236", "중간": "#9abeb0", "미측정": "#999999"}


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
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def clean_time(s):
    # 잘못된 날짜나 24 이상의 시각은 해당 셀만 비웁니다.
    parts = s.str.extract(r"^(\d{8})_(\d{1,2})$")
    hours = pd.to_numeric(parts[1], errors="coerce")
    dates = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    stamp = dates + pd.to_timedelta(hours.where(hours.between(0, 23)), unit="h")
    return s.where(parts[0].isna(), stamp.dt.strftime("%Y-%m-%d %H:%M:%S"))


def standardize(raw):
    # 원본 표를 보존하고, 같은 뜻의 열은 왼쪽 열의 빈칸만 보완합니다.
    columns = {}
    for header in raw.columns:
        name = re.sub(r"\.\d+$", "", header.strip())
        key = std_cols(pd.DataFrame(columns=[name])).columns[0]
        if key not in {*ENV, *GROW, ID, "time"}:
            continue
        s = raw[header].astype("string").str.strip().replace("", pd.NA)
        if key in ENV or key in GROW:
            number = s.str.replace(",", "", regex=False).str.extract(
                r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:[a-zA-Z°℃%장]+)?$",
                expand=False,
            )
            s = pd.to_numeric(number, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        elif key == "time":
            s = clean_time(s)
        columns[key] = columns[key].combine_first(s) if key in columns else s
    df = std_cols(pd.DataFrame(columns, index=raw.index))
    return df, MASS in df and MASS not in columns


@st.cache_data(show_spinner=False, max_entries=16)
def read_csv(data):
    empty = pd.DataFrame()
    if data is None:
        return empty, empty, False, ""
    try:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp949")
        first = next((line for line in text.splitlines() if line.strip()), "")
        if not first:
            return empty, empty, False, "파일이 비어 있어요."
        # 실제 환경 CSV는 탭 구분, 생육 CSV는 쉼표 구분입니다.
        try:
            sep = csv.Sniffer().sniff(first, delimiters=",\t;").delimiter
        except csv.Error:
            sep = ","
        raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        df, derived = standardize(raw)
        return raw, df, derived, ""
    except Exception:
        return empty, empty, False, "CSV의 인코딩, 첫 행의 열 이름과 구분 기호를 확인해 주세요."


def numeric_columns(df, fields):
    return [c for c in fields if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def values(df, key):
    if key in numeric_columns(df, [key]):
        return df[key].dropna()
    return pd.Series(dtype=float)


def individual_means(df):
    cols = numeric_columns(df, GROW)
    if not cols:
        return pd.DataFrame()
    ids = df.get(ID, pd.Series(pd.NA, index=df.index, dtype="string"))
    ids = ids.astype("string").str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    rows = df.loc[ids.notna() | df[cols].notna().any(axis=1), cols].copy()
    if rows.empty:
        return pd.DataFrame()
    # 같은 번호의 반복 측정은 한 개체로, 번호가 없는 행은 각각 한 개체로 처리합니다.
    rows["_key"] = [(0, ids.at[i]) if pd.notna(ids.at[i]) else (1, i) for i in rows.index]
    result = rows.groupby("_key", sort=False)[cols].mean().reset_index()
    result[ID] = [str(v) if kind == 0 else f"미기재 · 자료 {v + 1}행" for kind, v in result.pop("_key")]
    return result


def plot(fig, key):
    fig.update_layout(
        font=dict(family="Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=12, r=12, t=48, b=32),
    )
    st.plotly_chart(fig, key=key, width="stretch")


def table(df):
    st.dataframe(df, width="stretch", hide_index=True)


def original(df, upload, name):
    if len(df.columns):
        with st.expander(f"{name} 원본 표 · CSV 다운로드"):
            table(df)
            st.download_button(
                "CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{Path(upload.name).stem}.csv",
                mime="text/csv", key=f"download_{name}",
            )


def mean_card(col, df, key, title, unit):
    s = values(df, key)
    if len(s) >= 2:
        col.metric(title, f"{s.mean():,.2f} {unit}")
    else:
        col.info(f"{title}을 표시할 측정값이 2개 이상 필요해요.")


def daily_plot(df, key):
    title = ENV[key]
    if "time" not in df or key not in numeric_columns(df, ENV):
        st.info(f"일별 {title} 그래프에는 시간 열과 해당 센서 열이 필요해요.")
        return
    pair = df[["time", key]].dropna()
    if len(pair) < 2:
        st.info(f"시간과 {title} 값이 함께 기록된 행이 2개 이상 필요해요.")
        return
    daily = pair.set_index("time")[key].sort_index().resample("D").mean()
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily, mode="lines+markers", connectgaps=False,
        line=dict(color="#287e69"), name=title,
    ))
    fig.update_layout(title=f"일별 평균 {title}", xaxis_title="측정일", yaxis_title=title)
    plot(fig, f"daily_{key}")


def scatter(df, xkey, ykey):
    if len(numeric_columns(df, [xkey, ykey])) != 2:
        st.info(f"{xkey}와 {ykey} 열이 모두 있어야 산점도를 그릴 수 있어요.")
        return
    pair = df[[xkey, ykey]].dropna()
    if len(pair) < 2:
        st.info(f"{xkey}와 {ykey}를 함께 측정한 개체가 2개 이상 필요해요.")
        return
    x, y = pair[xkey], pair[ykey]
    fig = go.Figure(go.Scatter(
        x=x, y=y, text=df.loc[pair.index, ID], mode="markers", name="개체",
        marker=dict(size=9, color="#287e69", opacity=0.75),
        hovertemplate="개체 %{text}<br>" + GROW[xkey] + " %{x}<br>" + GROW[ykey] + " %{y}<extra></extra>",
    ))
    if x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        ends = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=ends, y=slope * ends + intercept, mode="lines", name="추세선"))
    if x.nunique() > 1 and y.nunique() > 1:
        st.write(f"**{xkey}와 {ykey}** · 상관계수 {x.corr(y):.3f} · {len(pair)}개체")
    else:
        st.info(f"{xkey} 또는 {ykey}의 값이 모두 같아 상관계수를 계산할 수 없어요.")
    fig.update_layout(xaxis_title=GROW[xkey], yaxis_title=GROW[ykey])
    plot(fig, f"scatter_{xkey}_{ykey}")


def compare_correlations(df):
    cols = numeric_columns(df, [LEAF, SHOOT, MASS])
    if len(cols) != 3:
        st.info("두 관계를 비교하려면 잎 수, 지상부 길이, 생중량이 모두 필요해요.")
        return
    common = df[cols].dropna()
    if len(common) < 2 or common.nunique().min() < 2:
        st.info("같은 개체에서 세 항목이 모두 측정되고 각 항목에 서로 다른 값이 있어야 비교할 수 있어요.")
        return
    leaf_r = common[LEAF].corr(common[MASS])
    shoot_r = common[SHOOT].corr(common[MASS])
    st.write(f"세 항목을 모두 측정한 **{len(common)}개체**에서 생중량과의 상관계수는 "
             f"잎 수 **{leaf_r:.3f}**, 지상부 길이 **{shoot_r:.3f}**예요.")
    if np.isclose(abs(leaf_r), abs(shoot_r), atol=0.0005, rtol=0):
        st.write("두 항목은 생중량과 거의 같은 강도의 직선 관계를 보여요.")
    else:
        stronger = LEAF if abs(leaf_r) > abs(shoot_r) else SHOOT
        st.write(f"이 표본에서는 **{stronger}** 쪽의 상관계수 절댓값이 더 커요.")


def weight_groups(df):
    weight = values(df, MASS).sort_values(kind="stable")
    group = pd.Series("미측정", index=df.index)
    group.loc[weight.index] = "중간"
    size = len(weight) // 4
    if size:
        group.loc[weight.index[:size]] = "하위"
        group.loc[weight.index[-size:]] = "상위"
    return group, size


def bar_plot(df, key, group):
    s = values(df, key).sort_values(ascending=False, kind="stable")
    if s.empty:
        st.info(f"{key} 막대그래프에 사용할 측정값이 없어요.")
        return
    fig = go.Figure(go.Bar(
        x=df.loc[s.index, ID], y=s, marker_color=group.loc[s.index].map(COLORS),
        customdata=group.loc[s.index],
        hovertemplate="개체 %{x}<br>" + GROW[key] + " %{y}<br>생중량 그룹 %{customdata}<extra></extra>",
    ))
    fig.update_xaxes(type="category", title_text=ID)
    fig.update_layout(title=f"개체별 {GROW[key]}", yaxis_title=GROW[key])
    plot(fig, f"bar_{key}")


def compare_groups(df, group, size):
    if not size:
        st.info("상위·하위 그룹을 비교하려면 생중량이 있는 개체가 4개 이상 필요해요.")
        return
    st.caption(f"생중량을 기준으로 양 끝에서 {size}개체씩 골랐어요. "
               "각 그룹 크기는 전체 유효 개체 수의 25%를 내림한 값이에요. "
               "경계에서 무게가 같으면 원본 순서로 순위를 나눠요.")
    rows = []
    for key in numeric_columns(df, GROW):
        high = df.loc[group.eq("상위"), key].dropna()
        low = df.loc[group.eq("하위"), key].dropna()
        if high.empty and low.empty:
            continue
        h = high.mean() if len(high) else np.nan
        l = low.mean() if len(low) else np.nan
        rows.append({
            "측정 항목": GROW[key], "상위 측정 수": len(high), "상위 평균": h,
            "하위 측정 수": len(low), "하위 평균": l,
            "하위 평균 대비 차이 (%)": 100 * (h - l) / l if pd.notna(l) and l != 0 else np.nan,
        })
    if rows:
        table(pd.DataFrame(rows).round(2))
        st.caption("차이가 +20%라면 상위 그룹의 평균이 하위 그룹보다 20% 커요. "
                   "하위 평균이 0이거나 한쪽 값이 없으면 백분율을 비워 둬요.")


env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")
with st.spinner("CSV의 열 이름과 측정값을 확인하고 있어요."):
    env_raw, env_df, _, env_note = read_csv(env_file.getvalue() if env_file is not None else None)
    growth_raw, growth_df, derived, growth_note = read_csv(
        growth_file.getvalue() if growth_file is not None else None
    )
    plants = individual_means(growth_df)

options = numeric_columns(plants, GROW)
selected = st.sidebar.multiselect(
    "생육 측정 항목", options, default=[k for k in (MASS, LEAF, SHOOT, ROOT) if k in options],
    format_func=GROW.get, help="개체별 막대그래프로 볼 항목을 선택하세요.",
)

schools = []
for upload in (growth_file, env_file):
    if upload is not None:
        school = re.split(r"[_\s-]+", Path(upload.name).stem.strip(), maxsplit=1)[0]
        if school and school not in schools:
            schools.append(school)
# 파일명에 붙은 지역명은 학교명과 함께 보존합니다.
school = schools[0] if len(schools) == 1 else "우리 학교"
st.title(f"{school}에서는 극지 식물이 어떻게 자랄까? 잘 안 자란다.")
if len(schools) > 1:
    st.info("두 파일의 학교 표기가 달라요. 같은 학교의 자료인지 확인해 주세요.")
if env_file is None and growth_file is None:
    st.info("왼쪽에서 환경 CSV 또는 생육 CSV를 올려 주세요. 한 파일만 있어도 해당 탭을 볼 수 있어요.")

tab1, tab2, tab3 = st.tabs(["📖 실험 소개", "🌡️ 환경 데이터", "🌱 개체별 생장"])

with tab1:
    try:
        st.subheader("같은 시설에서 자란 식물은 어떻게 다를까요?")
        st.write("재배 기간의 온도와 습도를 확인하고, 우리 학교에서 기른 개체끼리 비교해요. "
                 "잎 수와 지상부 길이 중 무게와 더 밀접한 항목, 무거운 개체와 가벼운 개체의 차이, "
                 "지하부 길이와 지상부 길이의 관계를 살펴봐요.")
        st.caption("‘잘 안 자란다’는 탐구의 출발점이에요. 실제로 생장이 느렸는지 판단하려면 "
                   "시작 시점의 크기나 같은 기간에 기른 비교 집단의 기록이 필요해요.")
        a, b, c, d = st.columns(4)
        if len(plants):
            a.metric("측정 개체 수", f"{len(plants)}개")
        else:
            a.info("개체 수를 표시할 생육 자료가 없어요.")
        mean_card(b, env_df, "temperature", "평균 온도", "°C")
        mean_card(c, env_df, "humidity", "평균 습도", "%")
        mean_card(d, plants, MASS, "평균 생중량", "g")
        st.caption("평균은 유효한 값의 합을 그 개수로 나눈 값이에요. 같은 번호를 여러 번 측정했다면 "
                   "먼저 개체별 평균을 구해서 각 개체가 한 번씩 반영되도록 해요.")
        if "time" in env_df:
            dates = env_df["time"].dropna()
            if len(dates):
                st.write(f"환경 측정 기간 · {dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}")
        if derived:
            st.write("생중량은 같은 행의 지상부 질량과 지하부 질량을 더했어요. "
                     "어느 한쪽이 비어 있으면 해당 행의 생중량도 비워 둬요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if env_file is None:
            st.info("환경 CSV를 올리면 날짜별 온도·습도 변화와 원본 표를 볼 수 있어요.")
        elif env_note:
            st.info(env_note)
        else:
            st.subheader("날짜별 재배 환경")
            st.caption("점 하나는 하루 동안 센서가 기록한 값의 평균이에요. "
                       "측정값이 하루 종일 없는 구간은 선을 끊어 표시해요.")
            for key in ("temperature", "humidity"):
                daily_plot(env_df, key)
            original(env_raw, env_file, "환경 데이터")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if growth_file is None:
            st.info("생육 결과 CSV를 올리면 세 가지 탐구 질문을 살펴볼 수 있어요.")
        elif growth_note:
            st.info(growth_note)
        else:
            if plants.empty:
                st.info("인식할 수 있는 생육 자료가 없어요. 잎 수, 길이 또는 질량 열을 확인해 주세요.")
            else:
                group, size = weight_groups(plants)
                st.subheader("개체별 생장 비교")
                if size:
                    st.caption("막대 색은 생중량 순위예요. 진한 초록은 상위, 주황은 하위, "
                               "연한 초록은 중간이며 회색은 생중량 미측정이에요.")
                if selected:
                    for key in selected:
                        bar_plot(plants, key, group)
                else:
                    st.info("사이드바에서 막대그래프로 볼 생육 항목을 선택해 주세요.")

                st.subheader("잎 수와 지상부 길이 중 어느 항목이 생중량과 더 밀접할까요?")
                st.caption("산점도의 점 하나는 한 개체예요. 상관계수가 +1에 가까우면 두 값이 함께 커지고, "
                           "−1에 가까우면 한쪽이 클수록 다른 쪽이 작아지는 경향이 있어요. "
                           "절댓값이 클수록 직선 관계가 강해요.")
                compare_correlations(plants)
                for key in (LEAF, SHOOT):
                    scatter(plants, key, MASS)

                st.subheader("무거운 개체와 가벼운 개체는 어떤 특징이 다를까요?")
                compare_groups(plants, group, size)

                st.subheader("지하부가 긴 개체는 지상부도 길까요?")
                scatter(plants, ROOT, SHOOT)
                st.caption("가로축은 지하부 길이, 세로축은 지상부 길이예요. "
                           "한 시점의 길이를 비교하면 두 부분이 함께 긴 경향을 살펴볼 수 있어요. "
                           "지하부 생장이 지상부 생장의 원인인지는 별도 실험이 필요해요.")
            original(growth_raw, growth_file, "생육 데이터")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
