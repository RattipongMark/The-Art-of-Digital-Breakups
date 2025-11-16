# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from libHeartbreak import summarize_conditional, cluster_reasons, predict_emotion
# import warnings
# warnings.filterwarnings("ignore", category=SyntaxWarning)


plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["TH Sarabun New"],
})

st.set_page_config(
    page_title="Breakup Analysis Tool",
    page_icon="💔",
    layout="wide",
)

# -------------------- Page selection --------------------
page = st.tabs(
    ["Dashboard", "Analysis Tool"],
)


df_all = pd.read_csv("all_post+all_output.csv")

# แปลงคอลัมน์ post_time เป็น datetime
df_all['post_time'] = pd.to_datetime(df_all['post_time'], errors='coerce')

# -------------------- Dashboard Page --------------------
with page[0]:
    st.title("💔 Breakup Analysis Dashboard")

    # -------------------- Filters --------------------
    filter_container = st.container()
    with filter_container:
        cols = st.columns([1, 1])
        with cols[0]:
            genders = st.multiselect(
                "Select Gender:",
                options=df_all['gender'].dropna().unique(),
                default=df_all['gender'].dropna().unique()
            )
        with cols[1]:
            min_date = df_all['post_time'].min()
            max_date = df_all['post_time'].max()
            date_range = st.date_input(
                "Select Post Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

    # -------------------- Filter data --------------------
    df_filtered = df_all[df_all['gender'].isin(genders)]
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['post_time'] >= pd.to_datetime(start_date)) & 
            (df_filtered['post_time'] <= pd.to_datetime(end_date))
        ]

    # st.markdown(f"Filtered data: {df_filtered.shape[0]} rows")

    # -------------------- Prepare Reason Data --------------------
    all_reasons = []
    for row in df_filtered['reason_extrac'].dropna():
        parts = [x.strip() for x in row.split(",")]
        all_reasons.extend(parts)

    # -------------------- Layout 2 Columns --------------------
    col1, col2 = st.columns(2)

    # -------------------- Top 5 Reasons --------------------
    with col1:
        st.markdown("### Top 5 Reasons")

        if all_reasons:
            reason_counts = pd.Series(all_reasons).value_counts().head(5)
            reason_df = reason_counts.reset_index()
            reason_df.columns = ['Reason', 'Frequency']
            chart_reason = alt.Chart(reason_df).mark_bar().encode(
                x='Frequency:Q',
                y=alt.Y('Reason:N', sort='-x'),
                color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='redpurple'))
            ).properties(height=300)
            st.altair_chart(chart_reason, use_container_width=True)
        else:
            st.info("No reason_extrac data available.")

    # -------------------- Emotion Distribution --------------------
    with col2:
        st.subheader("Emotion Distribution")
        emotion_counts = df_filtered['emotion_label'].value_counts().head(5)
        if not emotion_counts.empty:
            emotion_df = emotion_counts.reset_index()
            emotion_df.columns = ['Emotion', 'Frequency']
            chart_emotion = alt.Chart(emotion_df).mark_bar().encode(
                x='Frequency:Q',
                y=alt.Y('Emotion:N', sort='-x'),
                color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='greens'))
            ).properties(height=300)
            st.altair_chart(chart_emotion, use_container_width=True)
        else:
            st.info("No emotion data available.")

   # -------------------- Time Series Selector --------------------
    st.markdown("---")
    st.subheader("⏱Time Series of Selected Reason / Emotion")

    # ให้เลือก reason
    unique_reasons = pd.Series(all_reasons).value_counts().index.tolist()
    selected_reason = st.selectbox("Select a Reason to plot over time:", options=unique_reasons)

    # ให้เลือก emotion
    unique_emotions = df_filtered['emotion_label'].dropna().unique().tolist()
    selected_emotion = st.selectbox("Select an Emotion to plot over time:", options=unique_emotions)

    # -------------------- Reason Time Series --------------------
    df_reason_time = df_filtered.dropna(subset=['reason_extrac']).copy()
    df_reason_time['reason_list'] = df_reason_time['reason_extrac'].apply(lambda x: [r.strip() for r in x.split(",")])
    df_reason_time['flag'] = df_reason_time['reason_list'].apply(lambda x: selected_reason in x)
    reason_time = df_reason_time.groupby(df_reason_time['post_time'].dt.date)['flag'].sum().reset_index()
    reason_time.columns = ['Date', 'Frequency']

    chart_reason_time = alt.Chart(reason_time).mark_line(point=True, color='red').encode(
        x='Date:T',
        y='Frequency:Q',
        tooltip=['Date', 'Frequency']
    ).properties(title=f"Time Series of Reason: {selected_reason}", height=300)
    st.altair_chart(chart_reason_time, use_container_width=True)

    # -------------------- Emotion Time Series --------------------
    df_emotion_time = df_filtered.dropna(subset=['emotion_label']).copy()
    df_emotion_time['flag'] = df_emotion_time['emotion_label'].apply(lambda x: x == selected_emotion)
    emotion_time = df_emotion_time.groupby(df_emotion_time['post_time'].dt.date)['flag'].sum().reset_index()
    emotion_time.columns = ['Date', 'Frequency']

    chart_emotion_time = alt.Chart(emotion_time).mark_line(point=True, color='green').encode(
        x='Date:T',
        y='Frequency:Q',
        tooltip=['Date', 'Frequency']
    ).properties(title=f"Time Series of Emotion: {selected_emotion}", height=300)
    st.altair_chart(chart_emotion_time, use_container_width=True)


    st.dataframe(df_filtered)

with page[1]:
    st.markdown("### Input your text")
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = ""
    if "tags_input" not in st.session_state:
        st.session_state["tags_input"] = []

    user_text = st.text_area(
        "Paste or type a long paragraph below to analyze it:",
        value=st.session_state["user_text"],
        height=250,
        key="user_text",
    )

    features = st.multiselect(
        "Select one or more features:",
        ["Summarization", "Reason Extraction", "Emotion Detection"],
        default=["Summarization"],
    )

    def clear_text():
        for key in ["user_text", "tags_input"]:
            if key in st.session_state:
                del st.session_state[key]

    analyze_col, reset_col = st.columns([2,1])
    with analyze_col:
        submit_btn = st.button("Analyze Now", use_container_width=True)
    with reset_col:
        clear_btn = st.button("Clear", use_container_width=True, on_click=clear_text)

    st.markdown("---")

    if submit_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        elif not features:
            st.warning("⚠️ Please select at least one feature.")
        else:
            if "Summarization" in features:
                st.subheader("✨ Summarization Result")
                summary_result = summarize_conditional(user_text)
                st.markdown(f"""
                <div style="padding:15px; border-radius:8px;">
                <strong>Summary:</strong><br>
                {summary_result}
                </div>
                """, unsafe_allow_html=True)

            if "Reason Extraction" in features:
                st.subheader("💡 Reason Extraction Result")
                reason_result = cluster_reasons(user_text)
                for reason, sents in reason_result.items():
                    st.markdown(f"""
                    <div style="padding:10px; margin-bottom:5px; border-radius:6px;">
                    <strong>{reason}:</strong> {sents}
                    </div>
                    """, unsafe_allow_html=True)

            if "Emotion Detection" in features:
                st.subheader("😢 Emotion Detection Result")
                label, score = predict_emotion(user_text)
                st.markdown(f"""
                <div style="padding:10px; border-radius:6px;">
                <strong>Predicted Emotion:</strong> {label} <br>
                <strong>Score:</strong> {score:.3f}
                </div>
                """, unsafe_allow_html=True)

