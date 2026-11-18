from pathlib import Path

import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(
    page_title="Oregon Wildlife Collision Risk",
    page_icon="🦌",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F7F7F2;
    color: #183D32;
}
h1, h2, h3 {
    color: #183D32 !important;
}
[data-testid="stSidebar"] {
    background-color: #E7EEE8;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(
        Path(__file__).resolve().parent / "oregon_dashboard_data.csv",
        dtype={"route_id": str, "route": str}
    )

df = load_data()

st.title("Oregon Wildlife Collision Risk")
st.caption(
    "Predicting deer and elk collision risk across Oregon highways"
)

st.sidebar.header("Map Filters")

risk_options = st.sidebar.multiselect(
    "Risk level",
    ["High", "Moderate", "Lower"],
    default=["High", "Moderate"]
)

route_options = sorted(df["route_id"].dropna().unique())

selected_routes = st.sidebar.multiselect(
    "Highway route (optional)",
    route_options
)

show_candidates = st.sidebar.checkbox(
    "Only sign-assessment candidates",
    value=False
)

filtered = df[df["risk_level"].isin(risk_options)].copy()

if selected_routes:
    filtered = filtered[
        filtered["route_id"].isin(selected_routes)
    ]

if show_candidates:
    filtered = filtered[
        filtered["sign_candidate"].astype(str).str.lower() == "true"
    ]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Segments displayed", f"{len(filtered):,}")
c2.metric("Highway miles", f"{filtered['length_mi'].sum():,.1f}")
c3.metric(
    "Predicted collisions/year",
    f"{filtered['predicted_annual_collisions'].sum():,.0f}"
)
c4.metric(
    "Sign-assessment segments",
    f"{filtered['sign_candidate'].astype(str).str.lower().eq('true').sum():,}"
)

colors = {
    "High": [196, 73, 52, 210],
    "Moderate": [232, 165, 55, 200],
    "Lower": [64, 139, 106, 160]
}

filtered["color"] = filtered["risk_level"].map(colors)

st.subheader("Highway Collision Risk Map")

if filtered.empty:
    st.warning("No segments match the selected filters.")
else:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=250,
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.8
    )

    view = pdk.ViewState(
        latitude=float(filtered["lat"].mean()),
        longitude=float(filtered["lon"].mean()),
        zoom=6,
        pitch=0
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="light",
        tooltip={
            "html": """
            <b>Segment:</b> {segment_id}<br/>
            <b>Risk:</b> {risk_level}<br/>
            <b>Route:</b> {route_id}<br/>
            <b>Predicted annual collisions:</b>
            {predicted_annual_collisions}<br/>
            <b>Predicted risk per mile:</b>
            {risk_per_mile}<br/>
            <b>Historical collisions (2015–2020):</b>
            {coll_deer_elk_2015_2020}
            """,
            "style": {
                "backgroundColor": "#183D32",
                "color": "white"
            }
        }
    )

    st.pydeck_chart(deck, use_container_width=True)

st.caption(
    "Map markers represent model segment locations. "
    "They are not precise road boundaries or approved sign locations."
)

st.subheader("Highest-Risk Segments")

st.dataframe(
    filtered.sort_values(
        "risk_per_mile",
        ascending=False
    )[
        [
            "segment_id",
            "route_id",
            "begin_mp",
            "end_mp",
            "risk_level",
            "predicted_annual_collisions",
            "risk_per_mile"
        ]
    ].head(25),
    use_container_width=True,
    hide_index=True
)
