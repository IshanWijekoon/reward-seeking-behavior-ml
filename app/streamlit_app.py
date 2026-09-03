"""
Digital Well-being MVP — Streamlit app.

Run from project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path when launched via `streamlit run app/...`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import build_feature_row, form_payload_from_inputs
from app.inference import LABEL_ORDER, analyze_checkin, load_pipeline
from app.interventions import suggest_actions
from app.storage import CheckinStore

st.set_page_config(
    page_title="Digital Well-being Check-in",
    page_icon="📱",
    layout="centered",
)

DISCLAIMER = (
    "This tool is **not a clinical diagnosis**. It is a digital well-being support "
    "demo that estimates early habit risk from self-entered usage stats and suggests "
    "practical actions. Do not use it to punish or label yourself or others."
)


@st.cache_resource
def get_pipeline():
    return load_pipeline()


@st.cache_resource
def get_store() -> CheckinStore:
    return CheckinStore()


def risk_color(level: str) -> str:
    return {"Low": "green", "Moderate": "orange", "High": "red"}.get(level, "gray")


def page_checkin():
    st.header("Weekly check-in")
    st.caption("Enter your typical daily averages for the past week.")

    user_id = st.text_input(
        "Nickname / local ID",
        value=st.session_state.get("user_id", ""),
        help="Used only on this device to track your risk trend. No cloud login.",
        key="checkin_user_id",
    )

    with st.form("weekly_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=35, value=22, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            daily_screen = st.number_input(
                "Daily screen time (hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.1
            )
            social = st.number_input(
                "Social media (hours/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.1
            )
            gaming = st.number_input(
                "Gaming (hours/day)", min_value=0.0, max_value=24.0, value=1.0, step=0.1
            )
        with c2:
            work_study = st.number_input(
                "Study / work apps (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.1,
            )
            sleep = st.number_input(
                "Sleep (hours/night)", min_value=0.0, max_value=14.0, value=6.5, step=0.1
            )
            notifications = st.number_input(
                "Notifications per day", min_value=0, max_value=1000, value=120, step=1
            )
            app_opens = st.number_input(
                "App opens / unlocks per day", min_value=0, max_value=500, value=80, step=1
            )
            stress = st.selectbox("Stress level", ["Low", "Medium", "High"], index=1)

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if not submitted:
        if "last_result" in st.session_state:
            _render_result(st.session_state["last_result"], allow_save=True)
        return

    if not user_id.strip():
        st.warning("Enter a nickname / local ID so we can save your trend.")
        return

    st.session_state["user_id"] = user_id.strip()

    inputs = form_payload_from_inputs(
        age=int(age),
        gender=gender,
        daily_screen_time_hours=float(daily_screen),
        social_media_hours=float(social),
        gaming_hours=float(gaming),
        work_study_hours=float(work_study),
        sleep_hours=float(sleep),
        notifications_per_day=int(notifications),
        app_opens_per_day=int(app_opens),
        stress_level=stress,
    )
    X = build_feature_row(**inputs)
    pipeline = get_pipeline()
    analysis = analyze_checkin(pipeline, X)
    actions = suggest_actions(
        analysis["drivers"],
        feature_values={
            "sleep_hours": float(sleep),
            "notifications_per_day": float(notifications),
            "daily_screen_time_hours": float(daily_screen),
        },
        max_actions=2,
    )

    result = {
        "user_id": user_id.strip(),
        "inputs": inputs,
        "features": X.to_dict(orient="records")[0],
        **analysis,
        "actions": actions,
    }
    st.session_state["last_result"] = result
    _render_result(result, allow_save=True)


def _render_result(result: dict, *, allow_save: bool):
    level = result["risk_level"]
    st.subheader("Your early risk estimate")
    st.markdown(
        f"### :{risk_color(level)}[**{level} risk**]"
    )

    probs = result["probabilities"]
    prob_df = pd.DataFrame(
        {"Risk level": list(probs.keys()), "Probability": list(probs.values())}
    ).set_index("Risk level")
    st.bar_chart(prob_df)

    st.subheader("Top 3 drivers (SHAP)")
    st.caption("Features that most influenced this week's prediction for your risk class.")
    for i, d in enumerate(result["drivers"], start=1):
        sign = "+" if d["shap_value"] >= 0 else ""
        st.markdown(
            f"**{i}. {d['display_name']}** — value `{d['feature_value']:.2f}`, "
            f"SHAP `{sign}{d['shap_value']:.3f}` ({d['direction']})"
        )

    st.subheader("Suggested actions (1–2)")
    for action in result["actions"]:
        st.markdown(f"- {action}")

    if allow_save:
        if st.button("Save this check-in", type="primary"):
            store = get_store()
            row_id = store.save_checkin(
                user_id=result["user_id"],
                inputs=result["inputs"],
                risk_level=result["risk_level"],
                risk_index=result["risk_index"],
                probabilities=result["probabilities"],
                drivers=result["drivers"],
                actions=result["actions"],
            )
            st.success(f"Saved check-in #{row_id}. Open **My trend** after your next weekly entry.")


def page_trend():
    st.header("My trend")
    st.caption("Compare weekly check-ins for the same nickname on this device.")

    store = get_store()
    users = store.list_users()
    default_user = st.session_state.get("user_id", "")
    options = users if users else ([default_user] if default_user else [])

    if not options:
        st.info("No saved check-ins yet. Complete a weekly check-in and click **Save**.")
        return

    index = 0
    if default_user in options:
        index = options.index(default_user)
    user_id = st.selectbox("Nickname", options, index=index)
    checkins = store.get_checkins(user_id)

    if not checkins:
        st.info("No check-ins for this nickname.")
        return

    st.write(f"**{len(checkins)}** check-in(s) for `{user_id}`")

    trend_df = pd.DataFrame(
        {
            "Date": [c["created_at"][:10] for c in checkins],
            "Risk": [c["risk_level"] for c in checkins],
            "Risk index": [c["risk_index"] for c in checkins],
            "High prob": [c["probabilities"].get("High", 0.0) for c in checkins],
        }
    )
    st.line_chart(trend_df.set_index("Date")[["Risk index"]])
    st.caption("Risk index: 0 = Low, 1 = Moderate, 2 = High")

    if len(checkins) >= 2:
        prev, curr = checkins[-2], checkins[-1]
        delta = curr["risk_index"] - prev["risk_index"]
        if delta < 0:
            st.success(
                f"Risk improved from **{prev['risk_level']}** → **{curr['risk_level']}** "
                f"since the previous check-in."
            )
        elif delta > 0:
            st.warning(
                f"Risk increased from **{prev['risk_level']}** → **{curr['risk_level']}** "
                f"since the previous check-in. Focus on the latest suggested actions."
            )
        else:
            st.info(
                f"Risk stayed at **{curr['risk_level']}** compared with the previous check-in."
            )

        st.subheader("Last two check-ins")
        left, right = st.columns(2)
        for col, item, title in (
            (left, prev, "Previous"),
            (right, curr, "Latest"),
        ):
            with col:
                st.markdown(f"**{title}** ({item['created_at'][:10]})")
                st.write(f"Risk: **{item['risk_level']}**")
                inp = item["inputs"]
                st.write(
                    f"Screen {inp.get('daily_screen_time_hours')}h · "
                    f"Social {inp.get('social_media_hours')}h · "
                    f"Sleep {inp.get('sleep_hours')}h · "
                    f"Notifs {inp.get('notifications_per_day')}"
                )
    else:
        st.info(
            "Save a second check-in (ideally after ~7 days) to see week-over-week change."
        )

    with st.expander("Full history table"):
        rows = []
        for c in checkins:
            rows.append(
                {
                    "Date": c["created_at"][:19],
                    "Risk": c["risk_level"],
                    "Screen h": c["inputs"].get("daily_screen_time_hours"),
                    "Social h": c["inputs"].get("social_media_hours"),
                    "Sleep h": c["inputs"].get("sleep_hours"),
                    "Notifications": c["inputs"].get("notifications_per_day"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_about():
    st.header("About")
    st.markdown(
        """
This MVP wraps the research **Dataset 1 XGBoost** model for early Low / Moderate / High
risk of problematic reward-seeking smartphone habits.

**Loop**
1. Enter weekly usage averages  
2. Get risk + top SHAP drivers  
3. Try 1–2 concrete actions  
4. Re-check after about a week and view your trend  

**Model note:** Trained on anonymised public young-adult usage data (research prototype).
Predictions are probabilistic and imperfect — treat them as coaching signals, not medical advice.
"""
    )


def main():
    st.title("Digital Well-being Check-in")
    st.info(DISCLAIMER)

    page = st.sidebar.radio(
        "Navigate",
        ["Weekly check-in", "My trend", "About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Local-only storage · Dataset 1 XGBoost")

    if page == "Weekly check-in":
        page_checkin()
    elif page == "My trend":
        page_trend()
    else:
        page_about()


if __name__ == "__main__":
    main()
