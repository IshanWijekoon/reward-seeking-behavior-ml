"""Map SHAP driver features to concrete digital well-being actions."""
from __future__ import annotations

from typing import Any


# Each rule: feature -> list of action strings (picked when that driver ranks high)
DRIVER_ACTIONS: dict[str, list[str]] = {
    "notifications_per_day": [
        "Batch notifications: turn off non-essential alerts and check messages 2–3 times/day.",
        "Mute social apps during study blocks (Focus / Do Not Disturb for 45–90 minutes).",
    ],
    "social_media_hours": [
        "Set a 30–45 minute daily social-media limit and remove those apps from your home screen.",
        "Replace one evening scroll session with an offline activity (walk, stretch, or reading).",
    ],
    "gaming_hours": [
        "Schedule gaming only after study goals are done; set a weekday time cap.",
        "Use a hard stop alarm: end gaming at least 60 minutes before bed.",
    ],
    "reward_app_share": [
        "Cut reward-app share of screen time: move social/gaming apps into a folder off the home screen.",
        "Keep productive apps easy to reach; add friction (search required) for social/gaming apps.",
    ],
    "app_opens_per_day": [
        "Enable grayscale mode to reduce impulse unlocks and app hopping.",
        "Increase unlock friction (longer PIN) and hide red notification badges for a week.",
    ],
    "sleep_hours": [
        "Start a wind-down: no screens for 60 minutes before bed; charge the phone outside the bedroom.",
        "Set a consistent bedtime reminder and use Night Mode / scheduled Downtime after 10pm.",
    ],
    "stress_level_ord": [
        "Take two short offline breaks today (5–10 minutes) instead of opening social apps.",
        "Swap one stress-scroll session for a short walk or breathing exercise.",
    ],
    "daily_screen_time_hours": [
        "Set a daily screen budget about 1 hour below your current average and review it each evening.",
        "Use app timers for your top three time-consuming apps this week.",
    ],
    "work_study_hours": [
        "Protect study time with Focus mode; batch entertainment only after a completed study block.",
    ],
    "age": [
        "Track habits weekly rather than comparing yourself to others; focus on sleep and notification load.",
    ],
    "gender_Male": [],
    "gender_Other": [],
}

# Prefer these when sleep is low even if SHAP rank is mixed
SLEEP_LOW_THRESHOLD = 6.5


def suggest_actions(
    drivers: list[dict[str, Any]],
    *,
    feature_values: dict[str, float] | None = None,
    max_actions: int = 2,
) -> list[str]:
    """
    Pick 1–2 concrete actions from the top SHAP drivers.
    Deduplicate overlapping suggestions. Prefer sleep advice when sleep is low.
    """
    feature_values = feature_values or {}
    actions: list[str] = []
    seen: set[str] = set()

    def add_from_feature(feat: str) -> None:
        for action in DRIVER_ACTIONS.get(feat, []):
            if action not in seen and len(actions) < max_actions:
                actions.append(action)
                seen.add(action)

    # If sleep is critically low, surface sleep advice first
    sleep = feature_values.get("sleep_hours")
    if sleep is not None and sleep < SLEEP_LOW_THRESHOLD:
        add_from_feature("sleep_hours")

    for driver in drivers:
        if len(actions) >= max_actions:
            break
        feat = driver["feature"]
        # Skip empty demographic action lists
        if feat in {"gender_Male", "gender_Other", "age"} and DRIVER_ACTIONS.get(feat) == []:
            continue
        # For sleep: only push "low sleep" style actions when value is actually low
        if feat == "sleep_hours" and sleep is not None and sleep >= SLEEP_LOW_THRESHOLD:
            continue
        add_from_feature(feat)

    # Fallback if drivers were demographic-only
    if not actions:
        actions.append(
            "Review your top usage apps this week and set one concrete limit "
            "(notifications, social time, or bedtime screens)."
        )

    return actions[:max_actions]
