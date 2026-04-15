"""
OCC taxonomy (simplified for our 4-class setup):
    - Distress  (unpleasant event for self)       → sad
    - Anger     (disapproval of agent's action)    → angry / ang
    - Joy       (pleasant event for self)          → positive / hap
    - Neutral   (no strong appraisal)              → neutral / neu

Extended OCC attributes per emotion:
    - appraisal_type: "event" | "agent" | "object"
    - valence: "positive" | "negative" | "neutral"
    - arousal_tendency: "high" | "low" | "neutral"
    - desirability: how desirable the event is (for event-based emotions)
    - praiseworthiness: how praiseworthy the agent's action is (for agent-based)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OCCCategory:
    """OCC appraisal category for one emotion."""
    occ_name: str           # OCC emotion name
    appraisal_type: str     # "event" | "agent" | "object"
    valence: str            # "positive" | "negative" | "neutral"
    arousal_tendency: str   # "high" | "low" | "neutral"
    desirability: float     # -1.0 (undesirable) to +1.0 (desirable)
    praiseworthiness: float # -1.0 (blameworthy) to +1.0 (praiseworthy)
    description: str        # Human-readable explanation


# Dusha emotion → OCC category
DUSHA_OCC_MAP: dict[str, OCCCategory] = {
    "angry": OCCCategory(
        occ_name="Anger (Reproach + Distress)",
        appraisal_type="agent",
        valence="negative",
        arousal_tendency="high",
        desirability=-0.8,
        praiseworthiness=-0.9,
        description="Disapproval of someone's blameworthy action combined with distress",
    ),
    "sad": OCCCategory(
        occ_name="Distress",
        appraisal_type="event",
        valence="negative",
        arousal_tendency="low",
        desirability=-0.7,
        praiseworthiness=0.0,
        description="Displeasure at an undesirable event",
    ),
    "positive": OCCCategory(
        occ_name="Joy (Happy-for)",
        appraisal_type="event",
        valence="positive",
        arousal_tendency="high",
        desirability=0.8,
        praiseworthiness=0.0,
        description="Pleasure at a desirable event",
    ),
    "neutral": OCCCategory(
        occ_name="No Appraisal",
        appraisal_type="event",
        valence="neutral",
        arousal_tendency="neutral",
        desirability=0.0,
        praiseworthiness=0.0,
        description="No significant cognitive appraisal triggered",
    ),
}


def get_occ_category(emotion: str, dataset: str = "dusha") -> OCCCategory:
    mapping = DUSHA_OCC_MAP
    if emotion not in mapping:
        raise KeyError(f"Unknown {dataset} emotion: '{emotion}'. "
                       f"Valid: {list(mapping.keys())}")
    return mapping[emotion]


def get_occ_features(emotion: str, dataset: str = "dusha") -> dict[str, float]:
    cat = get_occ_category(emotion, dataset)
    valence_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    arousal_map = {"high": 1.0, "low": -1.0, "neutral": 0.0}
    return {
        "desirability": cat.desirability,
        "praiseworthiness": cat.praiseworthiness,
        "arousal_numeric": arousal_map[cat.arousal_tendency],
        "valence_numeric": valence_map[cat.valence],
    }
