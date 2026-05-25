from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path("artifacts/ufc_extra_trees_calibrated.joblib")

FEATURE_LABELS = {
    "age_diff_years": "Разница возраста, лет",
    "height_diff_cm": "Разница роста, см",
    "reach_diff_cm": "Разница размаха рук, см",
    "prior_fights_diff": "Разница числа прошлых UFC-боев",
    "prior_win_rate_diff": "Разница win rate в UFC",
    "prior_finish_win_rate_diff": "Разница finish-win rate в UFC",
    "avg_sig_strike_diff": "Разница среднего перевеса в значимых ударах",
    "avg_total_strike_diff": "Разница среднего перевеса во всех ударах",
    "avg_takedown_diff": "Разница среднего перевеса по тейкдаунам",
    "avg_submission_att_diff": "Разница среднего перевеса по сабмишн-попыткам",
    "avg_ctrl_time_min_diff": "Разница среднего контроля, минут",
    "avg_knockdown_diff": "Разница среднего перевеса по нокдаунам",
    "same_stance": "Одинаковая стойка",
    "title_fight": "Титульный бой",
    "scheduled_rounds": "Запланированные раунды",
}

FIGHTER_FACT_LABELS = {
    "age_years": "Возраст",
    "height_cm": "Рост, см",
    "reach_cm": "Размах, см",
    "prior_fights": "UFC боев",
    "prior_win_rate": "UFC win rate",
    "prior_finish_win_rate": "Finish-win rate",
    "avg_sig_strike_diff": "Средний +/- значимых ударов",
    "avg_takedown_diff": "Средний +/- тейкдаунов",
    "avg_ctrl_time_min_diff": "Средний +/- контроля, мин",
}


@st.cache_resource
def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        st.error(
            "Не найден model artifact. Запусти сначала: "
            "`python train_ufc_model.py`."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


def add_display_names(fighters: pd.DataFrame) -> pd.DataFrame:
    fighters = fighters.copy()
    fighters["display_name"] = fighters["name"].astype(str)
    duplicates = fighters["display_name"].duplicated(keep=False)
    fighters.loc[duplicates, "display_name"] = (
        fighters.loc[duplicates, "name"].astype(str)
        + " ["
        + fighters.loc[duplicates, "fighter_id"].astype(str).str[-7:]
        + "]"
    )
    return fighters


def default_index(options: list[str], preferred: str, fallback: int) -> int:
    for idx, option in enumerate(options):
        if preferred.lower() in option.lower():
            return idx
    return min(fallback, len(options) - 1)


def build_matchup_features(
    fighter_a: pd.Series,
    fighter_b: pd.Series,
    feature_columns: list[str],
    scheduled_rounds: int,
    title_fight: bool,
) -> pd.DataFrame:
    row = {
        "age_diff_years": fighter_a["age_years"] - fighter_b["age_years"],
        "height_diff_cm": fighter_a["height_cm"] - fighter_b["height_cm"],
        "reach_diff_cm": fighter_a["reach_cm"] - fighter_b["reach_cm"],
        "prior_fights_diff": fighter_a["prior_fights"] - fighter_b["prior_fights"],
        "prior_win_rate_diff": fighter_a["prior_win_rate"]
        - fighter_b["prior_win_rate"],
        "prior_finish_win_rate_diff": fighter_a["prior_finish_win_rate"]
        - fighter_b["prior_finish_win_rate"],
        "avg_sig_strike_diff": fighter_a["avg_sig_strike_diff"]
        - fighter_b["avg_sig_strike_diff"],
        "avg_total_strike_diff": fighter_a["avg_total_strike_diff"]
        - fighter_b["avg_total_strike_diff"],
        "avg_takedown_diff": fighter_a["avg_takedown_diff"]
        - fighter_b["avg_takedown_diff"],
        "avg_submission_att_diff": fighter_a["avg_submission_att_diff"]
        - fighter_b["avg_submission_att_diff"],
        "avg_ctrl_time_min_diff": fighter_a["avg_ctrl_time_min_diff"]
        - fighter_b["avg_ctrl_time_min_diff"],
        "avg_knockdown_diff": fighter_a["avg_knockdown_diff"]
        - fighter_b["avg_knockdown_diff"],
        "same_stance": int(str(fighter_a["stance"]) == str(fighter_b["stance"])),
        "title_fight": int(title_fight),
        "scheduled_rounds": scheduled_rounds,
    }
    return pd.DataFrame([row], columns=feature_columns)


def predict_probability(model, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[0, 1])


def explain_locally(
    model,
    X: pd.DataFrame,
    feature_medians: dict,
    fighter_a_name: str,
    fighter_b_name: str,
    top_n: int = 8,
) -> pd.DataFrame:
    base_probability = predict_probability(model, X)
    rows = []

    for feature in X.columns:
        perturbed = X.copy()
        perturbed.loc[0, feature] = feature_medians.get(feature, np.nan)
        perturbed_probability = predict_probability(model, perturbed)
        delta = base_probability - perturbed_probability
        rows.append(
            {
                "Признак": FEATURE_LABELS.get(feature, feature),
                "Значение": float(X.loc[0, feature]),
                "Медиана train": float(feature_medians.get(feature, np.nan)),
                "Влияние на P(левого)": delta,
                "Поддерживает": fighter_a_name if delta >= 0 else fighter_b_name,
            }
        )

    explanation = pd.DataFrame(rows)
    explanation["abs_delta"] = explanation["Влияние на P(левого)"].abs()
    explanation = explanation.sort_values("abs_delta", ascending=False).head(top_n)
    return explanation.drop(columns=["abs_delta"]).reset_index(drop=True)


def fighter_fact_table(fighter: pd.Series) -> pd.DataFrame:
    rows = []
    for feature, label in FIGHTER_FACT_LABELS.items():
        value = fighter.get(feature, np.nan)
        if pd.isna(value):
            formatted = "n/a"
        elif "rate" in feature:
            formatted = f"{value:.1%}"
        elif feature == "prior_fights":
            formatted = f"{int(value)}"
        else:
            formatted = f"{value:.2f}"
        rows.append({"Показатель": label, "Значение": formatted})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="UFC Matchup Predictor", layout="wide")
    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    feature_medians = artifact["feature_medians"]
    metadata = artifact["metadata"]
    fighters = add_display_names(artifact["fighter_table"])

    st.title("UFC Matchup Predictor")
    st.caption(
        "Baseline-модель: честные pre-fight признаки, ExtraTrees и калибровка вероятностей. "
        f"Данные в артефакте: {metadata['dataset_min_date']} - {metadata['dataset_max_date']}."
    )

    with st.sidebar:
        st.header("Настройки боя")
        scheduled_rounds = st.segmented_control(
            "Раунды",
            options=[3, 5],
            default=3,
        )
        title_fight = st.checkbox("Титульный бой", value=False)
        min_fights = st.slider(
            "Минимум прошлых UFC-боев в списке",
            min_value=0,
            max_value=20,
            value=1,
        )
        st.markdown(
            "Локальное объяснение считается perturbation-методом: "
            "каждый признак по очереди заменяется на train-медиану."
        )

    selectable = fighters.loc[fighters["prior_fights"].fillna(0) >= min_fights].copy()
    if len(selectable) < 2:
        st.warning("Слишком строгий фильтр по числу боев. Уменьши минимум в сайдбаре.")
        st.stop()

    options = selectable["display_name"].tolist()
    left_default = default_index(options, "Alex Pereira", 0)
    right_default = default_index(options, "Magomed Ankalaev", 1)
    if left_default == right_default:
        right_default = 1 if left_default == 0 else 0

    left_col, right_col = st.columns(2)
    with left_col:
        left_name = st.selectbox("Боец слева", options, index=left_default)
    with right_col:
        right_name = st.selectbox("Боец справа", options, index=right_default)

    fighter_a = selectable.loc[selectable["display_name"].eq(left_name)].iloc[0]
    fighter_b = selectable.loc[selectable["display_name"].eq(right_name)].iloc[0]

    if fighter_a["fighter_id"] == fighter_b["fighter_id"]:
        st.warning("Выбери двух разных бойцов.")
        st.stop()

    X = build_matchup_features(
        fighter_a,
        fighter_b,
        feature_columns,
        scheduled_rounds=scheduled_rounds,
        title_fight=title_fight,
    )
    p_left = predict_probability(model, X)
    p_right = 1.0 - p_left

    st.subheader("Прогноз")
    prob_left, prob_right = st.columns(2)
    with prob_left:
        st.metric(f"Победа: {fighter_a['name']}", f"{p_left:.1%}")
        st.progress(p_left)
    with prob_right:
        st.metric(f"Победа: {fighter_b['name']}", f"{p_right:.1%}")
        st.progress(p_right)

    st.subheader("Карточки бойцов")
    facts_left, facts_right = st.columns(2)
    with facts_left:
        st.markdown(f"**{fighter_a['name']}**")
        st.caption(f"Стойка: {fighter_a.get('stance', 'n/a')}")
        st.dataframe(fighter_fact_table(fighter_a), hide_index=True, width="stretch")
    with facts_right:
        st.markdown(f"**{fighter_b['name']}**")
        st.caption(f"Стойка: {fighter_b.get('stance', 'n/a')}")
        st.dataframe(fighter_fact_table(fighter_b), hide_index=True, width="stretch")

    st.subheader("Топ признаков, сдвинувших прогноз")
    explanation = explain_locally(
        model,
        X,
        feature_medians,
        str(fighter_a["name"]),
        str(fighter_b["name"]),
        top_n=8,
    )
    styled_explanation = explanation.copy()
    styled_explanation["Значение"] = styled_explanation["Значение"].map("{:.3f}".format)
    styled_explanation["Медиана train"] = styled_explanation["Медиана train"].map(
        "{:.3f}".format
    )
    styled_explanation["Влияние на P(левого)"] = styled_explanation[
        "Влияние на P(левого)"
    ].map("{:+.1%}".format)
    st.dataframe(styled_explanation, hide_index=True, width="stretch")

    with st.expander("Технические детали модели"):
        st.json(
            {
                "model": metadata["model_name"],
                "fighter_count": metadata["fighter_count"],
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "leakage_policy": metadata["leakage_policy"],
            }
        )


if __name__ == "__main__":
    main()
