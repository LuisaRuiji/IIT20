import json
import random

import joblib
import pandas as pd
import streamlit as st


# Fixed artifact names based on the current project files.
PIPELINE_PATH = "preprocessing_pipeline.pkl"
LOGISTIC_MODEL_PATH = "logistic_regression_model.pkl"
GB_MODEL_PATH = "gradient_boosting_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.json"
DEMO_PRESETS_PATH = "demo_presets.json"
MISSING_VALUE = -1


# Visible demo inputs grouped by signal family.
FORM_SECTIONS = [
    (
        "User Activity",
        [
            "user_view_count_30d",
            "user_cart_count_30d",
            "user_purchase_count_30d",
            "user_view_count_7d",
            "user_cart_count_7d",
        ],
    ),
    (
        "User-Product Interaction",
        [
            "user_product_view_count_30d",
            "user_product_cart_count_30d",
            "user_product_view_count_7d",
            "user_product_cart_count_7d",
        ],
    ),
    (
        "Product Demand",
        [
            "product_view_count_30d",
            "product_cart_count_30d",
            "product_purchase_count_30d",
            "product_view_count_7d",
            "product_cart_count_7d",
            "product_purchase_count_7d",
        ],
    ),
    (
        "Recency",
        [
            "hours_since_last_user_event",
            "hours_since_last_user_product_event",
            "days_since_last_purchase",
        ],
    ),
    (
        "Customer Profile",
        [
            "account_age_days",
            "lifetime_order_count",
            "lifetime_total_spend",
        ],
    ),
    (
        "Behavior Breadth",
        [
            "distinct_products_viewed_7d",
            "distinct_products_carted_7d",
            "user_category_view_count_30d",
        ],
    ),
    (
        "Product Info",
        [
            "price",
            "currentStock",
            "categoryId",
        ],
    ),
]

DEFAULT_INPUT_VALUES = {
    "user_view_count_30d": 20,
    "user_cart_count_30d": 4,
    "user_purchase_count_30d": 1,
    "user_view_count_7d": 5,
    "user_cart_count_7d": 1,
    "user_product_view_count_30d": 4,
    "user_product_cart_count_30d": 1,
    "user_product_view_count_7d": 2,
    "user_product_cart_count_7d": 1,
    "product_view_count_30d": 120,
    "product_cart_count_30d": 20,
    "product_purchase_count_30d": 8,
    "product_view_count_7d": 35,
    "product_cart_count_7d": 6,
    "product_purchase_count_7d": 2,
    "hours_since_last_user_event": 12.0,
    "hours_since_last_user_product_event": 6.0,
    "days_since_last_purchase": 14,
    "account_age_days": 365,
    "lifetime_order_count": 5,
    "lifetime_total_spend": 350.0,
    "distinct_products_viewed_7d": 8,
    "distinct_products_carted_7d": 2,
    "user_category_view_count_30d": 10,
    "price": 49.99,
    "currentStock": 25,
}

FLOAT_FIELDS = {
    "hours_since_last_user_event",
    "hours_since_last_user_product_event",
    "lifetime_total_spend",
    "price",
}

VISIBLE_INPUT_FIELDS = [field for _, fields in FORM_SECTIONS for field in fields]

HELPER_RANGE_FIELDS = [
    "user_view_count_30d",
    "user_cart_count_30d",
    "user_product_view_count_30d",
    "product_view_count_30d",
    "product_cart_count_30d",
    "hours_since_last_user_event",
    "lifetime_order_count",
    "lifetime_total_spend",
]


@st.cache_resource
def load_artifacts():
    # Load the saved files once so Streamlit does not reload them on every interaction.
    pipeline = load_optional_pipeline()
    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    gb_model = joblib.load(GB_MODEL_PATH)
    feature_columns = load_feature_columns()
    return pipeline, logistic_model, gb_model, feature_columns


def load_optional_pipeline():
    # The pipeline is optional because the current folder does not contain it.
    try:
        return joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        return None


def load_feature_columns():
    # Read the full feature list from JSON.
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("feature_columns", "columns", "features"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported JSON format in {FEATURE_COLUMNS_PATH}")


@st.cache_data
def load_demo_metadata():
    # Load precomputed preset rows and helper ranges bundled with the app.
    with open(DEMO_PRESETS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def build_demo_presets():
    metadata = load_demo_metadata()
    return pd.DataFrame(metadata["presets"])


@st.cache_data
def load_helper_ranges():
    # Read precomputed training-range guidance bundled with the app.
    metadata = load_demo_metadata()
    return metadata["helper_ranges"]


def format_range_value(field_name, value):
    # Keep helper text compact and consistent with each field's input type.
    if field_name in FLOAT_FIELDS:
        return f"{value:.2f}"

    return str(int(round(value)))


def render_helper_text(feature_name, helper_ranges):
    # Show approximate training-data ranges under selected inputs.
    if feature_name not in helper_ranges:
        return

    summary = helper_ranges[feature_name]
    st.caption(
        "Typical training range: "
        f"{format_range_value(feature_name, summary['q25'])}-"
        f"{format_range_value(feature_name, summary['q75'])} "
        f"(median {format_range_value(feature_name, summary['q50'])}, "
        f"95th pct {format_range_value(feature_name, summary['q95'])})"
    )


def get_model_categorical_info(model):
    # Pull safe category defaults and all allowed options from the fitted encoder.
    defaults = {}
    options = {}

    if not hasattr(model, "named_steps"):
        return defaults, options

    preprocessor = model.named_steps.get("prep") or model.named_steps.get("preprocessor")
    if preprocessor is None:
        return defaults, options

    if not hasattr(preprocessor, "named_transformers_") or "cat" not in preprocessor.named_transformers_:
        return defaults, options

    cat_pipeline = preprocessor.named_transformers_["cat"]
    if not hasattr(cat_pipeline, "named_steps") or "onehot" not in cat_pipeline.named_steps:
        return defaults, options

    encoder = cat_pipeline.named_steps["onehot"]
    categorical_features = None
    for name, _, columns in preprocessor.transformers_:
        if name == "cat":
            categorical_features = columns
            break

    if categorical_features is None or not hasattr(encoder, "categories_"):
        return defaults, options

    for feature_name, categories in zip(categorical_features, encoder.categories_):
        values = [str(category) for category in categories]
        options[feature_name] = values
        defaults[feature_name] = "0" if feature_name == "price_bucket" and "0" in values else values[0]

    return defaults, options


def model_has_embedded_preprocessing(model):
    # The current saved models are sklearn Pipelines with a fitted "prep" step.
    return hasattr(model, "named_steps") and "prep" in model.named_steps


def safe_rate(numerator, denominator):
    # Avoid division-by-zero when deriving rate features.
    if not denominator:
        return 0.0

    value = float(numerator) / float(denominator)
    return max(0.0, min(1.0, value))


def derive_daily_count(window_value, window_days):
    # Approximate 1-day counts from visible multi-day windows for a less sparse demo row.
    if window_days <= 0:
        return 0

    return max(0, int(round(float(window_value) / float(window_days))))


def get_price_bucket(price):
    # Match the training-time price bucket logic from the dataset exporter.
    if price <= 0:
        return MISSING_VALUE
    if price < 100:
        return 0
    if price < 300:
        return 1
    if price < 1000:
        return 2
    return 3


def format_probability(probability):
    # Show probabilities as readable percentages for the main demo metrics.
    if probability == 0:
        return "0.00000000%"

    percent = probability * 100
    if percent >= 1:
        return f"{percent:.2f}%"
    if percent >= 0.01:
        return f"{percent:.4f}%"

    return f"{percent:.8f}%"


def is_effectively_zero(value):
    # Treat string and numeric zero-like values as empty signal for demo diagnostics.
    if value is None:
        return True

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return True

        try:
            return float(stripped) == 0.0
        except ValueError:
            return False

    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def count_non_zero_features(frame):
    # Count how many features in the single input row carry a non-zero value.
    row = frame.iloc[0].to_dict()
    return sum(0 if is_effectively_zero(value) else 1 for value in row.values())


def initialize_form_state(categorical_defaults):
    # Seed session state once so preset buttons can update widget values cleanly.
    for field in VISIBLE_INPUT_FIELDS:
        if field in st.session_state:
            continue

        if field == "categoryId":
            st.session_state[field] = categorical_defaults.get("categoryId", "1")
        else:
            st.session_state[field] = DEFAULT_INPUT_VALUES[field]


def normalize_visible_value(feature_name, value):
    # Match widget value types when loading real dataset rows into session state.
    if feature_name == "categoryId":
        return str(int(value))
    if feature_name in FLOAT_FIELDS:
        return float(value)
    return int(value)


def apply_preset_row(preset_row):
    # Copy a real dataset row into the visible form inputs while keeping them editable.
    for field in VISIBLE_INPUT_FIELDS:
        st.session_state[field] = normalize_visible_value(field, preset_row[field])

    st.session_state["selected_preset_label"] = preset_row["preset_label"]
    st.session_state["selected_preset_target"] = int(preset_row["willPurchase"])
    st.session_state["selected_preset_probability"] = float(preset_row["gb_probability"])
    st.session_state["selected_preset_reference"] = (
        f"userId={int(preset_row['userId'])}, "
        f"productId={int(preset_row['productId'])}, "
        f"referenceTimeUtc={preset_row['referenceTimeUtc']}"
    )


def randomize_hidden_features(row, user_inputs, feature_columns, categorical_options, seed):
    # Randomize only the features that are not explicitly provided in the UI.
    rng = random.Random(seed)
    hidden_features = set(feature_columns) - set(user_inputs)

    for feature in hidden_features:
        if feature in categorical_options:
            row[feature] = rng.choice(categorical_options[feature])
            continue

        if feature.endswith("_flag") or feature.startswith("is_") or feature.endswith("Flag"):
            row[feature] = rng.randint(0, 1)
        elif "rate" in feature:
            row[feature] = round(rng.uniform(0.0, 1.0), 3)
        elif "hours_since" in feature:
            row[feature] = round(rng.uniform(0.0, 72.0), 2)
        elif "days_since" in feature:
            row[feature] = rng.randint(0, 365)
        elif "age_days" in feature:
            row[feature] = rng.randint(1, 3650)
        elif "spend" in feature or "price" in feature:
            row[feature] = round(rng.uniform(1.0, 500.0), 2)
        else:
            row[feature] = rng.randint(0, 50)

    return row


def add_derived_features(values):
    # Fill engineered features so the demo row better matches training-time feature patterns.
    values["user_view_count_1d"] = derive_daily_count(values["user_view_count_7d"], 7)
    values["user_cart_count_1d"] = derive_daily_count(values["user_cart_count_7d"], 7)
    values["user_product_view_count_1d"] = derive_daily_count(values["user_product_view_count_7d"], 7)
    values["user_product_cart_count_1d"] = derive_daily_count(values["user_product_cart_count_7d"], 7)
    values["product_view_count_1d"] = derive_daily_count(values["product_view_count_7d"], 7)
    values["product_cart_count_1d"] = derive_daily_count(values["product_cart_count_7d"], 7)
    values["cart_abandon_count_30d"] = max(
        0, int(values["user_cart_count_30d"]) - int(values["user_purchase_count_30d"])
    )
    values["inStockFlag"] = 1 if values["currentStock"] > 0 else 0
    values["isActive"] = 1
    values["price_is_missing"] = 1 if values["price"] <= 0 else 0
    values["price_bucket"] = get_price_bucket(values["price"])
    values["is_returning_customer"] = 1 if values["lifetime_order_count"] > 1 else 0
    values["has_user_product_event_30d"] = 1 if (
        values["user_product_view_count_30d"] > 0
        or values["user_product_cart_count_30d"] > 0
    ) else 0
    values["user_product_cart_flag_30d"] = 1 if values["user_product_cart_count_30d"] > 0 else 0
    values["view_to_cart_rate_30d"] = safe_rate(
        values["user_cart_count_30d"], values["user_view_count_30d"]
    )
    values["cart_to_purchase_rate_30d"] = safe_rate(
        values["user_purchase_count_30d"], values["user_cart_count_30d"]
    )
    values["product_cart_to_view_rate_30d"] = safe_rate(
        values["product_cart_count_30d"], values["product_view_count_30d"]
    )
    values["product_purchase_to_cart_rate_30d"] = safe_rate(
        values["product_purchase_count_30d"], values["product_cart_count_30d"]
    )
    return values


def render_feature_input(feature_name, categorical_defaults, categorical_options, helper_ranges):
    # Render a single input widget using either numeric entry or model-safe category options.
    if feature_name == "categoryId":
        options = categorical_options.get(
            "categoryId", [categorical_defaults.get("categoryId", "1")]
        )
        current_value = str(st.session_state.get(feature_name, categorical_defaults.get("categoryId", options[0])))
        if current_value not in options:
            current_value = options[0]
            st.session_state[feature_name] = current_value
        default_index = options.index(current_value)
        value = st.selectbox(feature_name, options=options, index=default_index, key=feature_name)
        render_helper_text(feature_name, helper_ranges)
        return value

    current_value = st.session_state[feature_name]
    if feature_name in FLOAT_FIELDS:
        value = st.number_input(
            label=feature_name,
            min_value=0.0,
            value=float(current_value),
            step=1.0,
            format="%.2f",
            key=feature_name,
        )
        render_helper_text(feature_name, helper_ranges)
        return value

    value = st.number_input(
        label=feature_name,
        min_value=0,
        value=int(current_value),
        step=1,
        format="%d",
        key=feature_name,
    )
    render_helper_text(feature_name, helper_ranges)
    return value


def build_input_frame(
    feature_columns,
    user_inputs,
    categorical_defaults,
    categorical_options,
    randomize_hidden,
    random_seed,
):
    # Start with every model feature set to zero.
    row = {feature: 0 for feature in feature_columns}

    # Fill hidden categorical fields with values that the trained encoder understands.
    row.update(categorical_defaults)

    if randomize_hidden:
        row = randomize_hidden_features(
            row, user_inputs, feature_columns, categorical_options, random_seed
        )

    # Overwrite only the fields entered by the user.
    for feature, value in user_inputs.items():
        if feature in row:
            row[feature] = value

    # Recompute engineered fields from the final row so derived values stay consistent
    # with the exact base features being sent to the model.
    row = add_derived_features(row)

    # Create a single-row DataFrame with the exact feature order used in training.
    return pd.DataFrame([row], columns=feature_columns)


def get_positive_probability(model, transformed_data):
    # Most scikit-learn classifiers expose predict_proba for class probabilities.
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(transformed_data)[0][1])

    # Simple fallback in case the saved model only supports predict.
    return float(model.predict(transformed_data)[0])


st.set_page_config(page_title="Purchase Likelihood Demo")
st.title("Purchase Likelihood Demo")
st.write(
    "This demo estimates how likely a customer is to purchase a product. "
    "It is intended for model demonstration only."
)

try:
    pipeline, logistic_model, gb_model, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

try:
    demo_presets = build_demo_presets(tuple(feature_columns))
except Exception as exc:
    demo_presets = None
    st.warning(f"Could not load demo presets from the training dataset: {exc}")

categorical_defaults, categorical_options = get_model_categorical_info(logistic_model)
helper_ranges = load_helper_ranges()
initialize_form_state(categorical_defaults)

if pipeline is None:
    st.info(
        "No separate preprocessing pipeline file was found. "
        "The saved model files already include preprocessing, so the app will use them directly."
    )

st.subheader("Enter Sample Feature Values")
st.caption(
    "Models loaded:\n"
    "- Logistic Regression baseline\n"
    "- Final Gradient Boosting deployment model"
)
st.caption(
    "These presets come from real dataset rows so the demo stays aligned with the model's "
    "training distribution."
)
st.caption("Derived flags and rate features are calculated automatically from the visible inputs.")

if demo_presets is not None:
    st.markdown("**Demo Presets**")
    preset_columns = st.columns(len(demo_presets))
    for column, preset in zip(preset_columns, demo_presets.to_dict("records")):
        with column:
            if st.button(preset["preset_label"], key=f"preset_{preset['preset_label']}"):
                apply_preset_row(preset)

    selected_preset_label = st.session_state.get("selected_preset_label")
    if selected_preset_label:
        st.info(
            f"Loaded preset: {selected_preset_label} | "
            f"Actual willPurchase: {st.session_state.get('selected_preset_target')} | "
            f"GB score on sampled row: {format_probability(st.session_state.get('selected_preset_probability', 0.0))}"
        )
        st.caption(st.session_state.get("selected_preset_reference", ""))

    with st.expander("Show preset table"):
        preset_summary = demo_presets[
            ["preset_label", "willPurchase", "gb_probability", "userId", "productId", "referenceTimeUtc"]
        ].copy()
        preset_summary["gb_probability"] = preset_summary["gb_probability"].map(format_probability)
        st.dataframe(preset_summary, hide_index=True, width="stretch")

user_inputs = {}
for section_title, fields in FORM_SECTIONS:
    st.markdown(f"**{section_title}**")
    columns = st.columns(2)
    for index, feature_name in enumerate(fields):
        with columns[index % 2]:
            user_inputs[feature_name] = render_feature_input(
                feature_name, categorical_defaults, categorical_options, helper_ranges
            )

with st.expander("Demo Controls"):
    randomize_hidden = st.checkbox("Randomize hidden features", value=False)
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1, format="%d")
    prediction_threshold = st.slider(
        "Prediction threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

if randomize_hidden:
    st.warning(
        "Warning: Hidden features are being randomized for demonstration only. "
        "Predictions may vary even with the same visible inputs."
    )
else:
    st.info(
        "Info: Hidden features are filled with deterministic defaults for a stable demo. "
        "In a production system, all features would come from the full data pipeline."
    )

if st.button("Predict Purchase Likelihood"):
    # Build a full feature row, filling unseen fields with zeros.
    input_df = build_input_frame(
        feature_columns,
        dict(user_inputs),
        categorical_defaults,
        categorical_options,
        randomize_hidden,
        random_seed,
    )

    # Avoid double preprocessing when the saved model already contains its own prep step.
    if pipeline is not None and not model_has_embedded_preprocessing(logistic_model):
        transformed_df = pipeline.transform(input_df)
    else:
        transformed_df = input_df

    non_zero_feature_count = count_non_zero_features(input_df)

    try:
        # Score both models.
        logistic_probability = get_positive_probability(logistic_model, transformed_df)
        gb_probability = get_positive_probability(gb_model, transformed_df)
    except Exception as exc:
        st.error(
            "Prediction failed. This usually means the saved models expect a separate "
            f"preprocessing pipeline or different feature names. Details: {exc}"
        )
        st.stop()

    # Use the GB score as the deployment decision; show the average only for comparison.
    average_probability = (logistic_probability + gb_probability) / 2
    final_prediction = (
        "Likely Purchase" if gb_probability >= prediction_threshold else "Not Likely"
    )

    st.subheader("Scenario Summary")
    scenario_summary = pd.DataFrame(
        [
            {
                "user_view_count_30d": user_inputs["user_view_count_30d"],
                "user_cart_count_30d": user_inputs["user_cart_count_30d"],
                "product_purchase_count_30d": user_inputs["product_purchase_count_30d"],
                "hours_since_last_user_event": user_inputs["hours_since_last_user_event"],
                "hours_since_last_user_product_event": user_inputs[
                    "hours_since_last_user_product_event"
                ],
                "price": user_inputs["price"],
                "currentStock": user_inputs["currentStock"],
            }
        ]
    )
    st.dataframe(scenario_summary, hide_index=True, width="stretch")

    st.subheader("Prediction Results")
    st.caption("Deployment decision uses the Final Gradient Boosting deployment model.")
    st.write("Raw GB probability:", f"{gb_probability:.12e}")
    st.write("Raw LR probability:", f"{logistic_probability:.12e}")
    st.write("Non-zero feature count:", non_zero_feature_count)
    primary_col, secondary_col = st.columns(2)
    with primary_col:
        st.metric("Gradient Boosting Deployment Probability", format_probability(gb_probability))
        st.metric("Final Prediction (GB model)", final_prediction)
    with secondary_col:
        st.metric("Logistic Regression Baseline Probability", format_probability(logistic_probability))

    st.caption("Comparison only")
    st.metric("Average Probability (comparison only)", format_probability(average_probability))
    st.caption(
        "The model outputs a probability score between 0 and 1. Because purchase events are rare in "
        "the training data, many predictions may appear as small percentages even when the model is "
        "working correctly."
    )

    with st.expander("Show model input row"):
        st.dataframe(input_df)

st.subheader("What this means")
st.write(
    "The prediction gives an estimated purchase likelihood that can support "
    "product recommendation demos or stakeholder presentations."
)
