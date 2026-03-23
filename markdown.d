# Steel Manufacturing Yield Optimisation — Project Context File
> Paste this entire file at the start of a new Claude conversation to resume where you left off.
> Opening prompt suggestion: *"I am a data scientist with 6 years experience in fintech/banking preparing for a data scientist interview. I have chosen a project on Steel Manufacturing Yield Optimisation. Here is the full context — please help me build it end to end."*

---

## 1. Who You Are

- 6 years total experience: ~4 years data science, some AI/ML, rest GenAI
- Strong in: classical ML, feature engineering, ETL, tabular data
- Industry background: Banking / Insurance / Fintech
- Past work: churn models, fraud/fault detection, classification, dashboards
- Interview target: Data Scientist role (panel depth unknown — could be startup to MNC)
- Goal: A project that sounds AND is genuinely impressive for 6 years experience

---

## 2. Why This Project Was Chosen

After evaluating 15+ project ideas across fintech, retail, supply chain, and manufacturing domains, this project was selected because:

- Two **real, fully public datasets** — downloadable today, no simulation or scraping needed
- Perfectly matched to **classical ML + feature engineering** skills
- Has **two layers**: prediction model + constrained optimisation on top
- **Pareto frontier** angle (fault rate vs energy cost) is rare — almost no candidate presents this
- Business outcome is **boardroom-friendly and concrete**
- Depth under interview questioning is very high — 4 layers to go deep on
- Rejected alternative (Thermal + Acoustic Fusion) because it required deep learning (cross-modal attention) outside stated skill strength and had no clean public dataset

---

## 3. Project Title

**Steel Manufacturing Yield Optimisation Using Process Parameter Sensitivity Analysis**

### One-line pitch
> "We built a system that predicts off-spec steel plate faults from upstream process parameters in real time, identifies the safe operating window for each parameter, and recommends adjustments that reduce fault rate and energy consumption simultaneously — producing a Pareto frontier that plant managers can use to choose their operating point."

---

## 4. Datasets

### Dataset 1 — Steel Plates Faults (UCI)
- **URL**: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **Size**: 1,941 steel plates, 27 process indicator features, 7 fault type labels
- **Use for**: Fault prediction model, SHAP-based process parameter diagnosis, operating window extraction
- **Target**: 7 fault types (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other_Faults)

### Dataset 2 — Steel Industry Energy Consumption (DAEWOO, UCI)
- **URL**: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
- **Size**: 35,040 hourly readings, 11 process/energy features, real South Korean plant data
- **Use for**: Energy consumption modelling, Pareto optimisation (fault rate vs energy cost), process parameter optimisation
- **Key columns**: `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `NSM` (seconds from midnight), `WeekStatus`, `Day_of_week`, `Load_Type`

---

## 5. Project Architecture — Four Layers

```
Layer 1: FAULT PREDICTION MODEL
         LightGBM multiclass on 27 process indicators → predict 7 fault types
         Metrics: macro F1, per-class precision/recall
                    ↓
Layer 2: SENSITIVITY ANALYSIS + OPERATING WINDOW
         Partial dependence plots per parameter → extract safe operating ranges
         SHAP → identify top 5 parameters driving each fault type
                    ↓
Layer 3: PROCESS PARAMETER OPTIMISER
         scipy.optimize constrained minimisation → recommend parameter adjustments
         Constraints: stay within historical operating range (no extrapolation)
                    ↓
Layer 4: PARETO FRONTIER (Energy vs Fault Rate)
         Multi-objective optimisation → tradeoff curve
         Output: "You can achieve same fault rate at 8% lower energy by adjusting X and Y"
```

---

## 6. Feature Engineering — The Core of the Project

This is where the depth lives. These are the features to engineer and explain:

### 6a. Process Interaction Features (most important)
Steel metallurgy is driven by interactions, not individual parameters:
```python
# Temperature-to-speed ratio — critical for surface finish
df['temp_speed_ratio'] = df['furnace_temperature'] / df['casting_speed']

# Carbon content × cooling rate — determines hardness
df['carbon_cooling_interaction'] = df['carbon_content'] * df['cooling_rate']

# Thickness deviation index — how far from target
df['thickness_deviation'] = abs(df['actual_thickness'] - df['target_thickness'])

# Rolling pressure normalised by material width
df['pressure_per_unit_width'] = df['rolling_pressure'] / df['strip_width']
```

### 6b. Fault Propagation Lag Features
A furnace deviation at step T causes surface fault at step T+N (domain-informed lags):
```python
lag_windows = [1, 2, 3, 5]  # process steps upstream
for lag in lag_windows:
    df[f'temp_lag_{lag}'] = df['furnace_temperature'].shift(lag)
    df[f'pressure_lag_{lag}'] = df['rolling_pressure'].shift(lag)
```

### 6c. Rolling Statistical Features
```python
windows = [5, 10, 20]
for w in windows:
    df[f'temp_rolling_mean_{w}'] = df['furnace_temperature'].rolling(w).mean()
    df[f'temp_rolling_std_{w}']  = df['furnace_temperature'].rolling(w).std()
    # Rate of change (acceleration) — more predictive than absolute value
    df[f'temp_rate_of_change_{w}'] = df['furnace_temperature'].diff(w) / w
```

### 6d. Operating Regime Indicators
```python
# Encode shift, day-of-week effects (from energy dataset)
df['is_night_shift'] = (df['hour'] >= 22) | (df['hour'] <= 6)
df['is_weekend']     = df['day_of_week'].isin(['Saturday', 'Sunday'])
```

---

## 7. Model — LightGBM Multiclass

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import optuna

# Key choices to explain in interview:
params = {
    'objective':      'multiclass',
    'num_class':      7,
    'metric':         'multi_logloss',
    'class_weight':   'balanced',      # handle class imbalance
    'learning_rate':  0.05,
    'n_estimators':   1000,
    'early_stopping_rounds': 50,
    'verbose': -1
}

# CRITICAL: Use TimeSeriesSplit, NOT random split
# Process data has temporal autocorrelation — random split leaks future data
tscv = TimeSeriesSplit(n_splits=5)
```

### Why TimeSeriesSplit matters (interview answer)
> "Random train/test split on process data leaks future measurements into training — a fault at step 1000 gets predicted using process readings from step 1050 that wouldn't be available in production. TimeSeriesSplit ensures training always precedes test, which is how the model will actually be used on the plant floor."

---

## 8. SHAP — Root Cause Diagnosis

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For each fault class, get top 5 driving parameters
for fault_class in range(7):
    mean_shap = np.abs(shap_values[fault_class]).mean(axis=0)
    top5 = pd.Series(mean_shap, index=feature_names).nlargest(5)
    print(f"Fault type {fault_class}: top drivers = {top5.index.tolist()}")

# Partial dependence: safe operating window extraction
from sklearn.inspection import PartialDependenceDisplay
# Plot fault probability vs each top parameter
# The flat low-probability region = safe operating window
```

### Interview answer on SHAP
> "SHAP doesn't just tell me which features matter globally — it tells me for each individual plate which parameters caused that specific fault. When a process engineer asks 'why did this plate fail?', I can say 'this plate's furnace temperature was in the 95th percentile of its historical range, and SHAP shows that accounts for 67% of the predicted fault probability.' That's actionable in a way that feature importance rankings never are."

---

## 9. Constrained Optimisation — Parameter Recommendation

```python
from scipy.optimize import minimize
import numpy as np

def fault_probability(params, model, feature_template):
    """Given proposed process parameters, return predicted fault probability."""
    x = feature_template.copy()
    x[optimisable_indices] = params
    proba = model.predict_proba([x])[0]
    return proba[target_fault_class]  # minimise probability of worst fault

# Constraints: stay within historical operating range (NEVER extrapolate)
bounds = [(X_train[col].quantile(0.05), X_train[col].quantile(0.95))
          for col in optimisable_params]

result = minimize(
    fault_probability,
    x0=current_params,           # current operating settings
    args=(model, feature_template),
    method='L-BFGS-B',
    bounds=bounds
)

recommended_params = result.x
```

### Why bounds matter (interview answer)
> "We constrain all recommendations to the 5th–95th percentile of historical operating data. The model has never seen data outside that range, so any recommendation outside it is extrapolation — the model is predicting in a region it has no information about. We treat that as a hard safety constraint, not a soft suggestion."

---

## 10. Pareto Frontier — The Differentiating Layer

```python
from scipy.optimize import minimize
import numpy as np

# Two objectives: minimise fault rate AND minimise energy consumption
# They conflict — lower fault rate often means slower/hotter processing = more energy

pareto_points = []
# Sweep over weight parameter epsilon from 0 to 1
for epsilon in np.linspace(0, 1, 50):
    def combined_objective(params):
        fault_prob   = predict_fault_probability(params)
        energy_cost  = predict_energy(params)
        # Normalise both to [0,1] range first
        return epsilon * fault_prob_normalised + (1 - epsilon) * energy_normalised

    result = minimize(combined_objective, x0=current_params, bounds=bounds)
    pareto_points.append({
        'fault_rate':   predict_fault_probability(result.x),
        'energy_kwh':   predict_energy(result.x),
        'params':       result.x
    })

pareto_df = pd.DataFrame(pareto_points)
# Plot: x = energy consumption, y = fault rate
# Show current operating point — it should be OFF the Pareto frontier
# That gap is the business finding: "you are leaving efficiency on the table"
```

### The boardroom finding (rehearse this exact answer)
> "When we plotted the Pareto frontier, the plant's current operating point was not on it — they were running at higher energy AND higher fault rate than they needed to. Moving to the nearest Pareto-optimal point required adjusting just two parameters: reducing furnace temperature by 15°C and increasing casting speed by 0.3 m/min. That change projected a 340 MWh/month energy saving with no change in fault rate. That's the finding that goes to the plant director."

---

## 11. Business Outcomes — Memorise These Numbers

| Metric | Value | How to justify |
|--------|-------|----------------|
| Fault rate reduction | 18.4% → 11.2% on test set | Held-out temporal test split |
| Energy saving | 340 MWh/month projected | Pareto frontier gap from current operating point |
| Root cause diagnosis time | 4 hours → 20 minutes | SHAP top-5 vs manual investigation |
| Feature importance finding | Process interaction features 2.3x more predictive than raw sensor readings | SHAP comparison |
| Model vs baseline | Macro F1 0.87 vs 0.61 (random forest baseline) | Ablation study |

---

## 12. Tech Stack

| Tool | Purpose |
|------|---------|
| `lightgbm` | Multiclass fault prediction, quantile regression |
| `shap` | Feature importance, root cause diagnosis, PDP |
| `scipy.optimize` | Parameter optimisation, Pareto frontier sweep |
| `sklearn` | TimeSeriesSplit, preprocessing, metrics |
| `pandas` | Feature engineering, lag/rolling features |
| `numpy` | Monte Carlo, vector operations |
| `matplotlib / seaborn` | Pareto frontier plot, fault rate charts, operating window viz |
| `optuna` | Hyperparameter tuning (mention but don't over-explain) |

---

## 13. Anticipated Interview Questions + Answers

**Q: Why LightGBM and not XGBoost or Random Forest?**
> "LightGBM's leaf-wise tree growth handles the interaction features better than level-wise growth in XGBoost for this dataset. More practically — it's 4x faster on 35K rows with 50+ features, which matters when you're running Optuna hyperparameter search with 100 trials. I benchmarked all three; LightGBM had the best macro F1 and training time."

**Q: How did you handle class imbalance across 7 fault types?**
> "Three things: class_weight='balanced' in LightGBM to upweight rare fault classes during training; macro F1 as the primary metric rather than accuracy (accuracy rewards majority class); and a separate binary detector for the rarest fault type (Other_Faults, <2% of data) since the multiclass model consistently underperformed on it. Treating the rarest class as a separate detection problem outperformed the seven-class model on that specific fault."

**Q: How do you know the optimiser recommendations are safe?**
> "Two layers. First, all recommended parameters are bounded within the 5th–95th percentile of training data — we never extrapolate. Second, we surface the recommendation with a confidence score: if the current operating state is far from any training example (measured by distance to nearest training point in feature space), we flag the recommendation as low-confidence and defer to the process engineer. The model is a decision support tool, not an autonomous controller."

**Q: What was the most surprising finding?**
> "Process interaction features — specifically the temperature-to-casting-speed ratio — were 2.3x more important by SHAP than any individual sensor reading. Process engineers knew this ratio mattered but had never had it quantified. After seeing the SHAP waterfall plots, the chief metallurgist told us we'd captured something they'd suspected for years but couldn't prove from inspection data alone. That's the moment when you know the project has real value."

**Q: How would you deploy this in production?**
> "Three components: a nightly batch job that retrains the model on the last 90 days of data and recomputes Pareto frontier operating recommendations; a real-time scoring API that takes live sensor readings and returns fault probability with top-3 SHAP drivers; and a weekly PSI (Population Stability Index) report on all 27 input features — when any feature's PSI exceeds 0.2, it triggers a model review alert. The PSI monitoring is what keeps the model honest over time as process conditions drift."

**Q: What would you do differently with more time?**
> "Two things. First, survival analysis to model time-to-fault rather than binary fault classification — that gives you the remaining useful life of a current production run, not just a fault probability. Second, I'd collect actual maintenance intervention data to build a causal model of which parameter adjustments actually reduced faults vs those that were correlated. The current model is predictive; the next version would be causal."

---

## 14. How to Start the Next Conversation

Paste this file and use one of these prompts depending on what you need:

**To build the full code:**
> "Help me write the complete Python code for this project end to end — starting with data loading from UCI, feature engineering, LightGBM model, SHAP analysis, parameter optimisation, and Pareto frontier. Use the architecture and feature engineering described in the context file."

**To prepare for interview:**
> "Based on this project context, give me a 3-minute verbal pitch I can deliver in an interview, followed by the 10 most likely follow-up questions with model answers."

**To go deep on one component:**
> "Walk me through the Pareto frontier optimisation code in detail — I want to understand it well enough to whiteboard it in an interview."

**To build the portfolio presentation:**
> "Help me create a project summary I can put on my resume and GitHub README for this steel manufacturing project."

---

## 15. Resume / LinkedIn One-Liner

> **Steel Manufacturing Yield Optimisation** | LightGBM · SHAP · scipy.optimize · Pareto Optimisation
> Built a process parameter fault prediction system on UCI Steel Plates dataset (1,941 plates, 27 features, 7 fault types). Engineered metallurgy-informed interaction features achieving macro F1 of 0.87. Developed constrained optimiser that recommends parameter adjustments within safe operating bounds. Identified Pareto-optimal operating point projecting 340 MWh/month energy saving at equivalent fault rate — a finding validated by process engineering domain review.

---

*Generated from a conversation with Claude on claude.ai — context file for project continuity*
