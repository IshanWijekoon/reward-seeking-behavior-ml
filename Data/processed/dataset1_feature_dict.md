# Feature dictionary — Dataset 1 (Smartphone Usage & Addiction)

- Rows: **7500**
- Columns: **13** (including target)
- Target: `risk_level` ∈ {Low, Moderate, High}

## Columns

| Column | dtype | role | transform / notes |
|--------|-------|------|-------------------|
| `age` | int64 | feature |  |
| `daily_screen_time_hours` | float64 | feature |  |
| `social_media_hours` | float64 | feature |  |
| `gaming_hours` | float64 | feature |  |
| `work_study_hours` | float64 | feature |  |
| `sleep_hours` | float64 | feature |  |
| `notifications_per_day` | int64 | feature |  |
| `app_opens_per_day` | int64 | feature |  |
| `risk_level` | str | target |  |
| `stress_level_ord` | int64 | feature | ordinal encoding |
| `gender_Male` | int64 | feature | one-hot |
| `gender_Other` | int64 | feature | one-hot |
| `reward_app_share` | float64 | feature | engineered ratio |

## Preparation notes

- Source: Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
- risk_level from addiction_level: NaN/Mild→Low, Moderate→Moderate, Severe→High
- Dropped IDs, academic_work_impact, addiction_level, addicted_label
- Unscaled features; apply scaling inside future LR/SVM/NN pipelines only
- Do not apply SMOTE to this file before train/test split