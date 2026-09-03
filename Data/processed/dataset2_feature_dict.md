# Feature dictionary — Dataset 2 (Mobile Addiction Data)

- Rows: **904**
- Columns: **26** (including target)
- Target: `risk_level` ∈ {Low, Moderate, High}

## Columns

| Column | dtype | role | transform / notes |
|--------|-------|------|-------------------|
| `Age` | int64 | feature |  |
| `Income_USD` | float64 | feature |  |
| `Daily_Screen_Time_Hours` | float64 | feature |  |
| `Phone_Unlocks_Per_Day` | float64 | feature |  |
| `Social_Media_Usage_Hours` | float64 | feature |  |
| `Gaming_Usage_Hours` | float64 | feature |  |
| `Streaming_Usage_Hours` | float64 | feature |  |
| `Messaging_Usage_Hours` | float64 | feature |  |
| `Work_Related_Usage_Hours` | float64 | feature |  |
| `Sleep_Hours` | float64 | feature |  |
| `Physical_Activity_Hours` | float64 | feature |  |
| `Stress_Level` | float64 | feature |  |
| `Time_Spent_With_Family_Hours` | float64 | feature |  |
| `Online_Shopping_Hours` | float64 | feature |  |
| `Has_Screen_Time_Management_App` | int64 | feature | binary yes/no |
| `Monthly_Data_Usage_GB` | float64 | feature |  |
| `Has_Night_Mode_On` | int64 | feature | binary yes/no |
| `Age_First_Phone` | float64 | feature |  |
| `Push_Notifications_Per_Day` | float64 | feature |  |
| `Tech_Savviness_Score` | float64 | feature |  |
| `risk_level` | str | target |  |
| `Education_Level_ord` | int64 | feature | ordinal encoding |
| `Gender_Male` | int64 | feature | one-hot |
| `Gender_Other` | int64 | feature | one-hot |
| `Urban_Urban` | int64 | feature | one-hot |
| `reward_app_share` | float64 | feature | engineered ratio |

## Preparation notes

- Source: mobile_addiction_data.csv
- Filtered to Age 18–35
- risk_level from Self_Reported_Addiction_Level with High+Severe→High
- Negatives set to NaN then median-imputed (re-fit medians on train in modelling)
- Education_Level missing → Unknown; ordinal encoded
- Dropped clinical scores, brand/connection, high-cardinality socio fields
- Unscaled; no SMOTE in this artifact