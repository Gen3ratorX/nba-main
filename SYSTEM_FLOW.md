# NBA Prediction System - Complete Flow

## High-Level Flow

```
1. DATA COLLECTION → 2. FEATURE ENGINEERING → 3. MODEL TRAINING → 4. PREDICTIONS → 5. TRACKING
```

## Detailed Flow

### PHASE 1: Data Collection
**File:** `fetch_data.py`  
**Class:** `NBADataCollector` / `FreeNBADataCollector`

```
Fetch from APIs:
├── Historical games → data/historical/historical_games_balldontlie.csv
├── Current season → data/current/current_season_espn_YYYYMMDD.csv
└── Upcoming games → data/upcoming/upcoming_games.csv
```

**Data Sources:**
- ESPN Scoreboard API (current/upcoming)
- BallDontLie API (historical)
- NBA Stats API (backup)

---

### PHASE 2: Feature Engineering
**File:** `data_processor.py`  
**Class:** `NBADataProcessor`

```
Input: CSV files (historical + current)
    ↓
Process games chronologically:
    ├── Update ELO ratings (rolling)
    ├── Calculate form (last 5 games win %)
    ├── Calculate rest days
    ├── Calculate advanced metrics (last 10 games):
    │   ├── Offensive rating (points per 100 possessions)
    │   ├── Defensive rating (points allowed per 100)
    │   ├── Net rating (off - def)
    │   ├── Pace (possessions per game)
    │   └── Win streak
    └── Situational features:
        ├── Back-to-back indicator
        ├── Head-to-head record
        └── Home court advantage
    ↓
Output: DataFrame with 28 features per game
```

**Key Principle:** Features use only past data (no future data leakage)

---

### PHASE 3: Model Training

#### 3A. Moneyline Model (Win/Loss)
**File:** `main.py` → `model_utils.py`  
**Target:** `home_won` (binary: 1 if home wins, 0 if away wins)

```
Features DataFrame
    ↓
Time-series split (80% train, 20% test)
    ↓
Train ensemble:
    ├── RandomForest (300 trees)
    ├── GradientBoosting (150 trees)
    ├── XGBoost (200 trees) - optional
    └── LogisticRegression
    ↓
Voting ensemble (weighted average)
    ↓
Calibrate probabilities (fix overconfidence)
    ↓
Save to models/:
    ├── nba_model_TIMESTAMP.pkl
    ├── scaler_TIMESTAMP.pkl
    └── features_TIMESTAMP.pkl
```

#### 3B. Totals Model (Over/Under)
**File:** `train_totals_model.py`  
**Target:** `total_points` = home_score + away_score

```
Features DataFrame + target (total_points)
    ↓
Time-series split (80% train, 20% test)
    ↓
Train regression models:
    ├── RandomForestRegressor
    └── GradientBoostingRegressor
    ↓
Average predictions
    ↓
Save to models/:
    ├── totals_rf_TIMESTAMP.pkl
    ├── totals_gb_TIMESTAMP.pkl
    ├── totals_scaler_TIMESTAMP.pkl
    └── totals_features_TIMESTAMP.pkl
```

---

### PHASE 4: Making Predictions
**File:** `predictions.py`  
**Class:** `UnifiedPredictor`

```
User runs: python predictions.py
    ↓
Load models via model_loader.py:
    ├── Load latest moneyline model
    └── Load latest totals model
    ↓
Get upcoming games (from data/upcoming/upcoming_games.csv)
    ↓
For each upcoming game:
    ├── Calculate features (using all historical data)
    │   ├── Current ELO ratings
    │   ├── Recent form
    │   ├── Advanced metrics
    │   └── Situational features
    │
    ├── Moneyline Prediction:
    │   ├── Scale features (using saved scaler)
    │   ├── Get win probabilities
    │   ├── Determine predicted winner
    │   └── Calculate bet size (Kelly Criterion)
    │
    └── Totals Prediction:
        ├── ML model prediction (if available)
        ├── OR formula-based: (pace/100) * ORTG
        └── Compare to typical line (220)
    ↓
Categorize games:
    ├── Confluence: Both ML + Totals have value
    ├── ML only: Just moneyline value
    └── Totals only: Just totals value
    ↓
Display ranked recommendations
```

---

### PHASE 5: Tracking Results
**File:** `update_results.py` → `performance_tracker.py`

```
User runs: python update_results.py --days 1
    ↓
Fetch recent game results (from data/current/)
    ↓
Load pending predictions (from data/performance_history.json)
    ↓
Match predictions to results:
    ├── Match by teams (normalized names)
    └── Match by date
    ↓
Record results:
    ├── Actual winner
    ├── Correct/incorrect
    ├── Profit/loss
    └── Update bankroll
    ↓
Save to data/performance_history.json
    ↓
Display performance report:
    ├── Win rate
    ├── ROI
    ├── Calibration analysis
    └── Alerts/warnings
```

---

## Complete End-to-End Example

**Scenario:** You want predictions for tomorrow's games

### Step 1: Data Collection (if needed)
```bash
python main.py --days-back 30 --days-ahead 7
```
**What happens:**
- Fetches historical games → `data/historical/`
- Fetches current season games → `data/current/`
- Fetches upcoming games → `data/upcoming/upcoming_games.csv`

### Step 2: Feature Engineering
**Automatically happens in `data_processor.py`:**
- Loads all CSV files
- Sorts games chronologically
- Calculates features for each game (using only past data)
- Returns DataFrame with features

### Step 3: Model Training (if needed)
```bash
python main.py --days-back 30
python train_totals_model.py
```
**What happens:**
- Moneyline: Trains ensemble on historical games
- Totals: Trains regression on historical totals
- Saves models to `models/` directory

### Step 4: Making Predictions
```bash
python predictions.py --tomorrow
```
**What happens:**
1. Loads models (via `model_loader.py`)
2. Gets tomorrow's games from `data/upcoming/`
3. Calculates features for each game
4. Makes predictions:
   - Moneyline: Win probabilities → bet recommendations
   - Totals: Expected total points → over/under recommendations
5. Finds confluence plays (both agree)
6. Displays ranked recommendations

### Step 5: Track Results (next day)
```bash
python update_results.py --days 1
```
**What happens:**
1. Fetches yesterday's results
2. Matches to your predictions
3. Records wins/losses
4. Calculates performance metrics
5. Updates tracking file

---

## Data Dependencies

```
CSV Files (raw data)
    ↓
Features DataFrame (processed)
    ↓
Trained Models (saved to disk)
    ↓
Predictions (in-memory)
    ↓
Performance History (JSON)
```

---

## Key Files & Their Roles

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `fetch_data.py` | Collect raw game data | APIs | CSV files |
| `data_processor.py` | Calculate features | CSV files | Features DataFrame |
| `model_utils.py` | Train moneyline model | Features DataFrame | Trained model files |
| `train_totals_model.py` | Train totals model | Features DataFrame | Trained model files |
| `model_loader.py` | Load saved models | Model files | Model objects |
| `predictions.py` | Make predictions | Models + upcoming games | Predictions display |
| `betting_strategy.py` | Calculate bet sizes | Win probabilities | Bet recommendations |
| `update_results.py` | Track actual results | Predictions + results | Performance stats |
| `performance_tracker.py` | Store/analyze performance | Prediction history | Performance reports |

---

## Weekly Cycle

**Sunday (Model Refresh):**
```
1. python main.py --days-back 60      # Retrain moneyline
2. python train_totals_model.py       # Retrain totals
3. python update_results.py --days 7 --export  # Review week
```

**Daily:**
```
Morning:  python predictions.py           # Get today's picks
Evening:  python update_results.py --days 1  # Track results
```

---

## Feature Calculation Order (Critical!)

**IMPORTANT:** Features must be calculated in chronological order to avoid data leakage.

```
Game 1 (2024-01-01): Uses no prior data (defaults)
Game 2 (2024-01-02): Uses Game 1 for features
Game 3 (2024-01-03): Uses Games 1-2 for features
...
Game N (2026-01-21): Uses Games 1-(N-1) for features
```

**Example for upcoming game:**
- ELO: Current rating based on ALL past games
- Form: Win rate in last 5 games
- Advanced metrics: Last 10 games' stats
- Rest: Days since last game
- H2H: Head-to-head record from past matchups

---

## Model Prediction Flow

**Moneyline:**
```
Game Features (28 features)
    ↓
StandardScaler (normalize)
    ↓
Ensemble Model (4 models voting)
    ↓
Calibrated Probabilities
    ↓
Win Probability (0-1)
    ↓
Kelly Criterion
    ↓
Bet Recommendation ($X, Y% edge)
```

**Totals:**
```
Game Features (12 totals-relevant features)
    ↓
StandardScaler (normalize)
    ↓
Regression Ensemble (RF + GB average)
    ↓
Predicted Total Points
    ↓
Compare to typical line (220)
    ↓
Over/Under Recommendation
```

---

## Data Storage Structure

```
data/
├── historical/
│   └── historical_games_balldontlie.csv  # Past seasons
├── current/
│   └── current_season_espn_YYYYMMDD.csv  # Recent games
├── upcoming/
│   └── upcoming_games.csv                # Future games
├── processed/
│   └── (intermediate processing files)
├── predictions/
│   └── (saved predictions)
└── performance_history.json              # Tracking results

models/
├── nba_model_TIMESTAMP.pkl              # Moneyline model
├── scaler_TIMESTAMP.pkl                 # Feature scaler
├── features_TIMESTAMP.pkl               # Feature names
├── totals_rf_TIMESTAMP.pkl              # Totals RF model
├── totals_gb_TIMESTAMP.pkl              # Totals GB model
├── totals_scaler_TIMESTAMP.pkl          # Totals scaler
└── totals_features_TIMESTAMP.pkl        # Totals features
```

---

## Error Handling & Edge Cases

**Missing Data:**
- Missing scores: Games skipped in processing
- Missing features: Default values used
- No model: Falls back to formula-based predictions

**Date Matching:**
- Predictions stored with game date
- Results matched by teams + date
- Handles date format variations

**Model Loading:**
- Loads latest model by timestamp
- Falls back gracefully if model missing
- Warns if model is outdated

---

## Performance Optimization

**Caching:**
- Models loaded once per session
- Features calculated once per game
- Data processed in batches

**Parallel Processing:**
- Model training uses n_jobs=-1 (all cores)
- Feature calculation is sequential (needs chronological order)

---

## Integration Points

```
main.py (full pipeline)
    ├──→ fetch_data.py
    ├──→ data_processor.py
    ├──→ model_utils.py
    └──→ predictions.py (via display method)

predictions.py (daily use)
    ├──→ model_loader.py
    ├──→ data_processor.py
    ├──→ betting_strategy.py
    └──→ performance_tracker.py

update_results.py
    ├──→ fetch_data.py
    ├──→ performance_tracker.py
    └──→ data/performance_history.json
```
