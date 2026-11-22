# AutismCare: Autism Diagnosis Age Predictor

## Mission and problem
**Mission:** help parents of autistic children get an early diagnosis and connect them to qualified professionals and therapists.
**Problem:** many autistic children are diagnosed years after the first signs and miss early intervention. This project predicts a
child's **age at autism diagnosis (in years)** from their co-occurring conditions, autism severity and family characteristics, so
outreach teams can contact the families most at risk of a late diagnosis first.

## Dataset
The **US National Survey of Children's Health (NSCH)**, a yearly survey of parents run by the US Census Bureau. We use the 2022 and 2024
public-use files and keep the **3,975 children whose parents were told they have autism** and reported the age of diagnosis. The target is
`diagnosis_age` (years), and the model uses 10 features: 1 numeric (family income as % of the poverty line) and 9 categorical (autism severity;
ADHD, anxiety, developmental delay, speech disorder and intellectual disability diagnoses; sex; insurance type; home language).
Source: [US Census Bureau, NSCH datasets](https://www.census.gov/programs-surveys/nsch/data/datasets.html)
(files: [2022](https://www2.census.gov/programs-surveys/nsch/datasets/2022/nsch_2022_topical_Stata.zip),
[2024](https://www2.census.gov/programs-surveys/nsch/datasets/2024/nsch_2024_topical_Stata.zip)).

## API
Swagger UI: **https://autism-diagnosis-age-api.onrender.com/docs**

The service is on Render's free plan, so the first request after a quiet period can take up to a minute while it wakes up.

## Video demo
YouTube: **https://youtu.be/X2_keELZejs**

## How to run the mobile app
1. Install the [Flutter SDK](https://docs.flutter.dev/get-started/install) and Android Studio (for the Android SDK and emulator).
2. Start an Android emulator (Android Studio > Device Manager > create and start a virtual device), **or** connect an Android phone with USB debugging turned on.
3. Clone the repository and open the app folder:
   ```bash
   git clone https://github.com/pmganga/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
   ```
4. Install the packages: `flutter pub get`
5. Check that Flutter sees the device: `flutter devices`
6. Run the app: `flutter run`
7. Fill in the 10 fields (the allowed values are shown under each field) and tap **Predict**. The predicted age at diagnosis appears below the button,
   or a message explaining which value is missing, has the wrong type or is out of range.

---

## Repository structure
```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml, uv.lock        # uv project (Python 3.12)
    ├── linear_regression/
    │   └── multivariate.ipynb         # data preparation, 4 models, tuning, loss curve, best model
    ├── API/
    │   ├── prediction.py              # FastAPI: /predict, /retrain, /docs
    │   ├── requirements.txt           # exported from uv for Render
    │   └── best_model.pkl             # random forest pipeline (preprocessing + model)
    └── FlutterApp/                    # one-page Flutter mobile app
```

## Running the notebook and API locally
```bash
cd summative
uv sync                                   # create the virtual environment from uv.lock
# Notebook: open linear_regression/multivariate.ipynb in VS Code and select the .venv Python as the kernel
cd API && uv run uvicorn prediction:app --reload   # then open http://127.0.0.1:8000/docs
```
