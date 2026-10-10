# GlucoTwin AI - Digital Twin for Type 2 Diabetes

## 1. Team & College Information
- **Team Name:** CodeHurry
- **College/Incubator:** Birla Institute Of Technology Mesra
- **Team Members:** Arsh Srivastava (Team Leader), Bhumika Shree

## 2. Project Title
GlucoTwin AI: An AI-Powered Digital Twin for Type 2 Diabetes Monitoring and Glucose Spike Prediction

## 3. Problem Statement & Healthcare Use Case
**Problem:** Managing Type 2 Diabetes requires constant monitoring and anticipation of how lifestyle choices (meals, activity, hydration) affect blood glucose levels. Unpredicted glucose spikes (hyperglycemia) can lead to severe long-term complications.
**Use Case:** GlucoTwin AI acts as a patient-specific Digital Twin that fuses clinical health records with time-series Continuous Glucose Monitor (CGM) and lifestyle data. It utilizes machine learning to predict potential glucose spikes up to two hours in advance, allowing for proactive, preventative action by both the patient and physician.

## 4. Technical Stack
- **Frontend/Dashboard:** Streamlit, Plotly (for interactive Radar & Time-Series charts)
- **Backend/Data Processing:** Python, Pandas, NumPy
- **AI/ML:** Scikit-learn (Random Forest Classifier) for time-series forecasting and spike prediction.
- **Reporting:** `fpdf2` and `kaleido` for automated, offline PDF clinical report generation.

## 5. Demonstration Video
[Link to Live Demonstration Video]

## 6. Open-Source License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Architecture Diagram
The complete system architecture diagram and data flow explanation can be found in the repository here:
📄 **[View Architecture Diagram PDF](./GlucoTwin_Architecture.pdf)**

## 8. Project Presentation
*(Please upload your final Presentation PDF/PPT to this repository and link it here)*

## 9. Instructions for Running the Application

### Prerequisites
Make sure you have Python 3.8+ installed.

### Setup
1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/arshsri0513/GlucoTwin-AI-Personalized-Digital-Twin.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate synthetic training data:
   ```bash
   python data_generator.py
   ```
4. Train the ML predictive model:
   ```bash
   python train_model.py
   ```
5. Run the interactive Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

## 10. Data Source, ML Evaluation, & Limitations
- **Data Source:** Due to healthcare privacy regulations, all data is purely synthetic. Base datasets are generated via `data_generator.py` simulating 24-hour CGM readings, meal carbohydrate intake, and physical activity intensity. The web UI also includes an on-the-fly synthetic data generator to match dynamic clinical inputs (HbA1c, current glucose).
- **Preprocessing:** Time-series feature engineering includes time-lagged glucose readings (up to 1 hour prior), glucose rate-of-change (ROC), rolling averages, and lagged lifestyle indicators (carbs/activity).
- **Model:** Scikit-Learn Random Forest Classifier trained to predict a binary outcome: whether glucose will exceed 180 mg/dL in the subsequent 2-hour window.
- **Evaluation:** The model achieves high accuracy on the synthetic validation set. Real-world deployment would require fine-tuning on IRB-approved clinical CGM datasets.
- **Advanced Features:** 
  - **Automated PDF Reporting** (`fpdf2`, `kaleido`) embedding live Plotly charts.
  - **Physiological What-If Engine:** Simulates the effects of Carbs, Activity, and Hydration (Renal Filtration) on predicted glucose peaks.
- **Limitations:** This is a proof-of-concept relying on synthetic statistical heuristics rather than true physiological metabolic differential equations. Not intended for actual medical use.
