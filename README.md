# 🧬 GlucoTwin AI

## 1. Team details & College/Incubator information
- **Team Name:** CodeHurry
- **College/Incubator:** Birla Institute Of Technology Mesra
- **Team Members:** Arsh Srivastava (Team Leader), Bhumika Shree

## 2. Project Title
GlucoTwin AI: A Personalized Digital Twin for Type 2 Diabetes

## 3. Problem Statement & Healthcare Use Case
**Problem Statement:** Managing Type 2 Diabetes requires constant monitoring and anticipation of how lifestyle choices (meals, activity, hydration) affect blood glucose levels. Unpredicted glucose spikes (hyperglycemia) can lead to severe long-term complications.
**Healthcare Use Case:** GlucoTwin AI acts as a patient-specific Digital Twin that fuses clinical health records with time-series Continuous Glucose Monitor (CGM) and lifestyle data. It allows for proactive, preventative action by both the patient and physician.

## 4. Technical Stack
- **Frontend/Dashboard:** Streamlit, Plotly (for interactive Radar & Time-Series charts)
- **Backend/Data Processing:** Python, Pandas, NumPy
- **Reporting:** `fpdf2` and `kaleido` for automated, offline PDF clinical report generation.

## 5. AI/ML Model or Framework Details
- **Framework:** Scikit-Learn (Random Forest Classifier).
- **Details:** The model extracts time-series features (time-lagged glucose readings up to 1 hour prior, glucose rate-of-change, rolling averages, and lagged lifestyle indicators) to predict a binary outcome: whether glucose will exceed 180 mg/dL in the subsequent 2-hour window.
- **Data Source:** Due to healthcare privacy regulations, the base data is purely synthetic (generated via `data_generator.py`).

## 6. Demonstration Video
*(Note: 15-20 minute Demo Video is uploaded and the link is added below)*
📺 **[View Demonstration Video](https://go.screenpal.com/watch/cO66IOnx9ae)**

## 7. Open-source License Details
This project is publicly licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 8. Architecture Diagram in PDF/PPT format
The complete system architecture diagram is included in the repository.
📄 **[View Architecture Diagram PDF](./GlucoTwin_Architecture.pdf)**

## 9. Presentation in PDF/PPT format covering project details and outcomes
The final project presentation covering details and outcomes is included in the repository.
📄 **[View Presentation PDF](./GlucoTwin_Presentation.pdf)**

## 10. Public Accessibility & Instructions (All files publicly accessible)
**Live Application Link:** https://glucotwin-ai-personalized-digital-twin.streamlit.app/

### Instructions to run locally
1. Clone the repository: `git clone https://github.com/arshsri0513/GlucoTwin-AI-Personalized-Digital-Twin.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Dashboard: `streamlit run app.py`
