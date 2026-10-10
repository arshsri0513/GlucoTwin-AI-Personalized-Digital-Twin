import base64
import requests
from fpdf import FPDF
import os

def create_pdf():
    print("Starting PDF generation...")
    # 1. Prepare the Mermaid diagram code
    mermaid_code = """
    flowchart TD
        User([Patient / Doctor]) --> Form[Clinical Form]
        Form --> DataGen[Synthetic Data Generator]
        DataGen --> SessionState[(Session State)]
        WhatIf[What-If Sliders] --> Features[Feature Engineering]
        SessionState --> Features
        Features --> ML[Random Forest Model]
        ML --> UI[Interactive Dashboard]
        SessionState --> ReportGen[FPDF2 Report Engine]
        UI --> ReportGen
        ReportGen --> User
    """
    
    # 2. Encode to base64 for mermaid.ink API
    encoded_str = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{encoded_str}"
    
    print(f"Fetching diagram from API...")
    has_image = False
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open("arch_diagram_temp.png", "wb") as f:
                f.write(response.content)
            has_image = True
            print("Successfully downloaded diagram image.")
        else:
            print("Warning: Could not fetch image, falling back to text only.")
    except Exception as e:
        print(f"Warning: {e}")
        
    # 3. Create the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(13, 110, 253)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 18)
    pdf.cell(190, 15, " GlucoTwin AI: System Architecture", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    current_y = pdf.get_y()
    
    if has_image:
        pdf.image("arch_diagram_temp.png", x=55, y=current_y, w=100)
        pdf.set_y(current_y + 175)
    
    pdf.ln(10)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(190, 10, "Data & Processing Flow", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    
    steps = [
        "1. Frontend (Streamlit): The user interacts with the sidebar to input medical vitals (HbA1c, BP, etc.) and uses the What-If sliders to simulate lifestyle changes.",
        "2. Backend Engine: Upon form submission, the system generates a mathematically synthetic 24-hour Continuous Glucose Monitor (CGM) curve aligned with the user's specific clinical inputs, storing everything in the app's secure Session State.",
        "3. Machine Learning Core: The system extracts time-series features (like rolling averages and lagged glucose changes) and feeds them into a trained Scikit-Learn Random Forest Classifier, which calculates the exact probability of a dangerous glucose spike happening in the next 2 hours.",
        "4. Offline Reporting Engine: The application uses Kaleido to capture headless screenshots of the interactive Plotly graphs, which are dynamically embedded into a highly structured FPDF-generated clinical report downloaded securely by the user."
    ]
    
    for step in steps:
        pdf.multi_cell(190, 8, step, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
    pdf.output("GlucoTwin_Architecture.pdf")
    print("Successfully generated GlucoTwin_Architecture.pdf")
    
    if os.path.exists("arch_diagram_temp.png"):
        os.remove("arch_diagram_temp.png")

if __name__ == "__main__":
    create_pdf()
