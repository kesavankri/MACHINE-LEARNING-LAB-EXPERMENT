import os
import io
import datetime
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)

def create_patient_pdf_report(patient_info, prep_arrays, features_dict, pred_result, output_path=None):
    """
    Generates a high-quality clinical diagnostic PDF report using ReportLab.
    """
    if output_path is None:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    else:
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    story = []
    styles = getSampleStyleSheet()

    # Custom typography styles
    header_title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=0,
        spaceAfter=2
    )
    header_sub_style = ParagraphStyle(
        'HeaderSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        alignment=0
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=6
    )
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#334155')
    )
    cell_normal = ParagraphStyle(
        'CellNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )
    cell_center = ParagraphStyle(
        'CellCenter',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=1, # Centered
        textColor=colors.HexColor('#0f172a')
    )

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>MedAI CadX Diagnostic Center</b><br/><font size=8>AI-Assisted Computer-Aided Radiology & Pathology</font>", header_title_style),
            Paragraph(f"<b>Report ID:</b> RPT-{datetime.datetime.now().strftime('%Y%m%d%H%M')}<br/><b>Date:</b> {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", cell_bold)
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,0), 'RIGHT')
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb'), spaceAfter=12))

    # 2. Patient Demographics Table
    story.append(Paragraph("<b>1. PATIENT DEMOGRAPHICS & CLINICAL DATA</b>", section_heading))
    patient_table_data = [
        [
            Paragraph("<b>Patient ID:</b>", cell_bold), Paragraph(str(patient_info.get("patient_id", "PT-9021")), cell_normal),
            Paragraph("<b>Full Name:</b>", cell_bold), Paragraph(str(patient_info.get("name", "Anonymous Patient")), cell_normal)
        ],
        [
            Paragraph("<b>Age:</b>", cell_bold), Paragraph(f"{patient_info.get('age', 'N/A')} yrs", cell_normal),
            Paragraph("<b>Gender:</b>", cell_bold), Paragraph(str(patient_info.get("gender", "N/A")), cell_normal)
        ],
        [
            Paragraph("<b>Scan Modality:</b>", cell_bold), Paragraph(str(patient_info.get("modality", "Magnetic Resonance Imaging (MRI)")), cell_normal),
            Paragraph("<b>Referring Physician:</b>", cell_bold), Paragraph("Dr. E. Vance, MD (Radiology)", cell_normal)
        ]
    ]
    pt_table = Table(patient_table_data, colWidths=[100, 170, 100, 170])
    pt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 10))

    # 3. Medical Imaging & Preprocessing Visuals
    story.append(Paragraph("<b>2. IMAGE PROCESSING & FEATURE EXTRACTION STAGES</b>", section_heading))
    
    # Save temporary images for ReportLab flowable insertion
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "temp_report")
    os.makedirs(temp_dir, exist_ok=True)
    orig_path = os.path.join(temp_dir, "report_orig.png")
    proc_path = os.path.join(temp_dir, "report_proc.png")
    edge_path = os.path.join(temp_dir, "report_edge.png")

    cv2.imwrite(orig_path, prep_arrays.get("original"))
    cv2.imwrite(proc_path, prep_arrays.get("overlay"))
    cv2.imwrite(edge_path, prep_arrays.get("edges"))

    img_table_data = [
        [
            RLImage(orig_path, width=140, height=140),
            RLImage(proc_path, width=140, height=140),
            RLImage(edge_path, width=140, height=140)
        ],
        [
            Paragraph("<b>(A) Raw Input Scan</b>", cell_center),
            Paragraph("<b>(B) Preprocessed & ROI</b>", cell_center),
            Paragraph("<b>(C) Canny Edge Map</b>", cell_center)
        ]
    ]
    img_table = Table(img_table_data, colWidths=[180, 180, 180])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 4. Extracted Quantitative Features
    story.append(Paragraph("<b>3. QUANTITATIVE RADIOMICS & TEXTURE DESCRIPTORS</b>", section_heading))
    feats_data = [
        [
            Paragraph("<b>Feature Metric</b>", cell_bold), Paragraph("<b>Computed Value</b>", cell_bold),
            Paragraph("<b>Feature Metric</b>", cell_bold), Paragraph("<b>Computed Value</b>", cell_bold)
        ],
        [
            Paragraph("Mean Pixel Intensity", cell_normal), Paragraph(str(features_dict.get("mean_intensity", "N/A")), cell_normal),
            Paragraph("GLCM Contrast", cell_normal), Paragraph(str(features_dict.get("glcm_contrast", "N/A")), cell_normal)
        ],
        [
            Paragraph("Shannon Entropy", cell_normal), Paragraph(str(features_dict.get("entropy", "N/A")), cell_normal),
            Paragraph("GLCM Homogeneity", cell_normal), Paragraph(str(features_dict.get("glcm_homogeneity", "N/A")), cell_normal)
        ],
        [
            Paragraph("Intensity Skewness", cell_normal), Paragraph(str(features_dict.get("skewness", "N/A")), cell_normal),
            Paragraph("GLCM Energy", cell_normal), Paragraph(str(features_dict.get("glcm_energy", "N/A")), cell_normal)
        ],
        [
            Paragraph("ROI Area Ratio", cell_normal), Paragraph(f"{round(features_dict.get('roi_area_ratio', 0.0)*100, 2)}%", cell_normal),
            Paragraph("Circularity Metric", cell_normal), Paragraph(str(features_dict.get("circularity", "N/A")), cell_normal)
        ]
    ]
    f_table = Table(feats_data, colWidths=[140, 130, 140, 130])
    f_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0e7ff')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#faf5ff')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 10))

    # 5. ML Diagnostic Classification & Findings
    story.append(Paragraph("<b>4. MACHINE LEARNING DIAGNOSTIC OUTCOME & RECOMMENDATION</b>", section_heading))
    is_pathology = pred_result.get("prediction_label", 0) == 1
    diag_bg = colors.HexColor('#fee2e2') if is_pathology else colors.HexColor('#dcfce7')
    diag_border = colors.HexColor('#ef4444') if is_pathology else colors.HexColor('#10b981')
    diag_text_color = "#991b1b" if is_pathology else "#166534"

    res_data = [
        [
            Paragraph(f"<font color='{diag_text_color}' size=11><b>DIAGNOSIS: {pred_result.get('diagnosis', 'INCONCLUSIVE')}</b></font><br/>"
                      f"<font size=9>Algorithm: <b>{pred_result.get('model_used', 'Ensemble ML')}</b> | Confidence: <b>{pred_result.get('confidence', 0)}%</b> | Risk Category: <b>{pred_result.get('risk_level', 'N/A')}</b></font><br/><br/>"
                      f"<font size=8.5><b>Clinical Guidance:</b> {pred_result.get('clinical_recommendation', 'Standard radiologist review.')}</font>", cell_normal)
        ]
    ]
    res_table = Table(res_data, colWidths=[540])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), diag_bg),
        ('BOX', (0,0), (-1,-1), 1.5, diag_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 12))

    # 6. Disclaimer & Doctor Signature Block
    footer_data = [
        [
            Paragraph("<font size=7 color='#64748b'><b>Clinical Notice:</b> This automated computer-aided diagnostic (CADx) report is generated using supervised machine learning algorithms (GLCM + Scikit-Learn Ensemble) for decision support. It does not replace independent evaluation by a licensed physician.</font>", cell_normal),
            Paragraph("____________________________<br/><b>Verified by Radiologist</b><br/><font size=7 color='#64748b'>MD, Radiodiagnosis & Imaging</font>", cell_bold)
        ]
    ]
    foot_table = Table(footer_data, colWidths=[360, 180])
    foot_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'CENTER')
    ]))
    story.append(foot_table)

    # Build PDF
    doc.build(story)
    
    if output_path is None:
        buffer.seek(0)
        return buffer.getvalue()
    return output_path
