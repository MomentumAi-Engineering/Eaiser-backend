import json
p = r"c:\\Users\\chris\\OneDrive\\Desktop\\Eaiser\\EAISER-BACKEND\\response-new.json"
with open(p,'r',encoding='utf-8') as f:
    data = json.load(f)
report_wrapper = data.get('report') or {}
report = report_wrapper.get('report') or report_wrapper
print('Top-level keys:', list(data.keys()))
print('Report keys:', list(report.keys()))
ov = report.get('issue_overview') or {}
print('Overview.type:', ov.get('type') or ov.get('issue_type'))
print('Overview.summary:', ov.get('summary'))
print('Overview.summary_explanation:', ov.get('summary_explanation'))

an = report.get('detailed_analysis') or {}
print('Detailed.potential_impact:', an.get('potential_impact'))
print('Detailed.potential_consequences_if_ignored:', an.get('potential_consequences_if_ignored'))

tmpl = report.get('template_fields') or {}
print('Template.address:', tmpl.get('address'))
print('Template.zip_code:', tmpl.get('zip_code'))
print('Template.map_link:', tmpl.get('map_link'))
print('Recommended_actions:', report.get('recommended_actions'))
