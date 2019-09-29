import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np

app = dash.Dash()

app.scripts.config.serve_locally=True
df = pd.read_csv('risk.csv')
x = []
y =  []
for i in df.course_id.unique():
	m = df.set_index('course_id',drop = False)
	rollno = m.loc['CPL501','rollno']
	y = list(rollno)
	print y
	rows =  m.loc['CPL501','risk_score']
	x = list(rows)
	print x

DF_SIMPLE = pd.DataFrame({
    'Student ID': y,
    'Risk Score': x
   
})
app.layout = html.Div([
    html.H4('Risk Score Table'),
    dt.DataTable(
        rows=DF_SIMPLE.to_dict('records'),

        # optional - sets the order of columns
        columns=sorted(DF_SIMPLE.columns),

        editable=True,
        

        id='editable-table'
    )
   
], className='container')

if __name__ == '__main__':
    app.run_server(debug=True)
	
