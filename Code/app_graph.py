import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input,Output

app = dash.Dash()
df = pd.read_csv(
    'graph_data.csv')
app.layout = html.Div([
html.H1(children='Predictive Student Analysis'),

	

    html.Div(children='''
        Student Report Card
    '''),
    
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
              {'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [39, 34, 11, 87, 88] ,'type': 'line', 'name':'  Ramesh Jaiswal '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [40, 70, 88, 37, 89] ,'type': 'line', 'name':' Siddharth Kumar '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [68, 41, 25, 70, 16] ,'type': 'line', 'name':' Aarushi Agarwal '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [13, 51, 40, 26, 55] ,'type': 'line', 'name':' Ketki Patil '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [41, 83, 40, 84, 73] ,'type': 'line', 'name':' Vidhi Ghokhkle '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [21, 57, 87, 36, 37] ,'type': 'line', 'name':' Shyam Bhatiya '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [16, 11, 62, 17, 89] ,'type': 'line', 'name':' Kirtan Sindhaye '},
{'x': ['Complier Design', 'Database Management', 'Computer Networks', 'Microprocessor and Interfacing', 'Discrete Structure and Graph Theory'] ,'y': [59, 48, 68, 56, 76] ,'type': 'line', 'name':' Samruddhi Deshpande '}
            
            ],
            'layout': go.Layout(
                xaxis={'title': 'Courses'},
                yaxis={'title': 'Risk Score'}
            )
        }
    )
    
])


if __name__ == '__main__':
    app.run_server(debug=True)
    
