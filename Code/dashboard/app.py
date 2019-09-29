import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Event,Input,Output
import plotly.graph_objs as go
import dash_table_experiments as dt
import sys
sys.path.append('../model/')
#from studentRiskScores import test
from studentRiskScores import test_predict
n_clicks = 0
prev_result = ""
t_input = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
app = dash.Dash()
app.config['suppress_callback_exceptions']=True
app.layout = html.Div(children = [

html.H1(children='PREDICTIVE STUDENT ANALYSIS',style = {
	'textAlign':'center',
	'color':'#7FDBFF'
    
	}),

	

    html.Div(style = {
	'textAlign':'center',
	'color':'#7FDBFF'
    
	},children=[html.H3(
		"Student Report Card")]

	),



  html.Div(
	dcc.Tabs(
	tabs = [
	    {'label':'Progress Report','value':'progress_tab'},
	    {'label':'Predict Now!!','value':'predict_tab'},
	    
	    {'label':'Class Performance','value':'class_tab'}
	],
	id = 'tabs',
	value = 'predict_tab',
	vertical = True,
	style = {
	'height':'100vh',
	'borderTop':'thin lightgrey solid',
	'borderRight':'thin lightgrey solid',
	'margin-right':'10',
	'textAlign':'left'
	}
	),
  style = {'width':'20%','float':'left'}
  ),
	html.Div(
	html.Div(id = 'tab-output'),
	style = {'width':'80%','float':'right'}
	)
     
],style = {
	
	'fontFamily':'Sans-Serif',
	'margin-left':'auto',
	'margin-right':'auto'
    
})


    	

@app.callback(
	Output(component_id='tab-output', component_property='children'),
	[Input(component_id='tabs',component_property='value')]

)
def display_tab_data(value):
	return_value = ""
	if value == 'predict_tab':
		return_value = html.Div([
		
		
		html.Div([
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'0','value': 0 },
					{'label':'1','value': 1 },
					{'label':'2','value': 2 },
					{'label':'3','value': 3 },
					{'label':'4','value': 4 },
					{'label':'5','value': 5 },
				], 	    
				multi=False,
				placeholder='study_hrs',
				id='study_hrs'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'UPTO 1 HOUR','value': 0 },
					{'label':'1 TO 2 HOURS','value': 1 },
					{'label':'2 TO 3 HOURS','value': 2 },
					{'label':'MORE THAN 3 HOURS','value': 3 },
				], 	    
				multi=False,
				placeholder='travel_time',
				id='travel_time'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'CPC501','value': 0 },
					{'label':'CPC502','value': 1 },
					{'label':'CPC503','value': 2 },
					{'label':'CPC504','value': 3 },
					{'label':'CPL501','value': 4 },
					{'label':'CPL502','value': 5 },
					{'label':'CSC401','value': 6 },
					{'label':'CSC402','value': 7 },
					{'label':'CSC403','value': 8 },
					{'label':'CSC404','value': 9 },
					{'label':'CSC405','value': 10 },
					{'label':'CSC406','value': 11 },
					{'label':'CSC301','value': 12 },
					{'label':'CSC302','value': 13 },
					{'label':'CSC303','value': 14 },
					{'label':'CSC304','value': 15 },
					{'label':'CSC305','value': 16 },
					{'label':'CSC306','value': 17 },
					{'label':'CPC601','value': 18 },
					{'label':'CPC602','value': 19 },
					{'label':'CPC603','value': 20 },
					{'label':'CPC604','value': 21 },
					{'label':'CPL605','value': 22 },
					{'label':'CPE6012','value': 23 },
				], 	    
				multi=False,
				placeholder='course_id',
				id='course_id'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'EXCELLENT','value': 0 },
					{'label':'GOOD','value': 1 },
					{'label':'AVERAGE','value': 2 },
					{'label':'POOR','value': 3 },
				], 	    
				multi=False,
				placeholder='attendance',
				id='attendance'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'NONE','value': 0 },
					{'label':'ANY ILL PERSON','value': 1 },
					{'label':'SALARIED EMPLOYEE','value': 2 },
					{'label':'FINANCES','value': 3 },
				], 	    
				multi=False,
				placeholder='challenges_family',
				id='challenges_family'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'HOUSEWIFE','value': 0 },
					{'label':'SALARIED EMPLOYEE','value': 1 },
					{'label':'BUSINESS','value': 2 },
				], 	    
				multi=False,
				placeholder='mother_occup',
				id='mother_occup'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),

		], className='row'),
		

		#Start of new row


			html.Div([
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'EXCELLENT','value': 0 },
					{'label':'POOR','value': 1 },
					{'label':'AVERAGE','value': 2 },
					{'label':'GOOD','value': 3 },
				], 	    
				multi=False,
				placeholder='health',
				id='health'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'-1','value': 0 },
					{'label':'0','value': 0 },
					{'label':'4','value': 4 },
					{'label':'5','value': 5 },
					{'label':'6','value': 6 },
					{'label':'7','value': 7 },
					{'label':'8','value': 8 },
					{'label':'9','value': 9 },
					{'label':'10','value': 1 },
				], 	    
				multi=False,
				placeholder='test',
				id='test'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'SALARIED EMPLOYEE','value': 0 },
					{'label':'BUSINESS','value': 1 },
					{'label':'DEFENCE SERVICE(AIRFORCE)','value': 2 },
					{'label':'NO JOB','value': 3 },
					{'label':'SERVICE','value': 4 },
					{'label':'PASSED AWAY','value': 5 },
					{'label':'HELPER,DRIVER','value': 6 },
					{'label':'SELF EMPLOYEE','value': 7 },
					{'label':'ELECTRICIAN','value': 8 },
					{'label':'BMC SERVICE (ASST. ENGINEER)','value': 9 },
				], 	    
				multi=False,
				placeholder='father_occup',
				id='father_occup'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'DISTINCTION','value': 0 },
					{'label':'FIRST CLASS','value': 1 },
					{'label':'SECOND CLASS','value': 2 },
				], 	    
				multi=False,
				placeholder='hsc',
				id='hsc'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'ENGLISH','value': 0 },
					{'label':'VERNACULAR','value': 1 },
				], 	    
				multi=False,
				placeholder='medium',
				id='medium'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'1','value': 1 },
					{'label':'2','value': 2 },
					{'label':'3','value': 3 },
					{'label':'4','value': 4 },
					{'label':'5','value': 5 },
				], 	    
				multi=False,
				placeholder='campus_feedback',
				id='campus_feedback'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),

		], className='row'),


		#Start of new row


			html.Div([
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'HINDI','value': 0 },
					{'label':'URDU','value': 1 },
					{'label':'MARATHI','value': 2 },
					{'label':'GUJARATI','value': 3 },
					{'label':'KONKANI','value': 4 },
					{'label':'TAMIL','value': 5 },
					{'label':'BENGALI','value': 6 },
				], 	    
				multi=False,
				placeholder='mother_tongue',
				id='mother_tongue'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'0','value': 0 },
					{'label':'1','value': 1 },
					{'label':'2','value': 2 },
					{'label':'3','value': 3 },
					{'label':'4','value': 4 },
				], 	    
				multi=False,
				placeholder='mother_edu',
				id='mother_edu'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'NUCLEAR','value': 0 },
					{'label':'JOINT','value': 1 },
					{'label':'EXTENDED','value': 2 },
				], 	    
				multi=False,
				placeholder='family_type',
				id='family_type'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'PARENTS','value': 0 },
					{'label':'SCHOLARSHIP','value': 1 },
					{'label':'MULTIPLE','value': 2 },
					{'label':'RELATIVES','value': 3 },
				], 	    
				multi=False,
				placeholder='source_fees',
				id='source_fees'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'NO','value': 0 },
					{'label':'YES','value': 1 },
				], 	    
				multi=False,
				placeholder='tuition',
				id='tuition'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'NO','value': 0 },
					{'label':'YES','value': 1 },
				], 	    
				multi=False,
				placeholder='drop_year',
				id='drop_year'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),

		], className='row'),


		#Start of new row


			html.Div([
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'0','value': 0 },
					{'label':'1','value': 1 },
					{'label':'2','value': 2 },
					{'label':'3','value': 3 },
					{'label':'4','value': 4 },
					{'label':'5','value': 5 },
				], 	    
				multi=False,
				placeholder='father_edu',
				id='father_edu'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'DISTINCTION','value': 0 },
					{'label':'FIRST CLASS','value': 1 },
					{'label':'SECOND CLASS','value': 2 },
				], 	    
				multi=False,
				placeholder='ssc',
				id='ssc'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'0','value': 0 },
					{'label':'1','value': 1 },
					{'label':'2','value': 2 },
					{'label':'3','value': 3 },
					{'label':'4','value': 4 },
					{'label':'5','value': 5 },
					{'label':'6','value': 6 },
					{'label':'7','value': 7 },
				], 	    
				multi=False,
				placeholder='backlogs',
				id='backlogs'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'LESS THAN 1 LAKH','value': 0 },
					{'label':'1 TO 1.99 LAKHS','value': 1 },
					{'label':'2 TO 2.99 LAKHS','value': 2 },
					{'label':'3 TO 3.99 LAKHS','value': 3 },
					{'label':'4 TO 4.99 LAKHS','value': 4 },
					{'label':'5 LAKHS AND ABOVE','value': 5 },
				], 	    
				multi=False,
				placeholder='annual_income',
				id='annual_income'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'MALE','value': 0 },
					{'label':'FEMALE','value': 1 },
				], 	    
				multi=False,
				placeholder='gender',
				id='gender'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'OPEN','value': 0 },
					{'label':'OBC','value': 1 },
					{'label':'GENERAL','value': 2 },
					{'label':'MUSLIM','value': 3 },
					{'label':'HINDU','value': 4 },
				], 	    
				multi=False,
				placeholder='caste',
				id='caste'
			)         
		], style={'width':'125','margin-top':'10','margin-bottom':'10' }, className='six columns'),

		], className='row'),


		#Start of new row


			html.Div([

		], className='row'),

		 html.Button('Predict Result',id='submit',style={'color':'#111111','margin-left':'350','margin-right':'auto','margin-top':'10','margin-bottom':'10'}),

		#End of dropdown
		html.Div(style={'color':'#7FDBFF'},id='output')

		])
	if value == 'progress_tab':
		return_value = html.Div([
			
		html.Div([
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
	] ,style= {'width': '100%', 'height': '200%'})
		
	if value == 'class_tab':
		return_value = html.Div([
		
		
		html.Div([
		html.Div([ 		    
			dcc.Dropdown( 	    
				options=[
					{'label':'CPC501','value': 'CPC501'},
					{'label':'CPC502','value': 'CPC502'},
					{'label':'CPC503','value': 'CPC503'},
					{'label':'CPC504','value': 'CPC504'},
					{'label':'CPL501','value': 'CPL501'},
				], 	    
				multi=False,
				placeholder='select Course',
				id='risk_course'
			)         
		], style={'width':'50%','margin':'15'})
		]),
		html.Div(id='risk_output',style={'color':'#7FDBFF'})

	
		])	

	return return_value

@app.callback(
	Output(component_id ='risk_output', component_property='children'),
	
	[Input(component_id='risk_course',component_property='value')]
)
def risk(value):
	return value


@app.callback(
	Output(component_id='output', component_property='children'),
	
	[Input(component_id='submit',component_property='n_clicks'),
	Input(component_id='study_hrs',component_property='value'),
	 Input(component_id='health',component_property='value'),
	 Input(component_id='tuition',component_property='value'),
	 Input(component_id='source_fees',component_property='value'),
	 Input(component_id='drop_year',component_property='value'),
	 Input(component_id='campus_feedback',component_property='value'),
	 Input(component_id='travel_time',component_property='value'),
	 Input(component_id='family_type',component_property='value'),
	 Input(component_id='annual_income',component_property='value'),
	 Input(component_id='father_edu',component_property='value'),
	 Input(component_id='mother_edu',component_property='value'),
	 Input(component_id='father_occup',component_property='value'),
	 Input(component_id='mother_occup',component_property='value'),
	 Input(component_id='challenges_family',component_property='value'),
	 Input(component_id='caste',component_property='value'),
	 Input(component_id='mother_tongue',component_property='value'),
	 Input(component_id='backlogs',component_property='value'),
	 Input(component_id='ssc',component_property='value'),
	 Input(component_id='hsc',component_property='value'),
	 Input(component_id='medium',component_property='value'),
	 Input(component_id='gender',component_property='value'),
	 Input(component_id='course_id',component_property='value'),
	 Input(component_id='attendance',component_property='value'),
	 Input(component_id='test',component_property='value')]
)

def predict(number_clicks,study_hrs,health,tuition,source_fees,drop_year,campus_feedback,travel_time,family_type,annual_income,father_edu,mother_edu,father_occup,mother_occup,challenges_family,caste,mother_tongue,backlogs,ssc,hsc,medium,gender,course_id,attendance,test):
	result = ""
	global n_clicks
	global t_input
	global prev_result 
	default_values = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
	 
	if number_clicks != None and number_clicks > n_clicks:
		test_input = [annual_income, father_edu, backlogs, course_id, attendance, test]
		if any(i == None for i in test_input):
			for ind,i in enumerate(test_input):
		                if i == None:
		                        test_input[ind] = default_values[ind]
		print test_input
		if t_input != test_input:
			result = test_predict(test_input)
		else:
			result = prev_result
		print result
		t_input = test_input
		prev_result = result
	n_clicks = number_clicks
	return result

external_css = ["https://cdnjs.cloudfare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
