attributes = {'study_hrs': 0, 'travel_time': 6, 'course_id': 21, 'attendance': 22, 'challenges_family': 13, 'mother_occup': 12, 'health': 1, 'test': 23, 'father_occup': 11, 'hsc': 18, 'medium': 19, 'campus_feedback': 5, 'mother_tongue': 15, 'mother_edu': 10, 'family_type': 7, 'source_fees': 3, 'tuition': 2, 'backlogs': 16, 'father_edu': 9, 'ssc': 17, 'drop_year': 4, 'annual_income': 8, 'gender': 20, 'caste': 14}

attribute = {0: ['0','1','2','3','4','5'],
1: ['EXCELLENT','POOR','AVERAGE','GOOD'],
2: ['NO','YES'],
3: ['PARENTS','SCHOLARSHIP','MULTIPLE','RELATIVES'],
4: ['NO','YES'],
5: ['1','2','3','4','5'],
6: ['UPTO 1 HOUR','1 TO 2 HOURS','2 TO 3 HOURS','MORE THAN 3 HOURS'],
7: ['NUCLEAR','JOINT','EXTENDED'],
8: ['LESS THAN 1 LAKH','1 TO 1.99 LAKHS','2 TO 2.99 LAKHS','3 TO 3.99 LAKHS','4 TO 4.99 LAKHS','5 LAKHS AND ABOVE'],
9: ['0','1','2','3','4','5'],
10: ['0','1','2','3','4'],
11: ['SALARIED EMPLOYEE','BUSINESS','DEFENCE SERVICE(AIRFORCE)','NO JOB','SERVICE','PASSED AWAY','HELPER,DRIVER','SELF EMPLOYEE','ELECTRICIAN','BMC SERVICE (ASST. ENGINEER)'],
12: ['HOUSEWIFE','SALARIED EMPLOYEE','BUSINESS'],
13: ['NONE', 'ANY ILL PERSON', 'SALARIED EMPLOYEE', 'FINANCES'],
14: ['OPEN','OBC','GENERAL','MUSLIM','HINDU'],
15: ['HINDI','URDU','MARATHI','GUJARATI','KONKANI','TAMIL','BENGALI'],
16: ['0','1','2','3','4','5','6','7'],
17: ['DISTINCTION','FIRST CLASS','SECOND CLASS'],
18: ['DISTINCTION','FIRST CLASS','SECOND CLASS'],
19: ['ENGLISH','VERNACULAR'],
20: ['MALE','FEMALE'],
21: ['CPC501', 'CPC502', 'CPC503', 'CPC504', 'CPL501', 'CPL502', 'CSC401', 'CSC402', 'CSC403', 'CSC404', 'CSC405', 'CSC406', 'CSC301', 'CSC302', 'CSC303', 'CSC304', 'CSC305', 'CSC306', 'CPC601', 'CPC602', 'CPC603', 'CPC604', 'CPL605',  'CPE6012'],
22: ['EXCELLENT','GOOD','AVERAGE','POOR'],
23: ['-1','0','4','5','6','7','8','9','10']}

c = 1
print "\n\thtml.Div(["
for key,val in attributes.items():	
	cnt=0
	l1 = attribute[val]
	print "html.Div([ \
		    \n\tdcc.Dropdown( \
	    \n\t\toptions=["
	for i in l1:
		if len(i) == 1:
			print "\t\t\t{'label':'"+i+"','value':",i,"},"
		else:
			print "\t\t\t{'label':'"+i+"','value':",cnt,"},"
			cnt = cnt+1;
	print "\t\t], \
	    \n\t\tmulti=False,"
	print "\t\tplaceholder='"+key+"',"
	print "\t\tid='"+key+"'"
	print "\t) \
        \n], style={'width':'150'}, className='six columns'),"

	if c%6==0:
		print "\n], className='row'),"
		print "\n\n#Start of new row\n"
		print "\n\thtml.Div(["
	c=c+1;
print "\n], className='row'),"
