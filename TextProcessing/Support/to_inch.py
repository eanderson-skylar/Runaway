from text_to_num import text2num
import sys
import traceback

def to_inch(input):
	## Define variables ##
	meter_to_inch = 0 # calcaulated inches from meters
	feet_to_inch = 0 # calcaulated inches from feet
	inches = [] # stated inches (Not calcaulated)
	result = [] # final result
	string_to_number_list = [] # Convert string list items `string` to integers (if applicable)
	string_to_number_final = [] # final converted list
	feet_ind = 0.0 # feet in case no individual inches

	# trim and then convert our input to list of words
	string = input.strip().split(" ")
	# lets convert written numbers into <int> data type
	# print(string)        
	for item in string: # Parse `Half` and `Quarter`
		if item in ["half", "Half"]:
			string_to_number_list.append(0.5)
		elif item in ["quarter", "Quarter"]:
			string_to_number_list.append(0.25)
		else: # Parse written digits   
			try:
				string_to_number_list.append(text2num(item, "en"))
			except:
				if item != ("an" or "a"): # Drop Indefinite Articles
					string_to_number_list.append(item)

	for item in string_to_number_list: # check if a digit still in <String> data type
		try:
			if type(item) != str:
				string_to_number_final.append(item)
			else:
				string_to_number_final.append(int(item))

		except:
			string_to_number_final.append(item)            
	# print(string_to_number_final)
	## Deeper check start
	for i in range(0,len(string_to_number_final)):
		# find feet in final list
		try:
			if ((string_to_number_final[i+1] in ["feet", "foot", "ft"]) and (type(string_to_number_final[i-1]) != int)) :
				feet_to_inch = string_to_number_final[i]*12
		except:
			pass        
		# find meters in final list and convert to feet
		try:
			if (string_to_number_final[i+1] in ["meter", "meters", "m"]) and (type(string_to_number_final[i-1]) != int):
				meter_to_inch = string_to_number_final[i]*39.3700787 # convert meter to inch
		except:
			pass
		# find individual inches (case 1 - separate from number)       
		try:
			if (string_to_number_final[i+1] in ["inch", "inches", "in"]) and (type(string_to_number_final[i-1]) != int):
				inches.append(string_to_number_final[i])
		except:
			pass        
		# find individual inches (case 2 - stick to number)       
		try:
			if string_to_number_final[i][-2:] == "in":            
				my_inches = string_to_number_final[i][:-2] # all except last to chars
				inches.append(int(my_inches))
		except:
			pass        
		# find individual inches in a range
		try:
			if string_to_number_final[i+1] == "or":            
				inches.append(int(string_to_number_final[i]))
		except:
			pass        
	# calcaulate total feet
	feet = feet_to_inch+meter_to_inch
	if len(inches):
		for inch in inches:
			result.append(inch+feet)
	else:
		result.append(feet)
	
	## Case no individual inches    
	if (input.find("in") == -1 and input.find("inch") == -1 ): # check if case happens
		if "or" in string_to_number_final: # if `OR` also exists in this case
			z = string_to_number_final.index("or")
			item_1 = string_to_number_final[z-1]*12
			item_2 = string_to_number_final[z+1]*12
			return_list = [item_1, item_2]
			return_list = [item for item in return_list if not isinstance(item, str)]
			
			if len(return_list) == 0:
				return  [None]
			else:
				return return_list

		i = 0 # initialize counter
		for x in range(0,len(string_to_number_final)):        
			try:
				if string_to_number_final[x-1] not in ["foot", "feet"]:
					feet_ind = feet_ind + string_to_number_final[x]
				i = i+1
			except:
				pass
				# print(traceback.format_exc())
		if i == 2:
			feet_ind = feet_ind*12
			if (feet_ind).is_integer() == True: # if no decimals convert to <Int> data type
				feet_ind = int(feet_ind)     
			#return [type(feet_ind).__name__+'('+str(feet_ind)+')']
			if type(feet_ind) == 'str':
				return [None]
			else:
				return [feet_ind]
		if i > 2: # Input string seems to be incorrect
			return [None]
		# if i equlas 1 will continue and ignore this block    

	 
	# return none if bad input or zero result
	if (len(result) == 0 or (len(result) == 1 and result[0] == 0)):
		return [None]
	# build new list to print output in proper format
	ls = [item for item in result if not isinstance(item, str)] #block strings
	
	if len(ls) == 0:
		return [None]
	else:
		return(ls)