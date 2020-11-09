import pandas as pd
import re
import logging
import inflect

class RunDate():
    def proof_string(self,text):
        patterns = {
            'wenesday': 'wednesday',
            '2d': '2nd',
            '3d': '3rd'
        }

        for key in patterns:
            text = text.replace(key, patterns[key])

        return text

    def generate_word_to_number_mapping(self):
        p = inflect.engine()
        word_to_number_mapping = {}
        for i in range(1, 32):
            word_form = p.number_to_words(i)
            ordinal_word = p.ordinal(word_form)
            word_to_number_mapping[ordinal_word.replace('-', ' ')] = i

        return word_to_number_mapping

    def generate_number_string(self):
        p = inflect.engine()
        number_string = {}
        for i in range(1, 32):
            word_form = p.number_to_words(i)
            number_string[word_form.replace('-', ' ')] = i

        return number_string

    def parse(self,input_string):
        month_dict = {
            'january': 1,
            'february': 2,
            'march': 3,
            'april': 4,
            'may': 5,
            'june': 6,
            'july': 7,
            'august': 8,
            'september': 9,
            'october': 10,
            'november': 11,
            'december': 12,
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12
        }

        number_string = {
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10
        }

        word_to_number_mapping = self.generate_word_to_number_mapping()
        number_string = self.generate_number_string()

        day_of_month_reg = ''
        for key in word_to_number_mapping:
            day_of_month_reg = day_of_month_reg + key + '|'
        day_of_month_reg = day_of_month_reg.rstrip('|')

        number_string_reg = '('
        for key in number_string:
            number_string_reg = number_string_reg + key + '|'
        number_string_reg = number_string_reg.rstrip('|') + ')'
        
        month_reg = '(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        # number_string_reg = '(one|two|three|four|five|six|seven|eight|nine|ten)'
        time_of_day_reg = '(morning|night|evening|afternoon)'
        day_of_week_reg = '(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'

        # converts all uppercase of input string to lowercase for convenience
        input_string = input_string.lower()
        input_string = self.proof_string(input_string)

        # completed parsing day
        # consider followings as day:
        #    * number that ends with st|nd|rd|th
        #    * middle as 15
        #    * 1~2 length number alone (doesn't have any trailing string)
        #    * 1~2 length number that doesn't have trailing string like "month|months|week|weeks|year|years"
        day = None
        if re.search(r'[\d]+(st|nd|rd|th)', input_string):
            temp = re.search(r'[\d]+(st|nd|rd|th)', input_string)
            day = temp.group(0).replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')
        elif re.search(f'{day_of_month_reg}', input_string):
            temp = re.search(f'{day_of_month_reg}', input_string).group(0)
            day = word_to_number_mapping[temp]
        elif re.search('middle', input_string):
            day = 15
        elif re.search('^[\d]{1,2}$', input_string) or re.search(' [\d]{1,2}$', input_string):
            day = re.search('[\d]+$', input_string).group(0)
        elif re.search('^[\d]{1,2} [a-z]+', input_string) or re.search(' [\d]{1,2} [a-z]+', input_string):
            if re.search('^[\d]{1,2} [a-z]+', input_string):
                temp = re.search('^[\d]{1,2} [a-z]+', input_string).group(0)
            else:
                temp = re.search(' [\d]{1,2} [a-z]+', input_string).group(0)
            
            if re.search('(months|weeks|month|week)', temp) == None:
                day = re.search('[\d]+', temp).group(0)

        # completed parsing month
        # parse month string
        month = re.search(month_reg, input_string)
        if month:
            month = month_dict[month.group(0)]

        # completed parsing year
        # consider 4 length number as year
        year = re.search('[\d]{4}', input_string)
        if year:
            year = year.group(0)

        # parsing day_offset
        # use the pattern "{number} day|days past" to parse the day_offset
        if re.search('([\d]+)[ ]+(day|days)[ ]+past', input_string):
            day_offset = re.search('([\d]+)[ ]+(day|days)[ ]+past', input_string).group(0)
            day_offset = re.search('[\d]+', day_offset).group(0)
            day_offset = 0 - day_offset
        elif re.search(f'{number_string_reg}[ ]+(day|days)[ ]+past', input_string):
            day_offset = re.search(f'{number_string_reg}[ ]+(day|days)[ ]+past', input_string).group(0)
            day_offset = re.search(f'{number_string_reg}', day_offset).group(0)
            day_offset = number_string[day_offset]
            day_offset = 0 - day_offset
        else:
            day_offset = None
        
        # parsing month_offset
        # use the pattern "{number} month|months ago|since" to parse the month_offset
        month_offset = re.search('(inst|instant|ult|ultimo)', input_string)
        if re.search('(inst|instant|present month)', input_string):
            month_offset = 0
        elif re.search('(ult|ultimo)', input_string):
            month_offset = -1
        elif re.search('([\d]+)[ ]+(month|months)[ ]+(ago|since)', input_string):
            month_offset = re.search('([\d]+)[ ]+(month|months)[ ]+(ago|since)', input_string).group(0)
            month_offset = re.search('[\d]+', month_offset).group(0)
            month_offset = 0 - int(month_offset)
        elif re.search(f'{number_string_reg}[ ]+(month|months)[ ]+(ago|since)', input_string):
            month_offset = re.search(f'{number_string_reg}[ ]+(month|months)[ ]+(ago|since)', input_string).group(0)
            month_offset = re.search(f'{number_string_reg}', month_offset).group(0)
            month_offset = number_string[month_offset]
            month_offset = 0 - month_offset
        
        # parsing week_offset
        # use pattern "{number} week|weeks ago" to parse the week_offset
        week_offset = re.search('([\d]+)[ ]+(week|weeks)[ ]+ago', input_string)
        if re.search('([\d]+)[ ]+(week|weeks)[ ]+ago', input_string):
            week_offset = re.search('([\d]+)[ ]+(week|weeks)[ ]+ago', input_string).group(0)
            week_offset = re.search('[\d]+', week_offset).group(0)
            week_offset = 0 - int(week_offset)
        elif re.search(f'{number_string_reg}[ ]+(week|weeks)[ ]+ago', input_string):
            week_offset = re.search(f'{number_string_reg}[ ]+(week|weeks)[ ]+ago', input_string).group(0)
            week_offset = re.search(f'{number_string_reg}', week_offset).group(0)
            week_offset = number_string[week_offset]
            week_offset = 0 - week_offset

        # parsing year_offset
        # use pattern "{number} year|years ago" to parse the year_offset
        year_offset = re.search('([\d]+)[ ]+(year|years)[ ]+ago', input_string)
        if re.search('([\d]+)[ ]+(year|years)[ ]+ago', input_string):
            year_offset = re.search('([\d]+)[ ]+(year|years)[ ]+ago', input_string).group(0)
            year_offset = re.search('[\d]+', year_offset).group(0)
            year_offset = 0 - int(year_offset)
        elif re.search(f'{number_string_reg}[ ]+(year|years)[ ]+ago', input_string):
            year_offset = re.search(f'{number_string_reg}[ ]+(year|years)[ ]+ago', input_string).group(0)
            year_offset = re.search(f'{number_string_reg}', year_offset).group(0)
            year_offset = number_string[year_offset]
            year_offset = 0 - year_offset

        # completed parsing day_of_week
        # parse week string
        day_of_week = re.search(f'{day_of_week_reg}', input_string)
        if day_of_week:
            day_of_week = day_of_week.group(0)

        # day_of_week_offset
        day_of_week_offset = re.search(f'last[ ]+ {day_of_week_reg}', input_string)
        if re.search(f'last[ ]+{day_of_week_reg}', input_string) or re.search(f'{day_of_week_reg}[ ]+last', input_string) or re.search(f'{day_of_week_reg}[ ]+{time_of_day_reg}[ ]+last', input_string):
            day_of_week_offset = -1
        elif re.search(f'next[ ]+{day_of_week_reg}', input_string) or re.search(f'{day_of_week_reg}[ ]+next', input_string) or re.search(f'{day_of_week_reg}[ ]+{time_of_day_reg}[ ]+next', input_string):
            day_of_week_offset = 1
        

        # completed parsing time_of_day
        time_of_day = re.search(f'{time_of_day_reg}', input_string)
        if time_of_day:
            time_of_day = time_of_day.group(0)
        
        
        result = {
            'input_string': [input_string], 
            'day': [day],
            'month': [month],
            'year': [year],
            'day_offset': [day_offset],
            'month_offset': [month_offset], 
            'week_offset': [week_offset], 
            'year_offset': [year_offset], 
            'day_of_week': [day_of_week], 
            'day_of_week_offset': [day_of_week_offset], 
            'time_of_day': [time_of_day]
        }
        return pd.DataFrame(result)

# input:   test.csv
# output:  result.csv
def main(debug):
    column_names = ['input_string', 'day', 'month', 'year', 'day_offset', 'month_offset', 'week_offset', 'year_offset', 'day_of_week', 'day_of_week_offset', 'time_of_day']

    input_df = pd.read_csv('./test.csv')
    output_df = pd.DataFrame(columns = column_names)

    for index, row in input_df.iterrows():
        new_row = parse(row['input_string'])
        output_df = output_df.append(new_row, ignore_index=True)

    print(output_df)
    output_df.to_csv('./result.csv')


if __name__ == '__main__':
    main(debug=True)