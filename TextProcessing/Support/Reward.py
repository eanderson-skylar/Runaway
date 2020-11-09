from word2number import w2n
import re
import pandas as pd
from price_parser import Price

class GetReward():
    def checkString(self,inputData, listData) :
        for strData in listData:
            if strData.lower() not in inputData:
                return False       
        return True
        
    def retDataFrame(self,resAmount, resUnit) :
        rowData = { 'amount' : [resAmount], 'unit' : [resUnit] }
        df = pd.DataFrame(rowData, columns = ['amount', 'unit'])
        return df
    
    # Function for parsing input string
    def parseData(self,inputString):
        # Default.xlsx : Excel file that contains currency information
        currencyDF = pd.DataFrame({'currency':['Dollars','Pounds','Shillings','Cents','Pistoles','Guineas','Pence','Texas Dollars'],
                                   'symbol':['$','£',None,None,None,None,None,'$ texas'],
                                   'string':['dollar','pound','shilling','cent','pistole','guineas','pence','texas dollar']})
        unit = ""
        amount = ""
        # Get amount and currency from the input string : ex: 20$ or £20
        inputString = inputString.replace(',', '')
        price = Price.fromstring(inputString)
        if ( price.amount != None and price.currency != None) :
            amount = price.amount
            unit = price.currency
        
        # Get currency data from string : ex: 20 dollars or twenty dollars
        if ( amount == "" ) :
            try:
                amount = w2n.word_to_num(inputString)
            except:
                try:
                    amount = int(re.search(r'\d+', inputString).group())
                except:
                    amount = ""
    
        # Converting currency string
        updatedUnit = ""
        # if (amount != "") :
        for index, row in currencyDF.iterrows():
            # Checking symbol columns from the Currency Excel File            
            if( row['symbol'] != "" and pd.isnull(row['symbol']) != True):
                symbolCheck = row['symbol'].split(' ')
                symbol = symbolCheck[0]
    
                if (len(symbolCheck) != 1) :
                    if ( unit == symbolCheck[0] and symbolCheck[1] in inputString):
                        return self.retDataFrame(amount, row['currency'])
                else :
                    inputString = inputString.replace(row['symbol'], row['string'])
                    if (unit == row['symbol']):
                        updatedUnit = row['currency']
    
            # Checking string columns from the Currency Excel File
            if (self.checkString(inputString.lower(), row['string'].split(' ')) == True):
                stringCheck = row['string'].split(' ')
                if (len(stringCheck) == 1) :
                    updatedUnit = row['currency']
                else:
                    return self.retDataFrame(amount, row['currency'])
    
        return self.retDataFrame(amount, updatedUnit)
