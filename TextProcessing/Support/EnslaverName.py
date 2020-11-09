import pandas as pd

class EnslaverName():
    def __init__(self):
        self.titles = ['mr','ms','miss','dr','col','mrs','capt','doct','esq','gen','general','doctor']
        self.first_names = ['john','anthony','james','lizzy','will','william','lewis','david','jacob','joseph','george','jack',
                       'tom','sam','charles','peter','joe','henry','dick','jim','harry','ben','isaac','frank','bill','daniel',
                       'richard']
    
    def main(self,text):
        titles = self.titles
        first_names = self.first_names
        
        #text = ' dr miller'
        text_list = text.split(' ')
        
        #remove blanks and single letters
        text_list = [item for item in text_list if len(item) > 1]
        
        #remove titles
        text_list = [item for item in text_list if item not in titles]
        
        #override first name designation
            #get first name
        first_name = [item for item in text_list if item in first_names]
        if len(first_name) > 0:
            first_name = first_name[0]
        else:
            first_name = None
            #remove from text list
        if first_name is not None:
            text_list.remove(first_name)
        
        #if only one name remains then assume last name else first name then last name
        if len(text_list) == 1: #assume last name
            last_name = text_list[0]
        elif len(text_list) > 1:
            first_name = text_list[0]
            last_name = text_list[-1]
        else:
            last_name = None
        
        return pd.DataFrame({'first_name':[first_name],'last_name':[last_name]})