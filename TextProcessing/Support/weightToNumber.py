# -*- coding: utf-8 -*-
from words2num import w2n
from spellchecker import SpellChecker
import pandas as pd

class weight2n:

    def main(self, value):
        number = None

        value = self.clean_text(value)
        value = self.correction_word(value)

        if ' or ' in value:
            values = value.split(' or ')
            word1 = values[0].strip()

            if word1.isnumeric():
                number1 = word1.strip()
            else:
                number1 = self.wordToNumber(word1)

            word2 = values[1].strip()
            if word2.isnumeric():
                number2 = word2.strip()
            else:
                number2 = self.wordToNumber(word2)

            number2 = self.check_number2(number1, number2)
            number = "{}, {}".format(number1, number2)

        elif ' or' in value:
            value = value.replace(' or', '').strip()
            if value.isnumeric():
                number = value.strip()
            else:
                number = self.wordToNumber(value)

        elif ' and ' in value and ' or ' not in value:
            values = value.split(' and ')
            word1 = values[0].strip()
            if word1.isnumeric():
                number1 = word1.strip()
            else:
                number = self.wordToNumber(value)
                return number

            word2 = values[1].strip()
            number2 = word2.strip()
            
            number2 = self.check_number2(number1, number2)
            number = "{}, {}".format(number1, number2)
            
        else:
            if value.isnumeric():
                number = value
            else:
                number = self.wordToNumber(value)
                
        return number

    def wordToNumber(self, value):
        
        """ This function converts word to number"""

        if ' or' in value:
            value = value.replace(' or', '').strip()
        if (' hundred' not in value) and ('hundred ' in value or 'hundred' in value):
            value = value.replace('hundred', 'one hundred')

        try:
            number = str(w2n(value))
            return number
        except:
            return ''

    def check_number2(self, number1, number2):

        """ This function checks second number following cases
            and updates it.

            150 or 60 pounds
            175 or 80 lbs
        """
        if int(number1) > int(number2):
            if len(number1) == 1:
                number2 = 10 + int(number2)
            elif len(number1) > 1:
                number2 = 100 + int(number2)
            elif len(number1) > 2:
                number2 = 1000 + int(number2)
            elif len(number1) > 3:
                number2 = 10000 + int(number2)

        return number2

    def clean_text(self, text):

        """This function works to clean the text for parsing"""

        text = text.lower()
        text = text.replace("'", '').strip()
        text = text.replace('between', '').strip()
        text = text.replace('fire pounds', '').strip()
        text = text.replace('pounds', '').strip()
        text = text.replace('pound', '').strip()
        text = text.replace('llbs', '').strip()
        text = text.replace('ibs', '').strip()
        text = text.replace('lbs', '').strip()
        text = text.replace('lb', '').strip()
        text = text.replace(' to ', ' or ').strip()
        text = text.replace('to ', '').strip()
        text = text.replace(' to', ' or').strip()
        return text

    def correction_word(self, text):

        """This function checks and corrects the word for incorrect words
            For example: nintey -> ninety
        """

        spell = SpellChecker()

        result = []
        for word in text.split(' '):
            cWord = spell.correction(word.strip())
            result.append(cWord)
        result = ' '.join(result)
        return result