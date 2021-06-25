import re
from nltk.tokenize import word_tokenize

def tokenize_sentence(sentence):
    """
    Arguments:
        sentence: string of texts
    Outputs:
        tok_text: list of tokens
    """
    tok_text = []
    tok = split_on_uppercase(sentence, True)
    for t in tok:        
        tok_text.extend([tt for tt in seperate_string_number(t) if tt!=' '])
    return tok_text
	
def replace_ponctuation_with_space(s):
    #s = re.sub(r"[^\w(),|!?\'\`\:\-\.;\$%#]", " ", s)
    s = re.sub(r"/", " ", s)
    s = re.sub(r"\*", " ", s)
    s = re.sub(r"'", " ", s)
    s = re.sub(r":", " ", s)
    return s
	
def split_paragraph(section):
    paragraphs = re.split('\n\n|\]\n',section)
    return paragraphs

def process(s):
    s = re.sub(r"[^\w(),|!?\'\`\:\-\.;\$%#]", " ", s)
    #s = re.sub(r"-", " ", s)
    return s

def split_on_uppercase(s, keep_contiguous=False):
    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or 
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])
    return parts
	
def spans(txt, tokens, start):
    #tokens=word_tokenize(my_process(txt))
    entity_start = []
    entity_end = []
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset) 
        entity_start.append(offset + start)
        entity_end.append(len(token)+ offset + start)
        offset += len(token)
    return entity_start, entity_end

def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    if len(string) == 1:
        groups.append(newword)
    elif len(string)>1:
        for x, i in enumerate(string[1:]):
            if i.isalpha() and previous_character.isalpha():
                newword += i
            elif i.isnumeric() and previous_character.isnumeric():
                newword += i
            else:
                groups.append(newword)
                newword = i

            previous_character = i

            if x == len(string) - 2:
                groups.append(newword)
                newword = ''
    return groups
