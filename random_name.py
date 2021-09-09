# 사물촬영시 보여줄 랜덤 단어 생성 
import random

def random_name():
    
    classesFile = 'coco.names'
    classNames = []

    with open(classesFile,'rt') as f :
        classNames = f.read().rstrip('\n').split('\n')

    spell_result = random.sample(classNames,1)
    return spell_result
