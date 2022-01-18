"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    words = {}                  # worddict: tagcounter : number of times seen
    tagCounter = Counter()      # keep track of total tags 
    for sen in train:
            for pair in sen:
                    word = pair[0]
                    tag = pair[1]
                    if word in words:
                        words[word].update([tag])
                    else: 
                            words[word] = Counter([tag])
                    tagCounter.update([tag])
                    
    outputData = []
    for sen in test:
            outputSen = []
            for word in sen:
                    pair = (word,tagCounter.most_common(1)[0][0])
                    if word in words:
                            pair = (word,words[word].most_common(1)[0][0])
                    outputSen.append(pair)
            outputData.append(outputSen)
    
    #print(outputData)
    return outputData