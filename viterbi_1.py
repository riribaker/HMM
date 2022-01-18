"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from collections import Counter
import math
import numpy as np

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagOut =[]
    wordTypes = {'UNK'}     # count number of unique words in train, initialize with "UNK" for unseen
    laplace = 0.001         # smoothing parameter - TEST DIFFERENT VALUES
    tagCount = Counter()    # count number of times tag occurs in train
    ftagCount = Counter()   # count number of times tag occurs for first word in sentence
    abCount = Counter()     # count number of times tagA transitions to tagB 
    tagWords = {}           # number of times word occurs for each tag
    
    for sen in train:
        for wordTag in sen:
            word = wordTag[0]
            tag = wordTag[1]
            tagCount.update([tag])
            wordTypes.add(word)
            if tag not in tagWords:
                tagWords[tag] = Counter()
            tagWords[tag].update([word])
        # for Pt:
        for idx in range(len(sen)-1):
            (word1,tagA) = sen[idx]
            (word2,tagB) = sen[idx+1]
            if idx == 0 : ftagCount.update([tagA])   
            abCount.update([(tagA,tagB)])

    tags = [t for t in tagCount]    # list of possible tags (provided we have seen all tags in train)
    # Precalculate probabilities:
    #Ps:
    Ps = {}
    for tag in tagCount:
        initialProb = (ftagCount[tag] + laplace) / (len(train) +1 + laplace*len(tagCount))
        Ps[tag] = math.log(initialProb)

    #Pt:
    Pt = {}
    for tagA in tagCount:
        for tagB in tagCount:
            tProb = (abCount[(tagA,tagB)] + laplace) / (tagCount[tagA] + laplace*(1+len(tagCount)))
            Pt[(tagA,tagB)] = math.log(tProb)

    #Pe:
    Pe = {}
    for tag in tags:
        Pe[tag] = {}
        for word in wordTypes:
            wProb = (tagWords[tag][word] + laplace) / (tagCount[tag] + laplace*(len(wordTypes)))
            Pe[tag][word] = math.log(wProb)
    
    for sen in test:
        #V = [[0 for i in range(len(tags))] for j in range(len(sen))]        # holds probability
        #B = [[0 for l in range(len(tags))] for m in range(len(sen))]        # holds backpointer
        V = [[0 for i in range(len(sen))] for j in range(len(tags))] 
        B = [[0 for i in range(len(sen))] for j in range(len(tags))]
        # Initialize V:
        for rdx in range(len(tags)):
            if sen[0] in Pe[tags[rdx]]:
                V[rdx][0] = Ps[tags[rdx]] + Pe[tags[rdx]][sen[0]]
            else:
                V[rdx][0] = Ps[tags[rdx]] + Pe[tags[rdx]]['UNK']
        
        for wIdx in range(1,len(sen)):
            for bIdx in range(len(tags)):
                word = sen[wIdx]
                Vinit = [0 for i in range(len(tags))]
                for aIdx in range(len(tags)):
                    if word in Pe[tags[bIdx]]:
                            Vinit[aIdx] = V[aIdx][wIdx-1] + Pt[(tags[aIdx],tags[bIdx])] + Pe[tags[bIdx]][word]
                    else:
                            Vinit[aIdx] = V[aIdx][wIdx-1] + Pt[(tags[aIdx],tags[bIdx])] + Pe[tags[bIdx]]['UNK']
                maxIdxValue = max(Vinit)
                maxIdx = Vinit.index(maxIdxValue)
                V[bIdx][wIdx] = maxIdxValue
                B[bIdx][wIdx] = maxIdx

        # now backtrack to return tags
        row = list(V[:][-1]).index(max(V[:][-1]))
        senOut = []
        for col in reversed(range(1,len(sen))):
            senOut.append((sen[col],tags[row]))
            row = B[row][col]
        senOut.append((sen[0],tags[row]))
        senOut.reverse()
        tagOut.append(senOut)

    return tagOut
   
