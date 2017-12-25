#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import re
import collections


#例用 贝叶斯 来进行单词拼写检查，并给出相应的建义单词


def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda :1)
    for f in features:
        model[f]+=1

    return model

NEWWORDS = train(words(open("words.txt").read()))


alphabet='abcdefghijklmnopqrstuvwsyx'

def edits1(word):
    n = len(word)

    return set([word[0:i] + word[i+1:] for i in range(n)] + # deletion
               [word[0:i] + word[i+1] +word[i] + word[i+2:] for i in range(n-1)] + #transposition
               [word[0:i] + c + word[i+1:]  for i in range(n) for c in alphabet] + # alteration
               [word[0:i] + c + word[i:] for i in range(n+1) for c in alphabet] # insertion
               )

def edit2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NEWWORDS)

def know(words):
    return set(w for w in words if w in NEWWORDS)

def correct(word):
    candidates = know([word]) or know(edits1(word)) or know(edit2(word)) or [word]
    return max(candidates, key=lambda w: NEWWORDS[w])



print correct("additionay")
print correct("agge")
print correct("absolutels")