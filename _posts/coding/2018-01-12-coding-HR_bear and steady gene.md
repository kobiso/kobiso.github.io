---
title: "HackerRank: Bear and Steady Gene"
categories:
  - Coding challenge
tags:
  - hackerrank
  - string
  - algorithm
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Bear and Steady Gene](https://www.hackerrank.com/challenges/bear-and-steady-gene/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Bear and Steady Gene (in Algorithm) 

## Problem Statement
A gene is represented as a string of length $$n$$ (where $$n$$ is divisible by 4), composed of the letters A, C, T, and G.
It is considered to be *steady* if each of the four letters occurs exactly $$n/4$$ times.
For example, GACT and AAGTGCCT are both steady genes.

Bear Limak is a famous biotechnology scientist who specializes in modifying bear DNA to make it steady.
Right now, he is examining a gene represented as a string $$s$$.
It is not necessarily steady.
Fortunately, Limak can choose one (maybe empty) substring of $$s$$ and replace it with any string of the same length.

Modifying a large substring of bear genes can be dangerous.
Given a string $$s$$, can you help Limak find the length of the smallest possible substring that he can replace to make $$s$$ a steady gene?

Note: A substring of a string $$S$$ is a subsequence made up of zero or more consecutive characters of $$S$$.

## Answer Code (in Python3) 
- Time complexity: $$O(n^2)$$

```python
#!/bin/python3

import sys
from collections import Counter as ctr
import math

def steadyGene(gene):
    # let's get input
    n = len(gene)
    cnt = ctr(gene)
    
    # If all element is less than n/4, substring length is 0
    if all(e<=n/4 for e in cnt.values()):
        return 0
    
    minSub = math.inf
    cnted = 0
    # Find the last sequence of the string satisfying the condition
    for i in range(n):
        cnt[gene[i]] -= 1
        while all(e<=n/4 for e in cnt.values()) and cnted <= i:
            minSub = min(minSub, i-cnted+1)
            cnt[gene[cnted]]+=1
            cnted += 1   
    return minSub    

if __name__ == "__main__":
    n = int(input().strip())
    gene = input().strip()
    result = steadyGene(gene)
    print(result)
```