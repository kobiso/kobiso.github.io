---
title: "HackerRank: Yet Another Minimax Problem"
categories:
  - Coding challenge
tags:
  - hackerrank
  - bit manipulation
  - algorithm
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Yet Another Minimax Problem](https://www.hackerrank.com/challenges/yet-another-minimax-problem/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Yet Another Minimax Problem (in Algorithm) 

## Problem Statement
You are given $$n$$ non-negative integers, $$a_0, a_1, ..., a_{n-1}$$.
We define the score for some permutation ($$p$$) of length $$n$$ to be the maximum of $$a_{p_i} \oplus a_{p_{i+1}}$$ for $$0\leq i < n-1$$.

Find the permutation with the minimum possible score and print its score.

## Answer Code (in Python3) 

```python
#!/bin/python3

import sys

def anotherMinimaxProblem(a):
    # Convert the input list into binary list
    bin_l = list(map(lambda x: bin(x)[2:], a))
    
    # Make a set of length of bin_l
    while True:
        len_s = set(map(lambda x: len(x), bin_l))
        if len(len_s) > 1:
            break
        if bin_l[0] == '0':
            return 0
        bin_l = list(map(lambda x: bin(int(x[1:], 2)), bin_l))
        
    big_s = set(map(lambda x: int(x,2), filter(lambda x: len(x)==max(len_s), bin_l)))
    small_s = set(map(lambda x: int(x,2), filter(lambda x: len(x)!=max(len_s), bin_l)))
    return min(big_c ^ small_c for big_c in big_s for small_c in small_s)

if __name__ == "__main__":
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    result = anotherMinimaxProblem(a)
    print(result)
```