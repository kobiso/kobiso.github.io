---
title: "HackerRank: Array Manipulation"
categories:
  - Coding challenge
tags:
  - hackerrank
  - array
  - data structure
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Array Manipulation](https://www.hackerrank.com/challenges/crush/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Array Manipulation (in Data Structures) 

## Problem Statement
You are given a list(1-indexed) of size $$n$$, initialized with zeroes.
You have to perform $$m$$ operations on the list and output the maximum of final values of all the $$n$$ elements in the list.
For every operation, you are given three integers $$a, b$$ and $$k$$ and you have to add value  to all the elements ranging from index $$a$$ to $$b$$(both inclusive).

For example, consider a list $$a$$ of size 3.
The initial list would be $$a = [0,0,0]$$ and after performing the update $$O(a,b,k)=(2,3,30)$$ the new list would be $$a = [0,30,30]$$.
Here, we've added value 30 to elements between indices 2 and 3. Note the index of the list starts from 1.

## Naive Answer Code (in Python3) 
- Time complexity: $$O(n^2)$$

```python
#!/bin/python3
# Input: the first line will contain two integer n and m seperated by a single space.
#  - Next m lines will contain three integers a,b and k separated by a single space.
#  - Numbers in list are numbered from 1 to n.
# Output: print in a single line the maximum value in the updated list.

import sys

if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    arr = [0]*n
    for a0 in range(m):
        a, b, k = input().strip().split(' ')
        a, b, k = [int(a), int(b), int(k)]
        for i in range(a-1,b):
            arr[i] += k

    print (max(arr))
```

## Better Answer Code (in Python3) 
- Time complexity: $$O(n)$$

```python
#!/bin/python3
# Input: the first line will contain two integer n and m seperated by a single space.
#  - Next m lines will contain three integers a,b and k separated by a single space.
#  - Numbers in list are numbered from 1 to n.
# Output: print in a single line the maximum value in the updated list.

import sys
from itertools import accumulate

if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    arr = [0]*(n+1)
    for a0 in range(m):
        a, b, k = input().strip().split(' ')
        a, b, k = [int(a), int(b), int(k)]
        arr[a-1] += k # Save one time and accumulate it later
        arr[b] -= k

    '''
    max = x = 0
    for i in arr:
        x = x+i
        if max < x:
            max = x
    '''
    print (max(accumulate(arr)))
```