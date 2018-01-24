---
title: "HackerRank: Game of Two Stacks"
categories:
  - Coding challenge
tags:
  - hackerrank
  - stack
  - data structure
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Game of Two Stacks](https://www.hackerrank.com/challenges/game-of-two-stacks/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Game of Two Stacks (in Data Structures) 

## Problem Statement
Alexa has two stacks of non-negative integers, stack $$A=[a_0,a_1,...,a_{n-1}]$$ and stack $$B=[b_0,b_1,...,b_{m-1}]$$ where index 0 denotes the top of the stack.
Alexa challenges Nick to play the following game:

- In each move, Nick can remove one integer from the top of either stack $$A$$ or stack $$B$$.
- Nick keeps a running sum of the integers he removes from the two stacks.
- Nick is disqualified from the game if, at any point, his running sum becomes greater than some integer $$x$$ given at the beginning of the game.
- Nick's final score is the total number of integers he has removed from the two stacks.

Given $$A, B,$$ and $$x$$ for $$y$$ games, find the maximum possible score Nick can achieve (i.e., the maximum number of integers he can remove without being disqualified) during each game and print it on a new line.

## Answer Code (in Python3) 
- Time complexity: $$O(n*m)$$

```python
#!/bin/python3

import sys

g = int(input().strip())
for a0 in range(g):
    n, m, x = map(int, input().strip().split(' '))
    a = list(map(int, input().strip().split(' ')))
    b = list(map(int, input().strip().split(' ')))

    sum, count, max_count = 0, 0, 0
    tempA = []
    
    # Inverse 'a' and 'b' list to use as stack
    a.reverse() 
    b.reverse()
    
    # Pop from stack A and sum until it exceeds the limit
    while len(a)!=0:
        if sum + a[-1] <= x:
            sum += a[-1]
            count += 1
            tempA.append(a.pop()) # Save pop-ed element from stack A
        else:
            break
    max_count = count # Save current max_count
    
    # Pop from stack B and plus it with 'sum' 
    while len(b)!=0:
        sum += b.pop()
        count += 1
        
        # If 'sum' exceeds the limit, discard one from tempA
        while sum > x and len(tempA)!=0:
            sum -= tempA.pop()
            count -= 1
        
        if sum <= x and max_count < count:
            max_count = count
        elif sum > x:
            break

    print (max_count)
```