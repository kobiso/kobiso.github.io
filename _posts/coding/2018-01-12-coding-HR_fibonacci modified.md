---
title: "HackerRank: Fibonacci Modified"
categories:
  - Coding challenge
tags:
  - hackerrank
  - dynamic programming
  - algorithm
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Fibonacci Modified](https://www.hackerrank.com/challenges/fibonacci-modified/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Fibonacci Modified (in Algorithm) 

## Problem Statement
We define a *modified Fibonacci sequence* using the following definition:

Given terms $$t_i$$ and $$t_{i+1}$$ where $$i \in [1,\infty) $$, term $$t_{i+2}$$ is computed using the following relation:

$$
t_{i+2} = t_i + (t_{i+1})^2
$${: .text-center}

For example, if term $$t_1 =0$$ and $$t_2 =1$$, term $$t_3 = 0 + 1^2 = 1$$, term $$t_4 = 1 + 1^2 = 2$$, term $$t_5 = 1 + 2^2 = 5$$, and so on.

Given three integers, $$t_1$$, $$t_2$$, and $$n$$, compute and print term $$t_n$$ of a modified Fibonacci sequence.

Note: The value of $$t_n$$ may far exceed the range of a 64-bit integer.
Many submission languages have libraries that can handle such large results but, for those that don't (e.g., C++),
you will need to be more creative in your solution to compensate for the limitations of your chosen submission language.

## Answer Code (in Python3) 
- Time complexity: $$O(n)$$

```python
#!/bin/python3

import sys

def fibonacciModified(t1, t2, n):
    # t_{i+2}=t_{i}+(t_{i+1})^2
    lookup=[t1, t2]
    for i in range(2, n):
        lookup.append(lookup[i-2] + (lookup[i-1]*lookup[i-1]))
        
    return lookup[n-1]    

if __name__ == "__main__":
    t1, t2, n = input().strip().split(' ')
    t1, t2, n = [int(t1), int(t2), int(n)]
    result = fibonacciModified(t1, t2, n)
    print(result)
```