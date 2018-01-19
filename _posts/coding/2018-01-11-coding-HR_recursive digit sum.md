---
title: "HackerRank: Recursive Digit Sum"
categories:
  - Coding challenge
tags:
  - hackerrank
  - recursion
  - data structure
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Recursive Digit Sum](https://www.hackerrank.com/challenges/recursive-digit-sum/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Recursive Digit Sum (in Algorithms) 

## Problem Statement
Given an integer, we need to find the super digit of the integer.

We define super digit of an integer  using the following rules:

- If $$x$$ has only 1 digit, then its super digit is $$x$$.
- Otherwise, the super digit of $$x$$ is equal to the super digit of the digit-sum of $$x$$. Here, digit-sum of a number is defined as the sum of its digits.

For example, super digit of 9875 will be calculated as:

```
super_digit(9875) = super_digit(9+8+7+5) 
                  = super_digit(29) 
                  = super_digit(2+9)
                  = super_digit(11)
                  = super_digit(1+1)
                  = super_digit(2)
                  = 2.
```

You are given two numbers $$n$$ and $$k$$.
You have to calculate the super digit of $$p$$.
$$p$$ is created when number $$n$$ is concatenated $$k$$ times.
That is, if $$n=123$$ and $$k=3$$, then $$p=123123123$$.

## Answer Code (in Python3) 
- Time complexity: $$O(n)$$

```python
# Recursive Digit Sum
#!/bin/python3

import sys

def digitSum(n, k):
    dig = int(n)
    while dig >= 10:
        dig_l = map(int, str(dig))
        dig = sum(dig_l)
        
    dig_f = dig * k
    while dig_f >= 10:
        dig_l = map(int, str(dig_f))
        dig_f = sum(dig_l)
    return dig_f
        

if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [str(n), int(k)]
    result = digitSum(n, k)
    print(result)
```