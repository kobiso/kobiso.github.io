---
title: "HackerRank: Lily's Homework"
categories:
  - Coding challenge
tags:
  - hackerrank
  - sorting
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Lily's Homework](https://www.hackerrank.com/challenges/lilys-homework/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Lily's Homework (in Algorithm) 

## Problem Statement
Whenever George asks Lily to hang out, she's busy doing homework. George wants to help her finish it faster, but he's in over his head!
Can you help George understand Lily's homework so she can hang out with him?

Consider an array of $$n$$ distinct integers, $$A=[a_0,a_1, ... , a_{n-1}]$$.
George can swap any two elements of the array any number of times.
An array is *beautiful* if the sum of $$\mid a_i - a_{i-1} \mid$$ among $$0 < i < n$$ is minimal possible (after, possibly, performing some swaps).

Given the array A, find and print the minimum number of swaps that should be performed in order to make the array beautiful.

## Answer Code (in Python3) 
- Time complexity: $$O(n\log n)$$ ($$n$$ is the number of elements in the list)
- Space complexity : $$O(n)$$

```python
#!/bin/python3
import sys

def lilysHomework(arr):
    org = list(arr)
    arr.sort()

    dic = {}
    for i in range(len(arr)):
        dic[org[i]]=i

    swp = 0
    for i in range(len(arr)):
        if arr[i] != org[i]:
            swp += 1
            #index = org.index(arr[i])
            index = dic[arr[i]]
            dic[org[i]] = index
            org[i], org[index] = org[index], org[i]

    return swp

if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    rev = list(arr)
    rev.reverse()
    asc = lilysHomework(arr)
    desc = lilysHomework(rev)
    print (min(asc,desc))

```