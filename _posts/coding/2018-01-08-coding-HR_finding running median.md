---
title: "HackerRank: Find the Running Median"
categories:
  - Coding challenge
tags:
  - hackerrank
  - heap
  - data structure
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Find the Running Median](https://www.hackerrank.com/challenges/find-the-running-median/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Find the Running Median (in Data Structures) 

## Problem Statement
The median of a set of integers is the midpoint value of the data set for which an equal number of integers are less than and greater than the value.
To find the median, you must first sort your set of integers in non-decreasing order, then:

- If your set contains an odd number of elements, the median is the middle element of the sorted sample. In the sorted set $${1,2,3}, 2$$ is the median.
- If your set contains an even number of elements, the median is the average of the two middle elements of the sorted sample. In the sorted set $${1,2,3,4}, (2+3)/2=2.5$$ is the median.

Given an input stream of $$n$$ integers, you must perform the following task for each $$i^th$$ integer:

Add the $$i^th$$ integer to a running list of integers.
Find the median of the updated list (i.e., for the first element through the $$i^th$$ element).
Print the list's updated median on a new line. The printed value must be a double-precision number scaled to 1 decimal place (i.e., 12.3 format).

## Answer Code (in Python3) 
- Time complexity: $$O(n)$$

```python
# Find the Running Median
# Enter your code here. Read input from STDIN. Print output to STDOUT
from heapq import heappush, heappop
import sys

n = input()
maxH = [] # Invert the sign of integers to use min-heap as max-heap
minH = [] # Python 'heapq' only support min-heap
for inp in sys.stdin:
    if len(maxH) == 0:
        heappush(maxH,-int(inp))
    elif int(inp) < -maxH[0]:
        heappush(maxH,-int(inp))
        if len(maxH) > len(minH)+1:
            heappush(minH,-heappop(maxH))        
    elif int(inp) >= -maxH[0]:
        heappush(minH,int(inp))
        if len(maxH) < len(minH):
            heappush(maxH,-heappop(minH))
            
    if (len(maxH)+len(minH))%2 == 0: # Even numbers
        print ((-maxH[0]+minH[0])/2)
    else: # Odd numbers
        print (format(-maxH[0], "0.1f"))
```