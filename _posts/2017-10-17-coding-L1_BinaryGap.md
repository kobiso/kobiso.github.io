---
title: "Codility Lesson1: Binary Gap"
categories:
  - Coding challenge
tags:
  - Codility
header:
  teaser: /assets/images/codility.png
  overlay_image: /assets/images/codility.png
  overlay_filter: 0.4
---

Sharing an answer code of mine about [BinaryGap problem of Codility lesson 1](https://codility.com/programmers/lessons/1-iterations/binary_gap/start/).

{% include toc title="Table of Contents" icon="file-text" %}

## Lesson 1: BinaryGap
A binary gap within a positive integer N is any maximal sequence of consecutive zeros that is surrounded by ones at both ends in the binary representation of N.

For example, number 9 has binary representation 1001 and contains a binary gap of length 2. The number 529 has binary representation 1000010001 and contains two binary gaps: one of length 4 and one of length 3. The number 20 has binary representation 10100 and contains one binary gap of length 1. The number 15 has binary representation 1111 and has no binary gaps.

Write a function:
```python
def solution(N)
```
that, given a positive integer N, returns the length of its longest binary gap. The function should return 0 if N doesn't contain a binary gap.

For example, given N = 1041 the function should return 5, because N has binary representation 10000010001 and so its longest binary gap is of length 5.

Assume that:
  * N is an integer within the range [1..2,147,483,647].

Complexity:
  * expected worst-case time complexity is O(log(N));
  * expected worst-case space complexity is O(1).

## Example answer code in Python 2.7

```python

def solution(N):
    # write your code in Python 2.7
    
    bin_str = bin(N)[2:] # getting binary representation
    max_count = 0 # for the maximum binary gap    
    bin_count = 0 # for the each binary gap
      
    for i in range(len(bin_str)):
        if bin_str[i] == '1': # when the value of index 'i' is '1'
            if bin_count >= max_count: 
                max_count = bin_count
            bin_count = 0
        else: # when the value of index 'i' is '0'
            bin_count = bin_count + 1
            
        if i+1 == len(bin_str): # if it is the last index, return 'max_count'
            return max_count
        
    return max_count
  
```