---
title: "Codility Lesson4: MissingInteger"
categories:
  - Coding challenge
tags:
  - Codility
header:
  teaser: /assets/images/codility.png
  overlay_image: /assets/images/codility.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing an answer code of mine about [MissingInteger problem of Codility lesson 4](https://codility.com/programmers/lessons/4-counting_elements/missing_integer/start/).

{% include toc title="Table of Contents" icon="file-text" %}

## Lesson 4: MissingInteger
This is a demo task.

Write a function:
```python
def solution(A, K)
```
that, given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.

For example, given A = [1, 3, 6, 4, 1, 2], the function should return 5.

Given A = [1, 2, 3], the function should return 4.

Given A = [−1, −3], the function should return 1.

Assume that:

- N is an integer within the range [1..100,000];
- each element of array A is an integer within the range [−1,000,000..1,000,000].

Complexity:

- expected worst-case time complexity is O(N);
- expected worst-case space complexity is O(N), beyond input storage (not counting the storage required for input arguments).

Elements of input arrays can be modified.

## Example answer code in Python 2.7

```python
def solution(A):
    # write your code in Python 2.7
    
    '''
    We can only consider N integers as we are tying to find the smallest positive integer.    
    '''
    
    occur = [False]*len(A)
    
    for value in A: # Save occured positive integer less than N
        if 0 < value <= len(A):
            occur[value-1] = True
        
    for i in range(len(occur)): # Check the smallest positive integer
        if occur[i] == False:
            return i+1
    
    return len(occur)+1 # All positive integer less than N exist, so next integer is the answer
```
- Detected time complexity: O(N) or O(N*log(N))