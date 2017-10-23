---
title: "Codility Lesson2: CyclicRotation"
categories:
  - Coding challenge
tags:
  - Codility
header:
  teaser: /assets/images/codility.png
  overlay_image: /assets/images/codility.png
  overlay_filter: 0.4
---

Sharing an answer code of mine about [CyclicRotation problem of Codility lesson 2](https://codility.com/programmers/lessons/2-arrays/cyclic_rotation/start/).

{% include toc title="Table of Contents" icon="file-text" %}

## Lesson 2: CyclicRotation
A zero-indexed array A consisting of N integers is given. Rotation of the array means that each element is shifted right by one index, and the last element of the array is also moved to the first place.

For example, the rotation of array A = [3, 8, 9, 7, 6] is [6, 3, 8, 9, 7]. The goal is to rotate array A K times; that is, each element of A will be shifted to the right by K indexes.

Write a function:
```python
def solution(A, K)
```
that, given a zero-indexed array A consisting of N integers and an integer K, returns the array A rotated K times.

For example, given array A = [3, 8, 9, 7, 6] and K = 3, the function should return [9, 7, 6, 3, 8].

Assume that:

N and K are integers within the range [0..100];
each element of array A is an integer within the range [−1,000..1,000].
In your solution, focus on correctness. The performance of your solution will not be the focus of the assessment.

## Example answer code in Python 2.7

```python
def solution(A, K):
    # write your code in Python 2.7
    
    if len(A) == 0 or K == 0: # no shifting needed, exceptions
        return A
    
    else: # shifting needed
        shift = K - ((K / len(A)) * len(A)) # calculate minimum number of shifting times
        if shift == 0: return A # same array after shifting
        else: # not same array after shifting
            temp = list(A) # list copy
            for i in range(len(A)):
                newIndex = i + shift
                if i+shift >= len(A): # when out of index
                    newIndex = newIndex - len(A)                
                A[newIndex] = temp[i]
    return A  
```