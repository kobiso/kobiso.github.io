---
title: "Codility Lesson14: MinMaxDivision"
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

Sharing an answer code of mine about [MinMaxDivision problem of Codility lesson 14](https://app.codility.com/programmers/lessons/14-binary_search_algorithm/min_max_division/).

{% include toc title="Table of Contents" icon="file-text" %}

# Lesson 14: MinMaxDivision
You are given integers K, M and a non-empty zero-indexed array A consisting of N integers. Every element of the array is not greater than M.

You should divide this array into K blocks of consecutive elements. The size of the block is any integer between 0 and N. Every element of the array should belong to some block.

The sum of the block from X to Y equals A[X] + A[X + 1] + ... + A[Y]. The sum of empty block equals 0.

The large sum is the maximal sum of any block.

For example, you are given integers K = 3, M = 5 and array A such that:

```
  A[0] = 2
  A[1] = 1
  A[2] = 5
  A[3] = 1
  A[4] = 2
  A[5] = 2
  A[6] = 2
```
  
The array can be divided, for example, into the following blocks:

- [2, 1, 5, 1, 2, 2, 2], [], [] with a large sum of 15;
- [2], [1, 5, 1, 2], [2, 2] with a large sum of 9;
- [2, 1, 5], [], [1, 2, 2, 2] with a large sum of 8;
- [2, 1], [5, 1], [2, 2, 2] with a large sum of 6.

The goal is to minimize the large sum. In the above example, 6 is the minimal large sum.

Write a function:

```python
def solution(K, M, A)
```

that, given integers K, M and a non-empty zero-indexed array A consisting of N integers, returns the minimal large sum.

For example, given K = 3, M = 5 and array A such that:

```
  A[0] = 2
  A[1] = 1
  A[2] = 5
  A[3] = 1
  A[4] = 2
  A[5] = 2
  A[6] = 2
```
  
the function should return 6, as explained above.

Assume that:

- N and K are integers within the range [1..100,000];
- M is an integer within the range [0..10,000];
- each element of array A is an integer within the range [0..M].

Complexity:

- expected worst-case time complexity is O(N*log(N+M));
- expected worst-case space complexity is O(1), beyond input storage (not counting the storage required for input arguments).

# Answer Code in Python 3
- Time complexity: $$O(N\log(N+M))$$
- Space complexity: $$O(1)$$

```python
# In Python 3.6
# Time complexity: O(N*log(N+M))
# Space complexity: O(1)
import math
def check(K, A, mid):
    n = len(A)
    ls = 0
    sum = 0
    block = 0
    for i in range(n):
        if sum+A[i]>mid:
            ls = max(ls, sum)
            block += 1
            sum = A[i]
        elif sum+A[i]==mid:
            sum = sum + A[i]
            ls = max(ls, sum)
            block += 1
            sum = 0
        else:
            sum+=A[i]
        if i == n-1 and sum != 0:
            ls = max(ls, sum)
            block += 1
            
    return block, ls

def solution(K, M, A):
    n = len(A)
    beg = 1
    end = M*n
    min_ = sum(A)
    while beg <= end:
        mid = (beg+end)//2
        block, ls = check(K, A, mid)
        #print (mid, block, ls)
        if block <= K:
            end = mid - 1
            min_ = min(min_, ls)
        else:
            beg = mid + 1
            
    return min_
```
