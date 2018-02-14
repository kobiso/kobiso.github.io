---
title: "Codility Lesson15: MinAbsSumOfTwo"
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

Sharing an answer code of mine about [MinAbsSumOfTwo problem of Codility lesson 15](https://app.codility.com/programmers/lessons/15-caterpillar_method/min_abs_sum_of_two/start/).

{% include toc title="Table of Contents" icon="file-text" %}

# Lesson 15: MinAbsSumOfTwo
Let A be a non-empty zero-indexed array consisting of N integers.

The abs sum of two for a pair of indices $$(P, Q)$$ is the absolute value $$\mid A[P] + A[Q]\mid$$, for $$0 ≤ P ≤ Q < N$$.

For example, the following array A:

$$
  A[0] =  1\\
  A[1] =  4\\
  A[2] = -3
$$

has pairs of indices (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2). \\
The abs sum of two for the pair (0, 0) is A[0] + A[0] = |1 + 1| = 2. \\
The abs sum of two for the pair (0, 1) is A[0] + A[1] = |1 + 4| = 5. \\
The abs sum of two for the pair (0, 2) is A[0] + A[2] = |1 + (−3)| = 2. \\
The abs sum of two for the pair (1, 1) is A[1] + A[1] = |4 + 4| = 8. \\
The abs sum of two for the pair (1, 2) is A[1] + A[2] = |4 + (−3)| = 1. \\
The abs sum of two for the pair (2, 2) is A[2] + A[2] = |(−3) + (−3)| = 6.
 
Write a function:
```python
def solution(A)
```
that, given a non-empty zero-indexed array A consisting of N integers, returns the minimal abs sum of two for any pair of indices in this array.

For example, given the following array A:

$$
  A[0] =  1\\
  A[1] =  4\\
  A[2] = -3
$$

the function should return 1, as explained above.

Given array A:

$$
  A[0] = -8\\
  A[1] =  4\\
  A[2] =  5\\
  A[3] =-10\\
  A[4] =  3
$$

the function should return $$\mid (−8) + 5\mid = 3 $$.

Assume that:

- N is an integer within the range [1..100,000];
- each element of array A is an integer within the range [−1,000,000,000..1,000,000,000].

Complexity:

- expected worst-case time complexity is O(N*log(N));
- expected worst-case space complexity is O(N), beyond input storage (not counting the storage required for input arguments).

# Answer Code in Python 3
- Time complexity: $$O(N\log(N))$$
- Space complexity: $$O(N)$$

```python
# Python 3.6, Time complexity: O(nlog(n))
def solution(A):
    # Exception for only one element in the list A
    if len(A) == 1:
        return abs(A[0]+A[0])
    
    A.sort() # Sort the list
    
    # Seperate list A into two lists
    #   1. minus: list with minus sign integers
    #   2. plus: list with plus sign integers and zero
    minus, plus = [], []
    for i in range(len(A)):
        if A[i]<0:
            minus.append(A[i])
        else:
            plus.append(A[i])
    minus.reverse()
    
    # Exceptions
    if len(minus) == 0: # there is no minus sign
        return A[0]+A[0]
    if len(plus) == 0: # there is no plus sign
        return abs(A[-1]+A[-1])
        
    min_ = min(abs(minus[0]+minus[0]), plus[0]+plus[0]) # Initial min_
    right = 0
    # Compare plus and minus lists
    for i in range(len(plus)):
        while right < len(minus):
            if right-1 < 0 and right+1 < len(minus):
                right += 1
                continue
            elif abs(plus[i]+minus[right-1])>=abs(plus[i]+minus[right]) and right+1 < len(minus):
                right += 1
            else: # Stop when the min for 'i' is found
                right -= 1
                break
            
        # Check if it is min for until now
        if min_ > abs(plus[i]+minus[right]):
            min_ = abs(plus[i]+minus[right])
                
    return min_     
```
