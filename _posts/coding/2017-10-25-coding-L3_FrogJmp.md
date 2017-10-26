---
title: "Codility Lesson3: FrogJmp"
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

Sharing an answer code of mine about [FrogJmp problem of Codility lesson 3](https://codility.com/programmers/lessons/3-time_complexity/frog_jmp/start/).

{% include toc title="Table of Contents" icon="file-text" %}

## Lesson 3: FrogJmp
A small frog wants to get to the other side of the road. The frog is currently located at position X and wants to get to a position greater than or equal to Y. The small frog always jumps a fixed distance, D.

Count the minimal number of jumps that the small frog must perform to reach its target.

Write a function:
```python
def solution(A, K)
```
that, given three integers X, Y and D, returns the minimal number of jumps from position X to a position equal to or greater than Y.

For example, given:

$$
  X = 10 \\
  Y = 85 \\
  D = 30
$${: .text-center}

the function should return 3, because the frog will be positioned as follows:
- after the first jump, at position 10 + 30 = 40
- after the second jump, at position 10 + 30 + 30 = 70
- after the third jump, at position 10 + 30 + 30 + 30 = 100

Assume that:
- X, Y and D are integers within the range [1..1,000,000,000];
- X ≤ Y.

Complexity:
- expected worst-case time complexity is O(1);
- expected worst-case space complexity is O(1).

## Example answer code in Python 2.7

```python
def solution(X, Y, D):
    # write your code in Python 2.7
    
    if X >= Y: # does not have to jump
        return 0;
        
    else: # have to jump
        range = Y-X
        if range % D == 0: # When the position equal to Y after jumps
            return range / D
        else: # When the position is greater than Y after jumps
            return (range / D) + 1
```
- Time complexity: O(1)