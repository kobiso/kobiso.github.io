---
title: "Stack and Queue"
categories:
  - Data structure
tags:
  - data structure
  - stack
  - queue
  - dequeue
header:
  teaser: /assets/images/stack&queue/stack.png
  overlay_image: /assets/images/stack&queue/stack.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Learn about stack, queue, dequeue, its implementation and running time.

{% include toc title="Table of Contents" icon="file-text" %}

# Stack
- **Stack**: a collection of objects that are inserted and removed according to the *last-in, first-out (LIFO)* principle.
![Stack]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/stack.png){: .align-center}

## Operations
  - S.push(e): Add element e to the top of stack S.
  - S.pop(): Remove and return the top element from the stack S
  - S.top(): Return a reference to the top element of stack S, without removing it.
  - S.is_empty(): Return True if stack S does not contain any elements.
  - len(S): Return the number of elements in stack S.
  
## Use list as stack in Python
![Stack python]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/stack python.png){: .align-center}

## Time Complexity
- Time complexity of array-based stack implementation
  - The bounds for *push* and *pop* are amortized due to similar bounds for the list class.
  - The space using is $$O(n)$$, where $$n$$ is the current number of elements in the stack.
  
![Stack time]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/stack time.png){: .align-center}

# Queue
- **Queue**: a collection of objects that are inserted and removed according to the *first-in, first-out (FIFO)* principle.
![Queue]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/queue.png){: .align-center}

## Operations
  - Q.enqueue(e): Add element e to the back of queue Q.
  - Q.dequeue(): Remove and return the first element from the queue Q.
  - Q.first(): Return a reference to the first element of queue Q, without removing it.
  - Q.is_empty(): Return True if queue Q does not contain any elements.
  - len(Q): Return the number of elements in queue Q.
  
## Queue in Python
- Using list as queue in python is possible but not efficient because all of the other elements have to be shifted by one after inserts or pops
- To implement queue, use 'collections.deque' which was designed to have fast appends and pops.
```python
from collectinos import deque
queue = dequeue(["Eric", "John", "Michael"])
queue.append("Terry") # Q.enequeue(e)
queue.popleft() # Q.dequeue()
queue[0] # Q.first()
len(queue)==0 # Q.is_empty()
len(queue) # len(Q)
```
  
## Time Complexity
- Time complexity of array-based queue implementation
  - The bounds for *enqueue* and *dequeue* are amortized due to the resizing of the array.
  - The space using is $$O(n)$$, where $$n$$ is the current number of elements in the queue.
  
![Queue time]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/queue time.png){: .align-center}

# Double-Ended Queue
- **Double-Ended Queue (deque)**: a queue-like data structure that supports insertion and deletion at both the front and the back of the queue.
![Deque]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/dequeue.gif){: .align-center}

## Operations
- D.add_first(e): Add element e to the front of deque D.
- D.add_last(e): Add element e to the back of deque D.
- D.delete_first(): Remove and return the first element from the deque D.
- D.delete_last(): Remove and return the last element from the deque D.
- D.first(): Return a reference to the first element of deque D, without removing it.
- D.last(): Return a reference to the last element of deque D, without removing it.
- D.is_empty(): Return True if deque D does not contain any elements.
- len(D): Return the number of elements in deque D.

## Deques in the Python Collections Module
![Deque python]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/deque python.png){: .align-center}

## Time Complexity
- 'n' is the number of elements currently in the container. 'k' is either the value of a parameter or the number of elements in the parameter.
![Deque time]({{ site.url }}{{ site.baseurl }}/assets/images/stack&queue/deque time.png){: .align-center}

# References
- Book: Cracking the coding interview [[Link](http://www.crackingthecodinginterview.com/)]
- Book: Data structures and algorithms in python [[Link](https://www.amazon.com/Structures-Algorithms-Python-Michael-Goodrich-ebook/dp/B00CTZ290I)]