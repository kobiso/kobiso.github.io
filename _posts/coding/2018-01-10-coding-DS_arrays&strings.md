---
title: "Array & String"
categories:
  - Data structure
tags:
  - data structure
  - array
  - string
header:
  teaser: /assets/images/array&string/hash table.png
  overlay_image: /assets/images/array&string/hash table.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Learn about hash tables and string builder which are derivatives of array and string.

{% include toc title="Table of Contents" icon="file-text" %}

# Hash Tables
- **Hash table**: a data structure that maps *keys* to *values* for highly efficient lookup.
![Hash table]({{ site.url }}{{ site.baseurl }}/assets/images/array&string/hash table.png){: .align-center}
  - **Hash function**: to map general keys to corresponding indices in a table
    - The goal of a *hash function* is to map each key $$k$$ to an integer in the range $$[0,N-1]$$.
    - Two parts of a hash function: a hash code and a compression function
    - **Hash code**: maps a key $$k$$ to an integer
    - **Compression function**: maps the hash code to an integer within a range of indices for a bucket array
  ![Hash function]({{ site.url }}{{ site.baseurl }}/assets/images/array&string/hash function.png){:height="80%" width="80%"}{: .align-center}
  - **Bucket array**: an array where each index obtains a bucket of collection of items for two different keys with the same index
  ![Bucket array]({{ site.url }}{{ site.baseurl }}/assets/images/array&string/bucket array.png){: .align-center}
  - **Collision**: if there are two or more keys with the same hash value, then two different items will be mapped to the same bucket and a collision has occurred.   

## Implementation
- For the simple implementation, we use an array of linked lists and a hash code function.
  1. Compute the key's hash code (Two different keys could have the same hash code)
  2. Map the hash code to an index in the array (Two different hash codes could map to the same index)
  3. Store the key and value in the index. (Use linked list because of collision) 

- To retrieve the value pair by its key, repeat the same process
  1. Compute the hash code from the key
  2. Compute the index from the hash code
  3. Search through the linked list for the value with this key
    
## Running Time
- Comparison of the running times between unsorted list and hash table where $$n$$ denote the number of items in the map.
![Running time]({{ site.url }}{{ site.baseurl }}/assets/images/array&string/running time.png){:height="70%" width="70%"}{: .align-center}
  - If the number of collisions is very high, the worst case is $$O(n)$$ and it is $$O(1)$$ for minimum collisions.
  
- Alternatively, hash table can be implemented with a *balanced binary search tree*.
  - It gives $$O(\log N)$$ lookup time.
  - It potentially uses less space and is able to iterate through the keys in order, which can be useful sometimes.

# References
- Book: Cracking the coding interview [[Link](http://www.crackingthecodinginterview.com/)]
- Book: Data structures and algorithms in python [[Link](https://www.amazon.com/Structures-Algorithms-Python-Michael-Goodrich-ebook/dp/B00CTZ290I)]