---
title: "Programmers Level5. Number Expression with 1,2,4"
categories:
  - Coding challenge
tags:
  - Programmers
header:
  teaser: /assets/images/programmers.png
  overlay_image: /assets/images/programmers.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [Programmers Level5. change124](https://programmers.co.kr/learn/challenge_codes/158).

{% include toc title="Table of Contents" icon="file-text" %}

# Level 5: Number Expression with 1,2,4

## Problem Statement
Given a natural number N, find the number expression of N with only 1, 2 and 4.
For example,

- Decimal number: 1 → 1
- Decimal number: 2 → 2
- Decimal number: 3 → 4
- Decimal number: 4 → 11
- Decimal number: 5 → 12
- Decimal number: 6 → 14
- Decimal number: 7 → 21

**Input:**<br/>
N = 10

**Output:**<br/>
Answer = 41

Return type is String.

## Answer Code (in C++) 

```cpp
#include<iostream>
#include<vector>
#include<string>
using namespace std;

// Think of it as 'chage123' and change 3 to 4 on the 'answer'
string change124(int no)
{
    string answer = "";

  while (no > 0){
    if (no % 3 == 0){
      answer.insert(0, to_string(4)); // Insert the head of string
      no = no/3 - 1; // Special case
    }
    else{
      int rem = no % 3;
      answer.insert(0, to_string(rem)); // Insert the head of string
      no = no/3;
    }
  }

    return answer;
}
int main()
{
    int testNo = 10;
    string testAnswer = change124(testNo);

    cout<<testAnswer;
}
```

## Clever Answer Code (in C++) 

```cpp
#include<iostream>
#include<vector>
using namespace std;

string change124(int no){
    string answer = "";
    int a;
    while(no > 0){
        a = no % 3;
        no = no / 3;
        if (a == 0){
            no -= 1;
        }
        // Interesting way to append
        answer = "412"[a] + answer;
    }

    return answer;
}

int main(){    
    int testNo = 10;
    string testAnswer = change124(testNo);

    cout<<testAnswer;
}
```