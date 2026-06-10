# 📘 C Programming — Coding Question Bank

> **Exam-ready question bank** for UE24CS151B / UE21CS141B — Problem Solving with C
> Sources: PYQs (2023, 2024, 2025) · Imp Code (UNIT 1–4) · My Programs

---

## 🔥 Most Repeated PYQ Topics

| Topic | 2023 | 2024 | 2025 | Frequency |
|-------|:----:|:----:|:----:|:---------:|
| Recursive functions (factorial, binary search, mystery) | ✅ | ✅ | ✅ | ⭐⭐⭐ |
| Callback / Function Pointers | ✅ | ✅ | ✅ | ⭐⭐⭐ |
| Enum output prediction | ✅ | ✅ | ✅ | ⭐⭐⭐ |
| Predefined macros (__FILE__, __DATE__, etc.) | ✅ | ✅ | ✅ | ⭐⭐⭐ |
| File handling (read/write/count lines) | ✅ | ✅ | ✅ | ⭐⭐⭐ |
| Binary search (iterative & recursive) | ✅ | ✅ | — | ⭐⭐ |
| SDLC / PDLC phases | ✅ | ✅ | — | ⭐⭐ |
| Structure declaration with typedef | ✅ | ✅ | — | ⭐⭐ |
| Dynamic memory allocation (malloc, realloc) | ✅ | — | ✅ | ⭐⭐ |
| Storage classes (static, register, extern, auto) | — | ✅ | ✅ | ⭐⭐ |
| Conditional compilation output (#ifdef, #ifndef) | — | ✅ | ✅ | ⭐⭐ |
| Macro expansion / output tracing | — | ✅ | ✅ | ⭐⭐ |
| String function emulation (strlen, strcat) | ✅ | — | ✅ | ⭐⭐ |
| Linked list operations (printList, insert, delete) | — | ✅ | ✅ | ⭐⭐ |
| Bitfields, VLA, enum true/false MCQs | ✅ | — | ✅ | ⭐⭐ |
| Struct vs Union differences | ✅ | — | ✅ | ⭐⭐ |
| Pointer output prediction | — | ✅ | ✅ | ⭐⭐ |

---

## 1. I/O, Operators, Control Structures

### PYQ Questions

**Q1.1** Based on the following code, answer the questions given below: [PYQ - 2025] (5M)
```c
#include <stdio.h>
int main() {
    int x, y, z, count;
    count = scanf("%d %d", &x, &y);
    printf("%d", count);
    return 0;
}
```
i) If the input is `5` and `15` separated by a space, what will be the output?
ii) If the input is `wi fi 25`, what will be the output?
iii) If the input is `25 -14`, what will be the output?
iv) If the input is `p` and then the user presses Enter, what will be the output?
v) If the input is `5 -10 35 50`, what will be the output?

**Q1.2** State whether the following is True or False. [PYQ - 2025] (5M)
i) Every C program must include meaningful comments to explain complex logic or important sections of the code.
ii) The variable name `float` is valid in C.
iii) The variable of type `long int` always takes 8 bytes of memory in the system.
iv) Given, `const int height = 100;` Any attempt to modify the value of height directly, will result in a compile-time error.
v) C allows direct access to memory through pointers.

**Q1.3** Identify the data types for the following values: [PYQ - 2024] (4M)
(Identify appropriate C data types for given values)

**Q1.4** i) Given a C program named p1.c, write the command used to compile and link p1.c. (2M)
ii) State true or false: continue is an unconditional control construct, which takes the control to the next iteration of the enclosed loop. (2M)
iii) Identify the compile time (if any) and link time errors (if any) in the below code. If there are no errors, write the expected output. (4M) [PYQ - 2024] (8M)
```c
#include<stdio.h>
int main()
{
    int a=2,
    int b=3;
    int *p1=&a;
    int *p2=&b;
    Print("%d+%d=%d\n",a,b,*p1+*p2);
}
```

**Q1.5** i) Given a C program named p1.c, write the command used to compile and link p1.c. (2M)
ii) State true or false: continue is an unconditional control construct, which takes the control to the next iteration of the enclosed loop. (2M)
iii) Identify the compile time (if any) and link time errors (if any) in the below code. If there are no errors, write the expected output. (4M) [PYQ - 2023] (8M)
```c
#include<stdio.h>
int main()
{
    int a=2,
    int b=3;
    int *p1=&a;
    int *p2=&b;
    Print("%d+%d=%d\n",a,b,*p1+*p2);
}
```

**Q1.6** Write the syntax of if-else statement with a sample. [PYQ - 2023] (6M)

**Q1.7** Choose the correct statement(s): [PYQ - 2025] (5M)
(A) The getchar function in C can be used to read a string of characters until a newline character is encountered.
(B) The default case in a switch statement is mandatory.

### Imp Code & My Programs Questions

**Q1.8** Write a C program that uses a `switch` statement inside a `while(1)` loop to implement a menu-driven program with the following options: [My Programs] [Easy]
1. Say Hello
2. Print a Number
3. Exit

The program should keep running until the user selects Exit. Handle invalid choices using the `default` case.

> **Variation 1:** Add options for basic arithmetic (add, subtract, multiply, divide) taking two numbers as input.
> **Variation 2:** Implement a calculator where the user can chain operations until they type 'q'.

**Q1.9** Write a C program to demonstrate `printf` return value and `scanf` return value. Show what happens when: [Imp Code] [Easy]
- `printf("one\n")` — what does it return?
- `scanf("%d,%d", &a, &b)` — what does it return for valid and invalid input?

> **Variation 1:** Write code that uses the return value of `scanf` to validate user input in a loop.
> **Variation 2:** Demonstrate undefined behavior in `printf` when format specifiers don't match arguments.

**Q1.10** Write a C program to print the following number pattern using nested loops: [Imp Code] [Easy]
```
1
1 2
1 2 3
1 2 3 4
1 2 3 4 5
```
Write three versions: using `for`, `while`, and `do-while` loops.

> **Variation 1:** Print the pattern in reverse (pyramid going from 5 rows to 1).
> **Variation 2:** Print the pattern using `*` instead of numbers.

**Q1.11** Write a C program to find the sum of numbers from 1 to n using all three loop types (for, while, do-while). [Imp Code] [Easy]

> **Variation 1:** Find the sum of only even numbers from 1 to n.
> **Variation 2:** Find the sum of squares from 1 to n.

**Q1.12** Predict the output of the following code and explain: [Imp Code] [Medium]
```c
#include <stdio.h>
int main() {
    printf("hello ", "world\n");
    printf("\n");
    printf("hello %s\n", "world\n");
    printf("%s %s %s %s\n", "one", "two", "three");
    printf("%5d and %5d is %6d\n", 20, 30, 20 + 30);
    return 0;
}
```

> **Variation 1:** What happens with `printf("what : %d\n", 2.5);`?

**Q1.13** Write a C program to count the number of words, lines, and characters in a given input (read until EOF). [Imp Code] [Medium]

> **Variation 1:** Read from a file instead of stdin.
> **Variation 2:** Also count the number of vowels and consonants.

---

## 2. Arrays and Pointers

### PYQ Questions

**Q2.1** Given, `int data[] = {68, 74, 59, 68, 74, 59, 80, 91, 65, 80, 91, 65, 68, 74, 59, 80, 91, 65};`
Write C statements for the following: [PYQ - 2025] (5M)
i) Print the number of integers in the array.
ii) Print the integer at index 5.
iii) Print the size of the last element in the array.
iv) Print the sum of first and last element of the array.
v) Compare whether the integer at zeroth position is same as the integer at third position?

**Q2.2** In the code given below, if the address of `arr[0][0]` is 5000 and the size of integer is 4 bytes, find the address of `arr[1][0]`. Also mention what gets printed. [PYQ - 2024] (4M)
```c
#include<stdio.h>
int main()
{
    int arr[][2] = {2,5,4,7,9,1,90,67,23,75,86};
    printf("%d",arr[1][0]);
    return 0;
}
```

**Q2.3** i. Consider the following function definition. What does this function do?
```c
#include <stdio.h>
void processArray(int *arr, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        *(arr + i) = *(arr + i) * 2;
    }
}
```
ii. Predict the output of the following code when it is run. [PYQ - 2024] (5M)
```c
#include <stdio.h>
int main()
{
    int arr[5] = {11, 21, 13, 44, 25};
    int *ptr = arr + 2;
    printf("%d", *ptr);
    return 0;
}
```

**Q2.4** Bring out the differences between arrays and pointers. [PYQ - 2023] (4M)

**Q2.5** Mention any 4 characteristics of an array. How array Declaration is done? [PYQ - 2023] (6M)

**Q2.6** What is the output of the below code execution? [PYQ - 2025] (MCQ)
```c
#include <stdio.h>
int main() {
    int a = 10, b = 20, c = 30;
    int *arr[3] = {&a, &b, &c};
    printf("%d\n", *arr[1]);
    return 0;
}
```
A) Error B) Undefined Behavior C) 10 D) 20

**Q2.7** A 2D array, 'matrix' represents the square matrix and 'n' is the number of elements in the matrix. Fill up the blanks with an appropriate expression so that the output would be the principal diagonal elements printed. [PYQ - 2024] (4M)
```
Sample: If the Matrix is:
[1 7 8
 5 4 6
 9 2 3]
expected output is 1 4 3
```
```c
#include <stdio.h>
void printPrincipalDiagonal(int matrix[3][3], int n) {
    for (int i = 0; _____________; i++) {
        printf("%d ", _________);
    }
    printf("\n");
}
```

### Imp Code & My Programs Questions

**Q2.8** Write a C program to demonstrate pointer arithmetic on an integer array `{10, 20, 30, 40, 50}`. Show: [Imp Code] [Medium]
- Increment (`p++`)
- Addition (`p = p + 2`)
- Decrement (`p--`)
- Subtraction (`p = p - 2`)
- Traversing array using pointer arithmetic in a for loop.

> **Variation 1:** Use pointer arithmetic to find the sum of all array elements without using index notation.
> **Variation 2:** Reverse an array using only two pointers.

**Q2.9** Write a C program to swap two numbers using pointers. [Imp Code] [Easy]
```
Expected:
Before: x = 10, y = 20
After:  x = 20, y = 10
```

> **Variation 1:** Swap two strings using pointers.
> **Variation 2:** Swap three numbers in a cyclic manner using pointers.

**Q2.10** Write a C program to sort an integer array using pointer arithmetic (without using array index notation `arr[i]`). Use `*(arr+i)` notation throughout. [My Programs] [Medium]

> **Variation 1:** Implement selection sort using only pointer variables (no index variables).
> **Variation 2:** After sorting, search for an element using pointer-based linear search.

**Q2.11** Write a C program that uses an array of pointers to store marks of students where each student has a different number of subjects. Print all marks using a function that accepts `int *students[]`. [Imp Code] [Hard]
```c
int student1[] = {85, 90, 78};   // 3 subjects
int student2[] = {88, 76};        // 2 subjects
int student3[] = {92, 81, 74, 89}; // 4 subjects
```

> **Variation 1:** Find the average marks of each student.
> **Variation 2:** Find the student with the highest average.

**Q2.12** Write a C program to perform matrix addition for two m×n matrices. [My Programs / PYQ - 2023] [Medium]

> **Variation 1:** Perform matrix multiplication.
> **Variation 2:** Find the transpose of a matrix.

**Q2.13** Write a C program to recast array data into a different shape. Given a 1D array of 12 elements, print it as a 3×4 matrix and a 4×3 matrix. [My Programs] [Easy]

> **Variation 1:** Flatten a 2D array into a 1D array.

---

## 3. Functions and Callbacks

### PYQ Questions

**Q3.1** Define a Function Pointer in C. Give an example code. [PYQ - 2025] (4M)

**Q3.2** What is callback in C? Give an example code. [PYQ - 2024] (5M)

**Q3.3** Write a simple callback program for addition and multiplication. [PYQ - 2023] (8M)

### Imp Code & My Programs Questions

**Q3.4** Write a C program that demonstrates function pointers and callbacks. Implement: [Imp Code] [Medium]
- A function `what(int x, int y, int z, int (*op)(int, int, int))` that takes a function pointer
- Two functions `add` and `mul` that operate on three integers
- Demonstrate calling through both direct function pointer and callback mechanism

> **Variation 1:** Add a `subtract` function and demonstrate all three operations via callbacks.
> **Variation 2:** Create an array of function pointers and iterate through them.

**Q3.5** Write a C program that implements a `filter` function using callbacks. The filter function should: [Imp Code] [Hard]
- Accept an array, a result array, size, and a condition function pointer
- Return filtered elements based on the condition (e.g., isEven)
- Return the count of filtered elements

> **Variation 1:** Add an `isOdd` condition and filter odd numbers.
> **Variation 2:** Add a `isPositive` condition to filter positive numbers from an array with mixed values.

**Q3.6** Write a C program that implements a `map` function using callbacks. The map function should apply a given function to each element of an array. [Imp Code] [Medium]
```
Input:  arr[] = {1, 2, 3, 4, 5}
Output: After square mapping: 1 4 9 16 25
```

> **Variation 1:** Implement a `double` mapping function that doubles each element.
> **Variation 2:** Chain map operations: first square, then increment.

**Q3.7** Write a C program that implements a `reduce` function using callbacks. The reduce function should combine all array elements into a single value. [Imp Code] [Hard]
```
Input:  arr[] = {1, 2, 3, 4, 5}, operation = add
Output: Reduced value (sum): 15
```

> **Variation 1:** Implement a `multiply` reduction to get the product.
> **Variation 2:** Use reduce to find the maximum element.

**Q3.8** Write a C program that uses `qsort()` with callback comparator functions to sort an array in both ascending and descending order. [Imp Code] [Medium]

> **Variation 1:** Sort an array of strings alphabetically using qsort.
> **Variation 2:** Sort an array of structures by a specific field using qsort.

**Q3.9** Write a C program that uses callback functions to apply transformations to array elements. Implement `square` and `increment` functions and a `processArray` function that accepts a function pointer `void (*func)(int*)`. [Imp Code] [Medium]
```
Input:  arr[] = {1, 2, 3, 4}
After square:    1 4 9 16
After increment: 2 5 10 17
```

**Q3.10** Write a C program that demonstrates a library late fee calculator using: [Imp Code] [Hard]
- A `getRate` function that returns fee rate based on book type (Fiction=2, Non-fiction=3, Reference=5)
- A `calculateTotal` function that computes total late fee
- A `displayFees` function that uses an array of pointers
- Array: `int days[5] = {2, 0, 5, 3, 1}`

> **Variation 1:** Add a discount for students who return within 1 day.
> **Variation 2:** Track fees per book type and print a summary report.

---

## 4. Recursion

### PYQ Questions

**Q4.1** Define Recursion. Trace the below code to get the output. What does the code do? [PYQ - 2025] (5M)
```c
#include <stdio.h>
int mystery(int x) {
    if(x == 0)
        return 0;
    else
        return x + mystery(x - 1);
}
int main() {
    int result = mystery(4);
    printf("%d\n", result);
    return 0;
}
```

**Q4.2** Write the user defined recursive function to find the factorial of a number. [PYQ - 2023] (6M)

### Imp Code & My Programs Questions

**Q4.3** Write a C program to find the factorial of a number using both iterative and recursive approaches. [Imp Code / My Programs] [Easy]
```
Input: 5
Output: factorial of 5 is = 120
```

> **Variation 1:** Handle edge cases: factorial of 0, and negative numbers (return -1).
> **Variation 2:** Use recursion to compute nCr = n! / (r! * (n-r)!).

**Q4.4** Write a recursive function `find_the_sum(int n)` that returns the sum of all numbers from n to 0. If n is negative, return -1. [Imp Code] [Easy]
```
Input: 5
Output: sum of numbers from 5 to 0 is 15
```

> **Variation 1:** Modify to find the sum of only even numbers from n to 0.
> **Variation 2:** Find the sum of digits of a number using recursion.

**Q4.5** Trace the output of the following recursive function and explain what it does: [Imp Code] [Medium]
```c
int what1(int n)
{
    if(n==0) return 0;
    else return (n%2)+10*what1(n/2);
}
// Called as: printf("%d", what1(10));
```

> **Variation 1:** What is the output for `what1(25)`?
> **Variation 2:** Write the inverse function that converts binary back to decimal.

**Q4.6** Trace the output of the following recursive function and explain what it does: [Imp Code] [Medium]
```c
int what2_v2(int n)
{
    if(!n) return 0;
    else if (!(n%2)) return what2_v2(n/2);
    else return 1+what2_v2(n/2);
}
// Called as: printf("%d", what2_v2(10));
```

> **Variation 1:** What is the output for n=15 (binary: 1111)?
> **Variation 2:** Write an iterative version of the same function.

**Q4.7** Write a C program to check whether a string is a palindrome using recursion. [Imp Code] [Medium]

> **Variation 1:** Check palindrome for an integer (e.g., 121 → palindrome).
> **Variation 2:** Ignore case and spaces while checking palindrome.

**Q4.8** Write a C program to check whether a given number is an Armstrong Number. [My Programs] [Easy]
```
Input: 153
Output: The number 153 is an Armstrong Number (1³ + 5³ + 3³ = 153)
```

> **Variation 1:** Find all Armstrong numbers between 1 and 1000.
> **Variation 2:** Generalize for n-digit Armstrong numbers.

**Q4.9** Write a C program to check if a given string is a palindrome using pointer arithmetic (start and end pointers). [My Programs] [Medium]

> **Variation 1:** Check if a given integer is a palindrome by converting to string first.
> **Variation 2:** Find the longest palindromic substring.

**Q4.10** Write a recursive function to generate all permutations of a string. [Imp Code] [Hard]
```
Input: "ABC"
Output: ABC ACB BAC BCA CBA CAB
```

> **Variation 1:** Generate permutations of only first k characters.
> **Variation 2:** Generate all combinations (not permutations) of a string.

**Q4.11** Write a recursive function to generate all combinations from a mobile keypad. Given a string of digits, print all possible letter combinations. [Imp Code] [Hard]
```
Input: "23"
Output: ad ae af bd be bf cd ce cf
```

---

## 5. Searching and Sorting

### PYQ Questions

**Q5.1** Write the definition of `binarySearchRecursive` function which takes the sorted (ascending order) integer array `arr` as its first argument, lowest index (0) as its second argument, highest index (n-1) as its third argument where n is the number of elements in the array and the element to be searched, `key` as its last argument. The function searches for the key element recursively in the given array and if the search is successful, returns the index of the element in an array, otherwise returns -1. [PYQ - 2024] (6M)
```c
int binarySearchRecursive(int arr[], int left, int right, int key);
```

**Q5.2** Implement Binary search using recursive method on an array of 100 integer elements which are stored in ascending order. Handle both successful and unsuccessful search.
Given the array: `int a[] = {100,98,76,54,44,43,42,40,31,30};` [PYQ - 2023] (8M)

### Imp Code & My Programs Questions

**Q5.3** Write a C program to implement Binary Search (iterative) on a sorted array. [Imp Code / My Programs] [Medium]
```
Input:  arr[] = {10, 20, 30, 40, 50}, key = 40
Output: Found at : 3
```

> **Variation 1:** Count the number of comparisons made during search.
> **Variation 2:** Search in a descending sorted array.

**Q5.4** Write a C program to implement Binary Search (recursive) that reads numbers from a file and searches for a key. [Imp Code] [Hard]
```c
int mysearch(int a[], int low, int high, int key);
```

> **Variation 1:** Find the first and last occurrence of a repeated element.
> **Variation 2:** Find the insertion position if element not found.

**Q5.5** Write a C program to implement Linear Search on an array. [Imp Code / My Programs] [Easy]
```
Input:  arr[] = {5, 3, 8, 1, 9}, key = 8
Output: Element found at position 3
```

> **Variation 1:** Return all positions where the key is found (for duplicates).
> **Variation 2:** Implement sentinel linear search.

**Q5.6** Write a C program to implement Random Search on an array. [Imp Code] [Easy]

> **Variation 1:** Limit maximum attempts and report success/failure.

**Q5.7** Write a C program to implement Bubble Sort. [My Programs] [Easy]
```
Input:  arr[] = {1, 3, 5, 4, 2}
Output: 1 2 3 4 5
```

> **Variation 1:** Add a flag to detect if array is already sorted (optimized bubble sort).
> **Variation 2:** Sort in descending order.

**Q5.8** Write a C program to implement Selection Sort. [My Programs] [Easy]
```
Input:  arr[] = {1, 3, 2, 5, 4}
Output: 1 2 3 4 5
```

> **Variation 1:** Implement using recursion.
> **Variation 2:** Sort an array of structures by a field using selection sort.

**Q5.9** Write a C program to implement Insertion Sort. [My Programs] [Easy]

> **Variation 1:** Sort strings alphabetically using insertion sort.
> **Variation 2:** Count the number of swaps performed.

**Q5.10** Write a C program to implement Selection Sort using recursion. [Imp Code] [Medium]
```c
void selectionSort(int arr[], int n, int index)
{
    // Base condition: if index >= n-1, return
    // Find minimum element in remaining array
    // Swap elements
    // Recursive call for remaining array
}
```

> **Variation 1:** Count recursion depth and comparisons.

**Q5.11** Write a C program that sorts and searches product IDs. Implement: [My Programs] [Medium]
- `selection_sort(int *arr, int N)` using only pointer notation
- `Search(int *arr, int N, int S)` using pointer notation
- Take input from user

---

## 6. Strings

### PYQ Questions

**Q6.1** i) Emulate the behavior of the standard library function `strlen()` by creating a custom function named `my_strlen`. This function should accept a string and return its length.
ii) What is the output of: [PYQ - 2025] (6M)
```c
char technique[100] = "diffusion";
printf("%d %d", sizeof(technique), strlen(technique));
```

**Q6.2** The below function emulates the `strcat()` in string.h. Complete the function by filling the missing code lines. [PYQ - 2023] (4M)
```c
void my_strcat(char* a, char* b)
{
    while (*a)
        //code1 - 1 Marks
    while(*b)
    {
        //code2 - 1 Marks
        a++;
        //code3 - 1 Marks
    }
    //code4 -- 1 Marks
}
```

**Q6.3** List any two string functions with its syntax. [PYQ - 2024] (4M)

**Q6.4** What gets printed when the code below is executed? [PYQ - 2024] (4M)
```c
#include<stdio.h>
int main()
{
    char str[] = "PESU";
    int i;
    for(i=0; str[i]; i++)
        printf("%c %c %c %c\n", str[i], *(str+i),
        *(i+str), i[str]+1);
    return 0;
}
```

### Imp Code & My Programs Questions

**Q6.5** Write a C program to check if a given string is a palindrome using two pointers (start and end). [My Programs] [Medium]
```
Input:  "madam"
Output: The Password is a Palindrome
```

> **Variation 1:** Make the check case-insensitive.
> **Variation 2:** Check palindrome for a sentence (ignore spaces and punctuation).

**Q6.6** Write a custom implementation of `strlen()` function named `my_strlen`. [Imp Code] [Easy]

> **Variation 1:** Implement using pointer arithmetic instead of array indexing.
> **Variation 2:** Implement a `my_strcmp` function.

**Q6.7** Write a custom implementation of `strcat()` function that appends string b to the end of string a. [Imp Code] [Medium]

> **Variation 1:** Implement `my_strncat` that appends at most n characters.
> **Variation 2:** Implement `my_strcpy` function.

---

## 7. Dynamic Memory Management

### PYQ Questions

**Q7.1** Write C Statements to perform the following tasks: [PYQ - 2025] (5M)
i) Prompt the user to enter the initial size of an integer array and dynamically allocate memory for the runtime Array.
ii) Accept integer values into the array from the user and display them.

**Q7.2** Explain briefly the following:
A) realloc
B) Priority Queue [PYQ - 2024] (6M)

**Q7.3** List any two functions related to dynamic memory allocation using C. Explain anyone with a code snippet. [PYQ - 2023] (6M)

### Imp Code & My Programs Questions

**Q7.4** Write a C program for a dynamic inventory system using structures and dynamic memory allocation. Create a `Product` struct with `id` and `price`. Allocate memory for N products using `malloc`, populate them, calculate total value, and free memory. [My Programs] [Medium]
```
Usage: ./program 5  (N from command line)
Output: Total is: 157.50
```

> **Variation 1:** Add `realloc` to expand inventory when more products arrive.
> **Variation 2:** Find the most expensive and cheapest product.

**Q7.5** Write a C program for an e-commerce tracker that: [My Programs] [Medium]
- Uses a struct with `emp_id`, `emp_name`, `emp_pro` (projects), `emp_per` (performance)
- Dynamically allocates memory for N employees
- Displays all employees, those with >5 projects, and those with performance ≥ 8.0

> **Variation 1:** Sort employees by performance score before displaying.
> **Variation 2:** Add a function to remove an employee and reallocate memory.

**Q7.6** Write a C program that demonstrates `malloc`, `calloc`, `realloc`, and `free` with appropriate examples. Show the difference between `malloc` and `calloc`. [Imp Code] [Medium]

> **Variation 1:** Demonstrate what happens when `realloc` returns NULL.
> **Variation 2:** Create a dynamically growing array that doubles in size when full.

---

## 8. Structures, Unions, Enums

### PYQ Questions

**Q8.1** Define Self referential structures in C. Also write the outputs of below code. [PYQ - 2025] (4M)
```c
#include<stdio.h>
struct Sample{int data; struct Sample *link;};
int main(){
    struct Sample s1, s2;
    s1.data = s2.data = 200;
    s1.link = &s2;
    int c = 100;
    struct Sample *sp = &s1;
    printf("%d\n", sp->data);
    sp->data = c;
    printf("%d\n", s1.data);
    sp->link = &s2;
    printf("%d\n", sp->link->data);
    return 0;
}
```

**Q8.2** List any 5 characteristics/properties of Unions in C. [PYQ - 2025] (5M)

**Q8.3** Answer the following MCQs: [PYQ - 2025] (5M)
i) Which of the following is correct about bitfields in C?
A) Array of bitfield member inside the structure is allowed.
B) You cannot take the address of a bitfield member using the & operator.
C) Bit field as the pointer member inside the structure is mandatory.
D) Float value can be used as the width of the bit field member inside the structure.

ii) What gets printed?
```c
#include <stdio.h>
enum Color { RED, GREEN = 5, BLUE };
int main() {
    enum Color c1 = RED;
    enum Color c2 = GREEN;
    enum Color c3 = BLUE;
    printf("%d %d %d\n", c1, c2, c3);
    return 0;
}
```
A) 0 1 2  B) 0 5 5  C) 1 5 7  D) 0 5 6

iii) Which keyword to create an alias name for a given type?
A) typedef  B) TYPEDEF  C) as  D) alias

iv) What is `argc` in `int main(int argc, char *argv[])`?
A) Return value  B) Chars in first arg  C) Number of arguments  D) Array of char pointers

**Q8.4** Create a structure called `Emp_attendance` with fields `Emp_Id`, `Emp_name`, `no_days_present`. [PYQ - 2024] (6M)
A) Give an alias name called `EMP_ATTD` for the defined structure.
B) Declare a variable to hold maximum 100 employee details.
C) Declare a pointer to the above structure. Store the address of the structure in this pointer.

**Q8.5** Declare a structure called `employee` with fields Id, name, designation, salary. Give an alias name called EMPLOYEE. Declare a variable capable of holding maximum 100 employee details. [PYQ - 2023] (4M)

**Q8.6** i. What gets printed when the below code runs?
```c
enum days { sunday = -3, monday, tuesday };
int main() {
    enum days d;
    printf("%d %d", d=monday, tuesday);
}
```
ii. Fill up the blanks to print principal diagonal elements. [PYQ - 2024] (4M)

**Q8.7** What will be the output of the following C code? [PYQ - 2023] (4M)
```c
#include<stdio.h>
enum colour { blue, red, yellow };
main() {
    enum colour c;
    c = yellow;
    printf("%d", c);
}
```
a) 1  b) 2  c) 0  d) Error

**Q8.8** Describe any three differences between structure and unions in C. [PYQ - 2023] (6M)

**Q8.9** Say True or False: [PYQ - 2023] (4M)
i) Array of bit fields is allowed
ii) Bit fields with a length of 0 must be unnamed
iii) Accessing the Variable length Arguments from the function body makes use of macros available in stdarg.h
iv) Storing the symbol of one enum in another enum variable is invalid in C.

### Imp Code & My Programs Questions

**Q8.10** Write a C program to declare a structure `Player` with fields `name` (char[20]) and `health` (int). Create an array of 3 players, and write a function `showParty(Player *p, int size)` that prints all player details using pointer notation. [My Programs] [Easy]

> **Variation 1:** Add a function to find the player with maximum health.
> **Variation 2:** Sort players by health using a function that accepts a struct pointer.

**Q8.11** Write a C program that uses a `union` inside a `struct` to represent a sensor reading. The union can store either a `float temp` or an `int error_code`. Write a function that prints the sensor data based on whether it's an error or not. [My Programs] [Medium]
```c
typedef union {
    float temp;
    int error_code;
} Reading;

typedef struct {
    int id;
    Reading val;
} Sensor;
```

> **Variation 1:** Store an array of 5 sensors, some with errors and some with valid readings.
> **Variation 2:** Add a flag field to the struct to track which union member is active.

**Q8.12** Write a C program that uses `enum` and `struct` together. Create an `Employee` struct with an `id` and a `Department` enum (HR, IT, SALES). Write a function to count employees in the IT department. [My Programs] [Medium]
```
Usage: ./program 6
Output: Number of IT employees: 3
```

> **Variation 1:** Count employees per department and print a summary.
> **Variation 2:** Add a function to find the department with the most employees.

**Q8.13** Write a C program using structures to manage crop data. Each crop has a name, cost of cultivation, selling price, and quantity. Calculate profit/loss for each crop and find the crop with maximum profit. [My Programs] [Hard]

> **Variation 1:** Sort crops by profit in descending order.
> **Variation 2:** Calculate break-even quantity for each crop.

**Q8.14** Write a C program using structures to analyze rainfall data for multiple cities. Each city has 7 days of rainfall data. Calculate total, average, risk level (High/Moderate/Low), and find the city with maximum rainfall. [My Programs] [Hard]

> **Variation 1:** Find the day with maximum rainfall across all cities.
> **Variation 2:** Print cities that had zero rainfall on any day.

**Q8.15** Write a C program to sort employees by salary using an array of structures and dynamic memory allocation. [My Programs] [Medium]

> **Variation 1:** Sort in descending order.
> **Variation 2:** Allow the user to search by employee ID after sorting.

**Q8.16** Write a C program to demonstrate the `sizeof` and `offsetof` for structures and unions. Show the difference between: [Imp Code] [Medium]
```c
union X { int i; int j; };
struct Y { int i; int j; double k; };
```

> **Variation 1:** Demonstrate `#pragma pack(1)` effect on struct size.
> **Variation 2:** Show padding in a struct with mixed types (int, char, int).

**Q8.17** Write a C program that demonstrates bitfields in a structure. Create a `Status` struct with `bin1:4` and `bin2:2`. Show what happens when: [Imp Code] [Hard]
- `bin1 = 15` (valid, max for 4 bits)
- `bin1 = -7` with unsigned (overflow)
- Attempt to take address of bitfield (`scanf("%d", &s.bin1)`)

**Q8.18** Write a C program that demonstrates enum behavior including: [Imp Code] [Medium]
- Default value assignment (starting from 0)
- Custom value assignment
- sizeof an enum variable
- Printing enum symbol names using switch

> **Variation 1:** Show that duplicate enum names across different enums cause errors.
> **Variation 2:** Use enum to represent months and print the month name.

**Q8.19** Write a C program demonstrating deep copy of a structure containing a dynamically allocated string. [My Programs] [Hard]
```c
typedef struct {
    int id;
    char *text;
} Document;

Document deepCopy(Document *original);
```

> **Variation 1:** Show the problem with shallow copy (modifying copy affects original).
> **Variation 2:** Implement deep copy for a struct containing an array of pointers.

**Q8.20** Write a C program that demonstrates passing structures to functions by value and by reference. Show that: [Imp Code] [Medium]
- Pass by value: modifications don't affect original
- Pass by reference: modifications affect original
- Pass by `const` pointer: prevents modifications

**Q8.21** Write a C program that sorts an array of structures (students with name, roll_no, marks) by roll number using selection sort through an array of pointers (without moving the actual structs). Then perform binary search by roll number. [Imp Code] [Hard]

---

## 9. Linked Lists, Stacks, Queues

### PYQ Questions

**Q9.1** What is a linked list? Consider the following structure for the node in a linked list. Write a function definition for `printList`, which takes the address of the first node as an argument to print all the nodes in a sequence. [PYQ - 2024] (5M)
```c
struct Node {
    int data;
    struct Node* next;
};
void printList(struct Node* node);
```

**Q9.2** List different types of queue. [PYQ - 2023] (4M)

### Imp Code & My Programs Questions

**Q9.3** Write a C program to implement a Singly Linked List with the following operations: [My Programs] [Medium]
- Insert at Beginning (`insertAtBeginning`)
- Insert at End (`insertAtEnd`)
- Delete from Beginning (`deleteFromBeginning`)
- Delete from End (`deleteFromEnd`)
- Display list

Use `typedef struct Node { int data; struct Node *next; } Node;`

> **Variation 1:** Add insert at a specific position.
> **Variation 2:** Add a function to reverse the linked list.

**Q9.4** Write a C program to search for a target value in a linked list. Return 1 if found, 0 otherwise. [My Programs] [Easy]
```c
int search_list(Node *head, int target);
```

> **Variation 1:** Return the position (1-indexed) of the element.
> **Variation 2:** Delete the node containing the target value.

**Q9.5** Write a C program to implement a Stack using a linked list with `push` (insert at beginning) and `pop` (delete from beginning) operations. [My Programs] [Medium]

> **Variation 1:** Add `peek` to view top element without removing.
> **Variation 2:** Check if a given expression has balanced parentheses using the stack.

**Q9.6** Write a C program to implement a hospital patient priority queue using linked lists. Each node stores patient_id, patient_name, age, and priority. Implement insert at beginning and insert at end. [My Programs] [Hard]

> **Variation 1:** Insert in sorted order by priority.
> **Variation 2:** Add a `dequeue` function that removes the highest priority patient.

**Q9.7** Write a C program to create a linked list, insert nodes at the head, and print all nodes. [My Programs] [Easy]
```c
Node* IAH(Node *head, int value) {
    Node *newnode = createnode(value);
    newnode->next = head;
    return newnode;
}
```
```
Expected: 10 -> 20 -> 30 -> NULL
```

> **Variation 1:** Insert at tail instead of head.
> **Variation 2:** Create a doubly linked list.

---

## 10. File Handling

### PYQ Questions

**Q10.1** Explain the concept of Formatted input and output (IO) operations in file handling. Describe the usage of these functions/operations with a suitable code snippet/s. [PYQ - 2025] (5M)

**Q10.2** Mr. Nith wants to write a C program that reads the contents of a text file line by line, removes the newline character from each line, adds a colon (:) at the end, and then displays the modified lines on the screen. [PYQ - 2025] (6M)
```
Input file: input.txt
Arjun likes coding
He practices every day
C is his favorite language

Expected output:
Updated lines:
Arjun likes coding:He practices every day:C is his favorite language:
```

**Q10.3** Write a C program that counts the total number of lines in a CSV file named "data.csv". Assume that the file data.csv is available to the program and maximum number of characters in each line is 200. Display appropriate message. [PYQ - 2024] (6M)

**Q10.4** Choose a correct statement about C file operation program: [PYQ - 2023] (4M)
```c
int main() {
    FILE *fp;
    char ch;
    fp=fopen("readme.txt","r");
    while((ch=fgetc(fp)) != EOF)
    {
        printf("%c",ch);
    }
}
```
A) FOPEN opens a file named readme.txt in Read Mode ("r).
B) EOF is End Of File. ch==EOF checks for end of file and while loop stops or exits.
C) FGETC(fp) is a function that returns one character and cursor goes to next character.
D) All the above

**Q10.5** i) List any two error handling functions related to file handling.
ii) Say True or False:
A) The fprintf function reads a data from a file.
B) The mode "w" in fopen(), creates a file if it is not existing. [PYQ - 2024] (4M)

### Imp Code & My Programs Questions

**Q10.6** Write a C program to write data to a file using `fprintf` and read it back using `fgets`. [My Programs] [Easy]
```c
// Write:
fprintf(fp, "Player: %s\n", name);
fprintf(fp, "Score: %d\n", score);
// Read:
while(fgets(buffer, sizeof(buffer), fp) != NULL)
    printf("%s", buffer);
```

> **Variation 1:** Write multiple records and read them back.
> **Variation 2:** Use `fscanf` instead of `fgets` for reading.

**Q10.7** Write a C program to copy one file to another using `getc` and `putc`. [Imp Code] [Easy]

> **Variation 1:** Copy using command line arguments for filenames.
> **Variation 2:** Copy only lines containing a specific keyword.

**Q10.8** Write a C program to write a structure to a binary file using `fwrite` and read it back using `fread`. [My Programs] [Medium]
```c
typedef struct {
    int id;
    float score;
} student;
// Write student {1, 90.9} to data.bin
// Read and print: Student ID: 1, Student score: 90.90
```

> **Variation 1:** Write an array of 5 students and read them back.
> **Variation 2:** Demonstrate `feof` — show that it becomes true only AFTER a failed read.

**Q10.9** Write a C program that demonstrates file error handling using `errno`, `strerror`, and `perror`. Attempt to open a non-existent file and handle the error gracefully. [My Programs] [Easy]
```
Output:
Error Code 2: No such file or directory
Failed to open file: No such file or directory
```

> **Variation 1:** Handle errors for write permission denied.
> **Variation 2:** Create a wrapper function for `fopen` that always handles errors.

**Q10.10** Write a C program that takes two filenames as command-line arguments and copies the content of the first file to the second file. [My Programs] [Medium]
```
Usage: ./cli-file input.txt output.txt
```

> **Variation 1:** Append instead of overwrite.
> **Variation 2:** Add error handling for missing arguments and file open failures.

**Q10.11** Write a C program that uses I/O redirection. The program reads two numbers from stdin and prints their sum to stdout. Demonstrate using `./program < input.txt > output.txt`. [My Programs] [Easy]

> **Variation 1:** Write errors to stderr while normal output goes to stdout.

**Q10.12** Write a C program to demonstrate `fwrite` and `fread` for a `Player` structure containing id and health. Save to `save.dat` and load it back. Show the `feof` behavior. [My Programs] [Medium]
```c
struct Player p1 = {1, 100};
// After save and load:
// Output: Loaded Player ID: 1, Health: 100
// After reading past end:
// Output: End of file reached!
```

---

## 11. Preprocessor Directives and Macros

### PYQ Questions

**Q11.1** In the kingdom of CodeLand, every young programmer received a magic compass from the Grand Compiler. This compass didn't point north — it revealed hidden clues within their code using predefined macros. List any four predefined macros revealed by this compass. [PYQ - 2025] (4M)

**Q11.2** What are preprocessor directives in C? What will be the output of the below program? [PYQ - 2025] (5M)
```c
#include <stdio.h>
#define VERSION 2
#define VERSION 4
#define FEATURE_B
int main() {
    #if VERSION == 1
        printf("Output A: Version 1\n");
    #elif VERSION == 2
        #ifdef FEATURE_B
            printf("Output B: Version 2 with Feature B\n");
        #else
            printf("Output C: Version 2 without Feature B\n");
        #endif
    #elif VERSION >= 3
        printf("Output D: Version 3 or higher\n");
    #else
        printf("Output E: Unknown Version\n");
    #endif
    return 0;
}
```

**Q11.3** Find the output: [PYQ - 2025] (5M)
```c
#include<stdio.h>
#define INC(x) x + 1
#define MUL(x) x * 2
int main() {
    int a = 4;
    printf("%d ", INC(MUL(a)));    // Line 1
    printf("%d ", MUL(INC(a)));    // Line 2
    printf("%d ", INC(a + 2));     // Line 3
    printf("%d ", MUL(a + 3));     // Line 4
    printf("%d ", INC(MUL(a + 1))); // Line 5
    return 0;
}
```

**Q11.4** With an example code, show that:
A) A string can be defined using macros
B) A macro can be used in another macro [PYQ - 2024] (6M)

**Q11.5** Define conditional compilation. Write the output of below code when it is run. [PYQ - 2024] (4M)
```c
#include <stdio.h>
int main()
{
    #ifndef CHECK
        #define CHECK 10
        printf("%d\n", CHECK);
    #endif
    #ifdef CHECK
        printf("%d", CHECK+2);
    #endif
    printf("\npreprocessing successful");
    return 0;
}
```

**Q11.6** Write a C program with six pre-defined macros (__FILE__, __DATE__, __TIME__, __STDC_VERSION__, __STDC__, __LINE__). [PYQ - 2023] (6M)

### Imp Code & My Programs Questions

**Q11.7** Write a C program to demonstrate all 6 predefined macros: `__FILE__`, `__DATE__`, `__TIME__`, `__STDC_VERSION__`, `__STDC__`, `__LINE__`. [Imp Code] [Easy]

> **Variation 1:** Use `__LINE__` in a conditional to print different messages.
> **Variation 2:** Create a custom `DEBUG_LOG` macro that includes file, line, and time.

**Q11.8** Write a C program that demonstrates macro pitfalls with plain text substitution: [Imp Code] [Medium]
```c
#define sqr(x) (x*x)
printf("%d", sqr(2+3));  // What is the output?
```

> **Variation 1:** Fix the macro using proper parenthesization: `((x)*(x))`.
> **Variation 2:** Show the pitfall with `#define sqr(x) x*x` and `sqr(2+3)`.

**Q11.9** Write a C program that defines a macro `cube(x)` using another macro `sqr(x)`. [Imp Code] [Easy]
```c
#define sqr(x) x*x
#define cube(x) sqr(x)*x
printf("The cube of 9 is %d\n", cube(9));
```

> **Variation 1:** Fix the macro with proper parenthesization and verify output.

**Q11.10** Write a C program demonstrating `#ifdef`, `#ifndef`, `#else`, `#endif` for conditional compilation. [Imp Code] [Medium]

> **Variation 1:** Use conditional compilation to switch between Windows (`fflush(stdin)`) and Linux (`__fpurge(stdin)`) buffer clearing.
> **Variation 2:** Use `#if`, `#elif` to select between debug, test, and production modes.

**Q11.11** Write a C program that demonstrates `#define` for symbolic constants: [Imp Code] [Easy]
```c
#define PI 3.14
#define MAX 10
#define STR "Hello All"
```
Show that macros cannot be modified with assignment operator.

> **Variation 1:** Demonstrate macro redefinition with `#define` (warning behavior).
> **Variation 2:** Use `#define` with expressions: `#define STR 2+5*1`.

**Q11.12** Write a C program that demonstrates `#pragma pack(1)` and its effect on structure size. [Imp Code] [Medium]
```c
struct sample {
    int a;    // 4 bytes
    char b;   // 1 byte
    int x;    // 4 bytes
};
// Default size vs packed size
```

**Q11.13** Write a multi-file C program with a header file, implementation file, and main file. Demonstrate: [Imp Code] [Hard]
- Header guards (`#ifndef`, `#define`, `#endif`)
- Function declarations in header
- Implementation in .c file
- Compilation: `gcc main.c mathutils.c -o output -lm`

---

## 12. Variable Length Arguments and Environment Variables

### PYQ Questions

**Q12.1** What is `argc` in the function signature of main function given here -> `int main(int argc, char *argv[]) { }` [PYQ - 2025] (MCQ)
A) The return value of the main function.
B) The number of characters in the first argument.
C) The number of arguments passed to the program.
D) The array of character pointers.

### Imp Code & My Programs Questions

**Q12.2** Write a C program using variable length arguments (`<stdarg.h>`) to create a function `print_no(int count, ...)` that prints `count` number of integers passed as arguments. [My Programs] [Medium]
```
Call: print_no(3, 10, 20, 30)
Output: 10 20 30
```

> **Variation 1:** Create a `sum(int count, ...)` function that returns the sum of all arguments.
> **Variation 2:** Create a `min(int count, ...)` function that returns the minimum value.

**Q12.3** Write a C program using variable length arguments to create a function `max(int count, ...)` that returns the maximum of `count` integers. [My Programs] [Medium]
```
Call: max(4, 10, 50, 30, 20)
Output: 50
```

> **Variation 1:** Create an `average(int count, ...)` function.
> **Variation 2:** Create a `print_strings(int count, ...)` that prints variable number of strings.

**Q12.4** Write a C program that prints environment variables using: [Imp Code / My Programs] [Medium]
```c
int main(int argc, char *argv[], char *envp[])
```
Print the first 5 environment variables.

> **Variation 1:** Use `getenv("PATH")` to print the PATH variable.
> **Variation 2:** Use `setenv` to modify PATH by appending the current directory, then verify the change.

**Q12.5** Write a C program that uses `extern char** environ` to print ALL environment variables without knowing how many there are. [Imp Code] [Medium]

> **Variation 1:** Count the total number of environment variables.
> **Variation 2:** Search for a specific environment variable by name.

---

## 📝 Quick Reference — Exam Patterns

| Section | Typical Marks | Question Types |
|---------|:------------:|----------------|
| Q1 (Basics) | 5–8M | True/False, Error ID, scanf behavior |
| Q2 (Arrays/Pointers) | 4–6M | Array operations, pointer output, 2D arrays |
| Q3 (Functions) | 4–8M | Callbacks, function pointers, code writing |
| Q4 (Recursion) | 5–6M | Trace output, write recursive function |
| Q5 (Search/Sort) | 6–8M | Binary search recursive, selection sort |
| Q6 (Strings) | 4–6M | Emulate string functions, output prediction |
| Q7 (Dynamic Memory) | 5–6M | malloc usage, realloc explanation |
| Q8 (Structs/Unions) | 4–6M | typedef declaration, enum output, bitfields |
| Q9 (Linked Lists) | 5–8M | printList, insert/delete operations |
| Q10 (Files) | 4–6M | Count lines, read/modify/display, binary I/O |
| Q11 (Macros) | 4–6M | Predefined macros, conditional compilation, expansion |
| Q12 (VLA/Env) | 4–5M | stdarg.h usage, argc/argv, envp |

## Unit 13: Prediction and Tracing Questions

### 101. IO Functions Return Values [Notes]
**Difficulty:** ⭐⭐  
**Question:** Predict the output of the following code.
```c
#include <stdio.h>
int main() {
    int i = printf("PES");
    printf("%d", i);
    return 0;
}
```
**Answer:** The first `printf` prints "PES" and returns 3 (the number of characters). The second `printf` prints the return value (3).  
**Output:** `PES3`

### 102. Data Types Memory Limitations [Notes]
**Difficulty:** ⭐⭐  
**Question:** Consider the following snippet on an architecture where `int` is 2 bytes. What happens?
```c
#include <stdio.h>
#include <limits.h>
int main() {
    int a = INT_MAX;
    printf("%d\n", a + 1);
    return 0;
}
```
**Answer:** `INT_MAX` for a 2-byte signed integer is 32767. Adding 1 causes integer overflow and wraps around to `INT_MIN` (-32768).  
**Output:** `-32768` (Assuming 16-bit int)

### 103. File Sorting via Array of Pointers [Notes]
**Difficulty:** ⭐⭐⭐  
**Question:** Write a C program to read an unknown number of lines from a text file. Store them dynamically using an array of character pointers. Sort the array of pointers alphabetically (using `qsort` and `strcmp`) without physically moving the strings in memory, and then write the sorted lines to a new file.  
> **Variation 1:** Perform a binary search (`bsearch`) on the sorted array of pointers to find a specific string.

---

*Last updated: June 2026 · Total questions: 100+*
