# C Syntax Reference

## Unit 1: Problem Solving Fundamentals

### 1. Program Structure
```c
#include <stdio.h> // Preprocessor directive for standard I/O

// Function prototype (optional if defined before main)
int add(int a, int b);

// Main function - execution starts here
int main() {
    // Variable declaration and initialization
    int sum = add(5, 3);
    
    // Output
    printf("Sum is: %d\n", sum);
    
    // Return 0 indicates successful execution
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}
```

### 2. Variables and Data Types
```c
int main() {
    // Integer types
    char c = 'A';               // 1 byte: -128 to 127
    short s = 32767;            // 2 bytes: -32,768 to 32,767
    int i = 2147483647;         // 4 bytes: -2,147,483,648 to 2,147,483,647
    long l = 2147483647L;       // 4 or 8 bytes depending on system
    long long ll = 9223372036854775807LL; // 8 bytes
    
    // Unsigned variants (cannot be negative, doubles max positive value)
    unsigned char uc = 255;
    unsigned int ui = 4294967295U;
    
    // Floating point types
    float f = 3.14159f;         // 4 bytes, ~6-7 decimal digits precision
    double d = 3.1415926535;    // 8 bytes, ~15 decimal digits precision
    
    return 0;
}
```

### 3. Qualifiers
```c
int main() {
    const float PI = 3.14159f;  // Cannot be modified after initialization
    // PI = 3.0; // Error: assignment of read-only variable
    
    volatile int sensor_value;  // Tells compiler value may change outside program control
                                // Prevents compiler optimization
                                
    signed int pos_or_neg;      // Default for int, can be +/-
    unsigned int only_pos;      // Only non-negative values
    
    return 0;
}
```

### 4. Operators
```c
int main() {
    int a = 10, b = 3, c;
    
    // Arithmetic: +, -, *, /, %
    c = a % b; // Modulo (remainder): 1
    
    // Relational: ==, !=, <, >, <=, >=
    int isEqual = (a == b); // 0 (false)
    
    // Logical: &&, ||, !
    int result = (a > 5 && b < 5); // 1 (true)
    
    // Bitwise: &, |, ^, ~, <<, >>
    c = a & b; // Bitwise AND: 1010 & 0011 = 0010 (2)
    c = a << 1; // Left shift (multiply by 2): 20
    
    // Assignment: =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
    a += 5; // Equivalent to: a = a + 5;
    
    // Ternary (Conditional): condition ? true_val : false_val
    int max = (a > b) ? a : b;
    
    // Comma
    c = (a=1, b=2, a+b); // Evaluates left to right, returns rightmost: 3
    
    // sizeof
    size_t int_size = sizeof(int); // Size in bytes
    size_t array_size = sizeof(int[10]); // 40 bytes
    
    // Type casting
    float div_result = (float)a / b; // Cast 'a' to float before division
    
    return 0;
}
```

### 5. Control Structures
```c
int main() {
    int score = 85;

    // if, else if, else
    if (score >= 90) {
        printf("A\n");
    } else if (score >= 80) {
        printf("B\n");
    } else {
        printf("C\n");
    }

    // switch-case
    int option = 2;
    switch(option) {
        case 1:
            printf("One\n");
            break; // Essential to prevent fall-through
        case 2:
        case 3: // Fall-through: cases 2 and 3 execute this
            printf("Two or Three\n");
            break;
        default:
            printf("Invalid\n");
    }

    // while loop (pre-tested)
    int i = 0;
    while (i < 5) {
        printf("%d ", i);
        i++;
    }

    // do-while loop (post-tested, executes at least once)
    int j = 0;
    do {
        printf("%d ", j);
        j++;
    } while (j < 5);

    // for loop
    for (int k = 0; k < 5; k++) {
        if (k == 2) continue; // Skip iteration
        if (k == 4) break;    // Exit loop
        printf("%d ", k);
    }
    
    // goto (use sparingly)
    if (score < 0) goto error;
    return 0;
    
error:
    printf("Invalid score\n");
    return 1;
}
```

### 6. I/O Functions
```c
#include <stdio.h>
int main() {
    int i = 42;
    float f = 3.14;
    char c = 'X';
    char str[50] = "Hello";
    
    // printf format specifiers
    printf("Int: %d, Float: %f, Char: %c, String: %s\n", i, f, c, str);
    printf("Long: %ld, Double: %lf, Unsigned: %u\n", 123456789L, 3.14159, 42U);
    printf("Hex: %x (or %X), Octal: %o, Pointer: %p\n", 255, 255, &i);
    printf("Scientific: %e\n", 12345.67);
    
    // Formatting: %-[width].[precision]specifier
    printf("Left align: '%-10d'\n", i);
    printf("Right align: '%10d'\n", i);
    printf("Precision: '%.2f'\n", f);
    
    // scanf
    int input_val;
    // Note: requires '&' for basic types
    scanf("%d", &input_val); 
    
    // Single character I/O
    char ch = getchar();
    putchar(ch);
    
    // String I/O
    // gets(str); // DANGEROUS! Removed in C11. Avoid.
    fgets(str, sizeof(str), stdin); // Safe alternative
    puts(str); // Prints string followed by newline
    
    return 0;
}
```

### 7. Language Behaviors
```c
int main() {
    // 1. Undefined Behavior (UB): Standard imposes no requirements. 
    // Anything can happen (crash, garbage values, seemingly normal execution).
    int arr[5];
    arr[10] = 42; // UB: Out-of-bounds array access
    
    int x;
    printf("%d", x); // UB: Using uninitialized automatic variable
    
    int a = 5;
    a = ++a + a++; // UB: Modifying variable multiple times between sequence points
    
    // 2. Unspecified Behavior: Standard provides two or more possibilities, 
    // but doesn't mandate which is chosen.
    // Example: Order of evaluation of arguments to a function.
    printf("%d %d\n", func1(), func2()); // Unspecified whether func1 or func2 runs first
    
    // 3. Implementation-defined Behavior: Like unspecified, but compiler must document its choice.
    size_t sz = sizeof(int); // Typically 4 bytes, but could be 2 or 8 depending on architecture/compiler
    
    return 0;
}
```

## Unit 2: Counting, Sorting and Searching

### 8. Arrays - 1D
```c
// Passing array to function (decays to pointer)
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
}

int main() {
    // Declaration
    int arr1[5]; 
    
    // Initialization
    int arr2[5] = {1, 2, 3, 4, 5};
    int arr3[] = {10, 20, 30}; // Size inferred (3)
    int arr4[5] = {1, 2};      // Remaining elements initialized to 0
    
    // Accessing
    arr1[0] = 100; // First element (0-indexed)
    
    printArray(arr2, 5);
    
    return 0;
}
```

### 9. Arrays - 2D
```c
// Passing 2D array: must specify columns
void print2DArray(int arr[][3], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // Declaration
    int matrix1[2][3];
    
    // Initialization (Row-major order in C)
    int matrix2[2][3] = {
        {1, 2, 3}, // Row 0
        {4, 5, 6}  // Row 1
    };
    
    // Equivalent initialization
    int matrix3[2][3] = {1, 2, 3, 4, 5, 6};
    
    // Accessing
    matrix1[0][1] = 42; // Row 0, Column 1
    
    print2DArray(matrix2, 2);
    
    return 0;
}
```

### 10. Pointers
```c
int main() {
    int a = 10;
    
    // Declaration and Initialization
    int *p = &a; // 'p' stores the address of 'a'
    
    // Dereferencing (accessing value at address)
    printf("Value of a: %d\n", *p);
    *p = 20; // Changes 'a' to 20
    
    // Pointer arithmetic
    int arr[5] = {10, 20, 30, 40, 50};
    int *ptr = arr; // Points to arr[0]
    
    printf("%d\n", *ptr);       // 10
    printf("%d\n", *(ptr + 2)); // 30 (Moves 2 * sizeof(int) bytes)
    ptr++;                      // Points to arr[1]
    
    // NULL pointer
    int *null_ptr = NULL; // Standard macro for 0/invalid memory
    // if (null_ptr != NULL) *null_ptr = 5; // Always check!
    
    // void pointer (Generic pointer)
    void *vptr = &a;
    // printf("%d", *vptr); // Error: cannot dereference void pointer
    printf("%d\n", *(int*)vptr); // Must cast first
    
    // Pointer to pointer
    int **pp = &p; // 'pp' stores address of 'p'
    printf("Value of a: %d\n", **pp);
    
    return 0;
}
```

### 11. Pointer to an Array
```c
int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    
    // p is a pointer to an array of 5 integers
    // Parentheses are crucial! Without them, it's an array of pointers.
    int (*p)[5] = &arr;
    
    // Accessing elements:
    // *p gives the array itself, which decays to a pointer to the first element
    printf("%d\n", (*p)[0]); // 1
    printf("%d\n", (*p)[2]); // 3
    
    // Pointer arithmetic moves by the size of the ENTIRE array (5 * 4 = 20 bytes)
    p++; // Points to memory after arr
    
    return 0;
}
```

### 12. Array of Pointers
```c
int main() {
    int a = 10, b = 20, c = 30;
    
    // arr is an array of 3 integer pointers
    int *arr[3];
    
    arr[0] = &a;
    arr[1] = &b;
    arr[2] = &c;
    
    for (int i = 0; i < 3; i++) {
        printf("%d ", *(arr[i])); // Prints 10 20 30
    }
    
    // Common use case: Array of strings
    char *names[] = {"Alice", "Bob", "Charlie"};
    printf("\n%s", names[1]); // Prints "Bob"
    
    return 0;
}
```

### 13. Functions
```c
// 1. Declaration (Prototype)
void swap(int *x, int *y);
int add(int a, int b);

int main() {
    int a = 5, b = 10;
    
    // 3. Call (Pass by Value)
    int sum = add(a, b); // a and b are not modified
    
    // 3. Call (Pass by Address/Reference)
    swap(&a, &b); // Passes addresses, allowing modification
    
    printf("a=%d, b=%d\n", a, b); // a=10, b=5
    return 0;
}

// 2. Definition
int add(int a, int b) {
    return a + b;
}

void swap(int *x, int *y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}
```

### 14. Function Pointers / Callback
```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

// Function accepting a callback (function pointer)
int compute(int a, int b, int (*operation)(int, int)) {
    return operation(a, b);
}

int main() {
    // Declare function pointer and assign
    int (*func_ptr)(int, int) = add;
    
    // Call via pointer
    printf("Add: %d\n", func_ptr(5, 3)); // 8
    
    // Pass as callback
    printf("Multiply: %d\n", compute(5, 3, multiply)); // 15
    
    return 0;
}
```

### 15. Storage Classes
```c
#include <stdio.h>

// 1. extern: Global scope, entire program lifetime, 0 default
// Defines the variable (allocates memory). Can be accessed in other files using 'extern int global_var;'
int global_var; 

void demo() {
    // 2. auto: Default for local variables. Local scope, function lifetime, garbage default
    auto int a = 1; // 'auto' keyword is rarely used explicitly
    
    // 3. register: Hint to compiler to store in CPU register for speed. Local scope/lifetime.
    // Cannot take address (&r) of a register variable.
    register int r = 10; 
    
    // 4. static: Local scope, ENTIRE PROGRAM lifetime, 0 default.
    // Initialized ONLY ONCE. Retains value between function calls.
    static int s = 1;
    
    a++; s++;
    printf("a=%d, s=%d\n", a, s);
}

int main() {
    demo(); // a=2, s=2
    demo(); // a=2, s=3 (s retained its value)
    return 0;
}
```

### 16. Recursion
```c
// 1. Factorial
int factorial(int n) {
    if (n == 0 || n == 1) return 1; // Base case
    return n * factorial(n - 1);    // Recursive case
}

// 2. Fibonacci
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 3. Binary Conversion (Print)
void printBinary(int n) {
    if (n == 0) return;
    printBinary(n / 2);
    printf("%d", n % 2);
}

// 4. Power (x^y)
int power(int x, int y) {
    if (y == 0) return 1;
    return x * power(x, y - 1);
}

// 5. Sum of digits
int sumDigits(int n) {
    if (n == 0) return 0;
    return (n % 10) + sumDigits(n / 10);
}

// 6. String Reverse (Print)
void printReverse(char *str) {
    if (*str == '\0') return;
    printReverse(str + 1);
    printf("%c", *str);
}
```

### 17. Searching
```c
// 1. Linear Search
int linearSearch(int arr[], int n, int key) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == key) return i; // Found
    }
    return -1; // Not found
}

// 2. Binary Search (Iterative) - Array must be sorted!
int binarySearchIterative(int arr[], int n, int key) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2; // Prevents overflow
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

// 3. Binary Search (Recursive)
int binarySearchRecursive(int arr[], int low, int high, int key) {
    if (low > high) return -1;
    
    int mid = low + (high - low) / 2;
    if (arr[mid] == key) return mid;
    
    if (arr[mid] < key)
        return binarySearchRecursive(arr, mid + 1, high, key);
    return binarySearchRecursive(arr, low, mid - 1, key);
}
```

### 18. Sorting
```c
void swap(int *xp, int *yp) {
    int temp = *xp; *xp = *yp; *yp = temp;
}

// 1. Bubble Sort (Compare adjacent, bubble largest to end)
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0; // Optimization
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped) break; // Array already sorted
    }
}

// 2. Selection Sort (Find min, swap with first unsorted element)
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(&arr[min_idx], &arr[i]);
    }
}

// 3. Insertion Sort (Insert element into its correct sorted position)
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}
```

## Unit 3: Text Processing and User-Defined Types

### 19. Strings
```c
#include <stdio.h>

int main() {
    // 1. Declaration and Initialization
    char str1[] = "Hello";        // Size inferred as 6 (includes '\0')
    char str2[10] = "World";      // Remaining chars initialized to '\0'
    char str3[] = {'C', 'o', 'd', 'e', '\0'}; // Explicit null termination
    
    char *str4 = "Pointer";       // String literal (Read-only memory!)
    // str4[0] = 'p'; // UB: Segfault on many systems
    
    // 2. Reading
    char input[50];
    
    // scanf("%s", input); // Stops at whitespace. Unsafe (buffer overflow).
    
    // Read string with spaces until newline
    // scanf("%[^\n]", input); 
    
    // Safest way to read strings
    printf("Enter string: ");
    fgets(input, sizeof(input), stdin);
    
    // 3. Traversal
    for (int i = 0; str1[i] != '\0'; i++) {
        printf("%c-", str1[i]);
    }
    
    return 0;
}
```

### 20. String Functions (string.h)
```c
#include <stdio.h>
#include <string.h>

int main() {
    char s1[50] = "Apple";
    char s2[] = "Banana";
    
    // 1. Length (excludes '\0')
    size_t len = strlen(s1); // 5
    
    // 2. Copy (dest, src)
    strcpy(s1, s2); // s1 becomes "Banana"
    // Safe copy: limits max characters copied
    strncpy(s1, "Cherry", sizeof(s1) - 1); 
    
    // 3. Concatenate (append src to dest)
    strcat(s1, " Pie"); // "Cherry Pie"
    strncat(s1, "!!!", 2); // Appends up to 2 chars: "Cherry Pie!!"
    
    // 4. Compare (returns 0 if equal, <0 if s1<s2, >0 if s1>s2)
    int cmp = strcmp("Cat", "Dog"); // < 0
    int ncmp = strncmp("Apple", "App", 3); // 0 (first 3 match)
    
    // 5. Search char in string (returns pointer to first occurrence, or NULL)
    char *ptr1 = strchr(s1, 'P'); 
    
    // 6. Search string in string (returns pointer to first occurrence, or NULL)
    char *ptr2 = strstr(s1, "Pie");
    
    // 7. Tokenize (split string by delimiters)
    char data[] = "apple,banana;cherry";
    char *token = strtok(data, ",;");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, ",;"); // Subsequent calls use NULL
    }
    
    // 8. sprintf (write formatted data to string)
    char buffer[100];
    sprintf(buffer, "Value: %d", 42);
    
    // 9. sscanf (read formatted data from string)
    int val;
    sscanf("Data 123", "Data %d", &val); // val = 123
    
    return 0;
}
```

### 21. Command Line Arguments
```c
#include <stdio.h>

// argc: Argument count (includes program name)
// argv: Array of strings (argument values)
int main(int argc, char *argv[]) {
    printf("Number of arguments: %d\n", argc);
    
    // argv[0] is typically the name of the executable
    for (int i = 0; i < argc; i++) {
        printf("Argument %d: %s\n", i, argv[i]);
    }
    
    // Usage: ./program arg1 arg2
    // argc = 3
    // argv[0] = "./program"
    // argv[1] = "arg1"
    // argv[2] = "arg2"
    
    return 0;
}
```

### 22. Dynamic Memory Management
```c
#include <stdio.h>
#include <stdlib.h> // Required for malloc, calloc, realloc, free

int main() {
    int n = 5;
    
    // 1. malloc: Allocates uninitialized memory
    int *arr1 = (int*)malloc(n * sizeof(int));
    if (arr1 == NULL) { // Always check for failure!
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // 2. calloc: Allocates memory AND initializes to zero
    // Takes 2 args: num_elements, size_of_element
    int *arr2 = (int*)calloc(n, sizeof(int));
    if (arr2 == NULL) { /* handle error */ }
    
    // 3. realloc: Resizes previously allocated memory
    // Data is preserved up to min(old_size, new_size)
    n = 10;
    int *temp = (int*)realloc(arr1, n * sizeof(int));
    if (temp == NULL) {
        printf("Reallocation failed. Original arr1 is untouched.\n");
    } else {
        arr1 = temp; // Assign back if successful
    }
    
    // 4. free: Releases memory back to heap
    free(arr1);
    free(arr2);
    
    // Best practice: prevent dangling pointers
    arr1 = NULL;
    arr2 = NULL;
    
    return 0;
}
```

### 23. Structures
```c
#include <stdio.h>
#include <string.h>

// Definition
struct Date {
    int day;
    int month;
    int year;
};

// Nested Structure
struct Student {
    int id;
    char name[50];
    struct Date dob; // Nested
};

// Self-referential structure (used in Linked Lists/Trees)
struct Node {
    int data;
    struct Node *next; // Pointer to same struct type
};

int main() {
    // Declaration & Initialization
    struct Date d1 = {15, 8, 2023};
    struct Date d2; // Uninitialized
    
    // Accessing members (.)
    d2.day = 10;
    
    struct Student s1 = {101, "Alice", {1, 1, 2000}};
    
    // Accessing nested members
    printf("Year: %d\n", s1.dob.year);
    
    // Pointer to Structure
    struct Student *ptr = &s1;
    
    // Accessing via pointer (->)
    printf("Name: %s\n", ptr->name);
    // Equivalent to: (*ptr).name
    
    return 0;
}
```

### 24. #pragma
```c
#include <stdio.h>

// 1. #pragma pack: Controls structure padding/alignment
#pragma pack(1) // Pack on 1-byte boundaries (no padding)
struct PackedStruct {
    char c;    // 1 byte
    int i;     // 4 bytes
               // Total = 5 bytes (Normally might be 8 due to padding)
};
#pragma pack() // Reset to default

// 2. #pragma startup/exit (Compiler specific, e.g., Turbo C/GCC via attributes)
void myStartup();
void myExit();
#pragma startup myStartup  // Runs before main()
#pragma exit myExit        // Runs after main() exits

// 3. #pragma warn (Compiler specific, suppresses warnings)
#pragma warn -rvl // Suppress "function should return a value" warning

int main() {
    printf("Size: %zu\n", sizeof(struct PackedStruct));
    return 0;
}
```

### 25 & 26. Array of Structures & Pointer to Structures
```c
#include <stdio.h>

struct Point {
    int x;
    int y;
};

int main() {
    // Array of Structures
    struct Point pts[3] = { {1,2}, {3,4}, {5,6} };
    
    // Iteration
    for (int i = 0; i < 3; i++) {
        printf("(%d,%d) ", pts[i].x, pts[i].y);
    }
    printf("\n");
    
    // Pointer to Array of Structures
    struct Point *p = pts; // Points to pts[0]
    
    // Pointer arithmetic with structures
    printf("First: %d\n", p->x);
    p++; // Moves to pts[1]
    printf("Second: %d\n", p->x);
    printf("Third: %d\n", (p+1)->x); // Access pts[2] without changing p
    
    return 0;
}
```

### 27. Passing Structures to Functions
```c
#include <stdio.h>

struct Rect {
    int w;
    int h;
};

// 1. By Value (Creates a full copy - slow for large structs)
void printArea(struct Rect r) {
    printf("Area: %d\n", r.w * r.h);
}

// 2. By Pointer (Efficient, allows modification)
void scaleRect(struct Rect *r, int factor) {
    r->w *= factor;
    r->h *= factor;
}

// 3. Array of Structures (Passed as pointer)
void initRects(struct Rect arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i].w = i + 1;
        arr[i].h = i + 1;
    }
}

int main() {
    struct Rect myRect = {5, 10};
    printArea(myRect);
    
    scaleRect(&myRect, 2); // Pass address
    
    struct Rect list[5];
    initRects(list, 5); // Decays to pointer
    
    return 0;
}
```

### 28. Bit Fields
```c
#include <stdio.h>

// Bit fields allow packing data tightly by specifying exact bit width
struct Flags {
    unsigned int is_ready : 1;  // Uses 1 bit
    unsigned int error_code: 3; // Uses 3 bits (values 0-7)
    int status : 4;             // Uses 4 bits (values -8 to 7)
    
    // Restrictions:
    // 1. Cannot use arrays of bit fields
    // 2. Cannot take address (&) of a bit field
    // 3. Type must be int, signed int, or unsigned int
    // 4. Cannot use float/double
};

int main() {
    struct Flags f;
    f.is_ready = 1;
    f.error_code = 5;
    
    // f.error_code = 10; // Warning: truncation (3 bits max is 7)
    
    printf("Size of struct: %zu bytes\n", sizeof(f)); // Usually 4 bytes (1 int)
    return 0;
}
```

### 29. Unions
```c
#include <stdio.h>

// Union: All members share the SAME memory location.
// Size of union is the size of its largest member.
union Data {
    int i;      // 4 bytes
    float f;    // 4 bytes
    char str[20]; // 20 bytes (Largest -> Size of union is 20)
};

int main() {
    union Data d;
    
    // You can only use ONE member reliably at a time.
    d.i = 10;
    printf("i: %d\n", d.i);
    
    // Overwrites memory used by 'i'
    d.f = 220.5;
    printf("f: %f\n", d.f);
    printf("i: %d\n", d.i); // i is now corrupted/garbage
    
    return 0;
}
```

### 30. Enums
```c
#include <stdio.h>

// Enum: Defines a set of named integer constants
enum Color { RED, GREEN, BLUE }; // Default: RED=0, GREEN=1, BLUE=2

// Custom values
enum State { 
    OFF = 0, 
    STARTING = 5, 
    RUNNING // Automatically 6
};

int main() {
    enum Color c = GREEN;
    
    // Usage in switch
    switch(c) {
        case RED:   printf("Stop\n"); break;
        case GREEN: printf("Go\n"); break;
        case BLUE:  printf("Water\n"); break;
    }
    
    return 0;
}
```

### 31. typedef
```c
#include <stdio.h>

// typedef creates an alias for an existing type

// Basic types
typedef unsigned long long int uint64;
uint64 large_number = 1000000;

// Structures (very common, eliminates need to write 'struct' everywhere)
typedef struct {
    int x, y;
} Point;

// Function pointers
typedef int (*MathFunc)(int, int);

int add(int a, int b) { return a + b; }

int main() {
    Point p1 = {10, 20}; // No 'struct' keyword needed
    
    MathFunc operation = add;
    printf("%d\n", operation(5, 5));
    
    return 0;
}
```

### 32. Linked List
```c
#include <stdio.h>
#include <stdlib.h>

// Node structure
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Create node
Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Insert at beginning
void insertFirst(Node** head, int data) {
    Node* newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

// Display
void display(Node* head) {
    Node* temp = head;
    while (temp != NULL) {
        printf("%d -> ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

int main() {
    Node* head = NULL;
    insertFirst(&head, 10);
    insertFirst(&head, 20);
    display(head); // 20 -> 10 -> NULL
    return 0;
}
```

### 33 & 34. Stack & Queue (Array-based)
```c
#include <stdio.h>
#define MAX 100

// STACK
typedef struct {
    int arr[MAX];
    int top;
} Stack;

void initStack(Stack* s) { s->top = -1; }
int isStackEmpty(Stack* s) { return s->top == -1; }
int isStackFull(Stack* s) { return s->top == MAX - 1; }

void push(Stack* s, int val) {
    if (isStackFull(s)) return;
    s->arr[++(s->top)] = val;
}

int pop(Stack* s) {
    if (isStackEmpty(s)) return -1;
    return s->arr[(s->top)--];
}

// QUEUE
typedef struct {
    int arr[MAX];
    int front, rear;
} Queue;

void initQueue(Queue* q) { q->front = -1; q->rear = -1; }

void enqueue(Queue* q, int val) {
    if (q->rear == MAX - 1) return;
    if (q->front == -1) q->front = 0;
    q->arr[++(q->rear)] = val;
}

int dequeue(Queue* q) {
    if (q->front == -1 || q->front > q->rear) return -1;
    return q->arr[(q->front)++];
}
```

### 35. Priority Queue
```c
#include <stdio.h>
#include <stdlib.h>

// Component struct
struct component { 
    char details[20];
    int priority;
};

// Node struct
struct node {
    struct component c;
    struct node *link;
};

// Priority Queue struct
struct priority_queue {
    struct node *head; 
};

// Enqueue (Ascending PQ - Lowest number = highest priority)
void enqueuePQ(struct priority_queue *pq, struct component new_c) {
    struct node *newNode = (struct node*)malloc(sizeof(struct node));
    newNode->c = new_c;
    newNode->link = NULL;

    if (pq->head == NULL || pq->head->c.priority > new_c.priority) {
        newNode->link = pq->head;
        pq->head = newNode;
    } else {
        struct node *temp = pq->head;
        while (temp->link != NULL && temp->link->c.priority <= new_c.priority) {
            temp = temp->link;
        }
        newNode->link = temp->link;
        temp->link = newNode;
    }
}
```

## Unit 4: File Handling and Portable Programming

### 36. File I/O Redirection
```bash
# Executed in terminal, not in C code
# < redirects standard input (stdin) from a file
./program < input.txt

# > redirects standard output (stdout) to a file (overwrites)
./program > output.txt

# Combine both
./program < input.txt > output.txt
```

### 37. File Handling Functions
```c
#include <stdio.h>

int main() {
    // 1. fopen modes:
    // "r" (read), "w" (write/overwrite), "a" (append)
    // "r+" (read/write), "w+" (write/read), "a+" (append/read)
    // Append "b" for binary files: "rb", "wb", etc.
    
    FILE *fp = fopen("data.txt", "w");
    if (fp == NULL) {
        perror("Error opening file"); // Prints OS error message
        return 1;
    }
    
    // 2. Formatted I/O
    fprintf(fp, "Name: %s, Age: %d\n", "Alice", 25);
    
    // 3. String I/O
    fputs("Hello File\n", fp);
    
    // 4. Character I/O
    fputc('X', fp);
    
    fclose(fp); // Always close!
    
    // Read example
    fp = fopen("data.txt", "r");
    if (fp == NULL) return 1;
    
    char buffer[100];
    // Read string until newline or EOF
    if (fgets(buffer, sizeof(buffer), fp) != NULL) {
        printf("Read: %s", buffer);
    }
    
    // Read formatted
    char name[20]; int age;
    fscanf(fp, "Name: %[^,], Age: %d", name, &age);
    
    // 5. Check End of File / Errors
    if (feof(fp)) printf("Reached EOF\n");
    if (ferror(fp)) printf("Read error occurred\n");
    
    fclose(fp);
    
    // 6. Binary I/O & File Positioning
    struct Data { int id; float val; } d = {1, 3.14};
    FILE *bfp = fopen("data.bin", "wb+");
    
    fwrite(&d, sizeof(struct Data), 1, bfp); // Write 1 element
    
    rewind(bfp); // Move pointer to start. Equivalent to fseek(bfp, 0, SEEK_SET);
    // fseek options: SEEK_SET (start), SEEK_CUR (current), SEEK_END (end)
    
    struct Data read_d;
    fread(&read_d, sizeof(struct Data), 1, bfp);
    
    long pos = ftell(bfp); // Get current position in bytes
    
    fclose(bfp);
    return 0;
}
```

### 38. Header Files (Include Guards)
```c
// mymath.h
#ifndef MYMATH_H  // If not defined
#define MYMATH_H  // Define it

// Declarations only!
int add(int a, int b);
#define PI 3.14159

#endif // MYMATH_H

// main.c
// #include "mymath.h"  // "" for local files
```

### 39. Comparing User-defined vs Built-in
```c
// Built-in: strlen(str) from <string.h>

// User-defined emulation:
int my_strlen(const char *str) {
    int count = 0;
    while (*str != '\0') {
        count++;
        str++;
    }
    return count;
}
```

### 40. Variable Length Arguments
```c
#include <stdio.h>
#include <stdarg.h> // Required

// Must have at least one fixed argument (count)
int sum(int count, ...) {
    va_list args;
    va_start(args, count); // Initialize args
    
    int total = 0;
    for (int i = 0; i < count; i++) {
        // Retrieve next argument, must specify type
        total += va_arg(args, int); 
    }
    
    va_end(args); // Cleanup
    return total;
}

int main() {
    printf("Sum: %d\n", sum(3, 10, 20, 30)); // 60
    return 0;
}
```

### 41. Environment Variables
```c
#include <stdio.h>
#include <stdlib.h> // Required for getenv, setenv

// envp is an array of strings like "PATH=/usr/bin..."
int main(int argc, char *argv[], char *envp[]) {
    // 1. Access using getenv
    char *path = getenv("PATH");
    if (path != NULL) printf("PATH: %s\n", path);
    
    // 2. Modify environment
    // 1 means overwrite if exists, 0 means don't overwrite
    setenv("MY_VAR", "HelloWorld", 1); 
    
    // putenv("MY_VAR=HelloWorld"); // Alternative (modifies string directly)
    
    // 3. Access using environ
    extern char **environ;
    // for (char **e = environ; *e != NULL; e++) printf("%s\n", *e);
    
    return 0;
}
```

### 42. Preprocessor Directives
```c
#include <stdio.h> // standard library (<>)
#include "my_header.h" // local file ("")

// Constants
#define PI 3.14
#define MAX 100

// Macros with parameters (Always parenthesize arguments!)
#define SQR(x) ((x) * (x))
// Without parens: SQR(2+3) -> 2+3*2+3 = 11 (Wrong!)
// With parens: ((2+3) * (2+3)) -> 25 (Correct!)

#undef MAX // Removes definition

int main() {
    // Predefined macros
    printf("File: %s\n", __FILE__);
    printf("Line: %d\n", __LINE__);
    printf("Date compiled: %s\n", __DATE__);
    printf("Time compiled: %s\n", __TIME__);
    
    // printf("Standard C: %d\n", __STDC__); 
    // printf("Version: %ld\n", __STDC_VERSION__);
    
    return 0;
}
```

### 43 & 44. Conditional Compilation & Portable Programming
```c
#include <stdio.h>

#define DEBUG 1

int main() {
    // 1. Feature flags
#if DEBUG
    printf("Debug mode active\n");
#else
    printf("Production mode\n");
#endif

    // 2. Portability
#ifdef __MINGW32__
    printf("Compiling for Windows\n");
#elif defined(__unix__)
    printf("Compiling for Unix/Linux\n");
#elif defined(__APPLE__)
    printf("Compiling for Mac\n");
#else
    printf("Unknown OS\n");
#endif

    // 3. #ifndef (commonly used in include guards)
#ifndef CONFIG_VAL
    #define CONFIG_VAL 100
#endif

    return 0;
}
```

### 45. qsort / bsearch
```c
#include <stdio.h>
#include <stdlib.h>

// Compare function for qsort and bsearch (ascending order)
// Returns < 0 if a < b, 0 if a == b, > 0 if a > b
int compare_ints(const void *a, const void *b) {
    int int_a = *(const int *)a;
    int int_b = *(const int *)b;
    return (int_a - int_b);
}

int main() {
    int arr[] = {40, 10, 100, 90, 20, 25};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    // qsort(array, element_count, element_size, compare_function)
    qsort(arr, n, sizeof(int), compare_ints);
    
    // bsearch(key_pointer, array, element_count, element_size, compare_function)
    // Array MUST be sorted first!
    int key = 40;
    int *item = (int*) bsearch(&key, arr, n, sizeof(int), compare_ints);
    
    if (item != NULL) {
        printf("Found item: %d\n", *item);
    } else {
        printf("Item not found\n");
    }
    
    return 0;
}
```

### 46. ctype.h functions
```c
#include <stdio.h>
#include <ctype.h>

int main() {
    char c = 'A';
    
    // Character testing functions (return non-zero if true, 0 if false)
    int is_alpha = isalpha(c);   // A-Z, a-z
    int is_digit = isdigit(c);   // 0-9
    int is_alnum = isalnum(c);   // A-Z, a-z, 0-9
    int is_space = isspace(' '); // space, \t, \n, \v, \f, \r
    int is_lower = islower(c);   // a-z
    int is_upper = isupper(c);   // A-Z
    int is_punct = ispunct('!'); // Punctuation characters
    
    // Character conversion functions
    char lower_c = tolower(c);   // Converts 'A' to 'a'
    char upper_c = toupper('b'); // Converts 'b' to 'B'
    
    return 0;
}
```

### 47. errno / strerror / perror
```c
#include <stdio.h>
#include <errno.h>    // For errno
#include <string.h>   // For strerror

int main() {
    // Attempt to open a non-existent file
    FILE *fp = fopen("unreal.txt", "r");
    
    if (fp == NULL) {
        // 1. Using errno directly (global variable set by library functions)
        printf("Error code: %d\n", errno);
        
        // 2. Using strerror (converts error code to string)
        printf("Error message: %s\n", strerror(errno));
        
        // 3. Using perror (prints prefix string + colon + space + error message)
        perror("File opening failed"); 
        // Output: "File opening failed: No such file or directory"
    } else {
        fclose(fp);
    }
    
    return 0;
}
```

### 48. Algorithms & Flowcharts
```c
/*
An Algorithm is a step-by-step procedure to solve a given problem.
Characteristics:
- Finiteness: Must terminate after a finite number of steps.
- Definiteness: Each step must be clearly defined and unambiguous.
- Input: Zero or more inputs.
- Output: One or more outputs.
- Effectiveness: Operations must be basic enough to be carried out exactly.

Example Algorithm (Find maximum of two numbers):
1. Start
2. Read two numbers, A and B
3. If A > B then
4.   Print A is maximum
5. Else
6.   Print B is maximum
7. End

Flowchart Symbols:
- Oval / Pill shape: Start / Stop
- Parallelogram: Input / Output
- Rectangle: Processing / Assignment
- Diamond: Decision / Condition (Yes/No branching)
- Arrows: Flow of control
- Circle: Connector (to connect different parts of a flowchart)
*/
```

### 49. Identifiers and Keywords
```c
// Identifiers: Names given to variables, functions, arrays, etc.
// Rules: 
// 1. Made of A-Z, a-z, 0-9, and _ (underscore).
// 2. Must start with a letter or underscore (not a digit).
// 3. Case-sensitive. Cannot be the same as keywords.

// Keywords: Reserved words in C (cannot be used as identifiers)
// Examples: auto, break, case, char, const, continue, default, do
// double, else, enum, extern, float, for, goto, if
// int, long, register, return, short, signed, sizeof, static
// struct, switch, typedef, union, unsigned, void, volatile, while
```

### 50. Data Types Limits (<limits.h> and <float.h>)
```c
#include <stdio.h>
#include <limits.h>
#include <float.h>

int main() {
    // Integer limits
    printf("Minimum int: %d\n", INT_MIN);
    printf("Maximum int: %d\n", INT_MAX);
    printf("Maximum unsigned int: %u\n", UINT_MAX);
    
    // Size guarantees (Secondary rule)
    // sizeof(short int) <= sizeof(int) <= sizeof(long int) <= sizeof(long long int)
    // sizeof(float) <= sizeof(double) <= sizeof(long double)
    
    return 0;
}
```

### 51. I/O Function Return Values
```c
#include <stdio.h>

int main() {
    // printf returns the number of characters successfully printed
    // Returns < 0 on failure
    int chars_printed = printf("Hello\n"); // Prints 6 ('H','e','l','l','o','\n')
    
    // scanf returns the number of items successfully read and assigned
    // Returns 0 if no match, or EOF (<0) on input error/end-of-file
    int a, b;
    int items_read = scanf("%d %d", &a, &b); 
    
    return 0;
}
```

### 52. File I/O Redirection (Command Line)
```c
// When compiling: gcc prog.c -o prog
// Running with input redirection: ./prog < input.txt
// Running with output redirection: ./prog > output.txt

#include <stdio.h>

int main() {
    int num;
    // Reads from stdin, but if run with `< input.txt`, it reads from the file instead
    while (scanf("%d", &num) != EOF) {
        // Prints to stdout, but if run with `> output.txt`, it writes to the file
        printf("Read: %d\n", num);
    }
    return 0;
}
// Note: Limited to standard input/output. No error handling for missing files in C code.
```
