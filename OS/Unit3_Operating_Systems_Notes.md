# Unit 3: Memory Management

**A Complete Exam Study Reference — PES University (UE24CS242B: Operating Systems)**

---

## Table of Contents

1. [Hardware and Control Structures of Main Memory](#1-hardware-and-control-structures-of-main-memory)
   - 1.1 [Memory Hierarchy, Registers, and Cache Stalls](#11-memory-hierarchy-registers-and-cache-stalls)
   - 1.2 [Uniprogramming vs. Multiprogramming Execution Environments](#12-uniprogramming-vs-multiprogramming-execution-environments)
   - 1.3 [Memory Protection: Base and Limit Registers](#13-memory-protection-base-and-limit-registers)
   - 1.4 [Address Binding Stages (Compile Time, Load Time, Execution Time)](#14-address-binding-stages-compile-time-load-time-execution-time)
   - 1.5 [Logical vs. Physical Address Space and the Memory Management Unit (MMU)](#15-logical-vs-physical-address-space-and-the-memory-management-unit-mmu)
   - 1.6 [Multistep Processing of a User Program (Compiler to Execution)](#16-multistep-processing-of-a-user-program-compiler-to-execution)
   - 1.7 [Dynamic Relocation Hardware Architecture](#17-dynamic-relocation-hardware-architecture)
2. [Dynamic Loading, Linking, and Shared Libraries](#2-dynamic-loading-linking-and-shared-libraries)
   - 2.1 [Dynamic Loading (Lazy Loading of Subroutines)](#21-dynamic-loading-lazy-loading-of-subroutines)
   - 2.2 [Static Linking vs. Dynamic Linking](#22-static-linking-vs-dynamic-linking)
   - 2.3 [Shared Libraries, DLLs, and Stub Execution Mechanics](#23-shared-libraries-dlls-and-stub-execution-mechanics)
   - 2.4 [Versioning, Security, and Memory Space Savings](#24-versioning-security-and-memory-space-savings)
3. [Memory-Mapped Files and mmap() System Call](#3-memory-mapped-files-and-mmap-system-call)
   - 3.1 [Motivation: Traditional Read/Write I/O vs. Memory Mapping](#31-motivation-traditional-readwrite-io-vs-memory-mapping)
   - 3.2 [The mmap() System Call Mechanics and API Signatures](#32-the-mmap-system-call-mechanics-and-api-signatures)
   - 3.3 [Page Cache Integration and Demand-Paged File I/O](#33-page-cache-integration-and-demand-paged-file-io)
   - 3.4 [MAP_SHARED vs. MAP_PRIVATE (Copy-on-Write) Semantics](#34-map_shared-vs-map_private-copy-on-write-semantics)
   - 3.5 [Disk Synchronization with msync() and Zero-Copy IPC](#35-disk-synchronization-with-msync-and-zero-copy-ipc)
4. [Swapping and Contiguous Memory Allocation](#4-swapping-and-contiguous-memory-allocation)
   - 4.1 [Classical Process Swapping and the Backing Store](#41-classical-process-swapping-and-the-backing-store)
   - 4.2 [Swapping Performance and Context Switch Latency Derivation](#42-swapping-performance-and-context-switch-latency-derivation)
   - 4.3 [Constraints on Swapping: Pending I/O and Double Buffering](#43-constraints-on-swapping-pending-io-and-double-buffering)
   - 4.4 [Modern OS Perspective: Whole-Process Swapping vs. Page Swapping](#44-modern-os-perspective-whole-process-swapping-vs-page-swapping)
   - 4.5 [Contiguous Allocation Models (Fixed vs. Dynamic Partitions)](#45-contiguous-allocation-models-fixed-vs-dynamic-partitions)
   - 4.6 [Dynamic Storage Allocation Algorithms: First-Fit, Best-Fit, Worst-Fit](#46-dynamic-storage-allocation-algorithms-first-fit-best-fit-worst-fit)
   - 4.7 [Performance, Memory Utilization, and Search Complexity Analysis](#47-performance-memory-utilization-and-search-complexity-analysis)
5. [Fragmentation: Internal vs. External and Remediation](#5-fragmentation-internal-vs-external-and-remediation)
   - 5.1 [Internal Fragmentation: Causes, Formulas, and Boundary Allocations](#51-internal-fragmentation-causes-formulas-and-boundary-allocations)
   - 5.2 [External Fragmentation and the 50-Percent Rule Analysis](#52-external-fragmentation-and-the-50-percent-rule-analysis)
   - 5.3 [Memory Compaction: Dynamic Relocation Requirements and I/O Cost](#53-memory-compaction-dynamic-relocation-requirements-and-io-cost)
   - 5.4 [Kernel Memory Allocation Alternatives: Buddy System and Slab Allocator](#54-kernel-memory-allocation-alternatives-buddy-system-and-slab-allocator)
6. [Segmentation](#6-segmentation)
   - 6.1 [Programmer's View of Memory (Logical Segments)](#61-programmers-view-of-memory-logical-segments)
   - 6.2 [Segmentation Architecture: Logical Address 2-Tuple (s, d)](#62-segmentation-architecture-logical-address-2-tuple-s-d)
   - 6.3 [Segment Table Hardware: Base, Limit, STBR, and STLR](#63-segment-table-hardware-base-limit-stbr-and-stlr)
   - 6.4 [Address Translation Flow and Hardware Protection Traps](#64-address-translation-flow-and-hardware-protection-traps)
   - 6.5 [Segment Sharing: Shared Reentrant Code and Data Segments](#65-segment-sharing-shared-reentrant-code-and-data-segments)
   - 6.6 [Segmentation in x86/x86-64: Historical GDT/LDT to Modern Flat Memory Model](#66-segmentation-in-x86x86-64-historical-gdtldt-to-modern-flat-memory-model)
7. [Paging Architecture and Hardware Support](#7-paging-architecture-and-hardware-support)
   - 7.1 [Paging Concept: Decoupling Logical and Physical Space](#71-paging-concept-decoupling-logical-and-physical-space)
   - 7.2 [Frames, Pages, and Hardware Translation Mapping](#72-frames-pages-and-hardware-translation-mapping)
   - 7.3 [Mathematical Address Decomposition: Page Number (p) and Offset (d)](#73-mathematical-address-decomposition-page-number-p-and-offset-d)
   - 7.4 [Internal Fragmentation in Paging: Mathematical Bound & Average Case](#74-internal-fragmentation-in-paging-mathematical-bound--average-case)
   - 7.5 [Operating System Support: Free-Frame Allocation and Frame Tables](#75-operating-system-support-free-frame-allocation-and-frame-tables)
   - 7.6 [Protection Bits and the Valid-Invalid Bit in Page Tables](#76-protection-bits-and-the-valid-invalid-bit-in-page-tables)
   - 7.7 [Shared Pages and Reentrant Code Execution](#77-shared-pages-and-reentrant-code-execution)
8. [Translation Lookaside Buffer (TLB) and Effective Access Time](#8-translation-lookaside-buffer-tlb-and-effective-access-time)
   - 8.1 [The Two-Memory-Access Problem with In-Memory Page Tables](#81-the-two-memory-access-problem-with-in-memory-page-tables)
   - 8.2 [TLB Architecture: Associative High-Speed Hardware Cache](#82-tlb-architecture-associative-high-speed-hardware-cache)
   - 8.3 [TLB Hit, TLB Miss, and Hardware vs. Software Handlers](#83-tlb-hit-tlb-miss-and-hardware-vs-software-handlers)
   - 8.4 [Context Switching, TLB Flushes, and Address Space Identifiers (ASIDs)](#84-context-switching-tlb-flushes-and-address-space-identifiers-asids)
   - 8.5 [Derivation of Effective Access Time (EAT) with TLB](#85-derivation-of-effective-access-time-eat-with-tlb)
   - 8.6 [Multi-Level Paging EAT Derivations and Impact on Performance](#86-multi-level-paging-eat-derivations-and-impact-on-performance)
9. [Hierarchical, Hashed, and Inverted Page Table Structures](#9-hierarchical-hashed-and-inverted-page-table-structures)
   - 9.1 [The Scalability Crisis of Flat Page Tables in 32-bit and 64-bit Systems](#91-the-scalability-crisis-of-flat-page-tables-in-32-bit-and-64-bit-systems)
   - 9.2 [Hierarchical Paging (Forward-Mapped / Multi-Level Page Tables)](#92-hierarchical-paging-forward-mapped--multi-level-page-tables)
   - 9.3 [32-bit Two-Level Paging (x86 Page Directory and Page Table)](#93-32-bit-two-level-paging-x86-page-directory-and-page-table)
   - 9.4 [64-bit Paging: 4-Level (x86-64 PML4) and 5-Level (PML5) Paging](#94-64-bit-paging-4-level-x86-64-pml4-and-5-level-pml5-paging)
   - 9.5 [Hashed Page Tables and Clustered Page Tables for Sparse Spaces](#95-hashed-page-tables-and-clustered-page-tables-for-sparse-spaces)
   - 9.6 [Inverted Page Tables: Architecture, Physical Frame Mapping, and Hash Anchor Tables](#96-inverted-page-tables-architecture-physical-frame-mapping-and-hash-anchor-tables)
   - 9.7 [Challenges of Inverted Page Tables: Search Latency and Shared Memory](#97-challenges-of-inverted-page-tables-search-latency-and-shared-memory)
10. [Virtual Memory and Demand Paging](#10-virtual-memory-and-demand-paging)
    - 10.1 [The Virtual Memory Abstraction and Sparse Address Spaces](#101-the-virtual-memory-abstraction-and-sparse-address-spaces)
    - 10.2 [Demand Paging vs. Pre-Paging and Pure Demand Paging](#102-demand-paging-vs-pre-paging-and-pure-demand-paging)
    - 10.3 [The Valid-Invalid Bit in Demand Paging](#103-the-valid-invalid-bit-in-demand-paging)
    - 10.4 [Detailed 6-Step Page-Fault Handling Sequence](#104-detailed-6-step-page-fault-handling-sequence)
    - 10.5 [Major (Hard) vs. Minor (Soft) Page Faults](#105-major-hard-vs-minor-soft-page-faults)
    - 10.6 [File-Backed vs. Anonymous Memory Management](#106-file-backed-vs-anonymous-memory-management)
    - 10.7 [Mobile OS Memory Constraints (iOS and Android Without Swap)](#107-mobile-os-memory-constraints-ios-and-android-without-swap)
    - 10.8 [Architectural Challenges: Instruction Restart and Auto-Increment/Decrement](#108-architectural-challenges-instruction-restart-and-auto-incrementdecrement)
    - 10.9 [Demand Paging Performance and EAT Mathematical Derivations](#109-demand-paging-performance-and-eat-mathematical-derivations)
11. [Copy-on-Write (COW) and Page Replacement Foundations](#11-copy-on-write-cow-and-page-replacement-foundations)
    - 11.1 [Process Creation Optimization: fork(), exec(), and vfork()](#111-process-creation-optimization-fork-exec-and-vfork)
    - 11.2 [Hardware Mechanism of Copy-on-Write: Read-Only Trap and Frame Duplication](#112-hardware-mechanism-of-copy-on-write-read-only-trap-and-frame-duplication)
    - 11.3 [Memory Over-Allocation and the Need for Page Replacement](#113-memory-over-allocation-and-the-need-for-page-replacement)
    - 11.4 [Victim Frame Eviction Sequence and Page Table Invalidation](#114-victim-frame-eviction-sequence-and-page-table-invalidation)
    - 11.5 [The Dirty (Modify) Bit Optimization: Halving I/O Transfer Overhead](#115-the-dirty-modify-bit-optimization-halving-io-transfer-overhead)
    - 11.6 [Kernel Paging Daemons: Linux kswapd, Watermarks (Min, Low, High)](#116-kernel-paging-daemons-linux-kswapd-watermarks-min-low-high)
12. [Page Replacement Algorithms](#12-page-replacement-algorithms)
    - 12.1 [Evaluation Metrics and Memory Reference Strings](#121-evaluation-metrics-and-memory-reference-strings)
    - 12.2 [First-In, First-Out (FIFO) Algorithm](#122-first-in-first-out-fifo-algorithm)
    - 12.3 [Belady's Anomaly: Definition, Demonstration, and Stack Algorithms](#123-beladys-anomaly-definition-demonstration-and-stack-algorithms)
    - 12.4 [Optimal (OPT / MIN / Clairvoyant) Page Replacement Algorithm](#124-optimal-opt--min--clairvoyant-page-replacement-algorithm)
    - 12.5 [Least Recently Used (LRU) Algorithm: Theory and Stack Property](#125-least-recently-used-lru-algorithm-theory-and-stack-property)
    - 12.6 [LRU Implementation Bottlenecks: Hardware Counters vs. Doubly Linked Stacks](#126-lru-implementation-bottlenecks-hardware-counters-vs-doubly-linked-stacks)
    - 12.7 [Approximating LRU: Reference Bits and Additional-Reference-Bits History Register](#127-approximating-lru-reference-bits-and-additional-reference-bits-history-register)
    - 12.8 [Clock (Second-Chance) Page Replacement Algorithm: Mechanism, Hand Movement, and Dynamics](#128-clock-second-chance-page-replacement-algorithm)
    - 12.9 [Enhanced Second-Chance Algorithm: The (Reference, Modify) 4-Class Selection](#129-enhanced-second-chance-algorithm-the-reference-modify-4-class-selection)
    - 12.10 [Counting-Based Algorithms: LFU and MFU](#1210-counting-based-algorithms-lfu-and-mfu)
13. [Frame Allocation, NUMA, and Thrashing](#13-frame-allocation-numa-and-thrashing)
    - 13.1 [Architecture-Enforced Minimum and Maximum Frame Constraints](#131-architecture-enforced-minimum-and-maximum-frame-constraints)
    - 13.2 [Equal vs. Proportional vs. Priority Frame Allocation](#132-equal-vs-proportional-vs-priority-frame-allocation)
    - 13.3 [Global vs. Local Page Replacement Trade-offs](#133-global-vs-local-page-replacement-trade-offs)
    - 13.4 [Non-Uniform Memory Access (NUMA): Topology, Node Latencies, and NUMA-Aware Placement](#134-non-uniform-memory-access-numa-topology-node-latencies-and-numa-aware-placement)
    - 13.5 [Thrashing: Definition, Cascade Dynamics, and CPU Utilization Collapse](#135-thrashing-definition-cascade-dynamics-and-cpu-utilization-collapse)
    - 13.6 [Locality Model of Program Execution](#136-locality-model-of-program-execution)
    - 13.7 [Working-Set Model: Parameter Δ, Working-Set Size (WSS), and Thrashing Prevention](#137-working-set-model-parameter-δ-working-set-size-wss-and-thrashing-prevention)
    - 13.8 [Page-Fault Frequency (PFF) Strategy: Upper and Lower Bound Triggers](#138-page-fault-frequency-pff-strategy-upper-and-lower-bound-triggers)
    - 13.9 [OS Rescue Mechanisms: Linux OOM Killer and Memory Compression](#139-os-rescue-mechanisms-linux-oom-killer-and-memory-compression)
14. [Official Slide Review Questions and Authoritative Answers](#14-official-slide-review-questions-and-authoritative-answers)
    - 14.1 [Module 1: Hardware, Protection, Address Binding, and Linking (Slide 24 Q1-Q7)](#141-module-1-hardware-protection-address-binding-and-linking-slide-24)
    - 14.2 [Module 2: Swapping, Contiguous Allocation, and Fragmentation (Slide 48 Q1-Q5)](#142-module-2-swapping-contiguous-allocation-and-fragmentation-slide-48)
    - 14.3 [Module 3: Segmentation Architecture and x86 (Slide 67 Q1-Q5)](#143-module-3-segmentation-architecture-and-x86-slide-67)
    - 14.4 [Module 4: Virtual Memory, Demand Paging, and Page Faults (Slide 141 Q1-Q9)](#144-module-4-virtual-memory-demand-paging-and-page-faults-slide-141)
    - 14.5 [Module 5: Copy-on-Write and Page Replacement Basics (Slide 142 Q1-Q7)](#145-module-5-copy-on-write-and-page-replacement-basics-slide-142)
    - 14.6 [Module 6: Page Replacement Algorithms and Clock (Slide 172 Q1-Q8)](#146-module-6-page-replacement-algorithms-and-clock-slide-172)
    - 14.7 [Module 7: Frame Allocation, NUMA, Thrashing, and OOM (Slide 190 Q1-Q9)](#147-module-7-frame-allocation-numa-thrashing-and-oom-slide-190)
15. [Comprehensive Solved Numerical Problems](#15-comprehensive-solved-numerical-problems)
    - 15.1 [Dynamic Memory Allocation Placement (First-Fit, Best-Fit, Worst-Fit)](#151-dynamic-memory-allocation-placement-first-fit-best-fit-worst-fit)
    - 15.2 [Segmentation Address Translation and Limit Fault Verification](#152-segmentation-address-translation-and-limit-fault-verification)
    - 15.3 [Paging Address Decomposition and Physical Translation](#153-paging-address-decomposition-and-physical-translation)
    - 15.4 [Internal Fragmentation in Paging Systems](#154-internal-fragmentation-in-paging-systems)
    - 15.5 [Effective Access Time (EAT) with TLB (Single-Level & Two-Level Paging)](#155-effective-access-time-eat-with-tlb-single-level--two-level-paging)
    - 15.6 [Demand Paging EAT & Maximum Acceptable Page Fault Rate Calculation](#156-demand-paging-eat--maximum-acceptable-page-fault-rate-calculation)
    - 15.7 [Step-by-Step Page Replacement Tracking: FIFO vs. OPT vs. LRU](#157-step-by-step-page-replacement-tracking-fifo-vs-opt-vs-lru)
    - 15.8 [Belady's Anomaly Step-by-Step Proof (3 Frames vs. 4 Frames)](#158-beladys-anomaly-step-by-step-proof-3-frames-vs-4-frames)
    - 15.9 [Clock (Second-Chance) Replacement Hand Trace](#159-clock-second-chance-replacement-hand-trace)
    - 15.10 [Proportional Frame Allocation Computation](#1510-proportional-frame-allocation-computation)
    - 15.11 [Working-Set Model Allocation and Thrashing Condition (D > m)](#1511-working-set-model-allocation-and-thrashing-condition-d--m)

---



## 1. Hardware and Control Structures of Main Memory

### 1.1 Memory Hierarchy, Registers, and Cache Stalls

The memory system of a modern computer is engineered as a hierarchical pyramid balancing access speed, capacity, and manufacturing cost. At the top of the hierarchy sit the CPU core registers, followed by hardware caches (L1, L2, L3), physical main memory (DRAM), and non-volatile secondary storage (Solid-State Disks, NVMe, and Hard Disk Drives).

```
                      +------------------------+
                      |     CPU Registers      |  (< 1 CPU clock cycle)
                      +------------------------+
                      |    L1 / L2 / L3 Cache  |  (1 - 20 CPU clock cycles)
                      +------------------------+
                      | Physical Main Memory   |  (50 - 200 CPU clock cycles -> Bus Stall!)
                      |        (DRAM)          |
                      +------------------------+
                      | Secondary Storage Disk |  (Millions of clock cycles -> I/O Interrupt)
                      |     (SSD / HDD)        |
                      +------------------------+
```

#### The Memory Access Bottleneck and Cache Stalls
- **Direct CPU Access**: The central processing unit (CPU) can directly execute instructions and manipulate data residing in only two hardware storage locations: **CPU general-purpose registers** and **physical main memory**. Any machine instruction requiring operands from persistent storage must first transfer those data bytes into main memory.
- **Clock Cycle Disparity**:
  - Registers built directly onto the silicon microprocessor die are accessible within **one CPU clock cycle** (or less, via pipelined register forwarding).
  - Main memory access traverses the system bus (Front-Side Bus / UPI / PCIe). Completing a memory transaction takes dozens or hundreds of CPU clock cycles (typically **50 to 200 nanoseconds**).
- **CPU Stall (Bus Stalling)**: Because the processor's arithmetic logic units (ALU) operate at gigahertz frequencies while the memory bus operates orders of magnitude slower, the CPU must suspend execution when an instruction references memory that is not immediately available. This involuntary idle period is termed a **memory stall**.
- **Hardware Caching Solution**: To bridge the speed chasm between processor registers and DRAM, computer architects place hardware **caches** (L1 Instruction/Data, L2, and shared L3) between the CPU cores and main memory. The cache operates entirely under hardware control; the operating system kernel does not intervene in individual cache hits or line evictions.

---

### 1.2 Uniprogramming vs. Multiprogramming Execution Environments

Operating systems historically evolved from single-task execution to concurrent multiprogramming, transforming how physical memory is organized and safeguarded:

#### 1. Uniprogramming Systems (No Translation or Protection)
In early single-user systems (e.g., classical MS-DOS or early batch monitors):
- Exactly one user application resided in physical memory alongside the operating system kernel.
- The application was always loaded into the exact same fixed physical address space (e.g., from `0x00000000` up to the OS boundary, or OS at `0xFFFFFFFF` down).
- **Lack of Hardware Protection**: The executing program had direct, unconstrained access to all physical memory lines. If an application contained a bug (e.g., an errant pointer overwrite), it could directly overwrite the operating system kernel, crashing the physical machine.
- The application was given the illusion of a dedicated physical machine by simply giving it the reality of a dedicated physical machine.

```
+------------------------------------+ 0xFFFFFFFF
|       Operating System Kernel      |
+------------------------------------+ 0x00100000
|                                    |
|      Single User Application       | (Full unconstrained physical access)
|                                    |
+------------------------------------+ 0x00000000
```

#### 2. Multiprogramming Systems (The Need for Isolation)
Modern computing requires concurrent execution of multiple independent processes to achieve high CPU utilization:
- **Inter-Process Protection**: The operating system must guarantee that Process $A$ cannot inspect or corrupt the private memory space of Process $B$.
- **OS Kernel Protection**: User-space processes must be physically barred from modifying kernel instructions, interrupt descriptor tables (IDT), or page directory structures.
- **Mandatory Hardware Enforcement**: Because the operating system does not execute between every CPU machine cycle (doing so would degrade instruction throughput by thousands of percent), memory boundary checks **must be implemented directly in hardware** by the processor's execution pipeline on every single memory reference.

---

### 1.3 Memory Protection: Base and Limit Registers

To achieve hardware-enforced process isolation in contiguous memory systems without complex virtual memory hardware, processors implement a pair of hardware boundary registers: the **Base Register** and the **Limit Register**.

```
+---------------------------------------------------------------------------------+
|                               CPU Execution Core                                |
+---------------------------------------------------------------------------------+
                                      │
                                      │ Generated Logical Address (e.g., 346)
                                      ▼
                             [   Limit Register   ]
                             [     (e.g., 2000)   ]
                                      │
                           Is Address < Limit ?
                                  /                                  YES   /         \  NO
                                ▼           ▼
                       [  Base Register  ]    [ TRAP TO OPERATING SYSTEM ]
                       [   (e.g., 14000) ]    [ (Addressing / Seg Fault) ]
                                │
                          Base + Address
                                │
                                ▼ Physical Address (14346)
                       [   Physical RAM   ]
```

#### Hardware Register Definitions:
1. **Base Register (Relocation Register)**: Stores the smallest legal physical memory address allocated to the executing process (e.g., `14000`).
2. **Limit Register**: Specifies the range or total size of the process's logical address space (e.g., `2000`).

#### The Hardware Translation and Validation Sequence:
Every memory reference generated by the CPU core in user mode undergoes two immediate hardware checks before traversing the physical memory bus:
1. **Range Validation**: The CPU verifies whether the generated logical address is strictly less than the Limit register:
   $$	ext{Logical Address} < 	ext{Limit}$$
2. **Trap on Violation**: If the generated address is greater than or equal to the limit ($\ge 	ext{Limit}$), the memory controller aborts the bus transaction and immediately fires a hardware interrupt: a **Trap to the Operating System** (specifically an illegal addressing trap / segmentation fault). The OS intercepts the trap and terminates the offending process with `SIGSEGV`.
3. **Physical Address Computation**: If the logical address is within bounds, the hardware adds the base register value to the logical address to construct the physical memory address:
   $$	ext{Physical Address} = 	ext{Base Register} + 	ext{Logical Address}$$

> [!IMPORTANT]
> **Privileged Instruction Enforcement**:
> To preserve the integrity of memory protection, instructions that load or modify the Base and Limit registers are strictly **privileged instructions**. They can execute only when the CPU is running in **Kernel / Supervisor Mode** (Ring 0). If a user-mode process attempts an instruction to rewrite its Base or Limit registers, the CPU traps immediately with an *Illegal Instruction Fault*. Only the OS kernel scheduler modifies these registers during a process context switch.

---

### 1.4 Address Binding Stages (Compile Time, Load Time, Execution Time)

A user program progresses through multiple representations—from human-readable source code to machine-executable binary bytes. In source code, addresses are purely **symbolic** (e.g., variable `count` or function `calculate_sum()`). The binding of these symbolic representations to concrete physical memory addresses occurs at one of three distinct stages:

```
[ Source Code ] ──(Compile Time)──> [ Object Module ] ──(Load Time)──> [ In-Memory Executable ] ──(Execution Time)──> [ Dynamic Physical RAM ]
```

| Binding Stage | Binding Mechanism | Resulting Machine Code Type | Mobility & Relocation Characteristics |
| :--- | :--- | :--- | :--- |
| **Compile Time** | If the physical memory location where the process will reside is known *a priori* at compilation time, the compiler generates absolute addresses directly. | **Absolute Code** (e.g., embedded systems, ROM bootloaders). | **Completely Rigid**: If the starting memory location changes, the entire source program must be recompiled. |
| **Load Time** | If the starting memory location is unknown at compile time, the compiler emits code with addresses relative to the start of the module. The system **Loader** binds these to physical addresses when inserting the program into RAM. | **Relocatable Code**. | **Static after Load**: If the process is swapped out or needs to move to another physical memory location during its run, it cannot move; addresses are hardcoded in RAM. |
| **Execution Time (Run Time)** | Binding is delayed until the exact clock cycle the instruction executes. If a process can move from one memory segment to another during execution, execution-time binding is mandatory. | **Dynamically Relocatable Code**. | **Completely Flexible**: Code can be relocated anywhere in physical memory at runtime. **Requires hardware support** (MMU with Base/Relocation register). |

---

### 1.5 Logical vs. Physical Address Space and the Memory Management Unit (MMU)

The distinction between logical addresses and physical addresses is the foundational bedrock of all modern memory management:

- **Logical Address (Virtual Address)**: An address generated directly by the CPU execution core during instruction fetching and operand decoding. To the executing application, memory appears as a contiguous sequence of bytes beginning at address `0x00000000` up to its private limit.
- **Physical Address**: The actual electrical address loaded into the **Memory Address Register (MAR)** of the physical memory bus and presented to the DRAM chips.

```
       +-----------------------+              +-------------------------+
       | Logical Address Space |              |  Physical Address Space |
       | (Set of all logical   |              |  (Set of all physical   |
       |  addresses generated  |              |   addresses corresponding|
       |  by a program)        |              |   to logical addresses) |
       +-----------------------+              +-------------------------+
```

#### The Memory Management Unit (MMU)
The **Memory Management Unit (MMU)** is a dedicated hardware module located directly on the CPU chip (between the execution pipeline and the system bus). The MMU performs run-time hardware translation mapping virtual/logical addresses to physical addresses.

In the simplest contiguous allocation MMU:
- The base register is formally designated as the **Relocation Register**.
- The value loaded in the relocation register is added to every logical address generated by a user process at the hardware level:
  $$	ext{Physical Address} = 	ext{Relocation Register} + 	ext{Logical Address}$$
- **Example**: If a process requests logical address `346`, and the OS loaded the relocation register with `14000`, the MMU generates physical address $14000 + 346 = \mathbf{14346}$. The user program never sees, computes, or manages physical address `14346`; it deals strictly with relative offset `346`.

---

### 1.6 Multistep Processing of a User Program (Compiler to Execution)

Before a C program executes in main memory, it undergoes a multistage compilation and transformation pipeline:

```
                  +--------------------------------+
                  |       Source Program           |
                  +--------------------------------+
                                  │
                                  ▼ Compiler / Assembler
                  +--------------------------------+
                  |        Object Module           |  (Unresolved external symbols,
                  +--------------------------------+   relocation tables)
                                  │
                    Other Object  │
                    Modules ────► │ Linkage Editor (Linker)
                                  │ ◄──── System Object Libraries (Static)
                                  ▼
                  +--------------------------------+
                  |         Load Module            |  (Self-contained binary executable)
                  +--------------------------------+
                                  │
                                  ▼ System Loader
                                  │ ◄──── Dynamically Linked Libraries (.so / .dll)
                                  ▼
                  +--------------------------------+
                  |   In-Memory Binary Process     |  (Allocated in physical DRAM)
                  +--------------------------------+
```

1. **Compilation / Assembly**: Converts high-level source instructions (`.c`, `.cpp`) into relocatable object files (`.o`, `.obj`). Addresses are mapped to relocatable offsets relative to module start.
2. **Linkage Editing (Linker)**: Combines multiple independent object modules into a single logical **Load Module** (executable binary). It resolves external symbols, links static library archives, and assigns contiguous virtual offsets.
3. **Loading (Loader)**: The OS loader reads the executable image from disk, allocates memory, maps binary segments (code, data, BSS) into memory, and initiates the program counter ($PC$) at `_start`.

---

### 1.7 Dynamic Relocation Hardware Architecture

In dynamic relocation, the operating system kernel maintains control over physical memory placement while the hardware executes translations at zero CPU clock penalty:

```
+-----------------------------------------------------------------------------------------+
|                                    CPU USER PROCESS                                     |
+-----------------------------------------------------------------------------------------+
                                           │
                                           │ Generates Logical Address: 0x00000100
                                           ▼
+-----------------------------------------------------------------------------------------+
|                               MEMORY MANAGEMENT UNIT (MMU)                              |
|                                                                                         |
|       +──────────────────────────────+         +──────────────────────────────+         |
|       |        Limit Register        |         |     Relocation Register      |         |
|       |       (Size: 0x00004000)     |         |      (Base: 0x00400000)      |         |
|       +──────────────────────────────+         +──────────────────────────────+         |
|                      │                                        │                         |
|         0x00000100 < 0x00004000 ?                             │                         |
|                      │                                        │                         |
|                     YES ───────────────────────────────► [+] (Hardware Adder)           |
|                                                               │                         |
+---------------------------------------------------------------┼-------------------------+
                                                                │ Physical Address:
                                                                │ 0x00400100
                                                                ▼
                                                [ PHYSICAL MAIN MEMORY (RAM) ]
```

- When the OS scheduler triggers a context switch from Process 1 to Process 2:
  1. The kernel saves Process 1's base and limit registers into its **Process Control Block (PCB)**.
  2. The kernel loads Process 2's base and limit values from its PCB into the CPU's relocation registers.
  3. When Process 2 runs, every instruction fetch and memory load is translated using its private relocation base, guaranteeing total address isolation.



## 2. Dynamic Loading, Linking, and Shared Libraries

### 2.1 Dynamic Loading (Lazy Loading of Subroutines)

In classical static memory loading, an entire executable program—including all functions, auxiliary modules, and error-handling routines—is loaded into physical RAM before execution begins.

#### The Problem with Static Loading
Large commercial software packages contain substantial blocks of code dedicated to infrequent operations:
- Emergency error-handling diagnostics.
- Rare hardware failure recovery handlers.
- Advanced configuration wizards and rarely invoked menu features.
Loading these thousands of lines into physical RAM wastes scarce physical frames, decreasing the system's degree of multiprogramming.

#### Mechanics of Dynamic Loading
Under **Dynamic Loading**:
1. Subroutines and functions are **not loaded into memory until they are explicitly called**.
2. All routines are maintained on secondary storage in a relocatable load format.
3. The main program is initially loaded into memory and begins execution.
4. When a routine calls another subroutine:
   - The calling routine first checks an internal module table to verify whether the target subroutine is already resident in memory.
   - If **not in memory**, the relocatable loader is invoked to dynamically fetch the required routine from disk into memory, update the process's internal address jump tables, and pass execution control.
5. **Key Advantage**: Unused routines are never loaded into physical RAM. Dynamic loading requires no special operating system kernel support; it can be implemented directly by application software design, though modern OS runtime environments provide standard library loader APIs (e.g., `dlopen()`, `dlsym()`, `dlclose()` in POSIX C).

---

### 2.2 Static Linking vs. Dynamic Linking

When an application invokes library functions (such as `printf()`, `malloc()`, or mathematical functions in `math.h`), the linkage can occur statically or dynamically:

```
             STATIC LINKING                                      DYNAMIC LINKING
   +---------------------------------+                 +---------------------------------+
   |      Application Object         |                 |      Application Object         |
   |           Code                  |                 |           Code                  |
   +---------------------------------+                 +---------------------------------+
   |    Full Embedded Copy of        |                 |      Small Jump Stub for        |
   |    Standard C Library (libc)    |                 |      Standard C Library (libc)  |
   |         (Several MBs)           |                 |          (A few bytes)          |
   +---------------------------------+                 +---------------------------------+
                 │                                                     │
                 ▼ Load into Memory                                    ▼ Load into Memory
   +---------------------------------+                 +---------------------------------+
   | Process A Memory Space (RAM)    |                 | Process A Memory Space (RAM)    |
   | [ App Code ] [ Full libc copy ] |                 | [ App Code ] [ Stub ]           |
   +---------------------------------+                 +---------------------------------+
                                                                       │ (Shared Reference)
   +---------------------------------+                                 ▼
   | Process B Memory Space (RAM)    |                 +─────────────────────────────────+
   | [ App Code ] [ Full libc copy ] |                 | Single Shared Physical Copy of  |
   +---------------------------------+                 | libc.so in RAM (Zero Waste!)    |
       (Massive Redundant Waste!)                      +─────────────────────────────────+
```

| Dimension | Static Linking | Dynamic Linking |
| :--- | :--- | :--- |
| **Linkage Point** | At compile/link time by the Linkage Editor. | Postponed until program execution time. |
| **Binary Executable Size** | Very large; every binary embeds full copies of all dependent library functions. | Very small; executable contains only lightweight pointer references (**stubs**). |
| **Physical Memory Footprint** | Massive duplication; if 50 processes call `printf()`, 50 identical copies of `libc` exist in RAM. | Highly optimal; exactly **one physical copy** of the shared library resides in RAM, mapped to all processes. |
| **Bug Fixes and Patching** | Severe drawback; fixing a security vulnerability in a library requires recompiling and relinking every single application binary on the system. | Trivial; replacing the shared library file on disk (`.so`) instantly updates all applications upon their next launch. |
| **Execution Performance** | Marginally faster subroutine invocation (direct function call without stub indirection). | Negligible overhead on the very first function call (stub resolution); identical speed thereafter. |

---

### 2.3 Shared Libraries, DLLs, and Stub Execution Mechanics

Dynamic linking is the technological foundation behind **Shared Libraries** in UNIX/Linux (`.so` - Shared Objects) and Microsoft Windows (`.dll` - Dynamic Link Libraries).

#### The Role of the "Stub"
In a dynamically linked executable:
- A **stub** is a miniature code sequence embedded in the binary image for each library function reference.
- The stub contains logic to:
  1. Determine whether the target shared library module is already resident in physical memory.
  2. If absent, request the OS kernel to load the library into memory.
  3. Replace itself with the absolute memory address of the loaded routine, and execute the routine.

#### Detailed Lifecycle of Stub Resolution:

```
[ Step 1: App Calls printf() ] ──► [ Step 2: Jump to Stub ] ──► [ Step 3: Check Memory Table ]
                                                                             │
                    ┌────────────────────────────────────────────────────────┴───────────────────┐
                    ▼ (Library Not in RAM)                                                       ▼ (Library Already in RAM)
   [ Step 4A: Invoke OS to load libc.so from disk ]                              [ Step 4B: Retrieve existing base address ]
                    │                                                                            │
                    └────────────────────────────────► [ Step 5 ] ◄──────────────────────────────┘
                                                         │
                                    [ Step 5: Overwrite Stub with Direct Jump ]
                                                         │
                                    [ Step 6: Jump directly to real printf() code ]
```

On any subsequent call to `printf()`, the program executes the direct jump instruction, bypassing the dynamic linker entirely with zero runtime penalty.

---

### 2.4 Versioning, Security, and Memory Space Savings

- **Memory Conservation**: In modern Linux systems running hundreds of background daemons, standard system libraries (e.g., `glibc`, `libpthread`, `libcrypto`) are mapped into the address space of nearly every running process. Dynamic linking saves gigabytes of physical RAM by sharing the executable code segments of these libraries.
- **Library Versioning (Solving "DLL Hell")**: If an operating system updates a library, new function signatures or altered return types can break legacy binaries compiled against older releases. Operating systems implement explicit version control:
  - In Linux, shared libraries employ **sonames** and symbolic links:
    ```bash
    libc.so.6 -> libc-2.35.so
    ```
  - An executable explicitly encodes the specific major library version it requires (e.g., `libc.so.6`). Both new and old library versions can coexist peacefully on disk and in memory, preserving backwards compatibility.



## 3. Memory-Mapped Files and mmap() System Call

### 3.1 Motivation: Traditional Read/Write I/O vs. Memory Mapping

In classical POSIX file manipulation, an application accesses file data using the sequential `read()` and `write()` system calls:

```
                    TRADITIONAL READ() FILE ACCESS (Double-Buffering)
+-----------------------------------------------------------------------------------------+
| User Space Application        char buffer[4096];                                        |
|                               read(fd, buffer, 4096); ◄──────────┐                      |
+----------------------------------------------------------------──┼──────────────────────+
| Kernel Space                      ▲                              │                      |
|                                   │ Data Copy                    │ User/Kernel Context  |
|                                   │ (CPU Overhead)               │ Switch Boundary      |
|                       +────────────────────────+                 │                      |
|                       |   Kernel Page Cache    | ◄───────────────┘                      |
|                       +────────────────────────+                                        |
+--------------------------------───▲─────────────────────────────────────────────────────+
| Hardware Disk Controller          │ Disk DMA Transfer                                   |
|                       +────────────────────────+                                        |
|                       |    Hard Disk / SSD     |                                        |
|                       +────────────────────────+                                        |
+-----------------------------------------------------------------------------------------+
```

#### Inefficiencies of Classical File I/O:
1. **Context Switch Overhead**: Every `read()` and `write()` call forces a transition from user mode to kernel mode and back.
2. **Double-Buffering and Memory Copy Penalty**: Data must first be transferred via DMA from disk into the kernel's **Page Cache**, and then explicitly copied via the CPU into the user-space process buffer (`buffer`).
3. **Random Access Complexity**: Seeking to arbitrary positions in a file requires repeated `lseek()` system calls, introducing substantial control overhead.

---

### 3.2 The mmap() System Call Mechanics and API Signatures

**Memory-Mapped File I/O** fundamentally resolves these bottlenecks by mapping a segment of a file on secondary storage directly into a process's virtual memory address space. Once mapped, file bytes are accessed using direct pointer dereferences in memory (e.g., `char byte = ptr[500];`), completely bypassing the `read()` and `write()` system calls!

```c
#include <sys/mman.h>

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
int msync(void *addr, size_t length, int flags);
```

#### Detailed Parameter Breakdown:
- **`addr`**: Suggested starting virtual address for the mapping. Typically set to `NULL`, directing the OS kernel to select an appropriate, unallocated virtual address region.
- **`length`**: Number of bytes to map into the process virtual address space.
- **`prot`**: Memory protection flags defining hardware access permissions:
  - `PROT_READ`: Pages may be read.
  - `PROT_WRITE`: Pages may be written.
  - `PROT_EXEC`: Pages may be executed as machine code.
  - `PROT_NONE`: Pages cannot be accessed (triggers trap if touched).
- **`flags`**: Determines mapping visibility and sharing semantics:
  - `MAP_SHARED`: Modifications are shared with other processes mapping the same file and are written back to the underlying disk file.
  - `MAP_PRIVATE`: Modifications are private (using **Copy-on-Write**). The underlying disk file is never altered.
  - `MAP_ANONYMOUS`: The mapping is not backed by any file; used by `malloc()` to allocate large contiguous blocks of heap RAM.
- **`fd`**: File descriptor of the open file (obtained via `open()`).
- **`offset`**: Byte offset within the file where the mapping begins (must be a strict multiple of the underlying page size, typically 4096 bytes).

---

### 3.3 Page Cache Integration and Demand-Paged File I/O

Memory mapping leverages the operating system's **Demand Paging** subsystem to perform file I/O lazily:

```
1. Process calls mmap() ──► OS creates Virtual Memory Area (VMA) without reading disk!
                                      │
2. App reads ptr[100]    ──► MMU detects page marked INVALID ('i') in Page Table
                                      │
3. Hardware Trap         ──► Page Fault Handler invoked
                                      │
4. Disk Transfer         ──► Kernel reads 4KB block from disk directly into Page Frame
                                      │
5. Page Table Updated    ──► Valid bit set to 'v', Physical Frame assigned
                                      │
6. Instruction Restarts  ──► Memory dereference succeeds at hardware DRAM speeds!
```

This architecture guarantees that only the specific portions of a multi-gigabyte file that are actually accessed by the application are ever transferred into physical RAM.

---

### 3.4 MAP_SHARED vs. MAP_PRIVATE (Copy-on-Write) Semantics

```
                     MAP_SHARED                                         MAP_PRIVATE (COW)
+---------------------------------------------------+  +---------------------------------------------------+
| Process A Virtual       Process B Virtual         |  | Process A Virtual       Process B Virtual         |
| Address Space           Address Space             |  | Address Space           Address Space             |
|   [ ptr_A ]                [ ptr_B ]              |  |   [ ptr_A ]                [ ptr_B ]              |
|        \                      /                   |  |        \                      /                   |
|         ▼                    ▼                    |  |         ▼                    ▼                    |
|       +────────────────────────+                  |  |       +────────────────────────+                  |
|       | Shared Physical Frame  |                  |  |       | Shared Physical Frame  | (Read-Only)      |
|       +────────────────────────+                  |  |       +────────────────────────+                  |
|                   │                               |  |        │ (Process A writes)                       |
|                   ▼ Written back to disk          |  |        ▼ Triggers Copy-on-Write                   |
|       [ Underlying File on Disk ]                 |  |       +────────────────────────+                  |
+---------------------------------------------------+  |       | Private Frame for A    | (File unchanged!)|
                                                       |       +────────────────────────+                  |
                                                       +---------------------------------------------------+
```

1. **`MAP_SHARED`**:
   - Both processes point to the exact same physical memory frames in the kernel page cache.
   - Any byte write executed by Process $A$ is instantly visible to Process $B$.
   - The OS dirty-page flusher writes modified frames back to the physical disk.
   - **Use Case**: High-performance Inter-Process Communication (IPC) and database storage engines.
2. **`MAP_PRIVATE`**:
   - Pages are initially mapped read-only, pointing to shared physical frames.
   - If a process attempts a write operation, the MMU triggers a protection fault. The OS intercepts the fault, allocates a brand new physical frame, duplicates the 4 KB page data, and grants write permission exclusively to the modifying process (**Copy-on-Write**).
   - The underlying disk file remains untouched.
   - **Use Case**: Dynamic program loaders mapping `.text` and initialized `.data` segments of binary executables.

---

### 3.5 Disk Synchronization with msync() and Zero-Copy IPC

When an application modifies memory mapped with `MAP_SHARED`, changes reside in volatile RAM cache frames until written back to persistent disk blocks. To guarantee ACID durability or crash resilience, applications invoke `msync()`:

```c
int msync(void *addr, size_t length, int flags);
```

- **`MS_ASYNC`**: Requests that dirty pages be scheduled for background writing to disk, returning immediately without blocking.
- **`MS_SYNC`**: Forces synchronous I/O; the system call blocks until all modified bytes across the designated address range are physically committed to non-volatile disk platters/flash.
- **`MS_INVALIDATE`**: Invalidates other cached copies of the mapping in memory, forcing subsequent reads to reload fresh data from storage.

#### High-Speed Zero-Copy Inter-Process Communication (IPC)
Because multiple distinct processes can map the exact same file using `MAP_SHARED`, memory-mapped files function as one of the fastest IPC mechanisms available in operating systems:
- Data written by Process 1 into its virtual memory buffer is immediately available to Process 2 without a single system call, context switch, or intermediate kernel buffer copy.
- Processes achieve raw memory bus throughput for multi-gigabyte data sharing.



## 4. Swapping and Contiguous Memory Allocation

### 4.1 Classical Process Swapping and the Backing Store

In systems where the cumulative memory requirements of all active processes exceed the physical capacity of installed RAM, the operating system must temporarily evict idle or lower-priority processes.

```
+-----------------------------------------------------------------------------------------+
|                                  PHYSICAL MAIN MEMORY (RAM)                             |
|                                                                                         |
|       +──────────────────────────────────+   +──────────────────────────────────+       |
|       |     Operating System Kernel      |   |       Active User Process 1      |       |
|       +──────────────────────────────────+   +──────────────────────────────────+       |
+-----------------------------------------------------------------------------------------+
                         ▲                                     │
                 Swap In │                                     │ Swap Out
                         │                                     ▼
+-----------------------------------------------------------------------------------------+
|                                    BACKING STORE DISK                                   |
|                                                                                         |
|       +──────────────────────────────────+   +──────────────────────────────────+       |
|       |    Process 2 Inactive Image      |   |       Process 3 Waiting Image    |       |
|       +──────────────────────────────────+   +──────────────────────────────────+       |
+-----------------------------------------------------------------------------------------+
```

#### Core Principles of Swapping:
- **Swapping**: A memory-management technique where an entire process image is moved temporarily out of physical RAM to a fast secondary storage device called the **Backing Store**, and subsequently brought back into RAM to resume execution.
- **Backing Store**: A high-speed disk or solid-state partition large enough to accommodate copies of all memory images for all active processes, providing direct access to these memory images.
- **Roll Out, Roll In**: A priority-based variant of swapping used in priority-driven preemptive schedulers. When a higher-priority process becomes ready, the scheduler swaps out a lower-priority process (**roll out**) to free memory, and loads the high-priority process (**roll in**). Once complete, the lower-priority process is swapped back in.

#### Address Binding Impact on Swapping:
A critical architectural question is: *Does a swapped-out process have to be restored to the exact same physical memory address space when swapped back in?*
1. **If Address Binding is at Compile Time or Load Time**: The process **must** be loaded into the exact same physical address space it occupied previously, because absolute or statically relocated memory pointers are embedded in its binary instructions. This severely constrains the OS memory scheduler.
2. **If Address Binding is at Execution Time (Run Time)**: The process can be swapped back into **any arbitrary, available physical memory partition**. Because addresses are translated dynamically using MMU relocation registers, the OS simply updates the base register to point to the new physical starting location.

---

### 4.2 Swapping Performance and Context Switch Latency Derivation

The primary performance cost of swapping is the massive transfer latency of secondary storage. When swapping is part of a context switch, process switching times degrade from microseconds to entire seconds.

#### Mathematical Derivation of Swapping Latency:
Let:
- $S$ = Size of the user process image (in Megabytes).
- $R$ = Data transfer rate of the backing store disk (in Megabytes per second).
- $L$ = Rotational disk latency and seek overhead (in milliseconds).

The time required to swap out the current process is:
$$T_{\text{swap-out}} = L + \frac{S}{R}$$

The time required to swap in the incoming process is:
$$T_{\text{swap-in}} = L + \frac{S}{R}$$

The total context switch swapping component is:
$$T_{\text{total}} = T_{\text{swap-out}} + T_{\text{swap-in}} = 2 \times \left( L + \frac{S}{R} \right) \approx \frac{2S}{R}$$

#### Step-by-Step Numerical Walkthrough (From Lecture Slides):
Consider a user process with a memory image of size $S = 100\text{ MB}$, and a hard disk with a sustained sequential transfer rate $R = 50\text{ MB/s}$:

1. **Swap-Out Time**:
   $$T_{\text{swap-out}} = \frac{100\text{ MB}}{50\text{ MB/s}} = \mathbf{2.0\text{ seconds}}$$
2. **Swap-In Time**:
   $$T_{\text{swap-in}} = \frac{100\text{ MB}}{50\text{ MB/s}} = \mathbf{2.0\text{ seconds}}$$
3. **Total Context Switch Time**:
   $$T_{\text{total}} = 2.0\text{ s} + 2.0\text{ s} = \mathbf{4.0\text{ seconds}}$$
4. **Scaling to Larger Processes**:
   If a modern process consumes $S = 3\text{ GB} = 3000\text{ MB}$:
   $$T_{\text{total}} = 2 \times \left( \frac{3000\text{ MB}}{50\text{ MB/s}} \right) = 2 \times 60\text{ s} = \mathbf{120\text{ seconds (2 full minutes!)}}$$

> [!WARNING]
> **Performance Conclusion**:
> Whole-process swapping incurs unacceptable latency. To mitigate this overhead, the OS must know the actual amount of memory actively used by a process rather than its maximum declared size (via system calls like `request_memory()` and `release_memory()`).

---

### 4.3 Constraints on Swapping: Pending I/O and Double Buffering

Swapping introduces severe hazards when processes interact with asynchronous hardware I/O devices:

#### The Pending I/O Hazard:
Suppose Process $P_1$ initiates an asynchronous read from a disk file into its private memory buffer. While the disk controller is reading the data, $P_1$ enters the waiting state. The CPU scheduler decides to swap out $P_1$ to load Process $P_2$. If $P_2$ is allocated the exact physical memory space previously occupied by $P_1$, the Direct Memory Access (DMA) controller completes the disk read and writes the data bytes directly into physical memory—**corrupting the memory space of Process $P_2$**!

```
Process P1 initiates I/O to Buffer at 0x1000 ──► P1 swapped out ──► P2 loaded at 0x1000
                                                                            │
DISK CONTROLLER COMPLETES I/O VIA DMA ──────────────────────────────────────┘
(CRITICAL DATA CORRUPTION: Overwrites P2's memory!)
```

#### Remediation Solutions:
1. **Never Swap Processes with Pending I/O**: The OS maintains an I/O state flag in the process PCB; swapping is strictly blocked until all pending I/O operations acknowledge completion.
2. **Double Buffering (Kernel Buffering)**: User I/O operations never write directly to user-space process memory. The DMA controller transfers data exclusively into OS **kernel buffers**. Once the I/O completes and the target process is confirmed resident in RAM, the kernel copies the data from kernel space into the process's memory space.
   - *Trade-off*: Adds significant CPU memory copying overhead, but guarantees memory safety.

---

### 4.4 Modern OS Perspective: Whole-Process Swapping vs. Page Swapping

In modern operating systems (such as Linux, Windows, and macOS), **standard whole-process swapping is virtually obsolete**:
- Modern systems employ **Demand Paging (Page Swapping)**: Instead of writing the entire multi-gigabyte address space of a process to disk, the operating system swaps out individual, fine-grained **4 KB memory pages** that are currently idle.
- Active processes remain resident in memory, with only inactive pages evicted to the swap partition/swap file. Whole-process swapping is retained only as an emergency mechanism in specialized embedded real-time systems under extreme starvation.

---

### 4.5 Contiguous Allocation Models (Fixed vs. Dynamic Partitions)

Main memory must simultaneously accommodate both the operating system kernel and multiple concurrent user processes. In contiguous memory allocation, each process is loaded into a single, continuous block of physical memory addresses.

```
LOW ADDRESSES                                                            HIGH ADDRESSES
+──────────────────────────+──────────────────────────+───────────────────────────────+
|     Operating System     |        Process P1        |          Process P2           |
|         Kernel           |      (Contiguous)        |         (Contiguous)          |
+──────────────────────────+──────────────────────────+───────────────────────────────+
0x00000000                 0x00100000                 0x00400000                      0x01000000
```

#### Memory Partitioning Architectures:
1. **Single-Partition Allocation**:
   - Memory is divided into two continuous regions: one holding the OS kernel (typically protected in lower memory), and the remaining partition dedicated to a single user process.
   - Supported only in single-tasking uniprogrammed systems.
2. **Fixed-Partition Allocation (MFT - Multiprogramming with a Fixed Number of Tasks)**:
   - Physical memory is statically partitioned into a fixed number of predetermined slots (which may be of equal or unequal sizes).
   - Each partition accommodates exactly one process at a time.
   - When a partition becomes free, a process from the input queue is loaded into it.
   - **Severe Limitation**: The degree of multiprogramming is strictly bounded by the number of partitions. Causes massive **Internal Fragmentation** when a process is smaller than its assigned partition.
3. **Variable-Partition Allocation (MVT - Multiprogramming with a Variable Number of Tasks)**:
   - Memory is treated as a continuous, dynamic pool.
   - The OS maintains a table recording which regions of memory are occupied and which regions are available (**holes**).
   - When a process arrives, the OS searches the free list for a hole large enough to satisfy the request, carves out exactly the required memory size, and returns the unused remainder to the free list.
   - When a process terminates, its memory partition is returned to the pool and coalesced with adjacent free holes.

---

### 4.6 Dynamic Storage Allocation Algorithms: First-Fit, Best-Fit, Worst-Fit

When a process requests a block of memory of size $n$, the operating system must choose a hole from the set of available free holes. Three primary placement strategies exist:

```
                DYNAMIC STORAGE-ALLOCATION STRATEGIES
+───────────────────────────────────────────────────────────────────────────────────+
| Free Holes:   [ 300 KB ]   [ 600 KB ]   [ 350 KB ]   [ 200 KB ]   [ 750 KB ]      |
+───────────────────────────────────────────────────────────────────────────────────+
Incoming Request: Process P (Size: 358 KB)

1. FIRST-FIT:   Scans from left to right. Selects the first hole >= 358 KB.
                -> Selects [ 600 KB ] (Fastest search; leaves 242 KB hole).

2. BEST-FIT:    Scans ENTIRE list. Selects the smallest hole >= 358 KB.
                -> Selects [ 750 KB ] (Produces the smallest leftover hole: 392 KB).

3. WORST-FIT:   Scans ENTIRE list. Selects the largest available hole.
                -> Selects [ 750 KB ] (Produces the largest leftover hole: 392 KB).
```

#### 1. First-Fit
- **Strategy**: Allocates the very first hole in the free list that is large enough ($\text{size} \ge n$).
- **Search Behavior**: Can start searching from the beginning of the list, or resume searching from where the previous first-fit search terminated (**Next-Fit**). Search stops immediately upon finding a viable hole.
- **Performance**: Generally the **fastest** algorithm because it minimizes list traversal time.

#### 2. Best-Fit
- **Strategy**: Allocates the smallest hole that is large enough to satisfy the request ($\text{size} \ge n$).
- **Search Behavior**: Must traverse and evaluate the **entire free list**, unless the list is kept pre-sorted in ascending order of hole size.
- **Characteristics**: Minimizes the unused space in the chosen block, producing the **smallest possible leftover hole**. However, these tiny residual holes are often too small to satisfy any future process, exacerbating external fragmentation.

#### 3. Worst-Fit
- **Strategy**: Allocates the largest available hole in the system.
- **Search Behavior**: Must traverse and evaluate the **entire free list**, unless pre-sorted in descending order of hole size.
- **Philosophy**: Produces the **largest possible leftover hole**, based on the heuristic that a large leftover hole will be far more useful for accommodating subsequent process requests.
- **Empirical Reality**: Simulations and mathematical proofs demonstrate that Worst-Fit is significantly inferior to both First-Fit and Best-Fit in terms of both execution speed and overall storage utilization.

---

### 4.7 Performance, Memory Utilization, and Search Complexity Analysis

| Criteria | First-Fit | Best-Fit | Worst-Fit |
| :--- | :--- | :--- | :--- |
| **Search Time Complexity** | $O(k)$ where $k \le N$ (Terminates on first match; fastest). | $O(N)$ (Must scan entire list of $N$ holes unless tree-indexed). | $O(N)$ (Must scan entire list of $N$ holes unless max-heap indexed). |
| **Storage Utilization** | Very High (Roughly tied with Best-Fit). | Very High (Slightly better on some distributions). | Lowest (Tends to destroy large blocks needed for big processes). |
| **Leftover Hole Size** | Variable. | Smallest leftover (often unusable shards). | Largest leftover. |
| **Practical Recommendation** | **Industry Standard** for general-purpose heaps. | Good for uniform, predictable request sizes. | Almost never used in modern production systems. |



## 5. Fragmentation: Internal vs. External and Remediation

### 5.1 Internal Fragmentation: Causes, Formulas, and Boundary Allocations

**Internal Fragmentation** is the condition where memory allocated to a process is larger than the memory requested by the process. The wasted space resides **inside** the allocated partition and cannot be used by any other process in the system.

```
+─────────────────────────────────────────────────────────────+
|               Allocated Partition: 512 KB                   |
| +────────────────────────────────────────+────────────────+ |
| |       Process Request: 400 KB          | Wasted: 112 KB | |
| |            (Actively Used)             | (Internal Frag)| |
| +────────────────────────────────────────+────────────────+ |
+─────────────────────────────────────────────────────────────+
```

#### Causes of Internal Fragmentation:
1. **Fixed-Size Partitions (MFT)**: When an OS utilizes fixed partitions of 512 KB, a process requesting 400 KB leaves $512 - 400 = \mathbf{112\text{ KB}}$ completely trapped and idle.
2. **Hardware Alignment and Granularity**: Modern memory architectures require allocations aligned to 4-byte, 8-byte, or 64-byte cache line boundaries. Even in dynamic allocators, requesting 33 bytes will result in a 40-byte or 64-byte allocation.
3. **Paging Systems**: In paging, memory is allocated in discrete integer multiples of page sizes (e.g., 4 KB). If a process requires 17 KB, the OS allocates 5 full pages ($5 \times 4\text{ KB} = 20\text{ KB}$), wasting $3\text{ KB}$ in the final frame.

$$\text{Internal Fragmentation} = \text{Allocated Partition Size} - \text{Requested Process Size}$$

---

### 5.2 External Fragmentation and the 50-Percent Rule Analysis

**External Fragmentation** occurs when total free memory space across the system is sufficient to satisfy a process allocation request, but the available memory is not contiguous. It is fragmented into a large collection of small, isolated holes scattered throughout RAM.

```
+──────────────+──────────────+──────────────+──────────────+──────────────+
|  Process P1  |  Hole 100MB  |  Process P2  |  Hole 200MB  |  Process P3  |
+──────────────+──────────────+──────────────+──────────────+──────────────+
Total Free Memory = 100 MB + 200 MB = 300 MB.
Incoming Request: Process P4 requires 250 MB contiguous memory.
RESULT: ALLOCATION FAILS! (Cannot allocate 250MB contiguously despite 300MB free).
```

#### The 50-Percent Rule (Knuth's Mathematical Proof):
Statistical analysis of dynamic memory allocation using First-Fit demonstrates that as memory undergoes continuous allocation and deallocation over time, the system approaches an equilibrium state.

Let:
- $N$ = Number of currently allocated blocks in memory.
- For every $N$ allocated blocks, approximately **$0.5 N$ blocks are lost to external fragmentation** as unusable holes between allocated segments.

$$\text{Number of Holes} \approx \frac{1}{2} N$$

**System Impact**: This means that approximately **one-third (33%) of physical memory is rendered completely unusable** due to external fragmentation! This catastrophic loss motivated the transition from contiguous allocation to non-contiguous paging.

---

### 5.3 Memory Compaction: Dynamic Relocation Requirements and I/O Cost

The standard technique to eliminate external fragmentation in contiguous systems is **Compaction**:

```
BEFORE COMPACTION (Fragmented):
+──────────────+──────────────+──────────────+──────────────+──────────────+
|  Process P1  |  Hole 100MB  |  Process P2  |  Hole 200MB  |  Process P3  |
+──────────────+──────────────+──────────────+──────────────+──────────────+

AFTER COMPACTION (Contiguous):
+──────────────+──────────────+──────────────+─────────────────────────────+
|  Process P1  |  Process P2  |  Process P3  |      Single Large Hole      |
|  (Unmoved)   |  (Relocated) |  (Relocated) |           300 MB            |
+──────────────+──────────────+──────────────+─────────────────────────────+
```

#### Compaction Constraints:
1. **Mandatory Dynamic Relocation**: Compaction is **only possible if relocation is dynamic and executed at run time**. If addresses are bound at compile time or load time, memory locations cannot be moved without invalidating absolute pointers.
2. **Massive Computational and I/O Overhead**:
   - To compact memory, the operating system must copy hundreds of megabytes or gigabytes of memory data from high physical addresses to low physical addresses.
   - During the copy operation, running processes must be halted to prevent data inconsistency.
   - For multi-gigabyte modern workloads, memory compaction induces massive multi-second system pauses, making it totally impractical for real-time or responsive interactive environments.

---

### 5.4 Kernel Memory Allocation Alternatives: Buddy System and Slab Allocator

Because standard first-fit/best-fit algorithms are too slow and suffer from severe fragmentation, modern operating system kernels utilize specialized low-level memory allocators:

#### 1. The Buddy System Allocator
- Allocates memory from a fixed-size segment of physically contiguous pages using a power-of-two allocation rule ($2^U$).
- **Splitting**: When an allocation request of size $S$ arrives, the allocator rounds $S$ up to the nearest power of 2. If no matching block exists, a larger block ($2^k$) is split into two equal halves called **Buddies** ($2^{k-1}$). Splitting continues until the smallest power-of-two block that accommodates the request is obtained.
- **Coalescing**: When a block is freed, the allocator inspects its immediate buddy. If the buddy is also free, the two buddies are instantly merged back into a single $2^k$ block.
- *Pros*: Extremely fast coalescing ($O(1)$ buddy arithmetic using bitwise XOR).
- *Cons*: High internal fragmentation (e.g., requesting 33 KB forces a 64 KB block, wasting 31 KB).

#### 2. The Linux Slab Allocator
- Eliminates internal and external fragmentation for recurrent kernel objects (e.g., process descriptors `task_struct`, file objects, network socket buffers).
- **Architecture**:
  - A **Cache** contains memory for one specific type of kernel data structure (e.g., a cache for `task_struct`).
  - Each cache consists of one or more **Slabs**. A slab consists of one or more physically contiguous memory pages.
  - Slabs are partitioned into pre-allocated, object-sized slots, classified into three states: **Full** (all slots allocated), **Empty** (all slots free), or **Partial** (mix of free and allocated slots).
- When the kernel requests an object, the slab allocator immediately returns a free slot from a partial slab. When freed, the slot is marked available without returning memory to the general pool.
- *Result*: Zero fragmentation and instantaneous $O(1)$ allocation times for kernel data structures.



## 6. Segmentation

### 6.1 Programmer's View of Memory (Logical Segments)

Programmers do not conceptualize software as a monotonous, linear array of physical bytes. Instead, human developers view programs as collections of distinct, logically related functional units:
- The main code routine.
- Independent functions, subroutines, and procedures.
- Global and static variables.
- Dynamic heap data structures.
- The thread execution stack and local activation records.
- Standard C runtime libraries.

```
PROGRAMMER'S LOGICAL VIEW                              PHYSICAL MEMORY
+─────────────────────────────────+                  +───────────────────+
|   Segment 1: Subroutines / Code |                  |     Segment 1     |
+─────────────────────────────────+                  +───────────────────+
|   Segment 2: Global Variables   |                  |     Segment 4     |
+─────────────────────────────────+                  +───────────────────+
|   Segment 3: Thread Call Stack  | ──(Segmentation)─►|     Segment 0     |
+─────────────────────────────────+                  +───────────────────+
|   Segment 4: Dynamic Heap Space |                  |     Segment 2     |
+─────────────────────────────────+                  +───────────────────+
|   Segment 0: Main Program Logic |                  |     Segment 3     |
+─────────────────────────────────+                  +───────────────────+
```

**Segmentation** is a memory-management architecture that directly supports this modular user view. A logical address space is defined as a collection of variable-length segments, each with a logical name/number and an associated length.

---

### 6.2 Segmentation Architecture: Logical Address 2-Tuple (s, d)

In a segmented memory architecture, the CPU generates addresses formatted as a two-dimensional tuple:

$$\text{Logical Address} = \langle s,\; d \rangle$$

- **$s$ (Segment Number)**: Used as an index into the process's **Segment Table**.
- **$d$ (Offset)**: The displacement byte within the chosen segment ($0 \le d < \text{Limit}$).

---

### 6.3 Segment Table Hardware: Base, Limit, STBR, and STLR

Because physical memory remains a one-dimensional sequence of physical bytes, the operating system and hardware maintain a **Segment Table** to translate two-dimensional user addresses into physical memory locations.

#### Segment Table Entry (STE) Structure:
Each row in the segment table contains two primary fields:
1. **Base**: The starting physical memory address where the segment resides in RAM.
2. **Limit**: The precise length of that segment in bytes.

```
                     SEGMENT TABLE STRUCTURE
           Index (s)     Limit (Length)       Base Address
          +───────────+──────────────────+───────────────────+
          |     0     |       600        |        219        |
          |     1     |        14        |       2300        |
          |     2     |       100        |         90        |
          |     3     |       580        |       1327        |
          |     4     |        96        |       1952        |
          +───────────+──────────────────+───────────────────+
```

#### Hardware Control Registers:
- **Segment-Table Base Register (STBR)**: A CPU hardware register storing the physical starting address of the currently executing process's segment table in main memory.
- **Segment-Table Length Register (STLR)**: Stores the total number of segments recognized for the process. A segment number $s$ is legal only if $s < \text{STLR}$.

---

### 6.4 Address Translation Flow and Hardware Protection Traps

The translation of logical address $\langle s, d \rangle$ by the MMU hardware follows a strict sequence of mathematical validations:

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU EXECUTION CORE                                       |
|                         Generates Logical Address: < s , d >                            |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                    │
                                    ├─── s (Segment Number)
                                    │        │
                                    │        ▼
                                    │   [ Check s < STLR ? ] ──► NO ──► [ TRAP: ILLEGAL SEGMENT ]
                                    │        │ YES
                                    │        ▼
                                    │   [ Index into Segment Table at entry s ]
                                    │   [ STBR + (s * Entry_Size)             ]
                                    │        │
                                    │        ├──────────────────┬─────────────────┐
                                    │        ▼ Limit            ▼ Base            ▼ Access Flags
                                    │      (Length)            (Base)            (r / w / x)
                                    ▼        │                  │                 │
                             d (Offset) ─────┘                  │                 │
                                    │                           │                 ▼
                             Is d < Limit ?                     │       Check Access Permission
                                    │                           │        (Read/Write Violation?)
                           ┌────────┴────────┐                  │                 │
                      NO   ▼            YES  ▼                  │                 ▼
          [ TRAP TO OPERATING SYSTEM ]       └───► [+] ◄────────┘       [ TRAP: PROTECTION VIOLATION ]
          [ (Addressing Error / SEGV)]              │
                                                    ▼
                                            Physical Address:
                                              Base + Offset (d)
                                                    │
                                                    ▼
                                         [ PHYSICAL MEMORY RAM ]
```

#### Step-by-Step Translation Logic:
1. The hardware verifies that segment number $s$ is valid: $s < \text{STLR}$.
2. The hardware fetches entry $s$ from the Segment Table using STBR: $\text{Entry Address} = \text{STBR} + (s \times \text{Size of Entry})$.
3. The hardware extracts the segment's **Limit** and **Base**.
4. **Boundary Validation**: The hardware compares offset $d$ against the Limit:
   - If $d \ge \text{Limit}$, the access attempts to reach memory beyond the legal boundary of the segment. The MMU halts instruction execution and raises a **Trap to the Operating System (Addressing Error / Segmentation Fault)**.
5. **Access Permission Validation**: The hardware checks whether the requested operation (read, write, execute) matches the segment's protection bits. A write to a read-only code segment triggers an immediate protection trap.
6. **Physical Address Calculation**: If both validations pass, the hardware adds the Base to the offset:
   $$\text{Physical Address} = \text{Base} + d$$

---

### 6.5 Segment Sharing: Shared Reentrant Code and Data Segments

Segmentation provides an intuitive mechanism for sharing common libraries and code routines across multiple independent processes:

```
Process P1 Segment Table                          Process P2 Segment Table
+─────+───────+──────+                            +─────+───────+──────+
| Seg | Limit | Base |                            | Seg | Limit | Base |
+─────+───────+──────+                            +─────+───────+──────+
|  0  | 25200 | 43060| ──┐                    ┌── |  0  | 25200 | 43060|
+─────+───────+──────+   │                    │   +─────+───────+──────+
|  1  |  5000 | 90000|   │                    │   |  1  |  8000 | 70000|
+─────+───────+──────+   │                    │   +─────+───────+──────+
                         ▼                    ▼
                    +──────────────────────────────+
                    | Physical Memory at 43060     |
                    | Shared Text Editor (Code)    | (Read-Only Reentrant Code)
                    +──────────────────────────────+
```

- **Reentrant Code**: Software code that never alters its own instructions during execution (pure code).
- **Mechanics of Sharing**: If two users run the text editor `vim`, both processes have an entry in their segment tables (e.g., Segment 0) with identical Base addresses (`43060`) and Limits (`25200`), marked read-only (`r-x`). Exactly one physical copy of the editor executable code exists in RAM.
- **Private Data Segments**: Each process maintains its own private data and stack segments (e.g., Segment 1) pointing to completely distinct physical memory locations (`90000` vs `70000`).

---

### 6.6 Segmentation in x86/x86-64: Historical GDT/LDT to Modern Flat Memory Model

The x86 architecture has a rich history tied to segmentation:

#### 1. 16-bit x86 Real Mode (Intel 8086)
- Addresses were formed using 16-bit segment registers (`CS` - Code, `DS` - Data, `SS` - Stack, `ES` - Extra).
- Physical address was computed as:
  $$\text{Physical Address} = (\text{Segment Register} \times 16) + \text{Offset} = (\text{Segment Register} \ll 4) + \text{Offset}$$
- Allowed access to 1 MB of physical RAM (`20` address lines) with no memory protection whatsoever.

#### 2. 32-bit x86 Protected Mode (Intel 80386 to Pentium)
- Segment registers held **Segment Selectors** pointing to 8-byte descriptors in the **Global Descriptor Table (GDT)** or **Local Descriptor Table (LDT)**.
- Descriptors specified 32-bit Base addresses, 20-bit Limits (scaled by a 4 KB granularity bit to 4 GB), and privilege levels (Rings 0 through 3).
- *Downside*: Caused severe external fragmentation and software engineering complexity across operating system compilers.

#### 3. Modern 64-bit x86-64 Architecture (The "Flat Memory Model")
- In x86-64 Long Mode, modern operating systems like Linux and Windows enforce a **Flat Memory Model**:
  - The segment bases for `CS`, `DS`, `ES`, and `SS` are **hardwired to `0`** by hardware, and segment limits are completely ignored.
  - The logical address generated by code is treated directly as a linear virtual address:
    $$\text{Linear Virtual Address} = \text{Offset}$$
  - **Segmentation is virtually deprecated in modern 64-bit hardware**: all memory isolation, protection, and translation are delegated to **Paging**.
  - *(Exception)*: The `FS` and `GS` segment registers are retained to store 64-bit base addresses for **Thread-Local Storage (TLS)** and kernel data structures.



## 7. Paging Architecture and Hardware Support

### 7.1 Paging Concept: Decoupling Logical and Physical Space

In contiguous memory allocation and pure segmentation, external fragmentation inevitably degrades memory utilization. Over time, physical memory is carved into small, disconnected holes that cannot satisfy large process requests.

**Paging** is a memory-management architecture that completely eliminates external fragmentation by **decoupling the logical address space from the physical address space**. Under paging:
- The physical address space of a process is permitted to be entirely **non-contiguous**.
- A process is allocated physical memory wherever free slots exist, without requiring continuous blocks.

```
       LOGICAL MEMORY (PROCESS)                         PHYSICAL MEMORY (RAM)
       +───────────────────────+                        +───────────────────+
       |        Page 0         | ────────┐              |      Frame 0      |
       +───────────────────────+         │              +───────────────────+
       |        Page 1         | ────┐   └─────────────►|  Frame 1 (Page 0) |
       +───────────────────────+     │                  +───────────────────+
       |        Page 2         | ──┐ └─────────────────►|  Frame 2 (Page 1) |
       +───────────────────────+   │                    +───────────────────+
       |        Page 3         | ─┐│                    |      Frame 3      |
       +───────────────────────+  ││                    +───────────────────+
                                  │└───────────────────►|  Frame 4 (Page 2) |
                                  │                     +───────────────────+
                                  └────────────────────►|  Frame 5 (Page 3) |
                                                        +───────────────────+
```

---

### 7.2 Frames, Pages, and Hardware Translation Mapping

The core abstraction of paging relies on dividing both physical and logical memory into identically sized blocks:

1. **Frames (Physical Memory)**: Physical main memory (DRAM) is partitioned into fixed-sized blocks called **Frames**. Frame sizes are strictly powers of two, typically ranging between **4 KB ($2^{12}\text{ bytes}$)** and **8 KB ($2^{13}\text{ bytes}$)**, with modern architectures supporting huge pages (2 MB and 1 GB).
2. **Pages (Logical Memory)**: The process's logical address space is partitioned into blocks of the **exact same size** called **Pages**.
3. **The Page Table**: Every process maintains a private hardware-indexed lookup table called the **Page Table**. The page table acts as a mathematical function mapping each logical page number $p$ to its corresponding physical frame number $f$ in RAM:
   $$f = \text{PageTable}[p]$$

---

### 7.3 Mathematical Address Decomposition: Page Number (p) and Offset (d)

An address generated by the CPU execution core is broken into two distinct binary components:

```
+─────────────────────────────────────────────+─────────────────────────────────────────+
|               Page Number (p)               |             Page Offset (d)             |
|                 (m - n bits)                |                (n bits)                 |
+─────────────────────────────────────────────+─────────────────────────────────────────+
```

#### Mathematical Formulation:
Let:
- $m$ = Total number of bits in the logical address space (total addressable size = $2^m$ bytes).
- $n$ = Number of bits required to address bytes within a page.
- $\text{Page Size} = 2^n\text{ bytes}$.
- Total number of addressable pages in logical space = $2^{m - n}$.

#### Hardware Translation Workflow:
1. **Extraction**: The hardware MMU splits the incoming $m$-bit logical address:
   $$p = \text{Logical Address} \gg n \quad \left( \text{High-order } m - n \text{ bits} \right)$$
   $$d = \text{Logical Address} \ \& \ (2^n - 1) \quad \left( \text{Low-order } n \text{ bits} \right)$$
2. **Lookup**: The MMU uses page number $p$ as an index into the process's page table to retrieve the physical frame number $f$.
3. **Physical Address Construction**: The physical frame number $f$ is concatenated with the unmodified offset $d$:
   $$\text{Physical Address} = (f \ll n) \mid d = (f \times 2^n) + d$$

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU EXECUTION CORE                                       |
|                         Generates Logical Address: < p , d >                            |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                    │
                                    ├─── p (Page Number)
                                    │        │
                                    │        ▼
                                    │   [ PROCESS PAGE TABLE ]
                                    │   Index p ──► Frame Number f
                                    │                    │
                                    ▼                    ▼
                             d (Offset) ────────────► [ Concatenation / Addition ]
                                                         │
                                                         ▼ Physical Address:
                                                           (f * 2^n) + d
                                                         │
                                                         ▼
                                              [ PHYSICAL MEMORY RAM ]
```

> [!NOTE]
> Because page sizes are powers of two, physical address generation requires **no arithmetic addition in hardware**. The MMU simply replaces the high-order $m - n$ bits of the logical address ($p$) with the frame bits ($f$), while copying the low-order $n$ offset bits ($d$) unchanged!

---

### 7.4 Internal Fragmentation in Paging: Mathematical Bound & Average Case

Because physical memory is allocated in discrete page-sized increments, paging **completely eliminates external fragmentation**. However, it suffers from **Internal Fragmentation**:

- **The Last Page Phenomenon**: A process's memory demand rarely aligns to an exact integer multiple of the page size. If a process requires $M$ bytes, it is allocated $\lceil M / 2^n \rceil$ pages. The final allocated frame will contain unused, trapped bytes.

#### Mathematical Derivations:
- **Worst-Case Internal Fragmentation**: Occurs when a process requires $k$ full pages plus exactly **1 byte** ($M = k \cdot 2^n + 1$). The OS must allocate $k + 1$ full frames, resulting in:
  $$\text{Worst-Case Fragmentation} = 2^n - 1\text{ bytes (Almost an entire frame!)}$$
- **Best-Case Internal Fragmentation**: Occurs when a process size aligns perfectly with a page boundary ($M = k \cdot 2^n$), yielding **0 bytes** of fragmentation.
- **Average-Case Internal Fragmentation**: Assuming process sizes are uniformly distributed across page boundaries:
  $$\text{Average Fragmentation} = \frac{1}{2} \times \text{Page Size} = 2^{n-1}\text{ bytes}$$

#### Step-by-Step Slide Example:
- **Given**:
  - $\text{Page Size} = 2{,}048\text{ bytes} = 2^{11}\text{ bytes}$ ($n = 11$).
  - $\text{Process Size} = 72{,}766\text{ bytes}$.
- **Calculation**:
  $$\text{Number of Pages} = \left\lceil \frac{72{,}766}{2{,}048} \right\rceil = \lceil 35.53 \rceil = \mathbf{36\text{ pages}}$$
  $$\text{Memory Allocated in Full Pages} = 35 \times 2{,}048 = 71{,}680\text{ bytes}$$
  $$\text{Bytes Occupied in 36th Page} = 72{,}766 - 71{,}680 = 1{,}086\text{ bytes}$$
  $$\text{Internal Fragmentation} = 2{,}048 - 1{,}086 = \mathbf{962\text{ bytes}}$$

#### The Page Size Dilemma:
- *Small Page Sizes (e.g., 512 bytes)*: Minimize internal fragmentation, but drastically increase the number of pages, causing the Page Table itself to balloon in memory and overburden the TLB.
- *Large Page Sizes (e.g., 4 KB, 2 MB)*: Increase internal fragmentation per process, but reduce Page Table overhead, improve disk transfer rates (fewer seek operations), and expand TLB memory coverage.

---

### 7.5 Operating System Support: Free-Frame Allocation and Frame Tables

To manage paging hardware, the operating system kernel maintains critical global data structures:

#### 1. The Free-Frame List
The kernel maintains a global tracking list (typically implemented as a **bitmap** or a **linked list of frame indices**) representing all physical memory frames currently unassigned:
- When a process is created or requests additional heap memory, the OS dequeues frames from the free-frame list, zeros their contents (to prevent information leakage from terminated processes), and writes their indices into the process's page table.
- When a process terminates, its allocated frames are returned to the free-frame list.

```
FREE-FRAME LIST (Before): [ Frame 14 ] ──► [ Frame 15 ] ──► [ Frame 18 ] ──► NULL
Process P requests 2 pages:
Allocates Frame 14 (maps to Page 0), Frame 15 (maps to Page 1)
FREE-FRAME LIST (After):  [ Frame 18 ] ──► NULL
```

#### 2. The System-Wide Frame Table
The operating system must manage physical hardware independently of any single process. The kernel maintains a **Frame Table** containing exactly one entry for each physical frame in RAM. Each entry records:
- Whether the frame is free or allocated.
- If allocated, which process (PID) owns the frame.
- Which logical page number of that process is currently occupying the frame.

---

### 7.6 Protection Bits and the Valid-Invalid Bit in Page Tables

Memory protection in paging is enforced on a per-page granularity by attaching hardware control flags to each **Page Table Entry (PTE)**:

```
+──────────────────────────┬───────┬───────┬───────┬────────────────────────────+
|   Physical Frame Number  | Valid | Read  | Write | Execute (NX / XD)          |
|          (f)             | (v/i) |  (r)  |  (w)  | (No-Execute / Instruction) |
+──────────────────────────┴───────┴───────┴───────┴────────────────────────────+
```

1. **The Valid-Invalid Bit ($v / i$)**:
   - **`Valid (v)`**: The page is legally within the process's logical address space and is currently mapped to a physical frame in RAM.
   - **`Invalid (i)`**: The page is either:
     - *Illegal*: The process generated an address outside its legal boundary (e.g., touching unmapped address `0x00000000`). Access triggers an immediate CPU exception: **Segmentation Fault (`SIGSEGV`)**.
     - *Valid but Not in Memory*: The page belongs to the process, but has been swapped to disk or has not yet been loaded. Access triggers a **Page Fault Exception**.
2. **Access Permission Bits ($r / w / x$)**:
   - `Read`: Allows data loading.
   - `Write`: Allows store instructions. Writing to a page marked read-only triggers a hardware protection trap.
   - `Execute (NX / XD Bit)`: Modern 64-bit architectures (AMD's No-Execute `NX`, Intel's eXecute Disable `XD`) mark stack and heap pages as non-executable, preventing malicious buffer-overflow attacks from executing injected shellcode.

---

### 7.7 Shared Pages and Reentrant Code Execution

Paging enables effortless, memory-efficient sharing of read-only code across multiple processes:

```
Process P1 Page Table                             Process P2 Page Table
+──────┬───────+                                  +──────┬───────+
| Page | Frame |                                  | Page | Frame |
+──────┬───────+                                  +──────┬───────+
|  0   |   3   | ──┐                          ┌── |  0   |   3   |
|  1   |   4   | ──┼──────┐            ┌──────┼── |  1   |   4   |
|  2   |   6   | ──┼──────┼──────┐     │      │   |  2   |   6   |
|  3   |   1   |   │      │      ▼     ▼      │   |  3   |   8   |
+──────┴───────+   │      │   +─────────────+ │   +──────┴───────+
                   │      └──►|   Frame 4   |◄┘
                   ▼          | (Shared C   |
            +─────────────+   |  Library)   |
            |   Frame 3   |   +─────────────+
            | (Shared Text|
            |  Editor)    |
            +─────────────+
```

- **Reentrant Code**: Non-self-modifying machine code. If a thousand users concurrently execute the C compiler `gcc` or the bash shell, the operating system maintains **exactly one physical copy** of the executable pages in RAM.
- **Independent Data Pages**: Each process maintains its own distinct, private data and stack frames (e.g., Page 3 maps to Frame 1 for $P_1$, and Frame 8 for $P_2$), guaranteeing data privacy.



## 8. Translation Lookaside Buffer (TLB) and Effective Access Time

### 8.1 The Two-Memory-Access Problem with In-Memory Page Tables

In early architectures, small page tables could be implemented using banks of ultra-fast hardware registers. However, modern page tables contain millions of entries and consume megabytes of storage; consequently, the **Page Table must reside in main memory (DRAM)**.

To manage the in-memory page table, the CPU provides two hardware control registers:
- **Page-Table Base Register (PTBR)**: Points to the starting physical memory address of the current process's page table in RAM.
- **Page-Table Length Register (PTLR)**: Indicates the total number of entries in the page table.

#### The Performance Catastrophe:
When the page table is stored in main memory, **every single logical memory access requires TWO separate physical memory references**:
1. **First Memory Access**: The MMU reads the Page Table Entry (PTE) from RAM at address $\text{PTBR} + (p \times \text{Entry Size})$ to retrieve the frame number $f$.
2. **Second Memory Access**: The MMU accesses the actual target data or instruction at physical address $(f \times 2^n) + d$.

$$\text{Total Access Latency} = \text{Access}(\text{Page Table}) + \text{Access}(\text{Data}) = 2 \times ma$$

> [!CAUTION]
> **Performance Impact**: Storing the page table in RAM slows memory throughput by **100% (a $2\times$ performance slowdown)**. A CPU capable of billions of operations per second is throttled to half-speed simply looking up translation addresses.

---

### 8.2 TLB Architecture: Associative High-Speed Hardware Cache

To solve the two-memory-access bottleneck, hardware architects incorporate a specialized, ultra-fast hardware associative cache inside the MMU: the **Translation Lookaside Buffer (TLB)**.

```
                  HARDWARE TLB (ASSOCIATIVE CACHE)
                  +─────────────┬──────────────┬───────────────+
                  | Page Number | Frame Number | Valid / Flags |
                  +─────────────┼──────────────┼───────────────+
                  |     12      |      45      |       v       |
                  |     84      |      02      |       v       |
                  |     03      |      99      |       v       |
                  +─────────────┴──────────────┴───────────────+
```

#### Key Properties of the TLB:
- **High-Speed Silicon**: Fabricated from SRAM directly on the CPU core die; access latency is typically **under 1 nanosecond (sub-clock cycle)**.
- **Small Capacity**: Typically holds between **32 and 1024 entries** due to power dissipation and silicon die cost constraints.
- **Associative Parallel Lookup**: When a page number $p$ is presented to the TLB, the hardware **compares $p$ against every single entry in the TLB simultaneously in parallel** using dedicated associative comparator circuits. If $p$ is present, the corresponding frame number $f$ is output instantaneously.

---

### 8.3 TLB Hit, TLB Miss, and Hardware vs. Software Handlers

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU GENERATES ADDRESS < p , d >                          |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ├─── p (Page Number)
                                           ▼
                             [ PARALLEL ASSOCIATIVE TLB LOOKUP ]
                                           │
                                 Is Page p in the TLB?
                                  /                 \
                          YES    /                   \   NO
                                ▼                     ▼
                         [   TLB HIT   ]       [   TLB MISS   ]
                                │                     │
                                │ Retrieve Frame f    │ Read Page Table from RAM (PTE)
                                │ from TLB            │ (Consumes 1 DRAM access cycle)
                                │                     │
                                │                     ├─► Install <p, f> into TLB
                                │                     │   (Evicts an old TLB entry)
                                └──────────┬──────────┘
                                           │
                                           ▼ Physical Frame f + Offset d
                                 [ PHYSICAL MEMORY ACCESS ]
                                 (Consumes 1 DRAM access cycle)
```

1. **TLB Hit**:
   - Page number $p$ is found in the TLB.
   - Frame number $f$ is immediately available.
   - Total physical DRAM accesses = **1** (only the actual data/instruction access).
2. **TLB Miss**:
   - Page number $p$ is absent from the TLB.
   - The MMU must perform a full page-table lookup in main memory (accessing RAM).
   - Once retrieved, the pair $\langle p, f \rangle$ is written into the TLB (replacing an existing entry via LRU or random replacement).
   - Total physical DRAM accesses = **2** (1 for Page Table in RAM + 1 for Data in RAM).
3. **Hardware vs. Software Miss Handling**:
   - **Hardware Walkers (CISC / x86, ARM)**: The CPU microcode contains dedicated hardware state machines ("Page Table Walkers") that automatically traverse page tables in RAM on a miss without interrupting the OS.
   - **Software Handlers (RISC / MIPS, SPARC)**: On a TLB miss, the CPU fires a hardware trap to an OS kernel handler. The OS loads the TLB using privileged instructions and resumes execution.

---

### 8.4 Context Switching, TLB Flushes, and Address Space Identifiers (ASIDs)

Because logical address spaces are private to each process, Process $A$ and Process $B$ may both use logical page number `5`, but map it to completely different physical frames (e.g., Frame `12` vs Frame `88`).

When the OS scheduler performs a context switch from Process $A$ to Process $B$, stale TLB entries must not be referenced by the incoming process:

#### Solution 1: TLB Flushing (Invalidation)
- On every context switch, the OS executes a privileged CPU instruction (e.g., reloading register `CR3` in x86) that invalidates every entry in the TLB.
- *Downside*: Causes massive performance penalties; the incoming process starts with a completely cold TLB and suffers an avalanche of TLB misses until its working set is re-cached.

#### Solution 2: Address Space Identifiers (ASIDs)
- Modern processors incorporate an **Address Space Identifier (ASID)** tag directly into each TLB entry.
- The ASID uniquely identifies the process that owns that translation entry (functioning like a hardware process ID).
- When resolving addresses, the MMU matches a TLB entry only if:
  $$\text{Entry Page} == p \quad \mathbf{AND} \quad \text{Entry ASID} == \text{Current Process ASID}$$
- *Benefit*: TLB entries from multiple processes coexist simultaneously across context switches, eliminating TLB flushes and maximizing hit rates.

---

### 8.5 Derivation of Effective Access Time (EAT) with TLB

The performance of a paging system with a TLB is characterized by its **Effective Access Time (EAT)**:

#### Mathematical Variables:
- $\alpha$ = **TLB Hit Ratio** (Fraction of memory accesses where page number is found in the TLB, $0 \le \alpha \le 1$).
- $\epsilon$ = **TLB Lookup Time** (Time to search associative TLB, typically $\approx 0\text{ to }1\text{ ns}$).
- $ma$ = **Main Memory Access Time** (DRAM access latency, typically $10\text{ to }100\text{ ns}$).

#### Rigorous Mathematical Derivation:
- **Time on TLB Hit**: The hardware searches the TLB ($\epsilon$) and then accesses physical DRAM once for data ($ma$):
  $$T_{\text{hit}} = \epsilon + ma$$
- **Time on TLB Miss**: The hardware searches the TLB ($\epsilon$), accesses DRAM to read the Page Table ($ma$), updates the TLB, and accesses DRAM a second time to read data ($ma$):
  $$T_{\text{miss}} = \epsilon + ma + ma = \epsilon + 2ma$$
- **Effective Access Time ($EAT$)**:
  $$EAT = \alpha \cdot T_{\text{hit}} + (1 - \alpha) \cdot T_{\text{miss}}$$
  $$EAT = \alpha(\epsilon + ma) + (1 - \alpha)(\epsilon + 2ma)$$
  $$EAT = \epsilon + \alpha \cdot ma + 2ma - 2\alpha \cdot ma$$
  $$\mathbf{EAT = \epsilon + (2 - \alpha) \cdot ma}$$

*(Note: When TLB lookup time is negligible or pipelined into instruction decode, $\epsilon \approx 0$, simplifying the equation to:)*
$$EAT = (2 - \alpha) \cdot ma$$

#### Step-by-Step Slide Examples:

**Example 1 (From Lecture Slide 88)**:
- Given: Main memory access time $ma = 10\text{ ns}$, TLB Hit ratio $\alpha = 80\% = 0.8$. (Assume $\epsilon = 0$).
  $$EAT = 0.8 \times (10) + (1 - 0.8) \times (20) = 8 + 4 = \mathbf{12\text{ nanoseconds}}$$
  $$\text{Slowdown} = \frac{12 - 10}{10} \times 100\% = \mathbf{20\%\text{ performance penalty}}$$

**Example 2 (Realistic 98% Hit Ratio)**:
- Given: Main memory access time $ma = 10\text{ ns}$, TLB Hit ratio $\alpha = 98\% = 0.98$.
  $$EAT = 0.98 \times (10) + (1 - 0.98) \times (20) = 9.8 + 0.4 = \mathbf{10.2\text{ nanoseconds}}$$
  $$\text{Slowdown} = \frac{10.2 - 10}{10} \times 100\% = \mathbf{2\%\text{ performance penalty (Near-native DRAM speed!)}}$$

---

### 8.6 Multi-Level Paging EAT Derivations and Impact on Performance

When an architecture implements $k$-level hierarchical paging (e.g., 2-level, 4-level, or 5-level paging), a TLB miss forces the MMU to traverse multiple page table levels in DRAM before reaching physical data:

- On a TLB Hit: Still requires only **1 memory access** ($ma$).
- On a TLB Miss: Requires **$k$ memory accesses** to traverse the page table hierarchy $+$ **1 memory access** to read target data $= \mathbf{(k + 1) \cdot ma}$.

$$EAT = \alpha(\epsilon + ma) + (1 - \alpha)(\epsilon + (k + 1) \cdot ma)$$

#### Impact on 4-Level 64-bit Systems ($k = 4, ma = 100\text{ ns}$):
- On a TLB Miss: $T_{\text{miss}} = (4 + 1) \times 100\text{ ns} = \mathbf{500\text{ ns}}$!
- If $\alpha = 95\%$:
  $$EAT = 0.95(100) + 0.05(500) = 95 + 25 = \mathbf{120\text{ ns}}$$
- If $\alpha = 99\%$:
  $$EAT = 0.99(100) + 0.01(500) = 99 + 5 = \mathbf{104\text{ ns}}$$

> [!IMPORTANT]
> This extreme penalty highlights why high TLB hit ratios ($\ge 98\%$) and large hardware TLBs are absolutely vital to the viability of modern 64-bit computing.



## 9. Hierarchical, Hashed, and Inverted Page Table Structures

### 9.1 The Scalability Crisis of Flat Page Tables in 32-bit and 64-bit Systems

A naive, flat page table allocates one entry for every conceivable page in the logical address space. This architecture collapses under basic memory sizing math:

#### 1. The 32-bit Address Space Sizing:
- Logical Address Space: $2^{32}\text{ bytes} = 4\text{ GB}$.
- Page Size: $4\text{ KB} = 2^{12}\text{ bytes}$.
- Total Number of Pages:
  $$\frac{2^{32}}{2^{12}} = 2^{20} = 1{,}048{,}576\text{ pages (1 Mega-pages)}$$
- If each Page Table Entry (PTE) consumes **4 bytes**:
  $$\text{Page Table Size} = 1{,}048{,}576 \times 4\text{ bytes} = \mathbf{4\text{ MB per process!}}$$
- In a system running 100 processes, $100 \times 4\text{ MB} = \mathbf{400\text{ MB}}$ of physical memory is consumed solely by page tables! Furthermore, the OS cannot easily allocate 4 MB of strictly contiguous physical DRAM for each process table.

#### 2. The 64-bit Catastrophe:
- Logical Address Space: $2^{64}\text{ bytes}$.
- Page Size: $4\text{ KB} = 2^{12}\text{ bytes}$.
- Total Pages: $2^{64} / 2^{12} = 2^{52}\text{ pages} \approx 4.5 \times 10^{15}\text{ pages}$.
- If each PTE is **8 bytes**:
  $$\text{Page Table Size} = 2^{52} \times 8\text{ bytes} = 2^{55}\text{ bytes} = \mathbf{32\text{ Petabytes per process!}}$$
- Storing a flat 32-petabyte table is physically impossible. Operating systems solve this crisis through three advanced structural paradigms: **Hierarchical Paging**, **Hashed Page Tables**, and **Inverted Page Tables**.

---

### 9.2 Hierarchical Paging (Forward-Mapped / Multi-Level Page Tables)

The foundational principle of **Hierarchical Paging** is to *"page the page table"*. Instead of storing a massive contiguous page table, the table itself is partitioned into standard 4 KB pages, which can be allocated non-contiguously in RAM and swapped out when idle.

Because address translation proceeds inward from the outer table to the inner tables, this scheme is known as a **Forward-Mapped Page Table**.

---

### 9.3 32-bit Two-Level Paging (x86 Page Directory and Page Table)

In a 32-bit system with 4 KB pages, a Two-Level Page Table breaks the logical address into three components:

```
+──────────────────────────┬──────────────────────────┬─────────────────────────+
|     Page Directory (p1)  |      Page Table (p2)     |        Offset (d)       |
|          10 bits         |          10 bits         |         12 bits         |
+──────────────────────────┴──────────────────────────┴─────────────────────────+
```

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU LOGICAL ADDRESS                                      |
|                        [ p1 (10 bits) | p2 (10 bits) | d (12 bits) ]                    |
+─────────────────────────────────────────────────────────────────────────────────────────+
        │                               │
        │ p1                            │
        ▼                               │
+────────────────+                      │
| Page Directory |                      │
| (Outer Table)  |                      │
+────────────────+                      │
| Entry p1 ──────┼──────┐               │
+────────────────+      │ Base of       │
                        │ Page Table    │ p2
                        ▼               ▼
                 +────────────────+
                 | Inner Page     |
                 | Table          |
                 +────────────────+
                 | Entry p2 ──────┼──────┐ Frame
                 +────────────────+      │ Number f
                                         ▼
                                  +──────────────────────────+
                                  | Physical Memory Frame f  |
                                  | Offset d Added           |
                                  +──────────────────────────+
```

1. **Outer Page Table (Page Directory)**:
   - Indexed by $p_1$ (10 bits $\implies 2^{10} = 1024$ entries).
   - Exactly $1024 \times 4\text{ bytes} = \mathbf{4\text{ KB}}$ (fits perfectly into a single physical page!).
   - Each entry stores the physical base frame address of an inner page table.
2. **Inner Page Table**:
   - Indexed by $p_2$ (10 bits $\implies 2^{10} = 1024$ entries).
   - Exactly $1024 \times 4\text{ bytes} = \mathbf{4\text{ KB}}$ (fits into a single page).
   - Each entry stores the physical frame number $f$ of the actual data page.
3. **Offset ($d$)**:
   - 12 bits $\implies 2^{12} = 4096\text{ bytes}$ displacement within the frame.

> [!TIP]
> **Massive Space Optimization for Sparse Spaces**:
> If a process only uses 2 MB of memory (e.g., small code and stack), it requires only **one Page Directory (4 KB)** and **two Inner Page Tables (8 KB)**—consuming only **12 KB of RAM**, rather than the full 4 MB demanded by flat paging!

---

### 9.4 64-bit Paging: 4-Level (x86-64 PML4) and 5-Level (PML5) Paging

In 64-bit architectures, a two-level scheme is grossly inadequate (the outer table alone would require $2^{42}$ entries $= 32\text{ Terabytes}$). Architectures implement 4-level or 5-level paging:

#### x86-64 4-Level Paging (PML4 Architecture):
Modern processors do not use all 64 address lines; they utilize a **48-bit canonical virtual address space** ($256\text{ TB}$), with the top 16 bits sign-extended:

```
+──────────────┬──────────────┬──────────────┬──────────────┬───────────────────+
| PML4 (Level 4| PDPT (Level 3|  PD (Level 2 |  PT (Level 1 |    Offset (d)     |
|   9 bits     |   9 bits     |   9 bits     |   9 bits     |     12 bits       |
+──────────────┴──────────────┴──────────────┴──────────────┴───────────────────+
```

- **PML4 (Page Map Level 4)**: 9 bits ($2^9 = 512$ entries).
- **PDPT (Page Directory Pointer Table)**: 9 bits (512 entries).
- **PD (Page Directory)**: 9 bits (512 entries).
- **PT (Page Table)**: 9 bits (512 entries).
- **Offset**: 12 bits (4 KB page).
- Every table at every level contains exactly $512 \times 8\text{ bytes} = \mathbf{4096\text{ bytes}}$ (1 frame).

#### 5-Level Paging (PML5):
To accommodate modern cloud data centers requiring more than 256 TB of memory, Intel Ice Lake and Linux introduced 5-level paging, adding a 9-bit Level 5 table to support **57-bit virtual addresses (128 Petabytes)**.

---

### 9.5 Hashed Page Tables and Clustered Page Tables for Sparse Spaces

For address spaces greater than 32 bits, **Hashed Page Tables** are widely adopted:

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU GENERATES ADDRESS < p , d >                          |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                    │
                                    ├─── p (Virtual Page Number)
                                    ▼
                         [ HASH FUNCTION h(p) ]
                                    │
                                    ▼
                         [ HASH TABLE BUCKET ]
                                    │
                                    ▼
                         +─────────────────────────────────────────+
                         | Linked List of Collisions:              |
                         | [ Page p | Frame f | Next Pointer ───┐] |
                         +──────────────────────────────────────┼──+
                                                                │
                                                                ▼
                                                 +─────────────────────────+
                                                 | [ Page p2 | Frame f2 | 0|
                                                 +─────────────────────────+
```

#### Mechanism:
1. The virtual page number $p$ is passed into a mathematical hash function $h(p)$.
2. The hash value indexes into a system hash table.
3. Each bucket contains a linked list of elements resolving hash collisions. Each element contains:
   $$\langle \text{Virtual Page Number } p,\; \text{Physical Frame Number } f,\; \text{Pointer to Next Element} \rangle$$
4. The hardware traverses the linked list, comparing page number $p$. When a match is found, frame number $f$ is extracted.

#### Clustered Page Tables (For 64-bit Systems):
A variation for 64-bit sparse address spaces where each entry maps **several contiguous pages** (e.g., 16 pages) instead of a single page, amortizing hashing overhead.

---

### 9.6 Inverted Page Tables: Architecture, Physical Frame Mapping, and Hash Anchor Tables

Traditional page tables maintain one entry for each *virtual* page. In stark contrast, an **Inverted Page Table** maintains **exactly ONE entry for each real PHYSICAL FRAME of main memory**!

```
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                CPU GENERATES ADDRESS                                    |
|                           < PID , Page Number p , Offset d >                            |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                    │
                                    ├─── Search < PID , p >
                                    ▼
                       [ INVERTED PAGE TABLE ]
                        Index = Physical Frame i
                       +───────┬─────────────+
                       |  PID  | Page Number |
                       +───────┬─────────────+
                       |  101  |      4      | ── Frame 0
                       |  540  |     12      | ── Frame 1
                       |  101  |     90      | ── Frame 2  <── MATCH FOUND AT INDEX i!
                       |  ...  |     ...     |
                       +───────┴─────────────+
                                  │
                                  ▼ Match at Entry i
                     Physical Address = (i * Page_Size) + d
```

#### Key Architecture Properties:
- **System-Wide Singleton**: There is only one inverted page table for the entire operating system, regardless of whether 10 or 10,000 processes are running!
- **Table Size Fixed by DRAM**: The size of the table depends strictly on the size of physical RAM, completely independent of the size of the virtual address space!
- **Entry Structure**: Each entry stores the tuple:
  $$\langle \text{Process-ID (PID)},\; \text{Virtual Page Number } p \rangle$$
  *(The PID acts as an address-space identifier).*
- **Commercial Systems**: Successfully implemented in IBM System/38, IBM RT, IBM RS/6000, PowerPC, and Sun UltraSPARC 64-bit.

---

### 9.7 Challenges of Inverted Page Tables: Search Latency and Shared Memory

Despite their phenomenal memory savings, inverted page tables present two severe engineering hurdles:

#### 1. Search Latency Bottleneck:
- In traditional paging, translation is an $O(1)$ direct array index ($\text{Table}[p]$).
- In an inverted page table, entries are sorted by *physical frame*, but lookups occur on *virtual addresses*. A naive lookup requires an **$O(N)$ linear scan of the entire table**!
- *Remediation*: Systems implement a **Hash Anchor Table (HAT)**: hashing $\langle \text{PID}, p \rangle$ into an anchor table that chains matching entries, bounding searches to 1 or 2 memory accesses.

#### 2. The Shared Memory Dilemma:
- Shared memory occurs when multiple distinct virtual addresses from different processes map to the exact same physical frame.
- **The Fatal Conflict**: Because each physical frame has **strictly ONE entry in the inverted page table**, a frame cannot simultaneously record two different $\langle \text{PID}, p \rangle$ mappings!
- *Workaround*: The page table stores only one virtual mapping at a time; when another process touches the shared frame, an unavoidable page fault occurs to rewrite the mapping, severely degrading shared memory performance.



## 10. Virtual Memory and Demand Paging

### 10.1 The Virtual Memory Abstraction and Sparse Address Spaces

In classical execution models, an entire program binary had to reside contiguously or completely in physical RAM before a single instruction could execute. However, analysis of real-world software reveals that **much of a program is rarely executed simultaneously**:
- Error-handling routines that handle catastrophic edge cases.
- Massive fixed-size arrays, tables, and structures allocated far larger than their actual usage.
- Rarely invoked configuration options, diagnostic tools, and administrative menus.

**Virtual Memory** is an architectural abstraction that separates the user's **Logical (Virtual) Memory** from **Physical Memory (RAM)**. It allows programs to execute **even when they are only partially loaded into physical memory**.

```
              VIRTUAL ADDRESS SPACE (SPARSE MEMORY LAYOUT)
      0xFFFFFFFFFFFFFFFF +────────────────────────────────────────+
                         |           Kernel Space                 |
      0x7FFFFFFFFFFF     +────────────────────────────────────────+
                         |           User Call Stack              |
                         |                  │                     |
                         |                  ▼ (Grows Downward)    |
                         |                                        |
                         |          [ UNALLOCATED HOLE ]          |
                         |          (Sparse Address Space         |
                         |           Zero Physical Frames!)       |
                         |                                        |
                         |                  ▲ (Grows Upward)      |
                         |                  │                     |
                         |           Dynamic Heap                 |
                         +────────────────────────────────────────+
                         |      BSS (Uninitialized Data)          |
                         +────────────────────────────────────────+
                         |      Data (Initialized Globals)        |
                         +────────────────────────────────────────+
                         |      Text (Executable Code Segment)    |
      0x0000000000000000 +────────────────────────────────────────+
```

#### Core Architectural Benefits of Virtual Memory:
1. **Programs Exceed Physical RAM**: Software is no longer constrained by the physical limit of installed DRAM chips. A computer with 8 GB of RAM can easily execute a 32 GB simulation process.
2. **Higher Degree of Multiprogramming**: Because each process occupies only a small active subset of its virtual pages in physical RAM, the CPU scheduler can load many more processes concurrently, dramatically increasing CPU utilization and system throughput.
3. **Reduced I/O Overhead**: Less program code and data need to be loaded into memory or swapped to disk, saving bandwidth and accelerating program startup.
4. **Sparse Address Spaces**: The virtual address space can leave a massive, unallocated chasm between the heap (growing upward) and the stack (growing downward). Physical frames are allocated by the kernel **strictly on demand** when the stack or heap actually expands into those addresses.

---

### 10.2 Demand Paging vs. Pre-Paging and Pure Demand Paging

#### 1. Demand Paging
Instead of loading an entire executable into memory at load time, the operating system employs a **Lazy Swapper (Pager)**. The pager brings a page into physical memory **only when that page is explicitly referenced during execution**:
- If a page is never referenced, it is **never brought into physical memory**.
- Avoids reading unneeded bytes into DRAM.

#### 2. Pre-Paging
An optimization where the OS attempts to predict and load contiguous or related pages into memory before they are referenced, attempting to amortize disk head seek latency. If predictions are incorrect, memory and I/O bandwidth are wasted.

#### 3. Pure Demand Paging
An extreme execution model where a process begins execution with **zero pages in physical memory**:
- The OS sets the instruction pointer ($PC$) to the first instruction of the executable and launches the process.
- The very first instruction fetch immediately triggers a **Page Fault**.
- The page is fetched from disk, and execution resumes.
- Subsequent operand fetches trigger additional page faults until the process accumulates its active execution set (**Locality of Reference**). Once the working set is loaded, page faults drop to near zero.

---

### 10.3 The Valid-Invalid Bit in Demand Paging

In demand paging, hardware support is provided via the **Valid-Invalid Bit** in each Page Table Entry (PTE):

```
+──────────────┬──────────────────┬─────────────────────────────────────────────+
| Page Number  | Valid / Invalid  | Physical Frame / Disk Backing Address       |
+──────────────┼──────────────────┼─────────────────────────────────────────────+
|      0       |      v (Valid)   | Frame 4 (Currently resident in RAM)         |
|      1       |      i (Invalid) | Page 1 on Backing Store (Swap / Executable) |
|      2       |      v (Valid)   | Frame 9 (Currently resident in RAM)         |
|      3       |      i (Invalid) | Illegal Address (Unmapped virtual region)   |
+──────────────┴──────────────────┴─────────────────────────────────────────────+
```

- When the valid-invalid bit is set to **`v` (Valid)**: The page is legal and **resident in physical RAM**. The MMU proceeds with address translation at hardware speeds.
- When the valid-invalid bit is set to **`i` (Invalid)**: The page is **not in physical memory**. Accessing an `i` entry causes the MMU hardware to trap immediately to the operating system: a **Page Fault**.

---

### 10.4 Detailed 6-Step Page-Fault Handling Sequence

When a process references a page marked with an invalid bit (`i`), the operating system executes a precise, six-step hardware-software fault-handling sequence:

```
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 1: CPU References Logical Page ] ──► MMU detects Valid-Invalid bit is 'i'        |
|                                         ──► Hardware TRAP: PAGE FAULT EXCEPTION         |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 2: OS Kernel Trap Handler Intercepts Fault ]                                     |
|  - Saves CPU registers, stack pointers, and state into PCB.                             |
|  - Inspects internal process memory tables (Linux vm_area_struct).                      |
|  - Check: Is address illegal?                                                           |
|       YES ──► Terminate process (SIGSEGV / Segmentation Fault).                         |
|       NO  ──► Valid page, but currently resident on disk backing store.                 |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 3: OS Finds a Free Frame in Physical RAM ]                                       |
|  - Checks system-wide Free-Frame List.                                                  |
|  - If Free Frame exists: Allocate it.                                                   |
|  - If NO Free Frame exists: Invoke PAGE REPLACEMENT ALGORITHM to select a Victim Frame! |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 4: Schedule Disk I/O Transfer ]                                                  |
|  - Issues I/O request to secondary storage (backing store / filesystem).                |
|  - Reads requested page contents into allocated physical frame.                         |
|  - While waiting for slow disk I/O, CPU context switches to another ready process!      |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 5: Disk Controller Raises I/O Completion Interrupt ]                             |
|  - Disk interrupt wakes up the faulted process.                                         |
|  - OS updates Page Table Entry: writes Frame Number, sets Valid-Invalid bit to 'v'.     |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| [ STEP 6: Restart the Interrupted Instruction ]                                         |
|  - Restores CPU registers and program counter ($PC$).                                   |
|  - Process re-executes the exact instruction that faulted; now finds page in RAM!       |
+─────────────────────────────────────────────────────────────────────────────────────────+
```

---

### 10.5 Major (Hard) vs. Minor (Soft) Page Faults

Modern operating systems distinguish between two fundamentally distinct classes of page faults based on the I/O cost required for resolution:

```
                                    PAGE FAULT CLASSIFICATION
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [ MAJOR (HARD) PAGE FAULT ]                       [ MINOR (SOFT) PAGE FAULT ]
             - Page exists ONLY on disk/swap.                  - Page ALREADY in physical RAM!
             - Requires physical disk I/O.                     - NO disk I/O required.
             - Latency: 5 to 10 MILLISECONDS.                  - Latency: 1 to 5 MICROSECONDS.
             - (Slows system down by 50,000x!)                 - (Resolved purely in kernel memory).
```

#### Causes of Minor Page Faults:
1. **Shared Memory / Page Cache**: Process $A$ references a shared library page (e.g., in `libc.so`) that was already brought into RAM by Process $B$. The kernel simply maps the existing physical frame into Process $A$'s page table and marks it valid.
2. **Reclaimed Free Pages**: The page was evicted from the active working set and placed on the standby/free list, but its physical frame was not yet overwritten. The kernel simply restores the valid bit.
3. **Copy-on-Write (COW)**: Duplicating a memory frame already resident in RAM without disk access.

---

### 10.6 File-Backed vs. Anonymous Memory Management

Operating systems classify process memory pages into two categories to determine eviction behavior:

1. **File-Backed Memory**:
   - Pages that correspond directly to a persistent file on disk (e.g., executable binary code `.text`, shared libraries `.so`, memory-mapped files via `mmap()`).
   - **Eviction Behavior**: If memory is needed, clean file-backed pages can be **dropped immediately from RAM without writing to swap**, because they can always be re-read directly from their source executable file on disk.
2. **Anonymous Memory**:
   - Memory pages that have no filesystem backing file (e.g., dynamic heap allocations from `malloc()`, local call stacks, uninitialized BSS variables).
   - **Eviction Behavior**: If the kernel evicts an anonymous page, it **must write the page to a dedicated swap partition or swap file**. If no swap space exists, anonymous pages cannot be evicted!

---

### 10.7 Mobile OS Memory Constraints (iOS and Android Without Swap)

Mobile operating systems deviate sharply from desktop/server operating systems regarding swap storage:

- **No Swap Partition**: Neither Apple iOS nor Google Android supports traditional swapping to disk partitions.
- **Why Mobile OSes Ban Swap**:
  1. *Flash Memory Wear-Out*: Mobile devices use eMMC / UFS NAND flash memory, which has strictly limited write-erase cycle endurance. Continuous swapping would destroy the flash storage within months.
  2. *Severe Throughput / Energy Constraints*: Paging I/O consumes massive battery power and introduces unacceptable UI frame stuttering.
- **Mobile Memory Reclamation Strategies**:
  - *iOS*: Read-only pages (executable code) are dropped from RAM. If free memory remains scarce, iOS sends memory warnings to running apps. If memory is exhausted, the iOS kernel terminates background apps immediately.
  - *Android (Low Memory Killer - LMK)*: Android uses the Low Memory Killer daemon, which categorizes apps by priority (Foreground, Visible, Service, Cached/Background). When RAM drops below thresholds, the kernel terminates background cached processes with `SIGKILL`. Android also utilizes **zRAM** (in-memory compressed swap).

---

### 10.8 Architectural Challenges: Instruction Restart and Auto-Increment/Decrement

A fundamental requirement of demand paging is **Instruction Restart**: the CPU must be able to restart the interrupted instruction and execute it to completion as if nothing occurred. However, complex computer architectures introduce difficult hardware edge cases:

#### 1. Auto-Increment and Auto-Decrement Addressing Modes (PDP-11, VAX, Motorola 68000)
Consider the instruction:
```assembly
ADD (R1)+, (R2)+
```
- This instruction fetches an operand from the address in register `R1`, increments `R1`, fetches the second operand from address `R2`, increments `R2`, and adds the operands.
- **The Page Fault Hazard**: Suppose the fetch from `R1` succeeds (and `R1` is successfully incremented), but the memory fetch from `R2` triggers a Page Fault!
- If the OS services the page fault and simply restarts the instruction from the beginning, **register `R1` will be incremented a second time**, producing corrupted pointer arithmetic and incorrect results!
- **Architectural Solutions**:
  - *Hardware Undo Buffers*: The CPU microcode maintains a shadow register tracking changes made to general-purpose registers during an instruction. On a page fault, the microcode rolls back all registers to their pre-instruction state.
  - *Software Recovery*: The OS kernel decodes the faulting instruction, inspects register increment fields, and manually decrements the register before restarting.

#### 2. Block Move Instructions Crossing Page Boundaries (IBM System/370)
- The IBM System/370 `MVC` (Move Characters) instruction can copy up to 256 contiguous bytes between memory locations.
- If the source and destination overlap, and a page fault occurs halfway through the 256-byte copy on the destination page, restarting the instruction would read partially overwritten source data!
- **Solution**: The CPU microcode pre-checks both the starting and ending addresses of both source and destination blocks to verify page validity *before* moving a single byte.

---

### 10.9 Demand Paging Performance and EAT Mathematical Derivations

The performance of a demand-paged memory system is governed by the probability of encountering a page fault. Even a minuscule page-fault rate causes staggering performance collapse.

#### Mathematical Formulation:
Let:
- $ma$ = Physical main memory access time ($10\text{ to }200\text{ nanoseconds}$).
- $p$ = **Page Fault Rate** ($0 \le p \le 1$):
  - If $p = 0$: No page faults occur; every access is a hit in RAM.
  - If $p = 1$: Every memory access triggers a page fault.
- $T_{\text{pfs}}$ = **Page Fault Service Time** (Average time to handle a page fault).

$$\mathbf{EAT = (1 - p) \times ma + p \times T_{\text{pfs}}}$$

#### Breakdown of Page Fault Service Time ($T_{\text{pfs}}$):
A page-fault service routine consists of three primary phases:
1. **Service Interrupt**: Trap overhead, saving registers, verifying address ($\approx 1\text{ to }5\text{ }\mu\text{s}$).
2. **Read the Page**: Disk seek, rotational latency, and data transfer ($\approx 8\text{ milliseconds} = 8{,}000{,}000\text{ ns}$). **(Dominates 99.9% of the time!)**
3. **Restart the Process**: Allocating CPU, context switch, restoring registers ($\approx 1\text{ to }5\text{ }\mu\text{s}$).

$$T_{\text{pfs}} \approx 8\text{ milliseconds} = 8 \times 10^6\text{ nanoseconds}$$

#### Step-by-Step Slide Example (From Lecture Slide 129):
- **Given**:
  - $ma = 200\text{ nanoseconds}$.
  - $T_{\text{pfs}} = 8\text{ milliseconds} = 8{,}000{,}000\text{ nanoseconds}$.

$$EAT = (1 - p) \times 200 + p \times (8{,}000{,}000)$$
$$EAT = 200 - 200p + 8{,}000{,}000p$$
$$\mathbf{EAT = 200 + 7{,}999{,}800 \times p\text{ nanoseconds}}$$

#### Case 1: Page Fault Rate $p = 0.001$ (1 fault in 1,000 accesses):
$$EAT = 200 + 7{,}999{,}800 \times 0.001 = 200 + 7{,}999.8 = \mathbf{8{,}199.8\text{ ns} \approx 8.2\text{ }\mu\text{s}}$$
$$\text{Slowdown Factor} = \frac{8{,}200}{200} = \mathbf{41\times\text{ SLOWER!}}$$
A system suffering just one page fault per 1,000 memory references slows down by a staggering **factor of 41**!

#### Case 2: Maximum Acceptable Page Fault Rate for $< 10\%$ Slowdown:
Suppose an engineer requires that memory degradation not exceed $10\%$ ($EAT < 220\text{ ns}$):
$$200 + 7{,}999{,}800 \times p < 220$$
$$7{,}999{,}800 \times p < 20$$
$$p < \frac{20}{7{,}999{,}800} \approx \mathbf{2.5 \times 10^{-6}}$$

> [!IMPORTANT]
> **Takeaway**: To keep memory performance degradation under 10%, the system must allow **fewer than 1 page fault for every 400,000 memory accesses**!



## 11. Copy-on-Write (COW) and Page Replacement Foundations

### 11.1 Process Creation Optimization: fork(), exec(), and vfork()

In UNIX and Linux operating systems, a new process is instantiated using the `fork()` system call:

#### The Inefficiency of Classical `fork()`:
- Classical `fork()` created a completely new child process by allocating physical memory frames and **copying every single data, heap, and stack page** of the parent process into the child's address space.
- In modern software, the child process almost immediately invokes the `exec()` system call to overwrite its address space with a new executable binary (e.g., executing a command in bash).
- *Massive Waste*: Duplicating hundreds of megabytes of parent memory only to instantly destroy and overwrite them milliseconds later is an enormous waste of CPU cycles and DRAM.

#### The Solution: Copy-on-Write (COW)
**Copy-on-Write (COW)** postpones page duplication until the exact moment a process attempts to modify a page.

---

### 11.2 Hardware Mechanism of Copy-on-Write: Read-Only Trap and Frame Duplication

```
                  INITIAL STATE: IMMEDIATELY AFTER FORK()
+─────────────────────────────────+             +─────────────────────────────────+
|      Process 1 (Parent)         |             |       Process 2 (Child)         |
|  Page C: Marked READ-ONLY ('r') |             |  Page C: Marked READ-ONLY ('r') |
+─────────────────────────────────+             +─────────────────────────────────+
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            +─────────────────────────+
                            |     Physical Frame      |
                            |         Page C          |
                            +─────────────────────────+
```

```
               AFTER PROCESS 1 ATTEMPTS TO WRITE TO PAGE C
1. Process 1 executes: Page_C[10] = 'X';
2. MMU detects Write to READ-ONLY page -> Raises Hardware PROTECTION FAULT TRAP!
3. OS Kernel recognizes page is flagged as Copy-on-Write.
4. OS allocates a brand new physical frame from Free-Frame List.
5. OS copies the 4KB contents of Page C into the new frame.
6. OS updates Process 1's Page Table: points to new frame, marks READ-WRITE ('rw').
7. Process 2's Page Table remains pointing to original frame.

+─────────────────────────────────+             +─────────────────────────────────+
|      Process 1 (Parent)         |             |       Process 2 (Child)         |
|  Page C: Marked READ-WRITE ('rw')|            |  Page C: Marked READ-ONLY ('r') |
+─────────────────────────────────+             +─────────────────────────────────+
                 │                                               │
                 ▼                                               ▼
    +─────────────────────────+                     +─────────────────────────+
    |   New Physical Frame    |                     | Original Physical Frame |
    |   (Private Copy of P1)  |                     | (Now Private to P2)     |
    +─────────────────────────+                     +─────────────────────────+
```

#### The `vfork()` System Call:
- An ultra-fast, historical variant of `fork()` where the parent process is completely suspended, and the child **directly shares the parent's address space without even copying page tables**.
- The child must not modify any parent memory and must immediately call `exec()` or `_exit()`. Any memory modification by the child corrupts the parent's state! COW has made `vfork()` largely unnecessary.

---

### 11.3 Memory Over-Allocation and the Need for Page Replacement

In a multiprogramming system with virtual memory, the operating system routinely **over-allocates physical memory**:
- Suppose a system has 40 physical frames. The OS may run 6 processes, each having a virtual address space of 10 pages (total virtual demand = 60 pages).
- The OS relies on the statistical fact that not all 6 processes will need all 10 pages simultaneously.

#### The Crisis of Frame Exhaustion:
What happens if all processes suddenly expand their active working sets, and a page fault occurs when **the Free-Frame List is completely empty (0 free frames)?**
- The OS cannot simply allocate a frame from the free list.
- Terminating the user process is unacceptable.
- **The Solution**: The operating system must invoke **Page Replacement**: selecting an existing page currently occupying a physical frame in RAM, evicting it, and reassigning that physical frame to the newly requested page!

---

### 11.4 Victim Frame Eviction Sequence and Page Table Invalidation

The process of page replacement proceeds through a rigorous four-step sequence:

```
+─────────────────────────────────────────────────────────────────────────────────────────+
| 1. Find the location of the desired page on the backing store disk.                     |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| 2. Find a free frame:                                                                   |
|    - If a free frame exists in the Free-Frame List ──► USE IT.                          |
|    - If NO free frame exists:                                                           |
|      Use Page Replacement Algorithm to select a VICTIM FRAME.                           |
|      Write the victim frame to disk IF DIRTY.                                           |
|      Update victim process Page Table: set Valid-Invalid bit to 'i' (Invalid).           |
|      Invalidate TLB entry for victim page.                                              |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| 3. Read the desired page from disk into the newly freed physical frame.                 |
|    Update faulting process Page Table: set Frame Number, set Valid-Invalid bit to 'v'.  |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| 4. Restart the user instruction that caused the page fault.                             |
+─────────────────────────────────────────────────────────────────────────────────────────+
```

---

### 11.5 The Dirty (Modify) Bit Optimization: Halving I/O Transfer Overhead

Page replacement potentially requires **two complete disk I/O operations**:
1. Writing the evicted victim page out to disk.
2. Reading the desired page from disk into RAM.

Because disk I/O is the primary bottleneck in demand paging, writing every evicted page to disk would double page-fault latency.

#### Hardware Support: The Modify (Dirty) Bit
Hardware architects incorporate a **Dirty Bit (Modify Bit)** directly into each Page Table Entry:
- When a page is first brought into a physical frame, the MMU hardware initializes the dirty bit to **`0`**.
- Whenever the CPU executes any store/write instruction to any byte within that page, the hardware MMU automatically flips the dirty bit to **`1`**.

```
                           DIRTY BIT EVICTION LOGIC
                                       │
                             Is Dirty Bit == 1 ?
                                    /     \
                            YES    /       \   NO
                                  ▼         ▼
                      [ PAGE IS MODIFIED ]   [ PAGE IS CLEAN ]
                      - Must write frame     - Page in RAM is IDENTICAL
                        contents to disk       to copy already on disk!
                        before eviction.     - DISK WRITE SKIPPED!
                      - Consumes 1 disk      - Frame immediately reassigned.
                        write I/O cycle.     - Eviction latency cut by 50%!
```

> [!TIP]
> By eliminating disk writes for clean pages (such as all executable code segments), the dirty bit effectively **halves the overhead of page replacement**.

---

### 11.6 Kernel Paging Daemons: Linux kswapd, Watermarks (Min, Low, High)

A naive operating system might wait until physical free memory reaches exactly **0 frames** before invoking page replacement. This causes severe application freezes, because the faulting thread must block synchronously while a victim page is written to disk.

#### Linux Background Page Reclamation (`kswapd`):
Modern Linux kernels maintain a background kernel daemon named **`kswapd`** that proactively preserves a pool of free frames using three memory watermarks:

```
HIGH MEMORY
     │
     ▲
     │    +────────────────────────────────────────+
     │    |   Abundant Memory (kswapd sleeps)      |
     │    +────────────────────────────────────────+ pages_high watermark
     │    |   kswapd wakes up and reclaims pages   |
     │    +────────────────────────────────────────+ pages_low watermark
     │    |   Memory pressure mounting             |
     │    +────────────────────────────────────────+ pages_min watermark
     ▼    |   SYNCHRONOUS DIRECT RECLAIM!          |
     │    |   (Allocating processes stall!)        |
LOW MEMORY+────────────────────────────────────────+
```

1. **`pages_high`**: Memory is abundant. `kswapd` sleeps.
2. **`pages_low`**: Free memory has dropped below safe operational buffers. `kswapd` wakes up in the background and quietly evicts clean/dirty pages to restore free memory back up to `pages_high`. User processes continue running uninterrupted.
3. **`pages_min`**: Free memory is critically exhausted. The kernel forces allocating processes into **Direct Reclaim**: processes stall synchronously to evict pages themselves.



## 12. Page Replacement Algorithms

### 12.1 Evaluation Metrics and Memory Reference Strings

When physical memory is fully saturated and a page fault occurs, the operating system must select a victim frame to evict. The policy governing this choice is the **Page Replacement Algorithm**.

#### Primary Objective:
The ultimate goal of any page replacement algorithm is to **minimize the total number of page faults** across the execution lifetime of a process. Fewer page faults directly translate to higher CPU throughput and lower disk I/O traffic.

#### Evaluation Methodology:
Operating system researchers evaluate replacement algorithms by executing them against a recorded trace of memory accesses called a **Reference String**:
- A reference string is a sequence of page numbers accessed by a process.
- Redundant contiguous references to the exact same page are consolidated (e.g., accessing address `0x1004`, `0x1008`, `0x100C` within page `1` produces a single reference to `1` in the string, because only the first access could ever cause a page fault).
- Algorithms are benchmarked by plotting **Page Faults vs. Number of Available Frames**:

```
Page Faults
    ▲
    │   \
    │     \
    │       \  (Expected Curve: As frames increase, page faults decline)
    │         \________________
    └─────────────────────────────► Number of Frames
```

---

### 12.2 First-In, First-Out (FIFO) Algorithm

The **FIFO algorithm** is the simplest replacement policy. When a frame must be replaced, the page that has resided in memory the longest (**the oldest page**) is selected as the victim.

- **Implementation**: The OS maintains a FIFO queue of all pages currently in memory.
- When a page is brought into RAM, it is enqueued at the tail.
- On a page fault requiring replacement, the page at the head of the queue is dequeued and evicted.

#### Step-by-Step Trace (From Lecture Slide 150):
- **Reference String**: `7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1`
- **Allocated Frames**: **3**

```
Ref:   7   0   1   2   0   3   0   4   2   3   0   3   2   1   2   0   1   7   0   1
F1:   [7] [7] [7] [2] [2] [2]  2  [4] [4] [4] [0]  0   0  [0]  0   0   0  [7] [7] [7]
F2:       [0] [0] [0] [0] [3]  3  [3] [2] [2] [2]  2   2  [1]  1   1   1  [1] [0] [0]
F3:           [1] [1] [1] [1]  0  [0] [0] [3] [3]  3   3  [3]  2  [2]  2  [2] [2] [1]
Fault: *   *   *   *       *       *   *   *   *           *       *       *   *   *
```
- **Total Page Faults for FIFO with 3 frames**: **15 Faults**.

---

### 12.3 Belady's Anomaly: Definition, Demonstration, and Stack Algorithms

Intuition dictates that giving a process more physical memory frames should always reduce—or at least keep constant—the number of page faults.

**Belady's Anomaly** is the startling phenomenon where, for certain page replacement algorithms, **the page-fault rate INCREASES as the number of allocated frames INCREASES**!

#### Formal Demonstration (From Lecture Slide 151):
Consider the reference string: `1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5` under FIFO:

#### Case 1: Execution with 3 Frames:
```
Ref:   1   2   3   4   1   2   5   1   2   3   4   5
F1:   [1]  1   1  [4]  4   4  [5]  5   5  [3]  3   3
F2:       [2]  2   2  [1]  1   1  [1]  1   1  [4]  4
F3:           [3]  3   3  [2]  2   2  [2]  2   2  [5]
Fault: *   *   *   *   *   *   *           *   *   *  ==> TOTAL: 9 PAGE FAULTS
```

#### Case 2: Execution with 4 Frames (Adding More Memory!):
```
Ref:   1   2   3   4   1   2   5   1   2   3   4   5
F1:   [1]  1   1   1   1   1  [5]  5   5   5  [4]  4
F2:       [2]  2   2   2   2   2  [1]  1   1   1  [5]
F3:           [3]  3   3   3   3   3  [2]  2   2   2
F4:               [4]  4   4   4   4   4  [3]  3   3
Fault: *   *   *   *           *   *   *   *   *   *  ==> TOTAL: 10 PAGE FAULTS!
```
With 3 frames: **9 page faults**. With 4 frames: **10 page faults**! Adding more RAM caused *worse* performance!

#### Why Belady's Anomaly Occurs: Stack Algorithms & Inclusion Property:
- **Stack Algorithms**: An algorithm is classified as a stack algorithm if the set of pages resident in memory for $n$ frames is **always a strict subset** of the set of pages that would be resident with $n+1$ frames at any time $t$:
  $$B(t, n) \subseteq B(t, n+1)$$
- **Theorem**: A stack algorithm **can NEVER exhibit Belady's Anomaly**. Adding more frames guarantees that page faults will strictly decrease or remain identical.
- **Why FIFO Fails**: FIFO is **not a stack algorithm**. The oldest page in memory is evicted regardless of whether it was referenced recently. Giving FIFO more frames completely scrambles the relative arrival order and eviction points, violating the inclusion property.

---

### 12.4 Optimal (OPT / MIN / Clairvoyant) Page Replacement Algorithm

Discovered by L.A. Belady, the **Optimal Page Replacement Algorithm (OPT or MIN)** provides the theoretical upper bound on replacement efficiency:

> [!NOTE]
> **Optimal Principle**:
> Replace the page that **will not be used for the longest period of time in the future**.

#### Step-by-Step Trace (From Lecture Slide 152):
- **Reference String**: `7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1`
- **Allocated Frames**: **3**

```
Ref:   7   0   1   2   0   3   0   4   2   3   0   3   2   1   2   0   1   7   0   1
F1:   [7]  7   7  [2]  2   2   2   2   2   2   2   2   2  [2]  2   2   2  [7]  7   7
F2:       [0]  0   0   0   0   0  [4]  4   4   0   0   0  [0]  0   0   0   0   0   0
F3:           [1]  1   1  [3]  3   3   3   3   3   3   3  [1]  1   1   1   1   1   1
Fault: *   *   *   *       *       *               *       *               *
```
- **Total Page Faults for OPT**: **9 Faults** (Guaranteed minimum possible!).

#### Practical Infeasibility:
- OPT requires **Clairvoyance**: absolute future knowledge of which memory addresses the program will reference.
- Because an OS cannot foretell future user inputs or conditional branches, **OPT is physically impossible to implement in production systems**.
- **Utility**: Serves as the ultimate mathematical benchmark against which all real-world algorithms are measured.

---

### 12.5 Least Recently Used (LRU) Algorithm: Theory and Stack Property

The **Least Recently Used (LRU) algorithm** approximates OPT by looking **backward in time** rather than forward, leveraging the **Principle of Locality** (if a page has not been used for a long time, it is unlikely to be needed soon).

- **Rule**: Replace the page that **has not been used for the longest period of time in the past**.
- **Stack Property**: LRU is a proven **stack algorithm**. It is mathematically immune to Belady's Anomaly!

#### Step-by-Step Trace (From Lecture Slides 153-154):
- **Reference String**: `7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1`
- **Allocated Frames**: **3**

```
Ref:   7   0   1   2   0   3   0   4   2   3   0   3   2   1   2   0   1   7   0   1
F1:   [7]  7   7  [2]  2   2   2  [4]  4   4  [0]  0   0   0   0   0   0  [7]  7   7
F2:       [0]  0   0   0   0   0   0  [2]  2   2   2   2  [1]  1   1   1   1  [0]  0
F3:           [1]  1   1  [3]  3   3   3  [3]  3   3   3  [3]  2  [2]  2   2   2  [1]
Fault: *   *   *   *       *       *   *   *   *           *       *       *   *   *
```
- **Total Page Faults for LRU**: **12 Faults**.
- Notice the hierarchy: $\text{OPT (9 faults)} < \text{LRU (12 faults)} < \text{FIFO (15 faults)}$.

---

### 12.6 LRU Implementation Bottlenecks: Hardware Counters vs. Doubly Linked Stacks

While conceptually elegant, pure LRU is rarely implemented in hardware due to staggering computational overhead:

#### 1. Hardware Counters Implementation:
- The CPU includes a global 64-bit hardware clock/counter incremented on every instruction.
- Every Page Table Entry has a `time-of-last-use` register.
- On every memory reference, hardware writes the current clock value into that page's counter.
- **Bottleneck**: On a page fault, the OS must perform an **$O(N)$ linear search across the entire page table** to find the minimum counter value.

#### 2. Doubly Linked Stack Implementation:
- The OS maintains a stack of page numbers represented as a doubly linked list.
- Whenever a page is referenced, it is unlinked from its current position and moved to the **top of the stack**.
- The page at the **bottom of the stack** is always the least recently used victim!
- **Bottleneck**: Moving a node to the top requires updating **6 pointers in memory on EVERY SINGLE MEMORY REFERENCE**. Executing pointer manipulations on billions of memory loads per second would cripple CPU performance.

---

### 12.7 Approximating LRU: Reference Bits and Additional-Reference-Bits History Register

Because pure LRU is too expensive, hardware architects provide low-cost primitive bits that allow the OS to closely approximate LRU:

#### 1. The Hardware Reference Bit (Accessed Bit)
- Each PTE contains a 1-bit hardware flag: the **Reference Bit** ($R$).
- Initialized to `0` by the OS. Whenever the CPU reads or writes to the page, the hardware MMU automatically flips $R \to 1$.
- Provides basic coarse information about which pages have been touched.

#### 2. Additional-Reference-Bits Algorithm
- The OS maintains an **8-bit history shift register** for each page in RAM.
- At regular timer interrupts (e.g., every 100 milliseconds), an OS kernel timer routine shifts each page's 8-bit register **1 bit to the right**, copying the current hardware Reference Bit into the High-Order (Most Significant) Bit, and clearing the hardware bit to `0`:
  $$\text{History Register} = (R \ll 7) \mid (\text{History Register} \gg 1)$$

```
Page A: 1 1 0 0 0 1 0 0  (Referenced recently in last 2 intervals; value = 196)
Page B: 0 0 0 0 1 1 1 1  (Not referenced in last 4 intervals; value = 15)
```
- The page with the **smallest unsigned integer value** is the least recently used victim!

---

### 12.8 Clock (Second-Chance) Page Replacement Algorithm

The **Clock Algorithm (Second-Chance Algorithm)** is the industry-standard, low-overhead LRU approximation implemented in modern kernels.

```
                      THE CLOCK (SECOND-CHANCE) CIRCULAR BUFFER
                                     [ Frame 0 ]
                                      (Use = 1)
                                   ┌─────────────┐
                    [ Frame 7 ] ───┘             └─── [ Frame 1 ]
                     (Use = 0)                         (Use = 1)
                        ▲                                  │
                        │        CLOCK HAND (-->)          │
                        │           ┌─────────┐            │
                    [ Frame 6 ]     │ Inspect │       [ Frame 2 ]
                     (Use = 1)      │ Use Bit │        (Use = 0)
                        │           └─────────┘            ▲
                        │                                  │
                    [ Frame 5 ] ───┐             ┌─── [ Frame 3 ]
                     (Use = 0)     └─────────────┘     (Use = 1)
                                     [ Frame 4 ]
                                      (Use = 1)
```

#### Step-by-Step Mechanism:
1. **Circular Queue**: Physical memory frames are conceptualized as arranged in a circular ring, with a single **Clock Hand** pointer pointing to the next frame to be evaluated.
2. **On a Page Fault Requiring Eviction**:
   - The OS inspects the page pointed to by the clock hand:
   - **Case A: $\text{Use Bit} == 1$**:
     - The page was used recently. It is granted a **"second chance"**!
     - The OS clears its use bit: $\text{Use Bit} \to \mathbf{0}$.
     - The clock hand advances to the next frame in the ring.
   - **Case B: $\text{Use Bit} == 0$**:
     - The page was not used recently. **This is our victim!**
     - If dirty, write to disk; invalidate PTE; load the incoming page into this frame; set the new page's $\text{Use Bit} = \mathbf{1}$.
     - Advance the clock hand to the next frame.
3. **Termination Guarantee**:
   - If all pages have their use bits set to `1`, the clock hand will make one full revolution around the ring, clearing every bit from `1` to `0`. On its second pass, it will immediately select the first frame it encounters (degrading cleanly to FIFO). It will **never loop indefinitely**.

#### Dynamics of Clock Hand Movement (From Slide 171):
- **Slowly Moving Hand**: Indicates that page faults are rare and the system has abundant free memory, or the hand quickly finds a page with $\text{Use} = 0$. Excellent system health.
- **Rapidly Moving Hand**: Indicates intense memory pressure and frequent page faults. The hand must clear large swaths of use bits to locate an eviction candidate.

---

### 12.9 Enhanced Second-Chance Algorithm: The (Reference, Modify) 4-Class Selection

The **Enhanced Second-Chance Algorithm** (utilized in macOS and classic UNIX) incorporates both the **Reference Bit ($R$)** and the **Modify/Dirty Bit ($M$)** into an ordered pair $\langle R, M \rangle$:

1. **Class 1: $\langle 0, 0 \rangle$**: Neither recently used nor modified. **Best possible victim!** Evicting requires zero disk write I/O.
2. **Class 2: $\langle 0, 1 \rangle$**: Not recently used, but modified. Requires writing to disk before eviction, but has not been referenced recently.
3. **Class 3: $\langle 1, 0 \rangle$**: Recently used, but clean. Likely to be used again soon, but clean.
4. **Class 4: $\langle 1, 1 \rangle$**: Recently used and modified. **Worst possible victim!** Actively in use and requires a slow disk write.

#### Multi-Pass Scan:
The clock hand scans the circular ring in up to 4 passes:
- **Pass 1**: Search for $\langle 0, 0 \rangle$. Do not modify reference bits. Evict first match found.
- **Pass 2**: Search for $\langle 0, 1 \rangle$. Clear $R \to 0$ for all visited pages. Evict first match found.
- **Pass 3 & 4**: Repeat Passes 1 and 2 if no victim was discovered.

---

### 12.10 Counting-Based Algorithms: LFU and MFU

Operating systems can maintain an access counter for each page:
- **Least Frequently Used (LFU)**: Evicts the page with the smallest access count. *Problem*: A page heavily used during process initialization will accumulate a huge count and remain permanently stuck in RAM even after it is never touched again.
- **Most Frequently Used (MFU)**: Evicts the page with the largest count, based on the heuristic that the page with the smallest count was just brought in and hasn't had time to execute.
- *Reality*: Both algorithms are extremely expensive to track in hardware and perform poorly compared to Clock and LRU.



## 13. Frame Allocation, NUMA, and Thrashing

### 13.1 Architecture-Enforced Minimum and Maximum Frame Constraints

When allocating physical memory among multiple concurrent processes, the operating system is bounded by strict upper and lower limits:

- **Maximum Number of Frames**: Bounded by the total amount of physical DRAM installed in the machine minus frames reserved by the OS kernel.
- **Minimum Number of Frames**: Strictly **dictated by the computer architecture's Instruction Set Architecture (ISA)**!

#### Why the Architecture Enforces a Minimum:
When a page fault occurs during an instruction, the instruction must be restarted. Therefore, the OS **must allocate enough frames to hold all the different pages that any single machine instruction can simultaneously reference**:
- *Direct Addressing*: Requires at least 2 frames (1 for the instruction opcode page, 1 for the data operand page).
- *Indirect Addressing (e.g., PDP-11 / IBM 370)*:
  ```assembly
  LOAD R1, @(R2)
  ```
  Requires at least **3 frames** (1 for instruction page, 1 for the pointer address page, 1 for the target operand page).
- *Multiple-Word Instructions (e.g., IBM 370 MVC)*: A move instruction that crosses page boundaries for both source and destination requires **up to 6 frames simultaneously**!
If a process is allocated fewer than its architecture-mandated minimum frames, an instruction can never complete, deadlocking the system in an infinite page-fault loop.

---

### 13.2 Equal vs. Proportional vs. Priority Frame Allocation

How should $m$ available physical frames be distributed among $n$ active processes?

#### 1. Equal Allocation
- Divides available memory equally among all processes:
  $$a_i = \frac{m}{n}\text{ frames per process}$$
- *Slide Example (Slide 179)*: If $m = 93$ frames and $n = 5$ processes:
  $$a_i = \lfloor 93 / 5 \rfloor = \mathbf{18\text{ frames}}$$
  The remaining $3$ frames are placed into the system free-frame buffer pool.
- **Fatal Flaw**: Completely ignores process size! A 10 KB utility tool receives 18 frames (wasting memory), while a 500 MB database process receives 18 frames and thrashes violently.

#### 2. Proportional Allocation
- Allocates memory proportionally based on the virtual address space size ($s_i$) of each process:
  $$S = \sum_{i=1}^n s_i$$
  $$a_i = \left( \frac{s_i}{S} \right) \times m$$
- *Slide Example (Slide 180)*: Allocating $m = 62$ frames between Process $P_1$ ($s_1 = 10\text{ pages}$) and Process $P_2$ ($s_2 = 127\text{ pages}$):
  $$S = 10 + 127 = 137\text{ pages}$$
  $$a_1 = \frac{10}{137} \times 62 \approx \mathbf{4\text{ frames (or 5)}}$$
  $$a_2 = \frac{127}{137} \times 62 \approx \mathbf{57\text{ frames}}$$

#### 3. Priority Allocation
- Allocates frames based on CPU scheduling priority rather than sheer size, ensuring high-priority interactive or real-time processes have ample memory to prevent page-fault latency spikes.

---

### 13.3 Global vs. Local Page Replacement Trade-offs

When a process suffers a page fault and must evict a victim, where can it select that victim from?

```
                 GLOBAL REPLACEMENT                                  LOCAL REPLACEMENT
+─────────────────────────────────────────────────+ +─────────────────────────────────────────────────+
| Physical RAM Frames:                            | | Physical RAM Frames:                            |
| [ P1 ] [ P1 ] [ P2 ] [ P2 ] [ P3 ] [ P1 ]       | | [ P1 ] [ P1 ] [ P1 ] | [ P2 ] [ P2 ] | [ P3 ]   |
|                                                 | | <── Allocated P1 ──> | <── Alloc P2 ─> | Alloc P3|
| Process 1 suffers page fault:                   | | Process 1 suffers page fault:                   |
| -> Can steal a frame from ANY process (e.g. P2)!| | -> Can ONLY evict its OWN frames!               |
+─────────────────────────────────────────────────+ +─────────────────────────────────────────────────+
```

| Dimension | Global Page Replacement | Local Page Replacement |
| :--- | :--- | :--- |
| **Eviction Scope** | Process can select a victim frame from the set of **all frames** in the system, stealing frames from other processes. | Process can select a victim **only from its own assigned set of frames**. |
| **System Throughput** | **Higher overall system throughput**; memory dynamically flows to the processes that need it most. | Lower overall throughput; a process may thrash while another process has idle frames. |
| **Execution Predictability** | **Non-deterministic**: A process's execution time depends wildly on the paging behavior of other processes running on the system. | **Deterministic**: A process's paging behavior is isolated from external system interference. |
| **Modern OS Usage** | **Standard in Linux, Windows, and macOS**. | Used in real-time kernels and mixed-criticality systems. |

---

### 13.4 Non-Uniform Memory Access (NUMA): Topology, Node Latencies, and NUMA-Aware Placement

Modern multi-socket enterprise servers utilize **Non-Uniform Memory Access (NUMA)** architectures:

```
                   NUMA SYSTEM ARCHITECTURE (TWO NODES)
+──────────────────────────────────────+      +──────────────────────────────────────+
|              NUMA NODE 0             |      |              NUMA NODE 1             |
|                                      |      |                                      |
|  [ CPU Socket 0 ] ──── Local Bus ──┐ |      |  [ CPU Socket 1 ] ──── Local Bus ──┐ |
|  (Cores 0 - 63)                    │ |      |  (Cores 64 - 127)                  │ |
|                                    ▼ |      |                                    ▼ |
|                        [ Local DRAM 0 ]      |                        [ Local DRAM 1 ]
+───────────────────────────────▲──────+      +───────────────────────────────▲──────+
                                │                                             │
                                └─────────── Interconnect Bus ────────────────┘
                                      (UPI / QPI / Infinity Fabric)
                                       (HIGHER ACCESS LATENCY!)
```

#### Core NUMA Concepts:
- **Local Memory vs. Remote Memory**:
  - A CPU core accessing DRAM located on its **own NUMA node** experiences low latency (typically $\approx 50\text{ ns}$).
  - A CPU core accessing DRAM attached to a **remote NUMA node** must traverse the inter-socket interconnect bus (Intel UPI / AMD Infinity Fabric), suffering **$2\times\text{ to }3\times\text{ higher access latency}$**!
- **NUMA-Aware Memory Allocation**:
  - Operating systems (Linux) must schedule threads on the CPU cores located on the exact same node where their physical memory pages reside.
  - When allocating a frame on a page fault, the kernel always attempts to allocate from the **local node's free-frame list** before resorting to remote memory.
- **Linux Diagnostics**: Administrators manage NUMA placement using `numactl --interleave` or `numactl --cpunodebind=0 --membind=0`.

---

### 13.5 Thrashing: Definition, Cascade Dynamics, and CPU Utilization Collapse

**Thrashing** is a pathological state of system collapse where **the operating system spends more time paging (swapping pages between RAM and disk) than executing actual user instructions**.

```
    CPU Utilization
        ▲
   100% │            /────────\
        │           /          \   (Catastrophic Thrashing Cliff!)
        │          /            \
        │         /              \
        │        /                \
        │       /                  \
        │      /                    \
        │     /                      \
        └────/────────────────────────\────────► Degree of Multiprogramming
                                       ▲
                                   Thrashing
                                   Begins Here!
```

#### The Deadly Thrashing Spiral (Snowball Effect):
1. **Memory Saturation**: As the degree of multiprogramming increases, each process receives fewer physical frames.
2. **Working Set Eviction**: Eventually, processes lose pages that belong to their active working sets.
3. **Exploding Page Faults**: Processes begin faulting continuously, waiting in the device queue for the disk backing store.
4. **CPU Idles**: While waiting for disk transfers, processes enter the blocked state, causing **CPU utilization to plummet**.
5. **Scheduler Mistake**: The OS CPU scheduler detects low CPU utilization and mistakenly concludes that the system is lightly loaded!
6. **Fatal Reaction**: To increase utilization, the scheduler **increases the degree of multiprogramming** by admitting even more processes into memory!
7. **Complete Collapse**: The new processes demand frames, stealing them from existing processes. System throughput collapses to near zero; disk heads thrash violently, and the machine freezes.

---

### 13.6 Locality Model of Program Execution

The foundational reason demand paging works—and the reason thrashing occurs—is governed by the **Locality Model**:

- **Locality of Reference**: As a process executes, it moves from one execution locality to another.
  - A **Locality** is a set of pages actively referenced together during a phase of execution (e.g., a function, a tight loop, an array manipulation, or a subroutines module).
  - Programs consist of multiple overlapping localities.

```
Why Thrashing Occurs (Mathematical Definition):
Thrashing occurs when:
                    SUM (Size of Locality of all active processes) > Total Physical Memory Size (m)
```
If the cumulative localities of active processes cannot fit simultaneously into RAM, thrashing is mathematically guaranteed.

---

### 13.7 Working-Set Model: Parameter Δ, Working-Set Size (WSS), and Thrashing Prevention

Developed by Peter J. Denning, the **Working-Set Model** utilizes the locality principle to prevent thrashing:

#### The Working-Set Window ($\Delta$):
- $\Delta$ is a fixed parameter defining a sliding window of the most recent **page references** (e.g., $\Delta = 10{,}000$ memory accesses).

```
Memory References:  ... 2 6 1 5 7 7 7 7 5 1 [ 6 2 3 4 1 2 3 4 4 4 3 4 3 4 ]  <-- Current Time t
                                            <──────── Window Δ ─────────────>
                                              Pages in Window: { 1, 2, 3, 4 }
                                              Working Set Size (WSS) = 4
```

#### Formulation:
1. **$WSS_i$ (Working-Set Size of Process $i$)**: The total count of distinct pages referenced by process $i$ within the last $\Delta$ memory references.
2. **$D$ (Total System Frame Demand)**:
   $$D = \sum_{i=1}^n WSS_i$$
3. **The Prevention Policy**:
   - Let $m$ = Total available physical frames in the system.
   - **If $D \le m$**: Memory demand is satisfied. Processes execute smoothly without thrashing.
   - **If $D > m$**: **Thrashing is imminent!** The operating system must immediately intervene by **suspending one or more processes**, rolling them out to disk, and reallocating their frames among the remaining active processes until $D \le m$.

---

### 13.8 Page-Fault Frequency (PFF) Strategy: Upper and Lower Bound Triggers

While the Working-Set model is theoretically sound, tracking sliding windows in hardware is complex. A more direct, practical approach is the **Page-Fault Frequency (PFF)** strategy:

```
    Page-Fault Rate
           ▲
           │   [ UPPER THRESHOLD ] ──► Process is Thrashing!
           │   ───────────────────     ACTION: Allocate MORE frames to process!
           │                           (If no free frames, SUSPEND a process).
           │
           │   [ ACCEPTABLE ZONE ] ──► Healthy operation.
           │
           │   ───────────────────
           │   [ LOWER THRESHOLD ] ──► Process has EXCESS memory!
           │                           ACTION: RECLAIM frames from process.
           └─────────────────────────► Time
```

1. **Upper Threshold Trigger**: If a process's page-fault rate climbs above the Upper Bound, it lacks enough frames to hold its locality. The OS allocates additional frames. If the free-frame list is empty, the OS suspends a process.
2. **Lower Threshold Trigger**: If the page-fault rate drops below the Lower Bound, the process has more frames than its locality requires. The OS safely reclaims frames to expand the system free pool.

---

### 13.9 OS Rescue Mechanisms: Linux OOM Killer and Memory Compression

When memory demand exceeds all algorithmic limits, modern kernels implement drastic survival mechanisms:

#### 1. The Linux Out-Of-Memory (OOM) Killer
When physical DRAM and swap space are 100% exhausted and page reclamation fails, the Linux kernel invokes the **OOM Killer** (`mm/oom_kill.c`) to rescue the machine from a fatal kernel panic:
- **`badness` Algorithm**: Scans all active processes and computes an `oom_score` (0 to 1000) based on:
  - Percentage of physical RAM consumed (Resident Set Size - RSS).
  - Swap space consumed.
  - Runtime duration (prefers saving long-lived daemons).
  - Administrative bias (`/proc/[pid]/oom_score_adj`).
- The process with the highest score is terminated immediately with **`SIGKILL`**, instantly freeing memory and saving the operating system.

#### 2. Memory Compression (zRAM / zswap in Linux and macOS)
Rather than writing victim pages to slow disk swap, modern OSes compress inactive pages in RAM:
- Uses blazing-fast hardware-accelerated algorithms (LZ4, ZSTD).
- Achieves a typical **$2:1\text{ to }3:1$ compression ratio**.
- Pushes the "thrashing cliff" significantly further out, providing high performance on memory-constrained systems (e.g., Apple Silicon Macs and Android smartphones).



## 14. Official Slide Review Questions and Authoritative Answers

---

### 14.1 Module 1: Hardware, Protection, Address Binding, and Linking (Slide 24)

#### Q1: Why must memory protection checks (like base and limit) be implemented in hardware rather than by the OS software?
**Authoritative Answer**:
Memory protection checks must be executed on **every single memory reference** generated by the CPU during instruction fetching and data manipulation. The operating system software does not execute between individual CPU machine cycles; handing control to an OS software routine on every memory cycle would degrade processor throughput by thousands of percent (inducing catastrophic context switch overhead). Only dedicated hardware comparators embedded in the CPU pipeline can validate addresses at the wire-speed of the microprocessor clock (< 1 nanosecond) without stalling execution.

#### Q2: What happens if a user process attempts to access a memory address that is greater than its limit register?
**Authoritative Answer**:
The hardware comparators in the CPU/MMU detect that $\text{Logical Address} \ge \text{Limit Register}$. The hardware immediately aborts the memory bus transaction and raises a CPU exception: a **Trap to the Operating System (Addressing Error / Segmentation Fault)**. The OS trap handler intercepts this exception and terminates the offending process with a signal such as `SIGSEGV` to safeguard system integrity.

#### Q3: Can a user process modify its own base and limit registers to access more memory? Why or why not?
**Authoritative Answer**:
**No, absolutely not.** The machine instructions that load and modify the Base and Limit registers are strictly **privileged instructions**. The CPU hardware permits these instructions to execute only when the processor is running in **Kernel Mode (Ring 0)**. If a user-mode process attempts to execute an instruction modifying these registers, the CPU immediately generates a hardware *Privileged Instruction Exception / General Protection Fault*, terminating the process. Only the OS kernel modifies these registers during a process context switch.

#### Q4: What is the primary disadvantage of static linking?
**Authoritative Answer**:
The primary disadvantages are **massive physical memory and disk bloat**, as well as **difficult maintenance**:
1. *Duplication*: Every binary executable file contains a complete, private copy of all linked system libraries (e.g., standard C library `libc`). If 100 processes run concurrently, 100 redundant copies of the library reside in physical RAM.
2. *Patching Overhead*: If a security vulnerability or bug is discovered and fixed in a library, every single application binary compiled on the system must be recompiled and relinked to receive the fix.

#### Q5: What is a "stub" in the context of dynamic linking?
**Authoritative Answer**:
A **stub** is a small piece of code embedded in the executable binary for each reference to a library function. When invoked for the first time, the stub:
1. Checks whether the required shared library is already loaded in physical RAM.
2. If not resident, requests the operating system loader to load the library image from disk into memory.
3. Replaces itself with the absolute memory address of the loaded library function and jumps directly to that code. Subsequent executions invoke the function directly without linker overhead.

#### Q6: Explain the concept of "Execution-time address binding."
**Authoritative Answer**:
Execution-time (or run-time) address binding delays the mapping of logical program addresses to physical memory addresses until the exact clock cycle the instruction executes. If a process can be moved from one memory segment to another during its execution (e.g., via swapping or compaction), execution-time binding is mandatory. It requires hardware support in the form of a **Memory Management Unit (MMU)** containing a dynamic Relocation Register:
$$\text{Physical Address} = \text{Relocation Register} + \text{Logical Address}$$

#### Q7: If a system uses a relocation (base) register set to 14000, what physical address is accessed if the program requests logical address 346?
**Authoritative Answer**:
$$\text{Physical Address} = \text{Relocation Register} + \text{Logical Address} = 14{,}000 + 346 = \mathbf{14{,}346}$$

---

### 14.2 Module 2: Swapping, Contiguous Allocation, and Fragmentation (Slide 48)

#### Q1: In early systems, standard swapping involved moving entire processes between main memory and a backing store. Why is this specific technique rarely used in modern operating systems like Linux?
**Authoritative Answer**:
Standard whole-process swapping incurs **prohibitive context switch latency**. Transferring a multi-gigabyte modern process across disk buses requires several seconds (e.g., swapping a 3 GB process over a 50 MB/s disk takes 120 seconds). Modern operating systems instead utilize **Demand Paging (Page Swapping)**, where only individual, fine-grained 4 KB idle pages are moved to the swap store, allowing the process to remain active in RAM with minimal latency.

#### Q2: What is the primary problem with swapping out a process that is currently waiting for an I/O operation to complete?
**Authoritative Answer**:
The **Pending I/O Hazard**: If a process initiates an asynchronous I/O operation into its memory buffer and is then swapped out, its physical memory may be allocated to a different incoming process. When the Direct Memory Access (DMA) controller completes the transfer, it writes the data directly into that physical memory address, **corrupting the data of the newly loaded process**. Solutions include blocking swaps during I/O or utilizing kernel double-buffering.

#### Q3: In contiguous memory allocation, what is the role of the Limit Register?
**Authoritative Answer**:
The Limit Register specifies the **exact size (range) of the process's logical address space**. The hardware verifies that every generated address is strictly less than the Limit register ($\text{Address} < \text{Limit}$), preventing the process from reading or overwriting memory belonging to other processes or the operating system.

#### Q4: A process requests 400KB of memory, but the OS allocates a fixed 512KB partition for it. What specific type of problem does this cause?
**Authoritative Answer**:
This causes **Internal Fragmentation**. The unused $512\text{ KB} - 400\text{ KB} = \mathbf{112\text{ KB}}$ resides inside the allocated partition and is completely trapped, unable to be allocated to any other process in the system.

#### Q5: A system has 500MB of total free memory, but a process requesting 300MB cannot be loaded. What is occurring here?
**Authoritative Answer**:
The system is suffering from severe **External Fragmentation**. While the cumulative free memory (500 MB) is greater than the requested size (300 MB), the free memory is not contiguous; it is fragmented into smaller, non-contiguous holes scattered across physical RAM, none of which is individually large enough to accommodate the 300 MB request.

---

### 14.3 Module 3: Segmentation Architecture and x86 (Slide 67)

#### Q1: In a segmented memory system, what two components make up a logical address?
**Authoritative Answer**:
A logical address consists of a two-dimensional tuple:
$$\langle s,\; d \rangle$$
where **$s$** is the **Segment Number** (used as an index into the Segment Table), and **$d$** is the **Offset** (displacement within the segment).

#### Q2: When the CPU translates a logical address in a segmented system, what causes a "trap to the operating system" (a Segmentation Fault)?
**Authoritative Answer**:
A trap to the operating system occurs when:
1. The segment number exceeds the table bounds: $s \ge \text{STLR}$ (Segment-Table Length Register).
2. The offset exceeds the segment limit: **$d \ge \text{Limit}$**.
3. The access violates protection permissions (e.g., executing a store instruction on a segment marked read-only).

#### Q3: Why is it advantageous to separate the Stack and the Code into different segments?
**Authoritative Answer**:
1. *Protection and Security*: Code segments can be marked **Read-Only and Executable (`r-x`)**, preventing accidental self-modification and malicious code-injection attacks. The Stack segment is marked **Read-Write and No-Execute (`rw-`)**, allowing local variable modification while preventing stack-smashing shellcode execution.
2. *Dynamic Growth and Sharing*: The code segment is fixed-size and can be shared among multiple processes (reentrant code), whereas the stack segment dynamically grows and shrinks during function calls and is private to each process.

#### Q4: What is the "Flat Memory Model" used by modern Linux on x86-64 architectures?
**Authoritative Answer**:
In x86-64 Long Mode, modern Linux configures hardware segment descriptor base registers (`CS`, `DS`, `ES`, `SS`) to be **fixed at `0`**, and disables segment limit checking. The logical address generated by user code is treated directly as a linear virtual address space spanning 0 to $2^{64}-1$. Modern OSes have completely deprecated hardware segmentation in favor of **pure Paging** for address translation and protection.

#### Q5: During process execution, which hardware component performs the addition of the segment base and the logical offset?
**Authoritative Answer**:
The hardware adder inside the **Memory Management Unit (MMU)** performs the addition ($\text{Physical Address} = \text{Base} + d$) after hardware comparators verify that $d < \text{Limit}$.

---

### 14.4 Module 4: Virtual Memory, Demand Paging, and Page Faults (Slide 141)

#### Q1: What is the fundamental definition of Virtual Memory?
**Authoritative Answer**:
Virtual Memory is a memory-management architecture that **separates user logical memory from physical main memory**, enabling the execution of processes that are only **partially loaded in physical RAM**. It provides software with the illusion of a massive, contiguous, private address space that can far exceed the physical capacity of installed DRAM.

#### Q2: In a virtual address space, why is it beneficial that the Stack and Heap grow towards each other from opposite ends?
**Authoritative Answer**:
Growing the Heap upward from low memory and the Stack downward from high memory creates a **Sparse Address Space**. The massive unallocated gap between them requires **zero physical memory frames** until touched. Both the heap and stack can dynamically expand and contract independently without pre-allocating contiguous physical RAM or imposing artificial size restrictions.

#### Q3: What does the term "Pure Demand Paging" mean?
**Authoritative Answer**:
Pure Demand Paging is an execution policy where a process begins execution with **zero pages in physical memory**. The OS sets the instruction pointer ($PC$) to the entry point and runs the process; the very first instruction fetch immediately causes a page fault, bringing in the first page, followed by faults for data, until the working set is resident.

#### Q4: How does the Memory Management Unit (MMU) differentiate between a page that is in RAM and a page that is swapped to disk?
**Authoritative Answer**:
The MMU inspects the **Valid-Invalid Bit** ($v / i$) in the corresponding Page Table Entry:
- `v (Valid)`: The page is currently resident in a physical RAM frame.
- `i (Invalid)`: The page is either unmapped (illegal address) or currently resides on the secondary storage backing store.

#### Q5: What is the immediate hardware response when a CPU attempts to access a page marked with an 'i' (invalid) bit?
**Authoritative Answer**:
The MMU hardware immediately halts instruction execution, prevents the bus transaction, and fires an internal hardware exception: a **Trap to the Operating System (Page Fault Trap)**, passing control to the OS kernel page-fault interrupt service routine.

#### Q6: Why is a "Minor Page Fault" much faster to resolve than a "Major Page Fault"?
**Authoritative Answer**:
- A **Major Page Fault** requires reading page data from slow persistent disk storage (SSD or HDD), incurring massive physical I/O transfer latencies of **5 to 10 milliseconds**.
- A **Minor Page Fault** occurs when the requested page **is already resident in physical RAM** (e.g., in the OS page cache, shared library, or unmapped zeroed pool). The kernel simply updates the process's page table entry and marks it valid in **1 to 5 microseconds** with **zero disk I/O**.

#### Q7: Describe the architectural difficulty of "Instruction Restart" during a page fault.
**Authoritative Answer**:
When a page fault occurs mid-instruction, the CPU must restart the instruction from scratch. The architectural difficulty arises when instructions have **side effects that alter registers or memory before the fault occurs**:
- *Auto-increment/decrement addressing*: In `ADD (R1)+, (R2)+`, if register `R1` is incremented and fetching `(R2)` causes a page fault, restarting naively increments `R1` a second time, corrupting pointers.
- *Overlapping block copies*: In instructions like IBM System/370 `MVC`, copying blocks across page boundaries could leave destination memory partially overwritten before the fault. Hardware must maintain undo log buffers or microcode checks to roll back state.

#### Q8: What is the difference between File-backed memory and Anonymous memory?
**Authoritative Answer**:
- **File-Backed Memory**: Pages that correspond directly to files on disk (e.g., `.text` executable code, shared libraries, `mmap` files). When evicted, clean pages are **discarded without writing to swap**, as they can be re-read from disk.
- **Anonymous Memory**: Pages with no underlying filesystem file (e.g., dynamic heap allocations, thread call stacks, BSS data). When evicted, they **must be written to dedicated swap storage**.

#### Q9: Why do mobile operating systems (like iOS/Android) typically avoid traditional swapping to a disk partition?
**Authoritative Answer**:
1. *Flash Memory Degradation*: Mobile devices rely on NAND flash storage, which suffers rapid wear-out under constant swap write-erase cycles.
2. *Throughput & Battery Drain*: Swapping causes massive power consumption and creates user-noticeable UI frame stuttering.
- Instead, mobile OSes drop read-only pages, compress memory (zRAM), and kill background processes via the Low Memory Killer daemon.

---

### 14.5 Module 5: Copy-on-Write and Page Replacement Basics (Slide 142)

#### Q1: In a Copy-on-Write (COW) system, what hardware permission does the OS apply to the shared pages immediately after a fork()?
**Authoritative Answer**:
The OS marks all shared pages in both the parent's and the child's page tables as strictly **Read-Only (`r`)**. Any subsequent attempt by either process to write to a page triggers an immediate hardware protection fault, prompting the kernel to allocate a new physical frame and copy the page data.

#### Q2: What specific problem necessitates the use of a Page Replacement algorithm?
**Authoritative Answer**:
**Memory Over-Allocation**: When the cumulative virtual memory demand of all active multiprogrammed processes exceeds the total capacity of physical RAM frames, and the system **Free-Frame List becomes completely empty (0 free frames)** during a page fault. The OS must replace an existing page to make room for the incoming page.

#### Q3: What is a "Victim Frame"?
**Authoritative Answer**:
A **Victim Frame** is a physical memory frame currently occupied by a page that has been selected by the page replacement algorithm to be evicted from RAM, allowing a newly faulted page to take its place.

#### Q4: What is the purpose of the "Dirty Bit" (Modify bit) in a page table entry?
**Authoritative Answer**:
The **Dirty Bit (Modify Bit)** tracks whether a page in RAM has been written to or modified since it was loaded from disk. The hardware MMU sets this bit to `1` on any write instruction. On eviction:
- If `Dirty == 0`: The page is clean (identical to the copy on disk); **the disk write is skipped**.
- If `Dirty == 1`: The page was modified; it **must be written back to disk**.

#### Q5: Why does replacing a dirty victim page effectively double the page-fault service time?
**Authoritative Answer**:
Replacing a dirty page requires **two sequential disk I/O operations**:
1. Writing the modified victim page out to disk storage.
2. Reading the newly requested page from disk into RAM.
Because disk I/O dominates page-fault handling latency, two disk operations double the service time compared to evicting a clean page (which requires only the read operation).

#### Q6: What must the OS do to the page table of the process whose page was just evicted as a victim?
**Authoritative Answer**:
The OS must:
1. Locate the victim's Page Table Entry.
2. Change the **Valid-Invalid bit from 'v' (Valid) to 'i' (Invalid)**.
3. Record the page's new backing store location (disk block address) in the entry.
4. Execute a **TLB Invalidation (Flush)** for that virtual page address to clear any stale hardware cache entries.

#### Q7: In modern systems like Linux, does the OS wait until there are exactly 0 free frames before executing page replacement?
**Authoritative Answer**:
**No.** Modern operating systems do not wait until free memory reaches 0 frames. Linux runs a background kernel daemon (**`kswapd`**) that activates when free memory drops below the **`pages_low` watermark**, quietly evicting pages in the background until free memory reaches **`pages_high`**, ensuring that incoming processes always find a pool of free frames immediately without blocking synchronously.

---

### 14.6 Module 6: Page Replacement Algorithms and Clock (Slide 172)

#### Q1: What is the primary objective of a page-replacement algorithm?
**Authoritative Answer**:
To **minimize the total number of page faults** encountered by executing processes, thereby maximizing CPU utilization and minimizing slow secondary storage I/O traffic.

#### Q2: How does the FIFO page replacement algorithm select a victim frame?
**Authoritative Answer**:
FIFO selects the page that has resided in physical memory for the **longest continuous period of time (the oldest page)**, regardless of how frequently or recently it was accessed.

#### Q3: Describe Belady's Anomaly.
**Authoritative Answer**:
Belady's Anomaly is the counter-intuitive phenomenon where increasing the number of physical memory frames allocated to a process results in an **INCREASE in the total number of page faults** for certain page replacement algorithms (specifically non-stack algorithms like FIFO).

#### Q4: If the Optimal algorithm guarantees the best performance, why do operating systems not use it?
**Authoritative Answer**:
The Optimal algorithm (OPT) requires **future knowledge (clairvoyance)** of the exact sequence of memory addresses that a process will reference. Because an operating system cannot predict future program execution paths or user inputs, OPT is physically impossible to implement in real systems; it is used solely as a theoretical benchmark.

#### Q5: What software behaviour makes the LRU (Least Recently Used) algorithm a good approximation of OPT?
**Authoritative Answer**:
The **Principle of Locality (Temporal Locality)**: Programs tend to reference memory locations that they have accessed recently. Therefore, looking backward into past references serves as a reliable statistical predictor of future memory access patterns.

#### Q6: In the stack implementation of LRU, what must happen to the stack every time a page is referenced?
**Authoritative Answer**:
Whenever a page is referenced, the corresponding page entry must be **removed from its current position in the doubly linked list and moved to the TOP of the stack**. The bottom of the stack always points to the least recently used victim.

#### Q7: Why is the pure stack implementation of LRU rarely used in modern operating systems?
**Authoritative Answer**:
Updating the doubly linked stack requires modifying **6 memory pointers on EVERY SINGLE MEMORY ACCESS**. Executing software pointer updates on billions of instructions per second would severely degrade CPU speed.

#### Q8: How does a hardware "Reference Bit" help the OS approximate LRU efficiently?
**Authoritative Answer**:
The Reference Bit provides minimal, high-speed 1-bit hardware support: the MMU flips the bit to `1` whenever a page is accessed. This bit allows the OS to determine which pages have been used and which have been idle during a clock interval, enabling algorithms like the **Clock Algorithm** to approximate LRU with virtually zero runtime penalty.

---

### 14.7 Module 7: Frame Allocation, NUMA, Thrashing, and OOM (Slide 190)

#### Q1: What is the main drawback of the Equal Allocation of frames among processes?
**Authoritative Answer**:
Equal allocation ignores differences in **process size and memory demand**. It allocates the exact same number of frames to a tiny 20 KB utility process and a colossal 2 GB database process, causing the large process to thrash violently due to memory starvation while the small process wastes unneeded physical frames.

#### Q2: In Global Page Replacement, what is a primary side effect on a process's execution time?
**Authoritative Answer**:
A process's execution time becomes **non-deterministic and unpredictable**. Because processes can steal frames from one another, a process's paging behavior and runtime depend heavily on the memory demands and scheduling of completely unrelated processes running concurrently.

#### Q3: Define the term "Thrashing."
**Authoritative Answer**:
Thrashing is a state of severe performance collapse where **the operating system spends more time paging (swapping pages in and out of disk) than executing user instructions**, causing system throughput and CPU utilization to plummet to near zero.

#### Q4: During a thrashing event, why does the Operating System often mistakenly make the problem worse?
**Authoritative Answer**:
When processes thrash, they spend most of their time waiting for disk I/O in the blocked state. The CPU scheduler observes that **CPU utilization has dropped** and mistakenly assumes the system is lightly loaded. To compensate, the scheduler **increases the degree of multiprogramming** by admitting new processes, which demand even more frames, accelerating the page-fault cascade and locking the system completely.

#### Q5: How does Local Page Replacement limit the effects of thrashing?
**Authoritative Answer**:
Under Local Page Replacement, a process can evict frames **only from its own allocated pool**. If one process begins thrashing, it cannot steal frames from other processes, confining the performance degradation strictly to the faulting process and preventing system-wide collapse.

#### Q6: According to the Locality Model, why does thrashing occur?
**Authoritative Answer**:
Thrashing occurs when the **sum of the sizes of the execution localities of all active processes exceeds the total capacity of physical memory**:
$$\sum_{i} \text{Size}(\text{Locality}_i) > \text{Total Memory Size } m$$

#### Q7: In the Working-Set Model, what does the parameter Δ represent?
**Authoritative Answer**:
$\Delta$ represents the **Working-Set Window**: a fixed parameter denoting a sliding window of the most recent **page references** examined to determine the active working set of a process.

#### Q8: In the Page-Fault Frequency (PFF) approach, what action does the OS take if a process's page-fault rate drops below the established lower bound?
**Authoritative Answer**:
The OS concludes that the process has been allocated **more frames than its active locality requires**. The kernel safely **reclaims (deallocates) frames from that process** and returns them to the system free-frame pool.

#### Q9: In modern Linux systems, what drastic measure does the kernel take when page-fault rates are critical and absolutely no free frames are available?
**Authoritative Answer**:
The Linux kernel invokes the **Out-Of-Memory (OOM) Killer**. The OOM killer computes a `badness` score for every process based on memory usage, swap consumption, and priority, and abruptly terminates the largest memory-hogging process using **`SIGKILL`**, instantly freeing physical frames and preventing an unrecoverable kernel panic.



## 15. Comprehensive Solved Numerical Problems

---

### 15.1 Dynamic Memory Allocation Placement (First-Fit, Best-Fit, Worst-Fit)

**Problem Statement (From Lecture Slide 44)**:
Given six contiguous memory partitions of sizes **300 KB, 600 KB, 350 KB, 200 KB, 750 KB, and 125 KB** (in that physical order).
How would the **First-Fit**, **Best-Fit**, and **Worst-Fit** algorithms place five arriving processes of sizes:
- $P_1 = 115\text{ KB}$
- $P_2 = 500\text{ KB}$
- $P_3 = 358\text{ KB}$
- $P_4 = 200\text{ KB}$
- $P_5 = 375\text{ KB}$ (arriving in order)?

Rank the algorithms in terms of how efficiently they utilize memory, and state whether any processes must wait.

#### Step-by-Step Solution:

Initial Partitions State:
- Slot 1: 300 KB
- Slot 2: 600 KB
- Slot 3: 350 KB
- Slot 4: 200 KB
- Slot 5: 750 KB
- Slot 6: 125 KB

---

#### 1. First-Fit Allocation (Scans from top to bottom, picks first hole $\ge$ request):
- **$P_1$ (115 KB)**: Scans Slot 1 (300 KB). Fits! Placed in **Slot 1** (Remaining: $300 - 115 = 185\text{ KB}$).
- **$P_2$ (500 KB)**: Scans Slot 1 (185 KB - too small), Slot 2 (600 KB). Fits! Placed in **Slot 2** (Remaining: $600 - 500 = 100\text{ KB}$).
- **$P_3$ (358 KB)**: Scans Slot 1 (185 KB), Slot 2 (100 KB), Slot 3 (350 KB - too small!), Slot 4 (200 KB - too small!), Slot 5 (750 KB). Fits! Placed in **Slot 5** (Remaining: $750 - 358 = 392\text{ KB}$).
- **$P_4$ (200 KB)**: Scans Slot 1 (185 KB), Slot 2 (100 KB), Slot 3 (350 KB). Fits! Placed in **Slot 3** (Remaining: $350 - 200 = 150\text{ KB}$).
- **$P_5$ (375 KB)**: Scans Slot 1 (185 KB), Slot 2 (100 KB), Slot 3 (150 KB), Slot 4 (200 KB), Slot 5 (392 KB). Fits! Placed in **Slot 5** (Remaining: $392 - 375 = 17\text{ KB}$).

**Result**: **All 5 processes successfully allocated!** No process waits.

---

#### 2. Best-Fit Allocation (Scans entire list, picks smallest hole $\ge$ request):
- Current Available: 300, 600, 350, 200, 750, 125.
- **$P_1$ (115 KB)**: Smallest hole $\ge 115\text{ KB}$ is **Slot 6 (125 KB)**. Placed in **Slot 6** (Remaining: $125 - 115 = 10\text{ KB}$).
- **$P_2$ (500 KB)**: Smallest hole $\ge 500\text{ KB}$ is **Slot 2 (600 KB)**. Placed in **Slot 2** (Remaining: $600 - 500 = 100\text{ KB}$).
- **$P_3$ (358 KB)**: Candidates $\ge 358\text{ KB}$: Slot 5 (750 KB). Placed in **Slot 5** (Remaining: $750 - 358 = 392\text{ KB}$).
- **$P_4$ (200 KB)**: Candidates $\ge 200\text{ KB}$: Slot 1 (300 KB), Slot 3 (350 KB), Slot 4 (200 KB), Slot 5 (392 KB). Smallest is exact fit: **Slot 4 (200 KB)**. Placed in **Slot 4** (Remaining: $0\text{ KB}$).
- **$P_5$ (375 KB)**: Candidates $\ge 375\text{ KB}$: Slot 5 has 392 KB left! Placed in **Slot 5** (Remaining: $392 - 375 = 17\text{ KB}$).

**Result**: **All 5 processes successfully allocated!** Zero waiting.

---

#### 3. Worst-Fit Allocation (Scans entire list, picks largest available hole):
- Current Available: 300, 600, 350, 200, 750, 125.
- **$P_1$ (115 KB)**: Largest hole is **Slot 5 (750 KB)**. Placed in **Slot 5** (Remaining: $750 - 115 = 635\text{ KB}$).
- Available: 300, 600, 350, 200, 635, 125.
- **$P_2$ (500 KB)**: Largest hole is **Slot 5 (635 KB)**. Placed in **Slot 5** (Remaining: $635 - 500 = 135\text{ KB}$).
- Available: 300, 600, 350, 200, 135, 125.
- **$P_3$ (358 KB)**: Largest hole is **Slot 2 (600 KB)**. Placed in **Slot 2** (Remaining: $600 - 358 = 242\text{ KB}$).
- Available: 300, 242, 350, 200, 135, 125.
- **$P_4$ (200 KB)**: Largest hole is **Slot 3 (350 KB)**. Placed in **Slot 3** (Remaining: $350 - 200 = 150\text{ KB}$).
- Available: 300, 242, 150, 200, 135, 125.
- **$P_5$ (375 KB)**: Largest available hole in entire system is **Slot 1 (300 KB)**.
  $$300\text{ KB} < 375\text{ KB} \implies \mathbf{CANNOT\ ALLOCATE!}$$

**Result**: **P5 must wait!** Worst-fit fails to place all processes.

#### Final Efficiency Ranking:
1. **Best-Fit** (Optimal packing, perfect zero-remainder fit for $P_4$).
2. **First-Fit** (Allocated all processes with fastest search time).
3. **Worst-Fit** (Failed; left $P_5$ stranded due to premature partitioning of large blocks).

---

### 15.2 Segmentation Address Translation and Limit Fault Verification

**Problem Statement (From Lecture Slides 63-64)**:
Consider a segmented memory architecture with the following Segment Table:

| Segment ($s$) | Base Address | Limit (Length) |
| :---: | :---: | :---: |
| **0** | 219 | 600 |
| **1** | 2300 | 14 |
| **2** | 90 | 100 |
| **3** | 1327 | 580 |
| **4** | 1952 | 96 |

Compute the physical addresses for the following logical addresses $\langle s, d \rangle$, or indicate if an addressing error (Segmentation Fault) occurs:
- a. $\langle 0, 430 \rangle$
- b. $\langle 1, 10 \rangle$
- c. $\langle 2, 500 \rangle$
- d. $\langle 3, 400 \rangle$
- e. $\langle 4, 112 \rangle$

#### Step-by-Step Solution:

Rule: For address $\langle s, d \rangle$, check if $d < \text{Limit}$. If valid, $\text{Physical Address} = \text{Base} + d$.

- **a. Logical Address $\langle 0, 430 \rangle$**:
  - For Segment 0: $\text{Limit} = 600$, $\text{Base} = 219$.
  - Check: $430 < 600$ (True $\implies$ Valid).
  - Physical Address = $219 + 430 = \mathbf{649}$.
- **b. Logical Address $\langle 1, 10 \rangle$**:
  - For Segment 1: $\text{Limit} = 14$, $\text{Base} = 2300$.
  - Check: $10 < 14$ (True $\implies$ Valid).
  - Physical Address = $2300 + 10 = \mathbf{2310}$.
- **c. Logical Address $\langle 2, 500 \rangle$**:
  - For Segment 2: $\text{Limit} = 100$, $\text{Base} = 90$.
  - Check: $500 < 100$ (False! $500 \ge 100$).
  - **Result**: **Illegal Address Access $\implies$ Trap to OS (Segmentation Fault)**.
- **d. Logical Address $\langle 3, 400 \rangle$**:
  - For Segment 3: $\text{Limit} = 580$, $\text{Base} = 1327$.
  - Check: $400 < 580$ (True $\implies$ Valid).
  - Physical Address = $1327 + 400 = \mathbf{1727}$.
- **e. Logical Address $\langle 4, 112 \rangle$**:
  - For Segment 4: $\text{Limit} = 96$, $\text{Base} = 1952$.
  - Check: $112 < 96$ (False! $112 \ge 96$).
  - **Result**: **Illegal Address Access $\implies$ Trap to OS (Segmentation Fault)**.

---

### 15.3 Paging Address Decomposition and Physical Translation

**Problem Statement**:
A computer system implements a 32-bit logical address space and uses a page size of **$4\text{ KB} = 4096\text{ bytes}$**.
Physical main memory is **$512\text{ MB}$**.
1. How many bits are used for the page offset ($d$), and how many bits represent the logical page number ($p$)?
2. How many total pages exist in the logical address space?
3. How many bits represent the physical frame number ($f$)?
4. If a process generates logical address **`0x0002A1FC`**, and logical page `42` is mapped to physical frame `150`, compute the resulting physical address in hexadecimal.

#### Step-by-Step Solution:

**1. Address Bit Allocation**:
- $\text{Page Size} = 4\text{ KB} = 4{,}096\text{ bytes} = 2^{12}\text{ bytes} \implies \mathbf{n = 12\text{ bits for Offset (d)}}$.
- Total logical address bits $m = 32$.
- Page Number bits:
  $$p = m - n = 32 - 12 = \mathbf{20\text{ bits for Page Number (p)}}$$

**2. Total Pages in Logical Address Space**:
$$\text{Total Pages} = 2^{20} = \mathbf{1{,}048{,}576\text{ pages (1 Mega-pages)}}$$

**3. Physical Frame Bits**:
- Physical Memory $= 512\text{ MB} = 512 \times 2^{20} = 2^9 \times 2^{20} = 2^{29}\text{ bytes}$.
- Total Frames $= \frac{2^{29}}{2^{12}} = 2^{17} = 131{,}072\text{ frames}$.
- Bits required for Frame Number $f = \mathbf{17\text{ bits}}$.

**4. Physical Address Translation for `0x0002A1FC`**:
- Convert logical address to binary/hex components:
  - Low-order 12 bits (last 3 hex digits) represent the offset:
    $$d = \text{0x1FC}$$
  - High-order 20 bits represent the page number:
    $$p = \text{0x0002A} = 2 \times 16^1 + 10 \times 16^0 = 32 + 10 = \mathbf{42\text{ (decimal)}}$$
- Given that Page `42` maps to Frame `150` ($150_{10} = \text{0x096}$):
- Construct Physical Address:
  $$\text{Physical Address} = (f \ll 12) \mid d = (\text{0x096} \ll 12) \mid \text{0x1FC} = \mathbf{\text{0x000961FC}}$$

---

### 15.4 Internal Fragmentation in Paging Systems

**Problem Statement (From Lecture Slide 77)**:
Consider a system with a page size of **$2{,}048\text{ bytes}$**.
A process of size **$72{,}766\text{ bytes}$** is loaded into memory:
1. How many pages must be allocated to this process?
2. How many bytes of internal fragmentation occur in the final page?
3. Express the internal fragmentation as a percentage of the final page's total capacity.
4. Calculate the worst-case and average-case internal fragmentation for this page size.

#### Step-by-Step Solution:

**1. Number of Allocated Pages**:
$$\text{Number of Pages} = \left\lceil \frac{72{,}766\text{ bytes}}{2{,}048\text{ bytes/page}} \right\rceil = \lceil 35.5303 \rceil = \mathbf{36\text{ pages}}$$

**2. Internal Fragmentation**:
- Memory allocated in 35 full pages:
  $$35 \times 2{,}048 = 71{,}680\text{ bytes}$$
- Bytes occupied in the 36th (final) page:
  $$\text{Bytes in Final Page} = 72{,}766 - 71{,}680 = 1{,}086\text{ bytes}$$
- Internal Fragmentation in final page:
  $$\text{Fragmentation} = 2{,}048 - 1{,}086 = \mathbf{962\text{ bytes}}$$

**3. Percentage Wasted in Final Frame**:
$$\text{Percentage} = \frac{962}{2{,}048} \times 100\% = \mathbf{46.97\%}$$

**4. Theoretical Bounds for 2,048-Byte Pages**:
- $\text{Worst-Case Fragmentation} = 1\text{ frame} - 1\text{ byte} = 2{,}048 - 1 = \mathbf{2{,}047\text{ bytes}}$.
- $\text{Average-Case Fragmentation} = \frac{1}{2} \times 2{,}048 = \mathbf{1{,}024\text{ bytes}}$.

---

### 15.5 Effective Access Time (EAT) with TLB (Single-Level & Two-Level Paging)

**Problem Statement (From Lecture Slide 88)**:
A computer system has a main memory access time of **$ma = 10\text{ nanoseconds}$**.
The hardware Translation Lookaside Buffer (TLB) has a search latency of **$\epsilon = 0\text{ ns}$** (pipelined).
1. If the TLB Hit ratio is **$\alpha = 80\%$**, compute the Effective Access Time ($EAT$).
2. If the TLB Hit ratio improves to **$\alpha = 98\%$**, compute the updated $EAT$.
3. Now suppose the architecture uses **Two-Level Paging**, where a TLB miss requires traversing two levels of page tables in memory before reading data. Compute the $EAT$ for both $\alpha = 80\%$ and $\alpha = 98\%$.

#### Step-by-Step Solution:

#### 1. Single-Level Paging ($EAT = \alpha(ma) + (1 - \alpha)(2ma)$):
- **For $\alpha = 80\% = 0.8$**:
  $$EAT = 0.8 \times 10 + (1 - 0.8) \times 20 = 8.0 + 0.2 \times 20 = 8.0 + 4.0 = \mathbf{12.0\text{ nanoseconds}}$$
  $$\text{Performance Slowdown} = \frac{12 - 10}{10} \times 100\% = \mathbf{20\%}$$
- **For $\alpha = 98\% = 0.98$**:
  $$EAT = 0.98 \times 10 + (1 - 0.98) \times 20 = 9.8 + 0.02 \times 20 = 9.8 + 0.4 = \mathbf{10.2\text{ nanoseconds}}$$
  $$\text{Performance Slowdown} = \frac{10.2 - 10}{10} \times 100\% = \mathbf{2\%}$$

---

#### 2. Two-Level Paging ($k = 2$):
- On a TLB Hit: 1 DRAM access for data $= 10\text{ ns}$.
- On a TLB Miss: 1 access for Page Directory $+ 1$ access for Page Table $+ 1$ access for Data $= 3 \times ma = 3 \times 10 = \mathbf{30\text{ ns}}$.
- General Formula: $EAT = \alpha \times 10 + (1 - \alpha) \times 30$.

- **For $\alpha = 80\%$**:
  $$EAT = 0.8 \times 10 + 0.2 \times 30 = 8.0 + 6.0 = \mathbf{14.0\text{ nanoseconds}}$$
  $$\text{Performance Slowdown} = \frac{14 - 10}{10} \times 100\% = \mathbf{40\%}$$
- **For $\alpha = 98\%$**:
  $$EAT = 0.98 \times 10 + 0.02 \times 30 = 9.8 + 0.6 = \mathbf{10.4\text{ nanoseconds}}$$
  $$\text{Performance Slowdown} = \frac{10.4 - 10}{10} \times 100\% = \mathbf{4\%}$$

---

### 15.6 Demand Paging EAT & Maximum Acceptable Page Fault Rate Calculation

**Problem Statement (From Lecture Slide 129)**:
A virtual memory demand-paged system has the following operational specifications:
- Memory Access Time: $ma = 200\text{ nanoseconds}$.
- Average Page-Fault Service Time: $T_{\text{pfs}} = 8\text{ milliseconds} = 8{,}000{,}000\text{ nanoseconds}$.
1. Derive the general linear equation for Effective Access Time as a function of the page-fault rate $p$.
2. Compute the EAT if one access out of every $1{,}000$ references causes a page fault ($p = 0.001$). Determine the slowdown factor relative to pure RAM access.
3. If management requires that total performance degradation not exceed **$10\%$** ($EAT \le 220\text{ ns}$), calculate the maximum permissible page-fault rate $p$.

#### Step-by-Step Solution:

**1. General Equation Derivation**:
$$EAT = (1 - p) \times ma + p \times T_{\text{pfs}}$$
$$EAT = (1 - p) \times 200 + p \times 8{,}000{,}000$$
$$EAT = 200 - 200p + 8{,}000{,}000p$$
$$\mathbf{EAT = 200 + 7{,}999{,}800 \times p\text{ nanoseconds}}$$

**2. Evaluation at $p = 0.001$**:
$$EAT = 200 + 7{,}999{,}800 \times 0.001 = 200 + 7{,}999.8 = \mathbf{8{,}199.8\text{ nanoseconds} \approx 8.2\text{ }\mu\text{s}}$$
$$\text{Slowdown Factor} = \frac{8{,}199.8\text{ ns}}{200\text{ ns}} \approx \mathbf{41\times\text{ slower!}}$$

**3. Maximum Permissible Page Fault Rate for $< 10\%$ Degradation**:
$$\text{Target EAT} \le 200 + (10\% \times 200) = 220\text{ nanoseconds}$$
$$200 + 7{,}999{,}800 \times p \le 220$$
$$7{,}999{,}800 \times p \le 20$$
$$p \le \frac{20}{7{,}999{,}800} = \frac{1}{399{,}990} \approx \mathbf{2.50 \times 10^{-6}}$$

**Conclusion**: To keep performance loss under 10%, no more than **1 out of every 400,000 memory references** may trigger a page fault!

---

### 15.7 Step-by-Step Page Replacement Tracking: FIFO vs. OPT vs. LRU

**Problem Statement (From Lecture Slides 150-155)**:
Consider a process allocated **3 physical frames** that are initially empty.
Trace the exact frame contents and total page faults for the reference string:
`7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1`
using:
1. **First-In, First-Out (FIFO)**
2. **Optimal Page Replacement (OPT)**
3. **Least Recently Used (LRU)**

#### Step-by-Step Solution:

#### 1. FIFO Algorithm Trace (Oldest page evicted):
| Step | Reference | Frame 1 | Frame 2 | Frame 3 | Page Fault? | Evicted Page |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **7** | **7** | — | — | **Fault** | — |
| 2 | **0** | 7 | **0** | — | **Fault** | — |
| 3 | **1** | 7 | 0 | **1** | **Fault** | — |
| 4 | **2** | **2** | 0 | 1 | **Fault** | 7 (Oldest) |
| 5 | **0** | 2 | 0 | 1 | Hit | — |
| 6 | **3** | 2 | **3** | 1 | **Fault** | 0 (Oldest) |
| 7 | **0** | 2 | 3 | **0** | **Fault** | 1 (Oldest) |
| 8 | **4** | **4** | 3 | 0 | **Fault** | 2 (Oldest) |
| 9 | **2** | 4 | **2** | 0 | **Fault** | 3 (Oldest) |
| 10 | **3** | 4 | 2 | **3** | **Fault** | 0 (Oldest) |
| 11 | **0** | **0** | 2 | 3 | **Fault** | 4 (Oldest) |
| 12 | **3** | 0 | 2 | 3 | Hit | — |
| 13 | **2** | 0 | 2 | 3 | Hit | — |
| 14 | **1** | 0 | **1** | 3 | **Fault** | 2 (Oldest) |
| 15 | **2** | 0 | 1 | **2** | **Fault** | 3 (Oldest) |
| 16 | **0** | 0 | 1 | 2 | Hit | — |
| 17 | **1** | 0 | 1 | 2 | Hit | — |
| 18 | **7** | **7** | 1 | 2 | **Fault** | 0 (Oldest) |
| 19 | **0** | 7 | **0** | 2 | **Fault** | 1 (Oldest) |
| 20 | **1** | 7 | 0 | **1** | **Fault** | 2 (Oldest) |

- **Total Page Faults for FIFO**: **15 Faults**.

---

#### 2. Optimal (OPT) Algorithm Trace (Longest future idle page evicted):
| Step | Reference | Frame 1 | Frame 2 | Frame 3 | Page Fault? | Evicted Page & Future Use Rationale |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **7** | **7** | — | — | **Fault** | — |
| 2 | **0** | 7 | **0** | — | **Fault** | — |
| 3 | **1** | 7 | 0 | **1** | **Fault** | — |
| 4 | **2** | **2** | 0 | 1 | **Fault** | Evict 7 (7 next used at step 18; 0 at 5; 1 at 14) |
| 5 | **0** | 2 | 0 | 1 | Hit | — |
| 6 | **3** | 2 | 0 | **3** | **Fault** | Evict 1 (1 next used at 14; 2 at 9; 0 at 7) |
| 7 | **0** | 2 | 0 | 3 | Hit | — |
| 8 | **4** | **4** | 0 | 3 | **Fault** | Evict 2 (2 next used at 9; 0 at 11; 3 at 10) |
| 9 | **2** | 4 | 0 | **2** | **Fault** | Evict 3 (3 next used at 10; 4 never used again!) |
| 10 | **3** | **3** | 0 | 2 | **Fault** | Evict 4 (4 is never referenced again!) |
| 11 | **0** | 3 | 0 | 2 | Hit | — |
| 12 | **3** | 3 | 0 | 2 | Hit | — |
| 13 | **2** | 3 | 0 | 2 | Hit | — |
| 14 | **1** | **1** | 0 | 2 | **Fault** | Evict 3 (3 never used again; 0 at 16; 2 at 15) |
| 15 | **2** | 1 | 0 | 2 | Hit | — |
| 16 | **0** | 1 | 0 | 2 | Hit | — |
| 17 | **1** | 1 | 0 | 2 | Hit | — |
| 18 | **7** | 1 | 0 | **7** | **Fault** | Evict 2 (2 never used again; 1 at 20; 0 at 19) |
| 19 | **0** | 1 | 0 | 7 | Hit | — |
| 20 | **1** | 1 | 0 | 7 | Hit | — |

- **Total Page Faults for OPT**: **9 Faults**.

---

#### 3. Least Recently Used (LRU) Algorithm Trace (Longest past idle page evicted):
| Step | Reference | Frame 1 | Frame 2 | Frame 3 | Page Fault? | Evicted Page & History Rationale |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **7** | **7** | — | — | **Fault** | — |
| 2 | **0** | 7 | **0** | — | **Fault** | — |
| 3 | **1** | 7 | 0 | **1** | **Fault** | — |
| 4 | **2** | **2** | 0 | 1 | **Fault** | Evict 7 (Past: 1@3, 0@2, 7@1 -> 7 is oldest) |
| 5 | **0** | 2 | 0 | 1 | Hit | — |
| 6 | **3** | 2 | 0 | **3** | **Fault** | Evict 1 (Past: 0@5, 2@4, 1@3 -> 1 is oldest) |
| 7 | **0** | 2 | 0 | 3 | Hit | — |
| 8 | **4** | **4** | 0 | 3 | **Fault** | Evict 2 (Past: 0@7, 3@6, 2@4 -> 2 is oldest) |
| 9 | **2** | 4 | 0 | **2** | **Fault** | Evict 3 (Past: 4@8, 0@7, 3@6 -> 3 is oldest) |
| 10 | **3** | 4 | **3** | 2 | **Fault** | Evict 0 (Past: 2@9, 4@8, 0@7 -> 0 is oldest) |
| 11 | **0** | **0** | 3 | 2 | **Fault** | Evict 4 (Past: 3@10, 2@9, 4@8 -> 4 is oldest) |
| 12 | **3** | 0 | 3 | 2 | Hit | — |
| 13 | **2** | 0 | 3 | 2 | Hit | — |
| 14 | **1** | **1** | 3 | 2 | **Fault** | Evict 0 (Past: 2@13, 3@12, 0@11 -> 0 is oldest) |
| 15 | **2** | 1 | 3 | 2 | Hit | — |
| 16 | **0** | 1 | **0** | 2 | **Fault** | Evict 3 (Past: 2@15, 1@14, 3@12 -> 3 is oldest) |
| 17 | **1** | 1 | 0 | 2 | Hit | — |
| 18 | **7** | 1 | 0 | **7** | **Fault** | Evict 2 (Past: 1@17, 0@16, 2@15 -> 2 is oldest) |
| 19 | **0** | 1 | 0 | 7 | Hit | — |
| 20 | **1** | 1 | 0 | 7 | Hit | — |

- **Total Page Faults for LRU**: **12 Faults**.

#### Comparative Summary:
$$\text{OPT (9 Faults)} < \text{LRU (12 Faults)} < \text{FIFO (15 Faults)}$$

---

### 15.8 Belady's Anomaly Step-by-Step Proof (3 Frames vs. 4 Frames)

**Problem Statement (From Lecture Slide 151)**:
Demonstrate mathematically why FIFO exhibits Belady's Anomaly using the reference string:
`1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5`
Prove that allocating 4 frames causes more page faults than allocating 3 frames.

#### Step-by-Step Solution:

#### 1. FIFO with 3 Frames:
| Ref | Frame 1 | Frame 2 | Frame 3 | Fault? | Queue State (Oldest to Newest) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **1** | — | — | **Fault** | [1] |
| **2** | 1 | **2** | — | **Fault** | [1, 2] |
| **3** | 1 | 2 | **3** | **Fault** | [1, 2, 3] |
| **4** | **4** | 2 | 3 | **Fault** | Evict 1 -> [2, 3, 4] |
| **1** | 4 | **1** | 3 | **Fault** | Evict 2 -> [3, 4, 1] |
| **2** | 4 | 1 | **2** | **Fault** | Evict 3 -> [4, 1, 2] |
| **5** | **5** | 1 | 2 | **Fault** | Evict 4 -> [1, 2, 5] |
| **1** | 5 | 1 | 2 | Hit | [1, 2, 5] |
| **2** | 5 | 1 | 2 | Hit | [1, 2, 5] |
| **3** | 5 | **3** | 2 | **Fault** | Evict 1 -> [2, 5, 3] |
| **4** | 5 | 3 | **4** | **Fault** | Evict 2 -> [5, 3, 4] |
| **5** | 5 | 3 | 4 | Hit | [5, 3, 4] |

$$\mathbf{\text{Total Page Faults (3 Frames)} = 9\text{ Faults}}$$

---

#### 2. FIFO with 4 Frames:
| Ref | Frame 1 | Frame 2 | Frame 3 | Frame 4 | Fault? | Queue State (Oldest to Newest) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **1** | — | — | — | **Fault** | [1] |
| **2** | 1 | **2** | — | — | **Fault** | [1, 2] |
| **3** | 1 | 2 | **3** | — | **Fault** | [1, 2, 3] |
| **4** | 1 | 2 | 3 | **4** | **Fault** | [1, 2, 3, 4] |
| **1** | 1 | 2 | 3 | 4 | Hit | [1, 2, 3, 4] |
| **2** | 1 | 2 | 3 | 4 | Hit | [1, 2, 3, 4] |
| **5** | **5** | 2 | 3 | 4 | **Fault** | Evict 1 -> [2, 3, 4, 5] |
| **1** | 5 | **1** | 3 | 4 | **Fault** | Evict 2 -> [3, 4, 5, 1] |
| **2** | 5 | 1 | **2** | 4 | **Fault** | Evict 3 -> [4, 5, 1, 2] |
| **3** | 5 | 1 | 2 | **3** | **Fault** | Evict 4 -> [5, 1, 2, 3] |
| **4** | **4** | 1 | 2 | 3 | **Fault** | Evict 5 -> [1, 2, 3, 4] |
| **5** | 4 | **5** | 2 | 3 | **Fault** | Evict 1 -> [2, 3, 4, 5] |

$$\mathbf{\text{Total Page Faults (4 Frames)} = 10\text{ Faults!}}$$

**Mathematical Conclusion**:
$$\text{Faults}(4\text{ frames}) = 10 > \text{Faults}(3\text{ frames}) = 9$$
Adding a fourth physical frame increased the page fault count by $11.1\%$, proving Belady's Anomaly.

---

### 15.9 Clock (Second-Chance) Replacement Hand Trace

**Problem Statement**:
A process is allocated **4 physical frames** managed by the **Clock Page Replacement Algorithm**.
Initially, the frames contain pages as follows:
- Frame 0: Page A (Use = 1)
- Frame 1: Page B (Use = 0)
- Frame 2: Page C (Use = 1)
- Frame 3: Page D (Use = 1)
The **Clock Hand** currently points to **Frame 0**.

Trace the clock hand movement, bit resets, and replacement actions when a page fault occurs requesting **Page E**.

#### Step-by-Step Solution:

1. **Inspect Frame 0**:
   - Current: Page A, $\text{Use Bit} = 1$.
   - Action: Page A used recently $\implies$ grant second chance!
   - Set $\text{Use Bit} \to \mathbf{0}$.
   - Advance clock hand to **Frame 1**.
2. **Inspect Frame 1**:
   - Current: Page B, $\text{Use Bit} = \mathbf{0}$.
   - Action: Use bit is 0 $\implies$ **VICTIM FOUND!**
   - Evict Page B (write to disk if dirty).
   - Load incoming **Page E** into Frame 1.
   - Set Page E's $\text{Use Bit} = \mathbf{1}$.
   - Advance clock hand to **Frame 2**.

#### Final System State:
- Frame 0: Page A ($\text{Use} = 0$)
- Frame 1: Page E ($\text{Use} = 1$)
- Frame 2: Page C ($\text{Use} = 1$)
- Frame 3: Page D ($\text{Use} = 1$)
- **Clock Hand Points to**: **Frame 2**.

---

### 15.10 Proportional Frame Allocation Computation

**Problem Statement (From Lecture Slide 180)**:
An operating system has **$m = 62$ free physical frames** available for user processes.
Two processes arrive:
- Process $P_1$ with a virtual memory size of $s_1 = 10\text{ pages}$.
- Process $P_2$ with a virtual memory size of $s_2 = 127\text{ pages}$.
Calculate the proportional frame allocation $a_1$ and $a_2$ for both processes.

#### Step-by-Step Solution:

1. **Calculate Total Virtual Memory Demand ($S$)**:
   $$S = \sum_{i=1}^n s_i = s_1 + s_2 = 10 + 127 = \mathbf{137\text{ pages}}$$
2. **Proportional Allocation Formula**:
   $$a_i = \left( \frac{s_i}{S} \right) \times m$$
3. **Compute Allocation for $P_1$**:
   $$a_1 = \left( \frac{10}{137} \right) \times 62 = \frac{620}{137} \approx \mathbf{4.5255\text{ frames}}$$
4. **Compute Allocation for $P_2$**:
   $$a_2 = \left( \frac{127}{137} \right) \times 62 = \frac{7874}{137} \approx \mathbf{57.4745\text{ frames}}$$
5. **Integer Adjustment**:
   - Rounding to integers such that $a_1 + a_2 = 62$:
   - $a_1 = \mathbf{4\text{ frames (or 5)}}$
   - $a_2 = \mathbf{57\text{ frames}}$
   - Check sum: $5 + 57 = 62$ (or $4 + 58 = 62$). Both satisfy minimum architectural frame requirements.

---

### 15.11 Working-Set Model Allocation and Thrashing Condition ($D > m$)

**Problem Statement**:
A multiprogrammed operating system has **$m = 12$ physical frames** available for user processes.
The system is currently running three processes: $P_1$, $P_2$, and $P_3$.
The OS tracks page references using a Working-Set Window of size **$\Delta = 10$ references**.
At the current clock tick $t$, the last 10 page references for each process are:
- $P_1$: `[ 2, 3, 2, 4, 3, 2, 4, 3, 2, 4 ]`
- $P_2$: `[ 1, 5, 6, 1, 7, 5, 8, 1, 6, 8 ]`
- $P_3$: `[ 9, 9, 10, 9, 11, 10, 11, 9, 10, 9 ]`

1. Determine the active Working Set and the Working-Set Size ($WSS_i$) for each process.
2. Compute the total system frame demand $D$.
3. Evaluate whether the system is in danger of thrashing.
4. If a fourth process $P_4$ arrives with working set $\{ 12, 13, 14 \}$, what action must the OS scheduler take?

#### Step-by-Step Solution:

**1. Determine Working Sets and $WSS_i$**:
- **Process $P_1$**:
  - Unique pages referenced: $\{ 2, 3, 4 \}$.
  - Working-Set Size: $WSS_1 = \mathbf{3\text{ frames}}$.
- **Process $P_2$**:
  - Unique pages referenced: $\{ 1, 5, 6, 7, 8 \}$.
  - Working-Set Size: $WSS_2 = \mathbf{5\text{ frames}}$.
- **Process $P_3$**:
  - Unique pages referenced: $\{ 9, 10, 11 \}$.
  - Working-Set Size: $WSS_3 = \mathbf{3\text{ frames}}$.

**2. Compute Total Demand ($D$)**:
$$D = \sum_{i=1}^3 WSS_i = WSS_1 + WSS_2 + WSS_3 = 3 + 5 + 3 = \mathbf{11\text{ frames}}$$

**3. Thrashing Evaluation**:
- Available physical frames $m = 12$.
- Condition check:
  $$D = 11 \le m = 12$$
- **Conclusion**: The total demand does not exceed available memory ($D \le m$). The system has **1 free frame** remaining in reserve. **No thrashing will occur.**

**4. Arrival of Process $P_4$**:
- $P_4$ requires $WSS_4 = 3\text{ frames}$.
- New Total Demand:
  $$D_{\text{new}} = 11 + 3 = \mathbf{14\text{ frames}}$$
- Condition check:
  $$D_{\text{new}} = 14 > m = 12 \implies \mathbf{D > m\text{ (THRASHING IMMINENT!)}}$$
- **OS Action**:
  The OS must **prevent admission of $P_4$** or **suspend one of the active processes** (e.g., suspend $P_1$ or $P_3$, rolling its pages out to swap). Admitting $P_4$ without suspending a process would trigger catastrophic thrashing across all four processes.



---

*End of Unit 3 Comprehensive Study Notes — PES University (UE24CS242B: Operating Systems).*

