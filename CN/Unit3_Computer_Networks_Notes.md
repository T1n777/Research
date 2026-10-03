# Unit 3: Transport Layer - TCP, Network Layer and Internet Protocol

**A Complete Exam Study Reference — PES University (UE25CS243A: Computer Networks)**  
*Core Textbooks: Computer Networking: A Top-Down Approach (8th Edition) by Kurose & Ross | Lecture Slides UE22CS252B & UE24CS252B*

---

## Table of Contents

1. [TCP Congestion Control & Modern Transport Protocols](#1-tcp-congestion-control--modern-transport-protocols)
   - 1.1 [The Principles of Congestion Control](#11-the-principles-of-congestion-control)
     - [What is Congestion? Causes and Manifestations](#what-is-congestion-causes-and-manifestations)
     - [Flow Control vs. Congestion Control](#flow-control-vs-congestion-control)
     - [Scenario 1: Two Senders, Two Receivers, Infinite Router Buffers](#scenario-1-two-senders-two-receivers-infinite-router-buffers)
     - [Scenario 2: Two Senders, Two Receivers, Finite Router Buffers and Drops](#scenario-2-two-senders-two-receivers-finite-router-buffers-and-drops)
     - [Scenario 3: Four Senders, Multihop Paths, Downstream Drops, and Capacity Waste](#scenario-3-four-senders-multihop-paths-downstream-drops-and-capacity-waste)
     - [Summary of Fundamental Costs of Congestion](#summary-of-fundamental-costs-of-congestion)
   - 1.2 [Approaches to Congestion Control](#12-approaches-to-congestion-control)
     - [End-to-End Congestion Control](#end-to-end-congestion-control)
     - [Network-Assisted Congestion Control](#network-assisted-congestion-control)
     - [Explicit Congestion Notification (ECN: RFC 3168) in IP and TCP](#explicit-congestion-notification-ecn-rfc-3168-in-ip-and-tcp)
   - 1.3 [TCP Congestion Control Mechanisms](#13-tcp-congestion-control-mechanisms)
     - [The Sender Transmission Rate Constraint](#the-sender-transmission-rate-constraint)
     - [Self-Clocking (ACK Clocking)](#self-clocking-ack-clocking)
     - [Additive Increase Multiplicative Decrease (AIMD) Philosophy](#additive-increase-multiplicative-decrease-aimd-philosophy)
     - [The AIMD Sawtooth Dynamic](#the-aimd-sawtooth-dynamic)
   - 1.4 [TCP Congestion Control Phases](#14-tcp-congestion-control-phases)
     - [Slow Start Phase (Exponential Growth)](#slow-start-phase-exponential-growth)
     - [Transition via Slow Start Threshold (ssthresh)](#transition-via-slow-start-threshold-ssthresh)
     - [Congestion Avoidance Phase (Linear Growth)](#congestion-avoidance-phase-linear-growth)
     - [Loss Detection and Dual Reaction Mechanisms](#loss-detection-and-dual-reaction-mechanisms)
   - 1.5 [TCP Tahoe vs. TCP Reno](#15-tcp-tahoe-vs-tcp-reno)
     - [TCP Tahoe (1988): Fast Retransmit without Fast Recovery](#tcp-tahoe-1988-fast-retransmit-without-fast-recovery)
     - [TCP Reno (1990): The Fast Recovery Algorithm](#tcp-reno-1990-the-fast-recovery-algorithm)
     - [Complete TCP Reno Finite State Machine (FSM)](#complete-tcp-reno-finite-state-machine-fsm)
     - [Comparative cwnd Evolution Graph & Side-by-Side Trace Table](#comparative-cwnd-evolution-graph--side-by-side-trace-table)
   - 1.6 [TCP Throughput Modeling & Macroscopic Description](#16-tcp-throughput-modeling--macroscopic-description)
     - [Mathematical Derivation of Average Throughput](#mathematical-derivation-of-average-throughput)
     - [The High BDP (Bandwidth-Delay Product) Dilemma](#the-high-bdp-bandwidth-delay-product-dilemma)
     - [Modern Congestion Control: TCP CUBIC and Google BBR](#modern-congestion-control-tcp-cubic-and-google-bbr)
   - 1.7 [TCP Fairness and Efficiency](#17-tcp-fairness-and-efficiency)
     - [Definition of Resource Allocation Fairness](#definition-of-resource-allocation-fairness)
     - [Vector State-Space Convergence Proof](#vector-state-space-convergence-proof)
     - [Real-World Fairness Disparities](#real-world-fairness-disparities)
   - 1.8 [The QUIC Protocol (RFC 9000) & HTTP/3](#18-the-quic-protocol-rfc-9000--http3)
     - [Why QUIC Runs Over UDP: Overcoming Middlebox Ossification](#why-quic-runs-over-udp-overcoming-middlebox-ossification)
     - [Stream-Level Multiplexing without Head-of-Line (HOL) Blocking](#stream-level-multiplexing-without-head-of-line-hol-blocking)
     - [0-RTT and 1-RTT Connection Establishment with Embedded TLS 1.3](#0-rtt-and-1-rtt-connection-establishment-with-embedded-tls-13)
     - [Connection Identifiers (CIDs) and Zero-Cost Connection Migration](#connection-identifiers-cids-and-zero-cost-connection-migration)
     - [Monotonically Increasing Packet Numbers](#monotonically-increasing-packet-numbers)
2. [Overview of the Network Layer](#2-overview-of-the-network-layer)
   - 2.1 [Network-Layer Services and Guiding Principles](#21-network-layer-services-and-guiding-principles)
     - [Host-to-Host vs. Process-to-Process Delivery](#host-to-host-vs-process-to-process-delivery)
     - [Network Layer in Every Host and Router](#network-layer-in-every-host-and-router)
   - 2.2 [Two Core Network-Layer Functions: Forwarding and Routing](#22-two-core-network-layer-functions-forwarding-and-routing)
     - [Forwarding (Data Plane): Local Hardware-Scale Switching](#forwarding-data-plane-local-hardware-scale-switching)
     - [Routing (Control Plane): Network-Wide Software Path Computation](#routing-control-plane-network-wide-software-path-computation)
     - [The Forwarding Table and Longest Prefix Matching](#the-forwarding-table-and-longest-prefix-matching)
   - 2.3 [Network Service Models](#23-network-service-models)
     - [Hypothetical Network Guarantees](#hypothetical-network-guarantees)
     - [The Internet Best-Effort Service Model & End-to-End Principle](#the-internet-best-effort-service-model--end-to-end-principle)
     - [Comparison: Internet vs. ATM vs. IntServ & DiffServ](#comparison-internet-vs-atm-vs-intserv--diffserv)
   - 2.4 [Control Plane Paradigms](#24-control-plane-paradigms)
     - [Traditional Per-Router Control Plane](#traditional-per-router-control-plane)
     - [Software-Defined Networking (SDN) Logically Centralized Control Plane](#software-defined-networking-sdn-logically-centralized-control-plane)
3. [What’s Inside a Router? (Router Architecture & Hardware)](#3-whats-inside-a-router-router-architecture--hardware)
   - 3.1 [High-Level Architectural Components](#31-high-level-architectural-components)
     - [Control Plane vs. Data Plane Operation](#control-plane-vs-data-plane-operation)
   - 3.2 [Input Port Functions](#32-input-port-functions)
     - [Line Termination & Physical Layer Reception](#line-termination--physical-layer-reception)
     - [Data Link Layer Decapsulation](#data-link-layer-decapsulation)
     - [Decentralized Forwarding and Shadow Forwarding Tables](#decentralized-forwarding-and-shadow-forwarding-tables)
     - [Destination-Based vs. Generalized Forwarding](#destination-based-vs-generalized-forwarding)
   - 3.3 [Longest Prefix Matching (LPM)](#33-longest-prefix-matching-lpm)
     - [Why LPM is Essential with CIDR](#why-lpm-is-essential-with-cidr)
     - [LPM Search Algorithm & Step-by-Step Prefix Evaluation](#lpm-search-algorithm--step-by-step-prefix-evaluation)
     - [Software Tries vs. Hardware TCAMs](#software-tries-vs-hardware-tcams)
   - 3.4 [Switching Fabrics](#34-switching-fabrics)
     - [1. Switching via Memory](#1-switching-via-memory)
     - [2. Switching via a Bus](#2-switching-via-a-bus)
     - [3. Switching via an Interconnection Network](#3-switching-via-an-interconnection-network)
   - 3.5 [Input Port Queuing and Head-of-Line (HOL) Blocking](#35-input-port-queuing-and-head-of-line-hol-blocking)
     - [Mechanism of Head-of-Line (HOL) Blocking](#mechanism-of-head-of-line-hol-blocking)
     - [Theoretical Throughput Derivation: The 58.6% Limit](#theoretical-throughput-derivation-the-586-limit)
     - [Eliminating HOL Blocking: Virtual Output Queuing (VOQ)](#eliminating-hol-blocking-virtual-output-queuing-voq)
   - 3.6 [Output Port Queuing and Buffer Sizing](#36-output-port-queuing-and-buffer-sizing)
     - [Where and Why Output Queuing Occurs](#where-and-why-output-queuing-occurs)
     - [Buffer Sizing: Traditional Rule vs. Stanford/Appenzeller Rule](#buffer-sizing-traditional-rule-vs-stanfordappenzeller-rule)
     - [Bufferbloat and Delay Penalties](#bufferbloat-and-delay-penalties)
     - [Active Queue Management (AQM): Drop-Tail vs. RED](#active-queue-management-aqm-drop-tail-vs-red)
   - 3.7 [Packet Scheduling Policies](#37-packet-scheduling-policies)
     - [1. First-In, First-Out (FIFO)](#1-first-in-first-out-fifo)
     - [2. Priority Queuing](#2-priority-queuing)
     - [3. Round Robin (RR)](#3-round-robin-rr)
     - [4. Weighted Fair Queuing (WFQ)](#4-weighted-fair-queuing-wfq)
4. [The Internet Protocol (IPv4)](#4-the-internet-protocol-ipv4)
   - 4.1 [IPv4 Datagram Format](#41-ipv4-datagram-format)
     - [Full 32-Bit Header Diagram](#full-32-bit-header-diagram)
     - [Comprehensive Field-by-Field Breakdown (All 13 Fields)](#comprehensive-field-by-field-breakdown-all-13-fields)
   - 4.2 [IP Fragmentation and Reassembly](#42-ip-fragmentation-and-reassembly)
     - [Maximum Transmission Unit (MTU) Constraints](#maximum-transmission-unit-mtu-constraints)
     - [Why Reassembly Occurs Exclusively at Destination Hosts](#why-reassembly-occurs-exclusively-at-destination-hosts)
     - [Fragmentation Mathematics & The 8-Byte Rule](#fragmentation-mathematics--the-8-byte-rule)
     - [Comprehensive Solved Numerical Examples](#comprehensive-solved-numerical-examples)
   - 4.3 [IPv4 Addressing, Subnets, and CIDR](#43-ipv4-addressing-subnets-and-cidr)
     - [Structure of an IPv4 Address](#structure-of-an-ipv4-address)
     - [Formal Definition of a Subnet](#formal-definition-of-a-subnet)
     - [Subnet Masks, Subnetting, and Host Allocation](#subnet-masks-subnetting-and-host-allocation)
     - [Special-Use IPv4 Addresses](#special-use-ipv4-addresses)
     - [Classful Addressing History and Its Collapse](#classful-addressing-history-and-its-collapse)
     - [Classless Inter-Domain Routing (CIDR — RFC 1519)](#classless-inter-domain-routing-cidr--rfc-1519)
   - [Interactive Exercise Problems (PES University Slides & Exams)](#interactive-exercise-problems-pes-university-slides--exams)
     - [Problem 1: Subnet Range and Boundary Calculations](#problem-1-subnet-range-and-boundary-calculations)
     - [Problem 2: Fixed-Length Subnetting of a Class B Address Block](#problem-2-fixed-length-subnetting-of-a-class-b-address-block)
     - [Problem 3: Determining Network and Broadcast from a Single Host IP](#problem-3-determining-network-and-broadcast-from-a-single-host-ip)
     - [Problem 4: Creating 500 Subnets from a Class A Block](#problem-4-creating-500-subnets-from-a-class-a-block)
     - [Problem 5: Subnet Analysis of Class B Address with Non-Trivial Mask](#problem-5-subnet-analysis-of-class-b-address-with-non-trivial-mask)
     - [Problem 6: Variable Length Subnet Masking (VLSM) Design](#problem-6-variable-length-subnet-masking-vlsm-design)
   - 4.4 [Hierarchical Addressing and Route Aggregation (Supernetting)](#44-hierarchical-addressing-and-route-aggregation-supernetting)
     - [Principles of Route Aggregation](#principles-of-route-aggregation)
     - [Supernetting Numerical & Binary Alignment](#supernetting-numerical--binary-alignment)
     - [More Specific Routing and Multihoming](#more-specific-routing-and-multihoming)
   - 4.5 [IPv6 Transition and Tunneling](#45-ipv6-transition-and-tunneling)
     - [IPv6 Architecture Overview](#ipv6-architecture-overview)
     - [Dual-Stack Transition Architecture](#dual-stack-transition-architecture)
     - [IPv6 Tunneling Over IPv4 Infrastructure (6in4 / Protocol 41)](#ipv6-tunneling-over-ipv4-infrastructure-6in4--protocol-41)
     - [Comprehensive Interactive Exercise Walkthrough](#comprehensive-interactive-exercise-walkthrough)
5. [Network Address Translation (NAT)](#5-network-address-translation-nat)
   - 5.1 [Motivation and Architecture](#51-motivation-and-architecture)
     - [The IPv4 Address Exhaustion Crisis](#the-ipv4-address-exhaustion-crisis)
     - [Private IP Address Blocks (RFC 1918)](#private-ip-address-blocks-rfc-1918)
     - [Unified NAT Edge Gateway Architecture](#unified-nat-edge-gateway-architecture)
   - 5.2 [NAT Router Operation & The NAT Translation Table](#52-nat-router-operation--the-nat-translation-table)
     - [NAPT (Network Address Port Translation / PAT)](#napt-network-address-port-translation--pat)
     - [Deep-Dive Packet Rewriting Engine](#deep-dive-packet-rewriting-engine)
     - [Comprehensive ASCII Diagram & Step-by-Step Packet Trace](#comprehensive-ascii-diagram--step-by-step-packet-trace)
   - 5.3 [NAT Traversal for Peer-to-Peer Applications](#53-nat-traversal-for-peer-to-peer-applications)
     - [The Inbound Connection Problem](#the-inbound-connection-problem)
     - [Traversal Solutions](#traversal-solutions)
   - 5.4 [Architectural Controversies and Trade-offs](#54-architectural-controversies-and-trade-offs)
6. [Software-Defined Networking (SDN)](#6-software-defined-networking-sdn)
   - 6.1 [Evolution and Motivation](#61-evolution-and-motivation)
     - [Legacy Networking: The Monolithic Era](#legacy-networking-the-monolithic-era)
     - [The Computing Paradigm Analogy: Mainframes to Personal Computers](#the-computing-paradigm-analogy-mainframes-to-personal-computers)
   - 6.2 [Key Characteristics of SDN](#62-key-characteristics-of-sdn)
   - 6.3 [SDN Architectural Layers](#63-sdn-architectural-layers)
   - 6.4 [The OpenFlow Protocol](#64-the-openflow-protocol)
     - [OpenFlow Message Taxonomy](#openflow-message-taxonomy)
     - [Flow Table Entry Anatomy](#flow-table-entry-anatomy)
     - [Practical Flow Table Configuration Examples](#practical-flow-table-configuration-examples)
     - [Control/Data Plane Interaction Scenario: Link Failure & Dynamic Rerouting](#controldata-plane-interaction-scenario-link-failure--dynamic-rerouting)
   - 6.5 [P4 (Programming Protocol-Independent Packet Processors)](#65-p4-programming-protocol-independent-packet-processors)
     - [Beyond OpenFlow: The Limits of Bottom-Up Protocol Standardization](#beyond-openflow-the-limits-of-bottom-up-protocol-standardization)
     - [The P4 Philosophy: Top-Down Software-Driven Packet Processing](#the-p4-philosophy-top-down-software-driven-packet-processing)
     - [The P4 Abstract Switch Architecture (PISA Model)](#the-p4-abstract-switch-architecture-pisa-model)
     - [Detailed Technical Comparison: OpenFlow vs. P4](#detailed-technical-comparison-openflow-vs-p4)
7. [Data Center Networking](#7-data-center-networking)
   - 7.1 [Architecture Requirements and Traffic Dynamics](#71-architecture-requirements-and-traffic-dynamics)
     - [Scale of Modern Hyperscale Data Centers](#scale-of-modern-hyperscale-data-centers)
     - [Traffic Dynamics: North-South vs. East-West Traffic](#traffic-dynamics-north-south-vs-east-west-traffic)
     - [Bisection Bandwidth and Non-Blocking Fabrics](#bisection-bandwidth-and-non-blocking-fabrics)
   - 7.2 [Traditional 3-Tier Architecture](#72-traditional-3-tier-architecture)
     - [Structural Flaws and Fatal Bottlenecks](#structural-flaws-and-fatal-bottlenecks)
   - 7.3 [Modern Clos / Fat-Tree (Spine-Leaf) Architecture](#73-modern-clos--fat-tree-spine-leaf-architecture)
     - [The 2-Tier Spine-Leaf Topology & Interconnection Invariants](#the-2-tier-spine-leaf-topology--interconnection-invariants)
     - [The $k$-Port Switch Fat-Tree Parameterization](#the-k-port-switch-fat-tree-parameterization)
     - [Equal-Cost Multi-Path (ECMP) Routing at Layer 3](#equal-cost-multi-path-ecmp-routing-at-layer-3)
     - [Comprehensive Side-by-Side Comparison: Traditional 3-Tier vs. Clos / Spine-Leaf](#comprehensive-side-by-side-comparison-traditional-3-tier-vs-clos--spine-leaf)
8. [Multicast Routing (Case Study)](#8-multicast-routing-case-study)
   - 8.1 [Principles and Addressing](#81-principles-and-addressing)
     - [Transmission Paradigms: Unicast vs. Broadcast vs. Multicast](#transmission-paradigms-unicast-vs-broadcast-vs-multicast)
     - [Class D IPv4 Multicast Addressing](#class-d-ipv4-multicast-addressing)
     - [Mapping Multicast IP to Ethernet Multicast MAC Addresses (RFC 1112)](#mapping-multicast-ip-to-ethernet-multicast-mac-addresses-rfc-1112)
   - 8.2 [Local Group Management: IGMP (Internet Group Management Protocol)](#82-local-group-management-igmp-internet-group-management-protocol)
     - [Scope and Role of IGMP](#scope-and-role-of-igmp)
     - [IGMP Protocol Mechanics](#igmp-protocol-mechanics)
   - 8.3 [Multicast Routing Tree Strategies](#83-multicast-routing-tree-strategies)
     - [Why Trees?](#why-trees)
     - [Strategy 1: Source-Based Trees (Shortest Path Trees - SPT)](#strategy-1-source-based-trees-shortest-path-trees---spt)
     - [Strategy 2: Shared Trees (Core-Based Trees / CBT)](#strategy-2-shared-trees-core-based-trees--cbt)
     - [Comprehensive Technical Comparison: Source-Based Trees vs. Shared Trees](#comprehensive-technical-comparison-source-based-trees-vs-shared-trees)
9. [Comprehensive Solved Numerical Problems (Step-by-Step with Diagrams)](#9-comprehensive-solved-numerical-problems-step-by-step-with-diagrams)
   - 9.1 [TCP Congestion Control Round-by-Round Evolution: TCP Tahoe vs. TCP Reno](#91-tcp-congestion-control-round-by-round-evolution-tcp-tahoe-vs-tcp-reno)
   - 9.2 [TCP Average Throughput & Loss Rate Calculation](#92-tcp-average-throughput--loss-rate-calculation)
   - 9.3 [IPv4 Datagram Fragmentation Across Multiple MTU Links](#93-ipv4-datagram-fragmentation-across-multiple-mtu-links)
   - 9.4 [IPv4 Subnetting & CIDR Block Analysis](#94-ipv4-subnetting--cidr-block-analysis)
   - 9.5 [Variable-Length Subnet Masking (VLSM) Enterprise Design](#95-variable-length-subnet-masking-vlsm-enterprise-design)
   - 9.6 [Longest Prefix Matching (LPM) Forwarding Table Lookup](#96-longest-prefix-matching-lpm-forwarding-table-lookup)
   - 9.7 [NAT Translation Table & Packet Rewriting Trace](#97-nat-translation-table--packet-rewriting-trace)
   - 9.8 [Router Buffer Sizing Calculations](#98-router-buffer-sizing-calculations)
   - 9.9 [Clos / Fat-Tree Data Center Architecture Calculations](#99-clos--fat-tree-data-center-architecture-calculations)

---

## 1. TCP Congestion Control & Modern Transport Protocols

### 1.1 The Principles of Congestion Control

#### What is Congestion? Causes and Manifestations
In computer networking, **network congestion** occurs when aggregate traffic demand across senders exceeds the available transmission capacity of intermediate communication links or the buffer processing capacities of intermediate packet switches (routers). Informally:

$$\text{Congestion} \iff \sum \text{Offered Traffic Rate} > \text{Available Path Capacity}$$

When senders inject packets into the network faster than intermediate routers can process and clear them onto outgoing links, two unavoidable physical manifestations occur:
1. **Excessive Queuing Delays:** Packets accumulate in router queue buffers, dramatically inflating end-to-end packet transmission latency.
2. **Buffer Overflow and Packet Dropping:** When router buffers become completely filled, incoming packets are dropped (*buffer overflow*). Dropped packets necessitate retransmissions by the sending end-systems, further compounding the load on the already saturated network.

If unchecked by an adaptive mechanism, uncoordinated sender transmissions lead to **congestion collapse**—a catastrophic degradation of network utility where the total network throughput drops almost to zero while packet delays approach infinity.

```
       UNCHECKED CONGESTION VICIOUS CYCLE
   +---------------------------------------+
   |   Senders inject packets at high rate |
   +---------------------------------------+
                      |
                      v
   +---------------------------------------+
   |   Router buffers fill; delays spike   |
   +---------------------------------------+
                      |
                      v
   +---------------------------------------+
   |  Buffers overflow; packets are dropped|
   +---------------------------------------+
                      |
                      v
   +---------------------------------------+
   |  Senders time out & retransmit copies | <----+
   +---------------------------------------+      | (amplifies
                      |                           |  congestion)
                      v                           |
   +---------------------------------------+      |
   | More duplicate packets enter pipeline |------+
   +---------------------------------------+
                      |
                      v
   +---------------------------------------+
   | Effective Goodput drops toward 0      |
   | (Total Congestion Collapse)           |
   +---------------------------------------+
```

---

#### Flow Control vs. Congestion Control
A foundational concept in transport layer engineering is distinguishing between **Flow Control** and **Congestion Control**:

| Dimension | Flow Control | Congestion Control |
| :--- | :--- | :--- |
| **Primary Objective** | Prevent a fast sender from overwhelming a single, slow **receiver**. | Prevent a collection of senders from overwhelming the **intermediate network fabric** (routers and transmission links). |
| **Scope of Mechanism** | **Point-to-point / End-to-end:** Strictly between two communicating endpoints. | **Network-wide / Multipoint:** Involves all competing senders, intermediate switches, and links across the Internet. |
| **Controlling Metric** | Receiver Advertised Window ($\text{rwnd}$), communicated in the TCP header. | Congestion Window ($\text{cwnd}$), dynamically estimated by the sender. |
| **Bottleneck Location** | The memory buffers of the destination host's operating system socket. | The queuing buffers and bandwidth capacities of intermediate routers. |
| **Feedback Mechanism** | Explicit: Receiver writes free buffer byte count directly into $\text{rwnd}$ field of ACK packets. | Implicit (in traditional TCP): Inferred by sender via packet loss, timeouts, or duplicate ACKs. |

---

#### Scenario 1: Two Senders, Two Receivers, Infinite Router Buffers
To analyze the fundamental causes and costs of congestion, we examine an idealized baseline model formulated by Kurose & Ross (Computer Networking: A Top-Down Approach, Section 3.6).

```
               SCENARIO 1: INFINITE ROUTER BUFFERS
       Host A                                    Host C (Receiver)
     [Sender 1] -----\                      /---- [Receiver 1]
     Rate: \lambda_in \                    /
                       v                  v
                      +--------------------+
                      |   Router: Shared   |
                      |   Output Buffer    |
                      |   (Capacity: R)    |
                      +--------------------+
                       ^                  ^
     Rate: \lambda_in /                    \
     [Sender 2] -----/                      \---- [Receiver 2]
       Host B                                    Host D (Receiver)
```

**Model Parameters:**
- Two sending hosts, Host A and Host B, generate independent data streams destined for Host C and Host D, respectively.
- Both connections share a single intermediate router with an outgoing link of capacity $R\text{ bits/second}$.
- Each host sends original data at an application-layer arrival rate of $\lambda_{in}\text{ bytes/sec}$.
- **Idealization 1:** The router possesses an **infinite queuing buffer**.
- **Idealization 2:** Packets are never lost; hence, no retransmissions occur ($\lambda'_{in} = \lambda_{in}$).

**Throughput Analysis:**
Since the outgoing link capacity $R$ is shared equally between the two symmetric flows, the maximum transmission rate available to each connection is:

$$\text{Capacity per flow} = \frac{R}{2}$$

1. **Light Load ($\lambda_{in} < \frac{R}{2}$):** The combined input rate $2\lambda_{in} < R$. Packets experience modest queuing. The outgoing rate per connection equals the offered load:
   $$\lambda_{out} = \lambda_{in}$$
2. **Heavy / Saturated Load ($\lambda_{in} \ge \frac{R}{2}$):** The aggregate arrival rate $2\lambda_{in} \ge R$. The outgoing link operates at 100% capacity ($R$). The throughput of each flow plateaus at:
   $$\lambda_{out} = \frac{R}{2}$$

**Delay Analysis (The First Cost of Congestion):**
Using classical queuing theory (modeling the router queue as an $\text{M/M/1}$ queuing system where arrival rate is $\lambda_{agg} = 2\lambda_{in}$ and service capacity is $\mu = R$), the average queuing delay $D_q$ is given by:

$$D_q = \frac{1}{\mu - \lambda_{agg}} = \frac{1}{R - 2\lambda_{in}}$$

As the per-sender offered rate $\lambda_{in}$ approaches the link boundary $\frac{R}{2}$:

$$\lim_{\lambda_{in} \to \frac{R}{2}} D_q = \lim_{\lambda_{in} \to \frac{R}{2}} \left( \frac{1}{R - 2\lambda_{in}} \right) = +\infty$$

```
   THROUGHPUT vs OFFERED LOAD                 AVERAGE DELAY vs OFFERED LOAD
   \lambda_out                                Delay
      ^                                         ^
  R/2 |           +--------------               |                     |
      |          /                              |                     |  Asymptote at
      |         /                               |                     |  \lambda_in = R/2
      |        /                                |                     |
      |       /                                 |                    /
      |      /                                  |                   /
      |     /                                   |                  /
    0 +----+------+----------> \lambda_in       0 +---------------+---+---------> \lambda_in
      0   R/4    R/2                            0               R/2
```

> **Fundamental Insight 1 (First Cost of Congestion):**  
> As aggregate arrival rates approach available link transmission capacity, packet queuing delays grow asymptotically toward infinity. Senders experience severe latency spikes even without packet drops.

---

#### Scenario 2: Two Senders, Two Receivers, Finite Router Buffers and Drops
In real-world networks, router buffers are finite. When the queue is full, arriving packets are dropped. Senders must retransmit dropped segments to provide reliable transport.

**Definitions:**
- $\lambda_{in}$: Original application-layer data rate injected into transport layer ($\text{bytes/sec}$).
- $\lambda'_{in}$: Offered load to the network layer, consisting of **original data plus retransmissions** ($\lambda'_{in} \ge \lambda_{in}$).
- $\lambda_{out}$: Receiver throughput of original application data, also known as **goodput** ($\text{bytes/sec}$).

```
   Host A (\lambda_in) ---> Transport Layer (\lambda'_in) ---> [Finite Buffer Router (R)] ---> Receiver (\lambda_out)
```

We evaluate three operational cases under this scenario:

##### Case 2a: Idealized Perfect Knowledge (No Losses)
Assume the sender has omniscient knowledge of the router buffer state and only transmits when buffer space is free.
- No packets are dropped; no retransmissions are required: $\lambda'_{in} = \lambda_{in}$.
- Senders smoothly pace packets. When $\lambda_{in} = \frac{R}{2}$, $\lambda_{out} = \frac{R}{2}$. All transmitted bits represent original useful data.

##### Case 2b: Idealized Retransmission Only on Real Loss
Assume the sender only retransmits when a packet is definitely dropped (e.g., via an instantaneous notification).
- Because buffers overflow at high rates, packets are dropped.
- To achieve an effective throughput of $\lambda_{out}$, the sender must transmit original packets plus necessary retransmissions: $\lambda'_{in} > \lambda_{in}$.
- When the offered load to the network is $\lambda'_{in} = \frac{R}{2}$, some fraction of that bandwidth is consumed by retransmitted packets. Consequently, the delivered original throughput $\lambda_{out}$ is strictly lower than $\frac{R}{2}$ (for example, $\lambda_{out} \approx \frac{R}{3}$ or $0.75 \frac{R}{2}$).
- Senders must perform extra transmission work to compensate for dropped packets.

##### Case 2c: Realistic Network with Premature Timeouts & Duplicate Packets
In realistic networks, senders determine loss via timers. When queue delays are large, a packet may still be in transit or waiting in a buffer when the sender's **Retransmission Timeout (RTO)** timer expires.
- The sender prematurely retransmits the packet.
- Both the original delayed packet and the duplicate retransmission eventually arrive at the receiver.
- The receiver's transport layer accepts the first copy, detects the duplicate sequence number of the second copy, and silently discards it!
- **Wasted Work:** The intermediate router expended precious CPU, memory, and link transmission bandwidth to carry a packet that was discarded upon arrival!
- If every packet is transmitted twice on average due to premature timeouts, the maximum achievable goodput is cut in half:
  $$\lambda_{out} = \frac{1}{2} \left( \frac{R}{2} \right) = \frac{R}{4}$$

```
                       THROUGHPUT (\lambda_out) vs OFFERED LOAD (\lambda'_in)
   \lambda_out
      ^
  R/2 |           +-------------------  Case 2a: Idealized (no loss, \lambda'_in = \lambda_in)
      |          /   - - - - - - - -    Case 2b: Retransmit ONLY on loss (\lambda_out < R/2)
  R/3 |         /  /
      |        / /   . . . . . . . .    Case 2c: Unneeded retransmissions / duplicates (\lambda_out -> R/4)
  R/4 |       //   .
      |      //  .
      |     // .
    0 +----+--+-----------------------> \lambda'_in
      0   R/4 R/2
```

> **Fundamental Insight 2 (Second Cost of Congestion):**  
> Senders must perform retransmissions to compensate for dropped packets resulting from buffer overflow.  
> **Fundamental Insight 3 (Third Cost of Congestion):**  
> Unneeded retransmissions caused by large queuing delays waste link capacity; routers forward duplicate packets that the destination host ultimately discards.

---

#### Scenario 3: Four Senders, Multihop Paths, Downstream Drops, and Capacity Waste
Consider a multihop network containing four routers arranged in a routing loop or pipeline, traversed by four competing flows.

```
       SCENARIO 3: MULTIHOP NETWORK TOPOLOGY & UPSTREAM WASTE
               
       Host A (Flow A)                                Host B (Flow B)
           |                                              |
           v                                              v
       +--------+      Link 1 (Capacity R)            +--------+      Link 2 (Capacity R)
       | Router |====================================>| Router |=========================> Receiver B
       |   R1   |                                     |   R2   |
       +--------+                                     +--------+
           ^                                              ^
           |                                              |
      Host D (Flow D)                                Host C (Flow C)
      (From Router R4)                              (Arriving at R2, heading to R3)
```

**Traffic Trajectory:**
- **Flow A (Host A to Receiver A):** Passes through Router $R_1$, traverses Link 1, passes through Router $R_2$, and traverses Link 2.
- **Flow B (Host B to Receiver B):** Arrives at Router $R_2$ and competes with Flow A for transmission across Link 2.

**Dynamics as Offered Load Increases:**
1. Suppose Host B ramps up its transmission rate $\lambda'_{in, B}$ to a very high level.
2. The buffer at Router $R_2$ facing Link 2 fills completely with packets from Flow B.
3. Meanwhile, Host A sends packets to Router $R_1$. Router $R_1$ has free buffer space, so it successfully buffers, switches, and transmits Flow A's packets over Link 1.
4. When Flow A's packets arrive at Router $R_2$, they encounter a saturated buffer dominated by Flow B's high arrival rate. Router $R_2$ drops Flow A's packets!
5. **The Upstream Capacity Waste Penalty:**
   - Link 1 expended its scarce transmission bandwidth to deliver Flow A's packets from $R_1$ to $R_2$.
   - Because those packets are discarded at $R_2$, **all transmission capacity and buffering used by Flow A at Router $R_1$ and Link 1 was completely wasted!**
   - That wasted bandwidth could have been used to carry traffic that terminated at $R_1$ or flowed elsewhere.

**Mathematical Limit of Multihop Collapse:**
As Host B's offered load $\lambda'_{in, B} \to \infty$, the probability that a packet from Flow A finds an open buffer slot at Router $R_2$ approaches zero:

$$P(\text{Buffer slot available for Flow A at } R_2) \approx \frac{\lambda'_{in, A}}{\lambda'_{in, A} + \lambda'_{in, B}} \xrightarrow[\lambda'_{in, B} \to \infty]{} 0$$

Consequently, Flow A's throughput drops to zero:

$$\lim_{\lambda'_{in, B} \to \infty} \lambda_{out, A} = 0$$

```
                   CONGESTION COLLAPSE IN MULTIHOP PATHS
   \lambda_out (Flow A)
      ^
  R/2 |        /\
      |       /  \
      |      /    \
      |     /      \  Severe throughput degradation
      |    /        \ due to downstream drops
      |   /          \
    0 +--+------------\--------------------------> \lambda'_in
      0               R/2
```

> **Fundamental Insight 4 (Fourth Cost of Congestion):**  
> When a packet is dropped along a multihop path, all upstream transmission capacity and buffer resources expended to carry that packet to the point of drop are completely wasted.

---

#### Summary of Fundamental Costs of Congestion
To synthesize the theoretical analysis:

1. **Infinite Latency Penalty:** Large queuing delays accumulate as link utilization approaches 100%.
2. **Retransmission Overhead:** Senders must retransmit dropped packets, consuming processing power and reducing effective throughput.
3. **Spurious Duplicate Penalty:** Premature timeouts force the transmission of unneeded duplicate packets, wasting link bandwidth on data destined for the receiver's trash bin.
4. **Upstream Resource Squandering:** Packets dropped at downstream nodes invalidate all upstream forwarding effort, starving other competing flows of link capacity.

---

### 1.2 Approaches to Congestion Control

Congestion control mechanisms are broadly classified into two categories based on whether intermediate network-layer switches provide explicit assistance to transport-layer end systems.

```
                    CONGESTION CONTROL TAXONOMY
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
[End-to-End Congestion Control]         [Network-Assisted Congestion Control]
 - No explicit router feedback           - Routers actively signal congestion
 - Inferred from loss & delay            - Direct Choke Packets (ICMP Source Quench)
 - Classic TCP (Tahoe, Reno, CUBIC)      - In-band Bit Marking (DECbit, ECN)
                                         - Rate-Based Allocation (ATM ABR)
```

#### End-to-End Congestion Control
In an **End-to-End Congestion Control** architecture:
- The network layer (IP) provides **zero explicit feedback** regarding internal congestion state.
- Intermediate routers remain stateless and passive with respect to transport-layer dynamics; they simply forward or drop datagrams.
- End hosts must infer network congestion entirely from observable transport-level phenomena:
  1. **Packet Loss Events:** Detected via the expiration of a Retransmission Timeout (RTO) or the arrival of **Triple Duplicate Acknowledgments** (3 duplicate ACKs).
  2. **Round-Trip Delay Variations:** Increases in smoothed Round-Trip Time ($\text{RTT}$) indicate queue buildup inside intermediate routers (used by TCP Vegas and FAST TCP).
- **Core Philosophy:** Adheres strictly to the Internet's **End-to-End Principle** (Saltzer, Reed, Clark 1984), ensuring that the core of the network remains simple, lightweight, and scalable, while complexity resides in end systems.

---

#### Network-Assisted Congestion Control
In a **Network-Assisted Congestion Control** architecture, intermediate routers actively monitor their internal queuing states and link utilization levels to provide direct, explicit signaling to end systems.

Common historical and modern implementations include:

1. **Direct Choke Packets:**
   - A congested router generates a special control packet (such as an ICMP (Internet Control Message Protocol) Source Quench packet) directed backward to the original sender.
   - *Limitation:* Generating choke packets injects additional traffic into an already congested network.
2. **In-Band Hop-by-Hop / End-to-End Bit Marking (DECbit):**
   - Developed by Digital Equipment Corporation for the Digital Network Architecture (DNA).
   - Routers calculate average queue length. If average queue length exceeds 1.0, the router sets a **Congestion Indicator (CI)** bit in the header of passing packets.
   - The destination host copies this bit into the returning acknowledgment. When the sender observes that more than 50% of returning ACKs have the bit set, it multiplicatively decreases its congestion window.
3. **ATM (Asynchronous Transfer Mode) ABR (Available Bit Rate):**
   - ATM networks divide data into 53-byte fixed cells and use dedicated **Resource Management (RM) cells** interleaved among data cells (typically 1 RM cell every 32 data cells).
   - **EFCI (Explicit Forward Congestion Indication):** Congested intermediate switches set the EFCI bit in data cell headers. The destination observes the marked cells and sets the **CI (Congestion Indication)** bit in returning backward RM cells.
   - **Explicit Rate (ER) Signaling:** A congested switch can directly inspect the ER field inside passing RM cells and overwrite it with a lower numerical transmission rate (e.g., $15.5\text{ Mbps}$). The sender immediately throttles its transmission rate to $\le \text{ER}$.

---

#### Explicit Congestion Notification (ECN: RFC 3168) in IP and TCP
The Internet Engineering Task Force (IETF) standardized **Explicit Congestion Notification (ECN)** in RFC 3168, integrating network-assisted congestion signaling into the IP and TCP architectures without violating layered protocol boundaries.

##### 1. IP Header ECN Bits
ECN utilizes the two least significant bits of the IPv4 **Type of Service (ToS)** field (or IPv6 **Traffic Class** field), located within the **Differentiated Services (DiffServ)** byte:

```
  IPv4 Type of Service (ToS) / Differentiated Services (DiffServ) Field (8 bits)
 +---------------------------------------------------+-------+-------+
 |     Differentiated Services Code Point (DSCP)     |  ECT  |  CE   |
 |                   (6 bits)                        | bit 6 | bit 7 |
 +---------------------------------------------------+-------+-------+
```

| Bit Pattern | Codepoint Name | Semantic Meaning |
| :---: | :--- | :--- |
| `00` | **Non-ECT** | Packet is sent by an application or transport protocol that is **not** ECN-capable. |
| `01` | **ECT(1)** | Endpoint is ECN-Capable Transport. Used by protocols like L4S (Low Latency, Low Loss, Scalable Throughput). |
| `10` | **ECT(0)** | Endpoint is ECN-Capable Transport. Standard indicator that sender and receiver support ECN. |
| `11` | **CE** | **Congestion Experienced.** Set by an intermediate router experiencing active queue buildup (e.g., via Random Early Detection - RED) instead of dropping the packet! |

##### 2. TCP Header ECN Flags
To relay congestion feedback back to the sender and confirm throttling, RFC 3168 introduced two flags into bits 8 and 9 of the TCP header flags byte:
- **ECE (ECN-Echo):**
  - In connection setup (SYN packet): Indicates that the host is ECN-capable.
  - In normal data transmission: Set by the **receiver** in its ACK packet to inform the sender that it received an IP datagram with the `CE` (`11`) codepoint set.
- **CWR (Congestion Window Reduced):**
  - Set by the **sender** in the header of the next transmitted data segment to acknowledge receipt of the `ECE` notification and confirm that it has halved its congestion window.

```
                  ECN SIGNALING INTERACTION FLOW
  Sender (Host A)               Router (R)                 Receiver (Host B)
     |                              |                              |
     | [IP: ECT(0)] Data Packet     |                              |
     |----------------------------->|                              |
     |                              | (Buffer queue exceeds threshold;
     |                              |  Router marks IP header: CE=11)
     |                              |                              |
     |                              | [IP: CE=11] Data Packet      |
     |                              |----------------------------->|
     |                              |                              | (Detects CE=11;
     |                              |                              |  Sets TCP flag ECE=1)
     |                              |                              |
     |             [TCP: ECE=1] Returning ACK Segment              |
     |<------------------------------------------------------------|
     |                                                             |
   (Reduces cwnd by half;                                          |
    Sets TCP flag CWR=1)                                           |
     |                                                             |
     | [TCP: CWR=1] Subsequent Data Segment                        |
     |------------------------------------------------------------>|
     |                                                             | (Clears ECE flag
     |                                                             |  in future ACKs)
```

**Benefits of ECN:**
- Avoids packet drops, eliminating costly retransmission delays.
- Decouples congestion signaling from packet loss, which is particularly beneficial for latency-sensitive applications (interactive gaming, financial trading, live audio/video).

---

### 1.3 TCP Congestion Control Mechanisms

#### The Sender Transmission Rate Constraint
A TCP sender regulates its transmission rate using an internal state variable called the **Congestion Window ($\text{cwnd}$)**.

The sender maintains the invariant that the volume of unacknowledged data in transit (the "in-flight" pipeline) never exceeds the minimum of the receiver's advertised buffer space and the network's perceived capacity:

$$\text{LastByteSent} - \text{LastByteAcked} \le \min(\text{cwnd}, \text{rwnd})$$

Where:
- $\text{LastByteSent}$: Byte sequence number of the most recent byte passed to the network layer.
- $\text{LastByteAcked}$: Byte sequence number of the highest cumulatively acknowledged byte received from the destination.
- $\text{rwnd}$: Receive Window advertised in the receiver's TCP header (enforcing **flow control**).
- $\text{cwnd}$: Congestion Window computed locally by the sender (enforcing **congestion control**).

Assuming the receiver has ample buffer space ($\text{rwnd} \gg \text{cwnd}$), the sender's transmission rate is governed by $\text{cwnd}$:

$$\text{Transmission Rate} \approx \frac{\text{cwnd}}{\text{RTT}} \quad (\text{bytes/sec})$$

```
                   SENDER SEQUENCE NUMBER SPACE
       |<- - - - - - - - - - - - - - cwnd - - - - - - - - - - - - ->|
  -----+-----------------------------+------------------------------+-------
       | Sent and Cumulatively       | Sent, Unacknowledged         | Usable Window
       | Acknowledged by Receiver    | ("In-Flight" Pipeline Bytes) | to Send Now
  -----+-----------------------------+------------------------------+-------
                                     ^                              ^
                               LastByteAcked                  LastByteSent
```

---

#### Self-Clocking (ACK Clocking)
TCP is a **self-clocking** (or **ACK-clocked**) protocol. A TCP sender uses the arrival of returning acknowledgments to pace the injection of new data into the network pipeline.

```
                          ACK CLOCKING PRINCIPLE
   Sender                                                    Bottleneck Link (Capacity R)
     |    P1        P2        P3        P4                     (Packets spaced out by
     |====|=========>|========>|========>|==================>  transmission delay L/R)
     |                                                             |
     |                                                             v
     |    A1        A2        A3        A4                     Receiver
     |<---|---------<|---------<|---------<|==================|
     |
  (Sender releases P5 upon receiving A1,
   automatically pacing transmission to link service rate R!)
```

1. When a connection begins, the sender transmits a burst of packets.
2. As these packets traverse the bottleneck link of capacity $R$, they are serialized and separated in time by a transmission delay:
   $$\Delta t = \frac{\text{MSS}}{R}$$
3. The receiver generates acknowledgments spaced by this same interval $\Delta t$.
4. When the acknowledgments arrive back at the sender, they trigger the release of new segments at the bottleneck link's processing rate.

ACK clocking prevents senders from injecting large, destructive packet bursts once the pipeline is full.

---

#### Additive Increase Multiplicative Decrease (AIMD) Philosophy
To discover available bandwidth without inducing congestion collapse, TCP uses the **Additive Increase Multiplicative Decrease (AIMD)** control algorithm.

The core heuristic consists of two complementary behaviors:
1. **Additive Increase (Linear Bandwidth Probing):**
   - As long as no packet loss occurs, the network is presumed to have spare capacity.
   - The sender gently increases its transmission window by **$1\text{ Maximum Segment Size (MSS)}$ every Round-Trip Time ($\text{RTT}$)**:
     $$\text{cwnd} \leftarrow \text{cwnd} + 1\text{ MSS} \quad (\text{per RTT})$$
   - Probing is deliberately linear and conservative to avoid overshooting available capacity.
2. **Multiplicative Decrease (Exponential Backoff):**
   - Upon detecting packet loss, the sender assumes intermediate queues have overflowed.
   - The sender immediately halves its congestion window:
     $$\text{cwnd} \leftarrow \frac{\text{cwnd}}{2} \quad (\text{upon loss detection})$$
   - The reduction is multiplicative because queue lengths grow non-linearly near saturation. A proportional reduction quickly drains congested buffers.

---

#### The AIMD Sawtooth Dynamic
Under steady-state conditions, the interplay between additive probing and multiplicative backoff produces a characteristic **sawtooth waveform**:

```
                       THE TCP AIMD SAWTOOTH PATTERN
   cwnd (MSS)
      ^
  W   |                 /|                /|                /|
      |                / | (Loss: cut    / |               / |
      |               /  |  in half)    /  |              /  |
  W/2 |   Additive   /   +-------------/   +-------------/   +---
      |   Increase  /                 /                 /
      |  (+1 MSS/  /                 /                 /
      |    RTT)   /                 /                 /
    0 +----------+-----------------+-----------------+-----------> Time
```

1. The window increases linearly by $+1\text{ MSS}$ each RTT, probing for bandwidth.
2. When the aggregate transmission rate exceeds the bottleneck link capacity $R$, router buffers overflow, causing packet loss.
3. TCP detects the loss and halves $\text{cwnd}$ to $\frac{W}{2}$.
4. The cycle repeats, oscillating around the optimal operating capacity of the path.

---

### 1.4 TCP Congestion Control Phases

TCP congestion control operates in three distinct phases: **Slow Start**, **Congestion Avoidance**, and **Fast Recovery**.

```
                   TCP CONGESTION CONTROL ROADMAP
                                  |
            +---------------------+---------------------+
            |                                           |
    [Slow Start (SS)]                         [Congestion Avoidance (CA)]
     - Initial exponential ramp-up             - Linear probing (+1 MSS/RTT)
     - cwnd < ssthresh                         - cwnd >= ssthresh
     - cwnd doubles every RTT                  - Probes carefully for capacity
            |                                           |
            +---------------------+---------------------+
                                  |
                           Packet Loss Event
                                  |
            +---------------------+---------------------+
            |                                           |
    [Retransmission Timeout]                    [Triple Duplicate ACKs]
     - Severe congestion                         - Mild, localized loss
     - ssthresh = cwnd / 2                       - ssthresh = cwnd / 2
     - cwnd = 1 MSS                              - Reno: Enter Fast Recovery
     - Drop to Slow Start                        - Tahoe: Drop to Slow Start (cwnd=1)
```

---

#### Slow Start Phase (Exponential Growth)
When a TCP connection is established, the sender does not know the path's transmission capacity. Probing additively (+1 MSS per RTT) from an initial window of 1 MSS on a high-speed Gigabit link would take thousands of round trips to reach capacity.

To solve this, TCP uses **Slow Start**:
- **Initial Value:** $\text{cwnd} = 1\text{ MSS}$ (modern RFC 6928 permits an Initial Window of $10\text{ MSS}$).
- **Growth Rule:** For every cumulatively received acknowledgment, $\text{cwnd}$ is incremented by $1\text{ MSS}$:
  $$\text{cwnd} \leftarrow \text{cwnd} + 1\text{ MSS} \quad (\text{per received ACK})$$
- **Net Effect per RTT:** If $\text{cwnd} = W\text{ MSS}$, the sender transmits $W$ segments. When the receiver acknowledges these $W$ segments, the sender processes $W$ distinct ACKs. Each ACK adds $1\text{ MSS}$, resulting in:
  $$\text{New cwnd} = W + W = 2W\text{ MSS}$$
  **The congestion window doubles every Round-Trip Time ($\text{RTT}$)** ($1 \to 2 \to 4 \to 8 \to 16\dots\text{ MSS}$).
- Thus, despite its name, Slow Start increases the sending rate **exponentially**.

```
                   SLOW START EXPONENTIAL RAMP-UP
   Sender                                                  Receiver
     |                    Segment 1 (cwnd = 1 MSS)            |
     |------------------------------------------------------->|
     |<-------------------------------------------------------|
     |                    ACK 1 (cwnd becomes 2 MSS)          |
     |                                                        |
     |                    Segment 2                           |
     |------------------------------------------------------->|
     |                    Segment 3                           |
     |------------------------------------------------------->|
     |<-------------------------------------------------------| ACK 2
     |<-------------------------------------------------------| ACK 3 (cwnd becomes 4 MSS)
     |                                                        |
     |                    Segments 4, 5, 6, 7                 |
     |=======================================================>| (Doubles every RTT!)
```

---

#### Transition via Slow Start Threshold (ssthresh)
Exponential growth cannot continue indefinitely without causing severe buffer overflow. TCP uses a state variable called **$\text{ssthresh}$ (Slow Start Threshold)** to manage the transition from exponential to linear growth.

- At connection initialization, $\text{ssthresh}$ is set to an arbitrary high value (historically $64\text{ KB}$).
- **State Selection Rule:**
  - If $\text{cwnd} < \text{ssthresh}$: The sender is in **Slow Start** (exponential increase).
  - If $\text{cwnd} \ge \text{ssthresh}$: The sender transitions to **Congestion Avoidance** (linear increase).

---

#### Congestion Avoidance Phase (Linear Growth)
In Congestion Avoidance, TCP probes for spare bandwidth conservatively.

- **Objective:** Increase $\text{cwnd}$ by exactly $+1\text{ MSS}$ over the course of one full $\text{RTT}$, regardless of how many individual ACKs are received.
- **Implementation:** Upon the arrival of each non-duplicate ACK, the window is updated as:
  $$\text{cwnd} \leftarrow \text{cwnd} + \text{MSS} \times \left( \frac{\text{MSS}}{\text{cwnd}} \right)$$

**Mathematical Proof of $+1\text{ MSS}$ per RTT:**
Suppose the current window is $\text{cwnd} = W\text{ bytes}$. The sender transmits $N = \frac{W}{\text{MSS}}$ segments. Assuming no drops, $N$ acknowledgments return during the subsequent RTT. The total window growth $\Delta \text{cwnd}$ across the RTT is:

$$\Delta \text{cwnd} = \sum_{i=1}^{N} \left[ \text{MSS} \times \left( \frac{\text{MSS}}{\text{cwnd}} \right) \right] = N \times \frac{\text{MSS}^2}{W} = \left( \frac{W}{\text{MSS}} \right) \times \frac{\text{MSS}^2}{W} = 1\text{ MSS}$$

---

#### Loss Detection and Dual Reaction Mechanisms
TCP detects packet loss via two mechanisms, each reflecting a different degree of network distress:

```
  +-----------------------------------------------------------------------------------+
  |                             LOSS DETECTION MECHANISMS                             |
  +-----------------------------------------+-----------------------------------------+
  |    1. Retransmission Timeout (RTO)      |    2. Triple Duplicate ACKs (3 Dup ACKs)|
  +-----------------------------------------+-----------------------------------------+
  | - Retransmission timer expires.         | - 3 duplicate ACKs received (4 total).  |
  | - Indicates severe congestion:          | - Indicates mild / localized loss:      |
  |   Packets or ACKs are completely stuck. |   Subsequent packets reached receiver!  |
  | - Reaction:                             | - Reaction:                             |
  |   ssthresh = cwnd / 2                   |   ssthresh = cwnd / 2                   |
  |   cwnd = 1 MSS                          |   Tahoe: cwnd = 1 MSS (Slow Start)      |
  |   Enter Slow Start                      |   Reno: Enter Fast Recovery             |
  +-----------------------------------------+-----------------------------------------+
```

##### 1. Loss via Retransmission Timeout (RTO)
- **Significance:** The timer expired without receiving any acknowledgment for in-flight data. This suggests intermediate router buffers are completely blocked, preventing forward progress.
- **Reaction (Universal across Tahoe & Reno):**
  1. Record half the current window as the new threshold:
     $$\text{ssthresh} \leftarrow \max\left( \frac{\text{cwnd}}{2}, 2\text{ MSS} \right)$$
  2. Reset the congestion window to $1\text{ MSS}$:
     $$\text{cwnd} \leftarrow 1\text{ MSS}$$
  3. Reset state and enter **Slow Start**.

##### 2. Loss via Triple Duplicate ACKs (3 Dup ACKs)
- **Why do duplicate ACKs occur?** TCP's receiver generates an immediate acknowledgment whenever an out-of-order segment arrives, repeating the last cumulatively acknowledged byte sequence number.
- **Significance:** If a sender receives **3 duplicate ACKs** (4 identical ACKs total), it knows that at least three subsequent segments reached the receiver safely and were buffered out of order.
- The pipeline is not completely blocked; packets are still getting through. Halving throughput to $1\text{ MSS}$ would unnecessarily underutilize the path.

```
                  TRIPLE DUPLICATE ACK GENERATION
   Sender                                                  Receiver
     |                     Segment 1 (Seq 100)                |
     |------------------------------------------------------->| (Received: ACKs Seq 200)
     |                     Segment 2 (Seq 200) [DROPPED!]     |
     |                 x   (Lost at congested router)         |
     |                     Segment 3 (Seq 300)                |
     |------------------------------------------------------->| (Out of order! ACKs Seq 200 - Dup 1)
     |                     Segment 4 (Seq 400)                |
     |------------------------------------------------------->| (Out of order! ACKs Seq 200 - Dup 2)
     |                     Segment 5 (Seq 500)                |
     |------------------------------------------------------->| (Out of order! ACKs Seq 200 - Dup 3)
     |<-------------------------------------------------------|
     |           TRIPLE DUPLICATE ACK ARRIVES!                |
     |  -> Sender infers Segment 2 was lost;                  |
     |     Segments 3, 4, 5 successfully arrived!             |
```

---

### 1.5 TCP Tahoe vs. TCP Reno

#### TCP Tahoe (1988): Fast Retransmit without Fast Recovery
Introduced by Van Jacobson in 1988 (4.3BSD Tahoe release):
- **Fast Retransmit:** When the sender receives 3 duplicate ACKs, it immediately retransmits the missing segment without waiting for the retransmission timer to expire.
- **Congestion Control Handling:** Tahoe makes **no distinction** between a Timeout and 3 Duplicate ACKs. In both cases, it treats the loss as severe congestion:
  $$\text{ssthresh} \leftarrow \max\left( \frac{\text{cwnd}}{2}, 2\text{ MSS} \right)$$
  $$\text{cwnd} \leftarrow 1\text{ MSS}$$
- The connection drops back to **Slow Start** in all loss scenarios, causing sharp throughput dips.

---

#### TCP Reno (1990): The Fast Recovery Algorithm
Introduced in the 1990 4.3BSD Reno release, TCP Reno added **Fast Recovery** to maintain high throughput following localized packet drops detected via 3 duplicate ACKs.

##### Algorithm Steps:
1. **Trigger:** Receive the 3rd duplicate ACK for an outstanding segment.
2. **Set Threshold:**
   $$\text{ssthresh} \leftarrow \max\left( \frac{\text{cwnd}}{2}, 2\text{ MSS} \right)$$
3. **Fast Retransmit:** Immediately retransmit the missing segment.
4. **Artificial Window Inflation:** Set the congestion window to:
   $$\text{cwnd} \leftarrow \text{ssthresh} + 3\text{ MSS}$$
   *(Rationale: The 3 duplicate ACKs indicate that 3 segments have left the network pipeline and are buffered at the receiver, freeing up space in the network).*
5. **Fast Recovery Maintenance:**
   - For every **additional duplicate ACK** received while in Fast Recovery:
     $$\text{cwnd} \leftarrow \text{cwnd} + 1\text{ MSS}$$
   - If allowed by the inflated $\text{cwnd}$, transmit a new data segment.
6. **Exit Fast Recovery (Arrival of Defrosting "New ACK"):**
   - When a "new" cumulative ACK arrives (acknowledging the retransmitted segment and missing data):
     - **Deflate the window:**
       $$\text{cwnd} \leftarrow \text{ssthresh}$$
     - Transition directly into **Congestion Avoidance** (linear growth), skipping Slow Start entirely!
7. **Timeout Handling:** If a timer expires while in Fast Recovery or Congestion Avoidance, Reno falls back to Slow Start:
   $$\text{ssthresh} \leftarrow \frac{\text{cwnd}}{2}, \quad \text{cwnd} \leftarrow 1\text{ MSS}$$

---

#### Complete TCP Reno Finite State Machine (FSM)
The following state transition diagram illustrates the complete control logic of TCP Reno across its three states:

```
                  COMPLETE TCP RENO FINITE STATE MACHINE (FSM)

                                 +-------------------------+
                                 |       INITIALIZE        |
                                 | cwnd = 1 MSS            |
                                 | ssthresh = High (64 KB) |
                                 | dupACKcount = 0         |
                                 +-------------------------+
                                              |
                                              v
          +-----------------------------------------------------------------------+
          |                                                                       |
          |                      +------------------------+                       |
          |                      |       SLOW START       |                       |
          |                      |    (Exponential)       |                       |
          |                      +------------------------+                       |
          |                        |        ^          ^                          |
          |     New ACK received:  |        |          |                          |
          |     cwnd = cwnd + 1 MSS|        |          |                          |
          |     dupACKcount = 0    |        |          |                          |
          |     (Loop in SS)       |        |          |                          |
          |                        v        |          |                          |
          |       +-------------------------+          |                          |
          |       | cwnd >= ssthresh                   |                          |
          |       +-------------------------+          |                          |
          |                    |                       |                          |
          |                    v                       |                          |
          |       +------------------------+           |                          |
          |       |  CONGESTION AVOIDANCE  |           |                          |
          |       |       (Linear)         |           |                          |
          |       +------------------------+           |                          |
          |         |                    ^             |                          |
          |         | New ACK received:  |             |                          |
          |         | cwnd = cwnd +      |             |                          |
          |         |   MSS*(MSS/cwnd)   |             |                          |
          |         | dupACKcount = 0    |             |                          |
          |         | (Loop in CA)       |             |                          |
          |         |                    |             | Timeout Event:           |
          |         |                    |             | ssthresh = cwnd / 2      |
          |         | 3 Dup ACKs:        | New ACK:    | cwnd = 1 MSS             |
          |         | ssthresh = cwnd / 2| cwnd =      | dupACKcount = 0          |
          |         | cwnd = ssthresh + 3|   ssthresh  | Retransmit missing       |
          |         | Retransmit packet  | (Deflate!)  | Enter Slow Start         |
          |         v                    |             | (From ANY State)         |
          |       +------------------------+           |                          |
          |       |     FAST RECOVERY      |-----------+                          |
          |       +------------------------+                                      |
          |         |                    ^                                        |
          |         | Additional Dup ACK:|                                        |
          |         | cwnd = cwnd + 1 MSS|                                        |
          |         | Send packet if ok  |                                        |
          |         +--------------------+                                        |
          +-----------------------------------------------------------------------+
```

---

#### Comparative cwnd Evolution Graph & Side-by-Side Trace Table
To see the operational differences between Tahoe and Reno, consider an identical network scenario:
- **Parameters:** Initial $\text{ssthresh} = 16\text{ MSS}$. Initial $\text{cwnd} = 1\text{ MSS}$.
- **Event 1:** At **Transmission Round 8**, $\text{cwnd} = 12\text{ MSS}$. A packet drop occurs, detected by **3 Duplicate ACKs**.
- **Event 2:** At **Transmission Round 16**, a severe **Retransmission Timeout** occurs.

```
                      CWND EVOLUTION: TCP TAHOE vs. TCP RENO
   cwnd (MSS)
      ^
   16 |-------+ ssthresh_0 = 16
      |      / \
   12 |     /   \   <-- Loss 1: Triple Duplicate ACKs at Round 8
      |    /     \
    8 |   /       \             +-- Reno (enters Fast Recovery, deflates to 6, grows linearly)
    6 |  /         \           / \
    4 | /           \         /   \
    2 |/             \       /     \
    1 +---------------+-----+-------+----------------------------------------> Transmission
      0   2   4   6   8  10    12   14   16                                       Round
                       \
                        +-- Tahoe (drops to cwnd = 1, restarts Slow Start)
```

##### Side-by-Side Trace Table:

| Transmission Round | Event / Description | TCP Tahoe cwnd | TCP Tahoe ssthresh | TCP Reno cwnd | TCP Reno ssthresh |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Connection setup | 1 | 16 | 1 | 16 |
| **1** | Slow Start (doubles) | 2 | 16 | 2 | 16 |
| **2** | Slow Start (doubles) | 4 | 16 | 4 | 16 |
| **3** | Slow Start (doubles) | 8 | 16 | 8 | 16 |
| **4** | Slow Start (doubles) | 16 | 16 | 16 | 16 |
| **5** | $\text{cwnd} \ge \text{ssthresh} \to$ Congestion Avoidance | 17 | 16 | 17 | 16 |
| **6** | Linear growth (+1) | 18 | 16 | 18 | 16 |
| **7** | Linear growth (+1) | 19 | 16 | 19 | 16 |
| **8** | **LOSS EVENT: 3 Duplicate ACKs ($\text{cwnd}=19$)** | **1** *(Reset!)* | **9** *(19/2)* | **12** *(9 + 3 inflated)* | **9** *(19/2)* |
| **9** | Tahoe in SS; Reno receives New ACK $\to$ exits FR | 2 | 9 | 9 *(Deflated to ssthresh)* | 9 |
| **10** | Tahoe in SS; Reno in CA (+1/RTT) | 4 | 9 | 10 | 9 |
| **11** | Tahoe in SS; Reno in CA (+1/RTT) | 8 | 9 | 11 | 9 |
| **12** | Tahoe hits ssthresh $\to$ CA; Reno in CA | 9 | 9 | 12 | 9 |
| **13** | Both in Congestion Avoidance | 10 | 9 | 13 | 9 |
| **14** | Both in Congestion Avoidance | 11 | 9 | 14 | 9 |
| **15** | Both in Congestion Avoidance | 12 | 9 | 15 | 9 |
| **16** | **LOSS EVENT: Timeout!** | **1** | **6** *(12/2)* | **1** | **7** *(15/2)* |
| **17** | Both enter Slow Start | 2 | 6 | 2 | 7 |

---

### 1.6 TCP Throughput Modeling & Macroscopic Description

#### Mathematical Derivation of Average Throughput
To model the macroscopic performance of a TCP connection operating in steady-state Congestion Avoidance, we analyze its AIMD sawtooth pattern.

```
                      STEADY-STATE SAWTOOTH CYCLE
   cwnd (MSS)
      ^
    W |                     /|
      |                    / |
      |                   /  |  Loss occurs at cwnd = W
      |                  /   |
  W/2 |                 /    +------------------  Window drops to W/2
      |                /
      |               /
    0 +--------------+-------+------------------> Time
                     |<-- T ->|
```

**Derivation:**
1. Let $W$ denote the peak congestion window size (in segments) immediately prior to packet loss.
2. Under AIMD, the window drops to $\frac{W}{2}$ segments upon loss detection and increases linearly at a rate of $1\text{ segment per RTT}$.
3. The number of round trips required to grow from $\frac{W}{2}$ back to $W$ is:
   $$\text{Duration of cycle } T = \left( W - \frac{W}{2} \right) = \frac{W}{2}\text{ RTTs}$$
4. The window size during each round trip forms an arithmetic progression. The total number of packets transmitted in one complete sawtooth cycle corresponds to the area under the curve:
   $$\alpha = \sum_{k=0}^{W/2} \left( \frac{W}{2} + k \right) = \left( \frac{W}{2} \right) \cdot \left( \frac{W}{2} \right) + \frac{(W/2)(W/2)}{2} = \frac{3}{8} W^2\text{ packets}$$
5. By definition, exactly one packet loss terminates this cycle. Thus, the packet loss probability $L$ is the reciprocal of the total packets sent:
   $$L = \frac{1}{\alpha} = \frac{1}{\frac{3}{8} W^2} = \frac{8}{3 W^2}$$
6. Solving for the peak window size $W$ in terms of the loss probability $L$:
   $$W^2 = \frac{8}{3L} \implies W = \sqrt{\frac{8}{3L}}$$
7. The average congestion window $\bar{W}$ over the cycle is the midpoint between $\frac{W}{2}$ and $W$:
   $$\bar{W} = \frac{W + \frac{W}{2}}{2} = \frac{3}{4} W = \frac{3}{4} \sqrt{\frac{8}{3L}} = \sqrt{\frac{9}{16} \cdot \frac{8}{3L}} = \sqrt{\frac{3}{2L}} \approx \frac{1.22}{\sqrt{L}}\text{ segments}$$
8. Converting segments to bytes ($\text{MSS}$ bytes per segment) and dividing by the round-trip duration $\text{RTT}$ yields the **Macroscopic TCP Throughput Formula** (often referred to as the Mathis et al. formula):

$$\text{Average Throughput} \approx \frac{1.22 \times \text{MSS}}{\text{RTT} \times \sqrt{L}} \quad (\text{bytes/sec})$$

---

#### The High BDP (Bandwidth-Delay Product) Dilemma
The **Bandwidth-Delay Product (BDP)** defines the volume of data that can be in flight across a network path:

$$\text{BDP} = \text{Bottleneck Link Bandwidth} \times \text{Path RTT}$$

Consider a modern high-speed long-distance transcontinental optical path:
- Link Capacity $C = 10\text{ Gbps} = 10^{10}\text{ bits/sec} = 1.25 \times 10^9\text{ bytes/sec}$.
- Round-Trip Time $\text{RTT} = 100\text{ ms} = 0.1\text{ sec}$.
- Standard Segment Size $\text{MSS} = 1500\text{ bytes} = 12,000\text{ bits}$.

1. The target BDP of this link is:
   $$\text{BDP} = 1.25 \times 10^9 \times 0.1 = 125\text{ Megabytes} \approx 83,333\text{ segments in-flight}$$
2. Using the macroscopic throughput formula, we calculate the required packet loss rate $L$ to sustain this $10\text{ Gbps}$ rate:
   $$\text{Throughput} = \frac{1.22 \times \text{MSS}}{\text{RTT} \times \sqrt{L}} \implies 1.25 \times 10^9 = \frac{1.22 \times 1500}{0.1 \times \sqrt{L}}$$
   $$\sqrt{L} = \frac{1.22 \times 1500}{0.1 \times 1.25 \times 10^9} = \frac{1830}{1.25 \times 10^8} \approx 1.464 \times 10^{-5}$$
   $$L \approx (1.464 \times 10^{-5})^2 \approx 2.14 \times 10^{-10}$$

> **The Problem:**  
> To utilize a 10 Gbps link over a 100 ms RTT using standard TCP Reno, the connection cannot tolerate more than **one packet loss event for every 5 billion transmitted packets**!  
> Furthermore, if a single drop occurs, Reno halves its window by 41,666 segments. At $+1\text{ MSS per RTT}$, recovering that window would take **41,666 round trips**, or **~70 minutes of continuous linear recovery** for a single loss event.

---

#### Modern Congestion Control: TCP CUBIC and Google BBR
To address these limitations on high-BDP links, modern operating systems have moved beyond standard Reno:

##### 1. TCP CUBIC (RFC 8312)
- Default in Linux, Android, and macOS.
- Replaces the linear growth function with a **cubic function** of the elapsed time $t$ since the last loss event:
  $$W_{\text{cubic}}(t) = C(t - K)^3 + W_{\max}$$
  Where $W_{\max}$ is the window size at the last loss, $C$ is a scaling factor, and $K = \sqrt[3]{\frac{W_{\max} \beta}{C}}$ is the time required to grow the window back to $W_{\max}$.
- **Key Advantage:** Window growth is **independent of RTT**, preventing short-RTT flows from unfairly dominating long-RTT flows. The cubic curve accelerates window growth when far from $W_{\max}$, flattens out near $W_{\max}$ for stability, and accelerates again to probe for new capacity.

##### 2. Google BBR (Bottleneck Bandwidth and RTT)
- Model-based congestion control that decouples rate control from packet loss.
- Concurrently measures two physical path parameters:
  1. Maximum path bottleneck bandwidth ($\text{BtlBw}$).
  2. Minimum round-trip propagation time ($\text{RTprop}$).
- Paces packet transmissions to maintain in-flight data at exactly $\text{BtlBw} \times \text{RTprop}$, keeping intermediate router queues empty and preventing **bufferbloat**.

---

### 1.7 TCP Fairness and Efficiency

#### Definition of Resource Allocation Fairness
Let $K$ independent TCP connections traverse a single shared bottleneck link of capacity $R$. A congestion control mechanism achieves **ideal fairness** if each active connection receives an equal share of the transmission bandwidth:

$$\text{Fair Throughput per Connection} = \frac{R}{K}$$

---

#### Vector State-Space Convergence Proof
The convergence of AIMD to fair and efficient bandwidth sharing was proven by Chiu and Jain (1989) using a two-dimensional vector state-space model.

Consider two connections, Flow 1 and Flow 2, sharing a bottleneck link of capacity $R$.
- Let $x_1$ and $x_2$ represent the transmission rates of Flow 1 and Flow 2, respectively.
- **Efficiency Line:** $x_1 + x_2 = R$. Points below this line underutilize the link; points above cause queue growth and packet drops.
- **Fairness Line:** $x_1 = x_2$. Points along this $45^\circ$ ray represent perfectly equal bandwidth sharing.
- **Optimal Operating Point:** The intersection of both lines: $(x_1, x_2) = \left( \frac{R}{2}, \frac{R}{2} \right)$.

```
                      CHIU & JAIN VECTOR FAIRNESS DIAGRAM
   Rate of Connection 2 (x_2)
      ^
      | Fairness Line: x_1 = x_2 (45 degrees)
    R |  \
      |   \
      |    \               Efficiency Line: x_1 + x_2 = R
      |     \            /
      |      \          /
      |       \  (Overloaded: Loss occurs!)
      |        \      /
  R/2 |         \ *  /  <-- Multiplicative Decrease points toward origin (0,0)
      |          \  /
      |           \/ Optimal Point (R/2, R/2)
      |           /\
      |          /  \
      |         / *  \  <-- Additive Increase moves parallel to 45 degree line (+1, +1)
      |        /      \
      |       /        \
    0 +------+----------+---------------------------------------------------->
      0     R/2         R                                     Rate of Connection 1 (x_1)
```

##### Step-by-Step Trajectory Analysis:
1. **Additive Increase Phase:**
   - Both connections increase their windows by $+1\text{ MSS}$ each RTT.
   - Vector displacement:
     $$\vec{\Delta}_{\text{AI}} = (+1, +1)$$
   - This moves the system along a path oriented at exactly $45^\circ$, parallel to the fairness line.
2. **Multiplicative Decrease Phase:**
   - When the combined rate crosses the efficiency line ($x_1 + x_2 > R$), a loss event occurs.
   - Both connections halve their windows:
     $$(x_1, x_2) \to \left( \frac{x_1}{2}, \frac{x_2}{2} \right)$$
   - This displacement points along a ray directed straight toward the origin $(0, 0)$.
3. **Iterative Convergence:**
   - Notice the geometric property: Shrinking toward the origin changes the ratio $\frac{x_1}{x_2}$ toward $1.0$ relative to the total distance from the fairness line.
   - Each successive cycle of additive increase ($45^\circ$ shift) followed by multiplicative decrease (radial shrinkage) walks the operating point closer to the optimal fair point $\left( \frac{R}{2}, \frac{R}{2} \right)$.

##### Why Alternative Algorithms Fail to Converge:

```
  +-----------------------------------------------------------------------------------+
  |                        CONVERGENCE PROPERTIES COMPARISON                          |
  +-----------------+------------------------------------+----------------------------+
  | Scheme          | Geometric Behavior                 | Convergence Outcome        |
  +-----------------+------------------------------------+----------------------------+
  | **AIMD**        | Additive: 45° parallel shift       | **CONVERGES** to fair and  |
  |                 | Multiplicative: Radial pull to (0,0)| optimal operating point.  |
  +-----------------+------------------------------------+----------------------------+
  | **AIAD**        | Both increase (+1,+1) and decrease | **FAILS:** Trapped in a    |
  |                 | (-1,-1) move along 45° slope.      | limit cycle; never shifts  |
  |                 |                                    | toward fairness line.      |
  +-----------------+------------------------------------+----------------------------+
  | **MIMD**        | Both increase (*a,*a) and decrease | **FAILS:** Multiplicative  |
  |                 | (/b,/b) trace rays through origin. | steps preserve ratio x1/x2;|
  |                 |                                    | initial unfairness persists|
  +-----------------+------------------------------------+----------------------------+
```

---

#### Real-World Fairness Disparities

##### 1. Round-Trip Time (RTT) Bias
The TCP throughput formula shows that transmission rate is inversely proportional to RTT:

$$\text{Throughput} \propto \frac{1}{\text{RTT}}$$

If Connection A has an RTT of $20\text{ ms}$ and Connection B has an RTT of $100\text{ ms}$, Connection A completes 5 additive increase cycles for every 1 cycle completed by Connection B. Connection A ramps up its window 5 times faster, taking a disproportionate share of bottleneck link bandwidth.

##### 2. Unresponsive UDP Traffic
UDP has no built-in congestion control. Applications using raw UDP (such as DNS, legacy real-time media, and VoIP) transmit at fixed rates regardless of network drop rates. Under heavy congestion:
- TCP detects drops and throttles its rate.
- UDP continues transmitting at full speed, filling the freed buffer space.
- The unresponsive UDP flow crowds out the responsive TCP flow.

##### 3. Multiple Parallel TCP Connections
Web applications often open multiple concurrent TCP connections to the same destination server to bypass browser per-domain limits.
- Suppose a bottleneck link of capacity $R$ is shared by Host 1 running a single TCP connection and Host 2 running 9 parallel TCP connections.
- The link arbitrates among 10 competing TCP connections, allocating $\frac{R}{10}$ to each.
- Host 1 receives $\frac{R}{10} = 10\%$ of the link capacity, while Host 2 receives $9 \times \frac{R}{10} = 90\%$ of the capacity!

---

### 1.8 The QUIC Protocol (RFC 9000) & HTTP/3

Standardized by the IETF in RFC 9000, **QUIC (Quick UDP Internet Connections)** is a modern transport-layer protocol designed to replace the legacy TCP + TLS protocol stack for the next generation of the web (**HTTP/3**).

```
          LEGACY WEB STACK                           MODERN HTTP/3 STACK
   +------------------------------+            +------------------------------+
   |    HTTP/2 (Application)      |            |    HTTP/3 (Application)      |
   +------------------------------+            +------------------------------+
   |   TLS 1.2 / 1.3 (Security)   |            |             QUIC             |
   +------------------------------+            | (Streams, Loss Recovery,     |
   |      TCP (Transport)         |            |  Integrated TLS 1.3, Cong.)  |
   +------------------------------+            +------------------------------+
   |       IP (Network)           |            |       UDP (Transport)        |
   +------------------------------+            +------------------------------+
                                               |       IP (Network)           |
                                               +------------------------------+
```

#### Why QUIC Runs Over UDP: Overcoming Middlebox Ossification
- **Protocol Ossification:** For decades, attempts to deploy improved transport protocols (such as SCTP - Stream Control Transmission Protocol) across the public Internet failed. Network middleboxes (firewalls, NAT devices, load balancers) routinely discard packets with unknown IP protocol numbers.
- Modifying OS kernels across billions of client devices and enterprise routers takes decades.
- **The QUIC Solution:** QUIC encapsulates its transport frames inside **UDP datagrams**, which pass through virtually all consumer firewalls and NAT gateways without modification.
- QUIC is implemented in **user space** as part of the browser or application runtime, enabling rapid deployment and independent version updates.

---

#### Stream-Level Multiplexing without Head-of-Line (HOL) Blocking
A major flaw of HTTP/2 over TCP is transport-level **Head-of-Line (HOL) blocking**:

```
        HTTP/2 OVER TCP: SINGLE PACKET LOSS BLOCKS ALL STREAMS
   TCP Stream: [ Stream A: P1 ] [ Stream B: P1 (LOST!) ] [ Stream C: P1 ] [ Stream A: P2 ]
                                         x
   Receiver TCP Buffer:
   [ Stream A: P1 ] ----> Held in OS TCP buffer! Cannot deliver Stream A or C to
                          the browser because Stream B byte sequence is missing!
```

- TCP enforces strict in-order delivery of an unstructured byte stream. If a packet containing bytes for Stream B is dropped, the receiving OS kernel pauses delivery of all subsequent bytes to the application, stalling Streams A, C, and D until the missing segment is retransmitted.
- **QUIC's Solution:** QUIC provides native transport-layer multiplexing. Each stream within a QUIC connection has its own stream identifier and flow-control limits.
- If a packet carrying data for Stream B is lost, **only Stream B is delayed**. Packets for Stream A and Stream C are immediately delivered to the application layer.

```
          QUIC MULTIPLEXING: INDEPENDENT DATA STREAMS
   Stream 1: [ Packet 1 ] [ Packet 2 (LOST!) ] [ Packet 3 ] ---> Only Stream 1 waits!
                                    x
   Stream 2: [ Packet 1 ] --------------------> Delivered immediately!
   Stream 3: [ Packet 1 ] --------------------> Delivered immediately!
```

---

#### 0-RTT and 1-RTT Connection Establishment with Embedded TLS 1.3
Traditional secure web connections over TCP suffer from high connection setup latency:

```
            TCP + TLS 1.3 LATENCY                    QUIC 0-RTT / 1-RTT LATENCY
   Client                     Server        Client                     Server
     |                          |             |                          |
     |------- TCP SYN --------->|             |--- Initial QUIC Packet ->|
     |<------ TCP SYN-ACK ------|             |    (Client Hello +       |
     |------- TCP ACK --------->|             |     Key Exchange)        |
     |   (1 Full RTT for TCP)   |             |<-- Initial Response -----|
     |                          |             |    (Server Hello +       |
     |--- TLS Client Hello ---->|             |     Encrypted Data)      |
     |<-- TLS Server Hello -----|             |  (Connection & Security  |
     |   (2nd Full RTT for TLS) |             |   established in 1 RTT!) |
     |                          |             |                          |
     |==== HTTP GET Request ===>|             |   SUBSEQUENT 0-RTT:      |
     |<=== HTTP Response Data ==|             |=== HTTP Data + Keys ====>|
     |  (Data at 3rd RTT!)      |             |  (Data on first packet!) |
```

- **1-RTT Handshake:** For a first-time connection, QUIC combines transport parameters and the cryptographic handshake into a single exchange, reducing setup latency to $1\text{ RTT}$.
- **0-RTT Handshake:** If the client has previously connected to the server, it uses cached cryptographic tokens to send encrypted application data in its **very first packet** ($0\text{ RTT}$ connection establishment).

---

#### Connection Identifiers (CIDs) and Zero-Cost Connection Migration
- A standard TCP connection is bound to a 4-tuple:
  $$(\text{Source IP}, \text{Source Port}, \text{Destination IP}, \text{Destination Port})$$
  When a user transitions from Wi-Fi to a cellular 5G network, the device's IP address changes. The 4-tuple breaks, terminating the connection and requiring a full re-handshake.
- **QUIC's Solution:** QUIC connections are identified by a 64-bit or 128-bit **Connection Identifier (CID)** independent of the underlying network addressing.
- When an IP address change occurs, the client sends a packet containing the existing CID from its new IP address. The server authenticates the packet cryptographic proof and migrates the connection state with zero disruption to active downloads or media streams.

---

#### Monotonically Increasing Packet Numbers
In TCP, a retransmitted segment carries the same sequence number as the original segment. If an ACK arrives, the sender cannot tell whether it acknowledges the original transmission or the retransmission (the **Retransmission Ambiguity Problem**), complicating RTT calculations.

- **QUIC's Solution:** Every QUIC packet carries a **new, strictly increasing 64-bit packet number**, even when carrying retransmitted frame payloads.
- An acknowledgment indicates the exact packet number it references, eliminating ambiguity and producing precise RTT estimates.

---

## 2. Overview of the Network Layer

### 2.1 Network-Layer Services and Guiding Principles

#### Host-to-Host vs. Process-to-Process Delivery
The fundamental role of the network layer is to provide logical communication between **end-system hosts**, complementing the transport layer's role above it:

```
  +-----------------------------------------------------------------------------------+
  |                          LAYER SERVICE COMPARISON                                 |
  +---------------------------------------+-------------------------------------------+
  | Transport Layer                       | Network Layer                             |
  +---------------------------------------+-------------------------------------------+
  | - **Process-to-Process** delivery.    | - **Host-to-Host** delivery.              |
  | - Runs exclusively on end hosts.      | - Runs on **every host and every router**.|
  | - Demultiplexes packets to sockets    | - Routes packets across intermediate      |
  |   using **port numbers**.             |   networks using **IP addresses**.        |
  | - Relies on underlying network layer. | - Interconnects heterogeneous link-layer  |
  |                                       |   technologies (Ethernet, Wi-Fi, Fiber).  |
  +---------------------------------------+-------------------------------------------+
```

##### The Household Mail Delivery Analogy:
- **Network Layer:** The national postal service. It moves letters from a source mailbox at House A to a destination mailbox at House B across postal distribution hubs. The postal service only cares about street addresses (IP addresses).
- **Transport Layer:** Siblings Ann and Bill inside the houses. Ann collects letters written by family members and drops them in the mailbox. Bill takes incoming letters from the mailbox and hands each letter to the intended person based on the name on the envelope (Port Numbers).

---

#### Network Layer in Every Host and Router
Unlike the application and transport layers, which execute purely in software on end hosts, the network layer is implemented in **every computing host and every intermediate forwarding router** throughout the Internet.

```
       END-TO-END NETWORK LAYER PACKET FLOW
   Source Host                                                         Destination Host
  +-------------+                                                       +-------------+
  | Application |                                                       | Application |
  +-------------+                                                       +-------------+
  |  Transport  |                                                       |  Transport  |
  +-------------+          Router 1                  Router 2           +-------------+
  |   Network   |       +-------------+           +-------------+       |   Network   |
  +-------------+       |   Network   |           |   Network   |       +-------------+
  |    Link     |======>|    Link     |==========>|    Link     |======>|    Link     |
  +-------------+       +-------------+           +-------------+       +-------------+
  |  Physical   |       |  Physical   |           |  Physical   |       |  Physical   |
  +-------------+       +-------------+           +-------------+       +-------------+
```

1. **Sender Host:** Encapsulates transport-layer segments into **IP datagrams**, populates header fields (source/destination IP addresses, TTL, protocol type), and passes them to the link layer.
2. **Intermediate Routers:** Inspect the IP header of each incoming datagram, consult a forwarding table, and switch the packet to the appropriate outgoing link.
3. **Destination Host:** Strips the IP header, performs validation checks, and extracts and delivers the transport-layer segment to the appropriate protocol handler (TCP, UDP, ICMP).

---

### 2.2 Two Core Network-Layer Functions: Forwarding and Routing

The network layer divides its responsibilities into two distinct operations: **Forwarding** and **Routing**.

```
              FORWARDING vs. ROUTING: PLANES OF OPERATION
   +----------------------------------------------------------------------+
   |                       CONTROL PLANE (Software)                       |
   |                                                                      |
   |                     [ Routing Algorithm Component ]                  |
   |                     (Dijkstra, Bellman-Ford, BGP)                    |
   |                                   |                                  |
   |           Network-wide path       | Writes to forwarding             |
   |           determination           v table                            |
   |                           +------------------+                       |
   |                           | Forwarding Table |                       |
   |                           +------------------+                       |
   +-----------------------------------|----------------------------------+
                                       |
   +-----------------------------------|----------------------------------+
   | DATA PLANE (Hardware)             v                                  |
   |                                                                      |
   |  Incoming Packet ---> [ Header Match ] ---> [ Output Port Switch ]    |
   |                          (TCAM)             (Switching Fabric)       |
   +----------------------------------------------------------------------+
```

#### Forwarding (Data Plane): Local Hardware-Scale Switching
- **Definition:** The local, per-router action of transferring an arriving packet from an input port interface to the appropriate output port interface.
- **Operational Scope:** Strictly local to a single router.
- **Execution Timescale:** Operates on **nanosecond timescales**, implemented directly in dedicated hardware (ASICs - Application-Specific Integrated Circuits, and TCAM - Ternary Content Addressable Memory).

---

#### Routing (Control Plane): Network-Wide Software Path Computation
- **Definition:** The network-wide coordination process that determines the end-to-end path packets follow from source to destination across multiple intermediate hops.
- **Operational Scope:** Distributed network-wide across autonomous systems.
- **Execution Timescale:** Operates on **millisecond to second timescales**, typically executed in software by the router's routing processor CPU or an external SDN controller.

##### Analogy: Driving a Vehicle
- **Routing:** Using a navigation system to plan an overall travel route from Bangalore to Mysore (Control Plane).
- **Forwarding:** Navigating through a single highway interchange or roundabout along that route (Data Plane).

---

#### The Forwarding Table and Longest Prefix Matching
Every router maintains a **Forwarding Table** that maps destination IP address prefixes to output link interfaces:

```
  Destination Address Range / CIDR Prefix          Output Interface
  -----------------------------------------------  ----------------
  11001000 00010111 00010000 00000000 /20 (Link 0) Interface 0
  11001000 00010111 00011000 00000000 /21 (Link 1) Interface 1
  11001000 00010111 00011000 10000000 /24 (Link 2) Interface 2
  Otherwise (Default Gateway)                     Interface 3
```

- When a packet arrives, the router examines its 32-bit destination IP address and matches it against table entries using **Longest Prefix Matching**:
- If a destination matches multiple prefixes of different lengths, the packet is forwarded to the interface corresponding to the **most specific entry** (the one with the largest subnet prefix length).

---

### 2.3 Network Service Models

#### Hypothetical Network Guarantees
A network layer could theoretically provide several types of service guarantees:

```
  +-----------------------------------------------------------------------------------+
  |                           CANDIDATE SERVICE GUARANTEES                            |
  +---------------------------------------+-------------------------------------------+
  | Guarantee Type                        | Meaning                                   |
  +---------------------------------------+-------------------------------------------+
  | **Guaranteed Delivery**               | Packets will eventually reach the         |
  |                                       | destination without loss.                 |
  +---------------------------------------+-------------------------------------------+
  | **Bounded Delay**                     | Packets will arrive within a guaranteed   |
  |                                       | latency bound (e.g., < 40 ms).            |
  +---------------------------------------+-------------------------------------------+
  | **In-Order Delivery**                 | Packets arrive in the exact order they    |
  |                                       | were sent.                                |
  +---------------------------------------+-------------------------------------------+
  | **Guaranteed Minimum Bandwidth**      | Senders are guaranteed a sustained        |
  |                                       | transmission rate along the path.         |
  +---------------------------------------+-------------------------------------------+
  | **Bounded Jitter**                    | The inter-arrival spacing between packets |
  |                                       | remains within strict bounds.             |
  +---------------------------------------+-------------------------------------------+
```

---

#### The Internet Best-Effort Service Model & End-to-End Principle
The Internet's network layer provides a **Best-Effort Service Model**:
- **Zero Guarantees:** Packets are not guaranteed to arrive, may experience arbitrary delays, can be delivered out of order, and can be duplicated or corrupted.

```
       WHY BEST-EFFORT SERVICE WON THE INTERNET
  +--------------------------------------------------+
  | Simple Core: Stateless, high-speed packet routers|
  +--------------------------------------------------+
                           |
                           v
  +--------------------------------------------------+
  | Link Heterogeneity: Works over fiber, copper,    |
  | satellite, wireless, carrier pigeons             |
  +--------------------------------------------------+
                           |
                           v
  +--------------------------------------------------+
  | End-to-End Principle: Implement reliability and  |
  | intelligence at endpoints (TCP, apps)            |
  +--------------------------------------------------+
```

##### Rationale for Best-Effort Architecture:
1. **Low Switching Overhead:** Routers do not maintain per-connection state or perform admission control, allowing them to forward packets at line rate.
2. **Link Heterogeneity:** The network can run over diverse link-layer technologies with widely differing reliability characteristics.
3. **The End-to-End Principle:** Functions that can be implemented completely and correctly at the endpoints (such as reliable transmission) should not be duplicated in the network core.

---

#### Comparison: Internet vs. ATM vs. IntServ & DiffServ
In contrast to the Internet's best-effort design, telecommunications architectures like **Asynchronous Transfer Mode (ATM)** were designed around **Virtual Circuit (VC)** networks with explicit Quality of Service (QoS) guarantees:
- **CBR (Constant Bit Rate):** Emulates a dedicated physical leased line with constant bandwidth, low latency, and zero loss.
- **VBR (Variable Bit Rate):** Guarantees peak and sustained rates for bursty traffic (such as compressed video).
- **ABR (Available Bit Rate):** Best-effort delivery with a guaranteed minimum rate and explicit congestion feedback.
- **UBR (Unspecified Bit Rate):** Pure best-effort without guarantees.

##### Quality of Service (QoS) Architectural Comparison Matrix:

| Network Architecture | Service Model | Bandwidth Guarantee | Loss Guarantee | Order Guarantee | Timing / Delay Bound | Congestion Feedback |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Internet (IP)** | **Best-Effort** | None | No | No | No | None (Implicit loss; optional ECN) |
| **ATM** | **CBR** (Constant Bit Rate) | Constant rate | **Yes** | **Yes** | **Yes** | Not needed (Dedicated rate) |
| **ATM** | **VBR** (Variable Bit Rate) | Guaranteed rate | **Yes** | **Yes** | **Yes** | Not needed (Resource allocated) |
| **ATM** | **ABR** (Available Bit Rate) | Guaranteed minimum | No | **Yes** | No | **Explicit** (RM cells, ER, EFCI) |
| **ATM** | **UBR** (Unspecified Bit Rate) | None | No | **Yes** | No | None |
| **Internet QoS** | **IntServ (RFC 1633)** | **Guaranteed** | **Yes** | **Yes** | **Yes** | RSVP reservation signaling |
| **Internet QoS** | **DiffServ (RFC 2475)** | Probabilistic / Class | Class-based | No | Class-based | DSCP packet marking / RED |

---

### 2.4 Control Plane Paradigms

Two architectural paradigms govern how the network control plane computes and installs forwarding tables:

```
            CONTROL PLANE ARCHITECTURAL PARADIGMS

   TRADITIONAL PER-ROUTER CONTROL PLANE      SOFTWARE-DEFINED NETWORKING (SDN)
   
        Routing            Routing                    +--------------------+
       Algorithm          Algorithm                   |   SDN Controller   |
       Component          Component                   | (Central Software) |
      +---------+        +---------+                  +--------------------+
      | Control |<======>| Control |                     |       |      |
      |  Plane  | Routing|  Plane  |           OpenFlow  |       |      |
      +---------+ Protocol+--------+           API       v       v      v
      |  Data   |        |  Data   |                  +----+  +----+  +----+
      |  Plane  |        |  Plane  |                  | SW |  | SW |  | SW |
      +---------+        +---------+                  +----+  +----+  +----+
        Router 1           Router 2                   Physical Forwarding Switches
```

#### Traditional Per-Router Control Plane
- **Characteristics:**
  - An independent routing algorithm component (such as OSPF - Open Shortest Path First, or BGP - Border Gateway Protocol) runs locally on every router's operating system.
  - Routers communicate with one another by exchanging distributed routing protocol messages.
  - Each router independently computes its shortest paths and populates its local forwarding table.
- **Strengths:** Decentralized and robust; no single point of failure.
- **Weaknesses:** Difficult to manage, vendor-dependent configuration syntax, and complex distributed convergence under topological churn.

---

#### Software-Defined Networking (SDN) Logically Centralized Control Plane
- **Characteristics:**
  - Forwarding devices (switches) are decoupled from the routing control logic, acting as simple packet-forwarding hardware (the **Data Plane**).
  - A logically centralized **SDN Controller** executes on external computing servers, maintains a global view of network topology, and runs routing applications.
  - The controller computes forwarding rules and installs them directly into switch flow tables using standardized southbound APIs (such as **OpenFlow**).
- **Strengths:** Programmable network control, rapid innovation, centralized policy enforcement, and optimized traffic engineering.
- **Resilience:** Logically centralized, but implemented physically across a distributed, fault-tolerant cluster of servers to eliminate single points of failure.

---

## 3. What’s Inside a Router? (Router Architecture & Hardware)

A packet switch operating at the network layer is formally designated as a **router**. Unlike an end-system host whose primary job is running user applications, a router is a specialized computing engine designed to forward incoming packets from its input links onto appropriate outgoing links at line speed.

---

### 3.1 High-Level Architectural Components

A commercial, carrier-grade router is architecturally partitioned into four fundamental hardware and software modules:

1. **Input Ports**: Perform physical-layer termination, data-link layer decapsulation, and decentralized forwarding lookups.
2. **Switching Fabric**: An ultra-high-speed hardware backplane that physically transfers datagrams from input ports to output ports.
3. **Output Ports**: Receive datagrams from the switching fabric, buffer them in queuing memory, and transmit them onto the outgoing link using data-link and physical layer protocols.
4. **Routing Processor**: The Central Processing Unit (CPU) executing control-plane tasks, including routing protocol computation, network management agent execution, and forwarding table generation.

```
                      +---------------------------------------+
                      |           ROUTING PROCESSOR           |
                      |   (Control Plane: Software / OS)      |
                      |   Routing Protocols (OSPF, BGP)       |
                      |   Management, Table Computation       |
                      +-------------------+-------------------+
                                          |
                        Management / Table Distribution Bus
                                          |
      +-----------------------------------+-----------------------------------+
      |                                   |                                   |
      v                                   v                                   v
+------------+                     +-------------+                     +------------+
| INPUT PORT |                     |  SWITCHING  |                     | OUTPUT PORT|
| Line Term. |                     |   FABRIC    |                     | Queuing    |
| Link Decap |====================>|             |====================>| Link Encap |====> Output
| Shadow FIB |   Data Plane        | (Crossbar / |   Data Plane        | Line Trans |      Links
+------------+   (Nanoseconds)     |  Multistage)|   (Nanoseconds)     +------------+
      |                            |             |                            ^
      v                            +-------------+                            |
+------------+                            ^                            +------------+
| INPUT PORT |============================+===========================>| OUTPUT PORT|
+------------+                                                         +------------+
  Input Links
```

#### Control Plane vs. Data Plane Operation

Modern network engineering enforces a strict separation between the **control plane** and the **data plane** (also known as the forwarding plane):

| Architectural Attribute | Control Plane (Routing Engine) | Data Plane (Forwarding Engine) |
|---|---|---|
| **Primary Responsibility** | Determining the end-to-end paths of packets across network topology | Moving packets from incoming physical interfaces to outgoing physical interfaces |
| **Implementation** | Software executing on a general-purpose Central Processing Unit (CPU) | Dedicated hardware: Application-Specific Integrated Circuits (ASICs) & Field-Programmable Gate Arrays (FPGAs) |
| **Operating Timeframe** | Millisecond ($\text{ms}$) to second ($\text{s}$) timescale | Nanosecond ($\text{ns}$) to microsecond ($\mu\text{s}$) timescale (line speed) |
| **Key Algorithms** | Dijkstra's Link-State algorithm, Bellman-Ford Distance-Vector, Border Gateway Protocol (BGP) policy engines | Longest Prefix Matching (LPM) via Ternary Content Addressable Memory (TCAM), packet scheduling, queue management |
| **Failure Impact** | Slow route convergence, temporary routing loops | Immediate packet drops, buffer overflow, link starvation |

---

### 3.2 Input Port Functions

An input port is not merely a physical socket; it is a complex pipeline consisting of three discrete functional blocks:

```
+-------------------+      +-------------------+      +-------------------+
|  LINE TERMINATION | ===> | DATA LINK LAYER   | ===> | LOOKUP, FORWARDING| ===> To Switching
|  (Physical Layer) |      | (Decapsulation)   |      | & INPUT QUEUING   |      Fabric
+-------------------+      +-------------------+      +-------------------+
  - Physical connector       - Ethernet frame           - Shadow FIB lookup
  - Bit-level reception        decapsulation            - Longest prefix match
  - Optical/electrical       - CRC / FCS check          - Line-speed switching
    conversion               - MAC address filtering    - Queuing if fabric busy
```

#### Line Termination & Physical Layer Reception
The incoming physical transmission medium (twisted-pair copper wire, coaxial cable, or optical fiber) terminates at the input port's physical transceiver. Here, analog electrical signals, radio frequencies, or light pulses are sampled, synchronized, and converted into discrete digital bitstreams.

#### Data Link Layer Decapsulation
The bitstream is parsed by link-layer logic (such as IEEE 802.3 Ethernet or High-Level Data Link Control — HDLC). The network device:
- Identifies framing boundaries (preamble and start-of-frame delimiter).
- Computes the Frame Check Sequence (FCS) using a Cyclic Redundancy Check (CRC) to verify bit integrity. Corrupted frames are silently dropped.
- Inspects the destination Media Access Control (MAC) address to ensure the frame was addressed to this router interface.
- Strips off the link-layer header and trailer (decapsulation), exposing the enclosed Layer 3 Internet Protocol (IP) datagram.

#### Decentralized Forwarding and Shadow Forwarding Tables
In early routers, forwarding was centralized: every input port handed arriving packets over a shared bus to the central routing processor CPU, which looked up the destination address in main memory. This created a severe bottleneck, capping aggregate router throughput at a fraction of line speed.

Modern routers implement **decentralized forwarding**:
1. The routing processor computes the global routing state using routing protocols and compiles a master Forwarding Information Base (FIB).
2. A duplicate, read-only copy—termed the **shadow forwarding table** (or shadow FIB)—is distributed directly into the local memory of **every individual input port**.
3. When a datagram arrives, the input port determines the appropriate output port locally, in hardware, without consulting the central routing processor or generating bus traffic.
4. **Line-Speed Goal**: Lookup processing must execute in fewer nanoseconds than the transmission time of the smallest allowable packet. For example, on a 40 Gigabits per second ($\text{Gbps}$) link with minimum-sized 64-byte (512-bit) Ethernet packets:
   $$t_{\text{packet}} = \frac{512\text{ bits}}{40 \times 10^9\text{ bps}} = 12.8\text{ nanoseconds}$$
   The entire lookup and forwarding pipeline must complete within $12.8\text{ ns}$ per packet!

#### Destination-Based vs. Generalized Forwarding
- **Destination-Based Forwarding**: The traditional Internet routing paradigm. The forwarding decision depends **solely and exclusively** on the 32-bit IPv4 (or 128-bit IPv6) destination address contained in the packet header.
- **Generalized Forwarding (Match-plus-Action)**: The foundational paradigm of Software-Defined Networking (SDN) and OpenFlow. Instead of examining only the destination IP address, the switch matches packets against arbitrary combinations of header fields across multiple protocol layers:
  - Layer 2: Source/Destination MAC address, Virtual Local Area Network (VLAN) Identifier.
  - Layer 3: Source/Destination IP address, Type of Service (TOS) / Differentiated Services Code Point (DSCP), Protocol field.
  - Layer 4: Transmission Control Protocol (TCP) / User Datagram Protocol (UDP) Source/Destination port numbers.
  - **Action**: Matched packets can be forwarded to one or more output ports, dropped (firewalling), modified (Network Address Translation — NAT), cloned (traffic monitoring / port mirroring), or redirected to the central SDN controller.

---

### 3.3 Longest Prefix Matching (LPM)

#### Why LPM is Essential with CIDR
Prior to Classless Inter-Domain Routing (CIDR — RFC 1519), IP addresses belonged to rigid classes (Class A, B, or C). In classful networks, a routing table entry had an unambiguous, fixed netmask ($/8$, $/16$, or $/24$). 

With CIDR, network prefixes can have arbitrary prefix lengths ranging from $/0$ to $/32$. Consequently, an Internet Service Provider (ISP) might advertise an aggregated block covering millions of addresses (e.g., `200.23.16.0/20`), while an organization inside that block has a dedicated link advertised with a more specific route (e.g., `200.23.18.0/24`). Both entries encompass the IP address `200.23.18.5`.

To resolve overlapping address spaces unambiguously, routers enforce the **Longest Prefix Matching (LPM)** rule:
> **Longest Prefix Match Rule**: When searching the forwarding table for an entry matching a given destination IP address, the router must select the routing table entry that shares the **longest contiguous bit prefix** with the destination address (i.e., the most specific route).

#### LPM Search Algorithm & Step-by-Step Prefix Evaluation

Consider the following canonical forwarding table from Kurose & Ross:

| Prefix Entry | Destination Address Prefix (Binary) | Outgoing Link Interface |
|---|---|---|
| **Entry 0** | `11001000 00010111 00010*** ********` (21-bit prefix) | Interface 0 |
| **Entry 1** | `11001000 00010111 00011000 ********` (24-bit prefix) | Interface 1 |
| **Entry 2** | `11001000 00010111 00011*** ********` (21-bit prefix) | Interface 2 |
| **Default** | Otherwise (0-bit prefix: `0.0.0.0/0`) | Interface 3 |

##### Example Evaluation 1:
Destination Address: `11001000 00010111 00010110 10100001` (Dotted decimal: `200.23.22.161`)
1. Compare against Entry 0:
   - Prefix 0: `11001000 00010111 00010...` (first 21 bits)
   - Dest IP:  `11001000 00010111 00010...` $\implies$ **Match (21 bits)**.
2. Compare against Entry 1:
   - Prefix 1: `11001000 00010111 00011000...`
   - Dest IP:  `11001000 00010111 00010110...` (mismatch at bit 21) $\implies$ **No Match**.
3. Compare against Entry 2:
   - Prefix 2: `11001000 00010111 00011...`
   - Dest IP:  `11001000 00010111 00010...` (mismatch at bit 21) $\implies$ **No Match**.
- **Decision**: Only Entry 0 matches. Packet forwarded to **Interface 0**.

##### Example Evaluation 2:
Destination Address: `11001000 00010111 00011000 10101010` (Dotted decimal: `200.23.24.170`)
1. Compare against Entry 0:
   - Prefix 0: `11001000 00010111 00010...`
   - Dest IP:  `11001000 00010111 00011...` (mismatch at bit 21) $\implies$ **No Match**.
2. Compare against Entry 1:
   - Prefix 1: `11001000 00010111 00011000...` (first 24 bits)
   - Dest IP:  `11001000 00010111 00011000...` $\implies$ **Match (24 bits)**.
3. Compare against Entry 2:
   - Prefix 2: `11001000 00010111 00011...` (first 21 bits)
   - Dest IP:  `11001000 00010111 00011...` $\implies$ **Match (21 bits)**.
- **Decision**: Both Entry 1 (24-bit match) and Entry 2 (21-bit match) match the destination address. Because $24 > 21$, Entry 1 is the longest prefix. Packet forwarded to **Interface 1**.

##### Example Evaluation 3:
Destination Address: `11001000 00010111 00011100 10101010` (Dotted decimal: `200.23.28.170`)
- Entry 0: Mismatch at bit 21.
- Entry 1: Mismatch at bit 22 (`0` in prefix vs. `1` in destination).
- Entry 2: Matches first 21 bits (`11001000 00010111 00011...`).
- **Decision**: Longest match is Entry 2. Packet forwarded to **Interface 2**.

#### Software Tries vs. Hardware TCAMs

##### Software Lookups: Tries and Radix Trees
In software, prefixes are organized in tree structures called **tries** (from retrieval). A binary trie represents each prefix as a path from the root, where a `0` branches left and a `1` branches right. While efficient for moderate table sizes ($O(K)$ lookup time where $K = 32$ for IPv4), software tree traversal requires sequential memory accesses, incurring a latency of $20\text{--}50\text{ ns}$ per lookup. This is far too slow for multi-gigabit and terabit core routers.

##### Hardware Implementation: TCAM (Ternary Content Addressable Memory)
High-performance routers implement LPM using **TCAM (Ternary Content Addressable Memory)**:
1. **Ternary Logic**: Standard memory (RAM) takes an address and returns data. Standard CAM (Content Addressable Memory) takes data (`0` or `1`) and returns the matching address in a single clock cycle. **TCAM** extends CAM by supporting three logic states: `0`, `1`, and `*` (wildcard / "don't care").
2. **Single-Cycle Parallel Search**: An incoming 32-bit IP address is broadcast simultaneously across hundreds of thousands of TCAM rows in a single clock cycle ($O(1)$ constant time complexity), regardless of routing table size.
3. **Priority Encoder**: Since multiple overlapping prefix entries can match simultaneously, the TCAM arrays are arranged in descending order of prefix length (longest prefixes at the lowest memory indices). The hardware priority encoder instantly returns the match located at the lowest index.
4. **Industrial Reality**: High-end enterprise switches (e.g., Cisco Catalyst 6500 and 7600 series) utilize TCAMs capable of storing upwards of $1{,}000{,}000$ routing table entries with search times under $5\text{ nanoseconds}$.
5. **Engineering Trade-offs**: TCAMs are physically expensive, occupy substantial silicon area, generate intense heat, and consume 10 to 15 times more electrical power than standard Static Random Access Memory (SRAM).

---

### 3.4 Switching Fabrics

The switching fabric is the physical core of the router. Its sole task is transferring datagrams from input buffers to output buffers. The **switching rate** is the aggregate rate at which packets can be transferred from inputs to outputs, typically measured as a multiple of the line rate $R$. For an $N$-port router, an ideal non-blocking switching fabric achieves a switching rate of:
$$\text{Ideal Fabric Switching Rate} = N \times R$$

There are three primary architectural generations of switching fabrics:

```
+--------------------+      +--------------------+      +--------------------+
| 1. VIA MEMORY      |      | 2. VIA A BUS       |      | 3. INTERCONNECTION |
|                    |      |                    |      |    NETWORK         |
|  [Input]  [Output] |      |  [Input]  [Output] |      |  [In 1]     [Out 1]|
|     \       ^      |      |     \       ^      |      |    \   |   /       |
|      v     /       |      | =====\=====/====== |      |  ----+---+----     |
|    +--------+      |      |     SHARED BUS     |      |    Crossbar / Clos |
|    | CPU/RAM|      |      |  (Single transfer  |      |  (Parallel paths,  |
|    +--------+      |      |   at a time)       |      |   non-blocking)    |
| (2 bus crossings)  |      +--------------------+      +--------------------+
+--------------------+
```

#### 1. Switching via Memory
- **Architecture**: First-generation commercial routers were general-purpose computers. Input and output ports operated as conventional Input/Output (I/O) peripheral cards plugged into a system bus.
- **Operation**:
  1. Arriving packet signals an interrupt to the Central Processing Unit (CPU).
  2. Packet is copied from the input port buffer over the shared system bus into CPU Random Access Memory (RAM).
  3. Routing processor reads the packet header, performs lookup, and writes the output port assignment into the packet descriptor.
  4. Packet is copied from CPU RAM over the shared system bus into the output port buffer.
- **Performance Bottleneck**: Every single datagram must traverse the system memory bus **twice** (Input Port $\to$ Memory, then Memory $\to$ Output Port). Consequently, the maximum switching throughput is strictly bounded by half the memory bus bandwidth:
  $$B_{\text{switching}} \le \frac{B_{\text{bus}}}{2}$$
  Two packets cannot be forwarded simultaneously, even if they arrive at different input ports and depart through different output ports.

#### 2. Switching via a Bus
- **Architecture**: Second-generation routers eliminated CPU intervention from the forwarding path. Input ports and output ports are connected directly to an internal shared high-speed data bus.
- **Operation**:
  1. Input port receives a datagram and determines the output port via its local shadow forwarding table.
  2. Input port prepends an internal switching label (header) specifying the destination output port.
  3. The packet is transmitted directly onto the shared bus.
  4. All output ports see the packet, but only the output port whose internal ID matches the label copies the packet into its queue and strips the label.
- **Performance Bottleneck**: The bus is a shared broadcast medium. **Only one packet can traverse the bus at any given instant.** If multiple packets arrive at different input ports simultaneously, all but one must wait for bus arbitration.
- **Capacity**: Sufficient for small-to-medium enterprise and access routers (e.g., the Cisco 5600 router operated with a $32\text{ Gbps}$ bus), but inadequate for telecommunications core routers.

#### 3. Switching via an Interconnection Network
To surpass the single-packet serialization bottleneck of a shared bus, modern core routers employ **interconnection networks**:

##### Crossbar Switches
A crossbar switch is a two-dimensional mesh of $2N$ buses interconnecting $N$ input ports with $N$ output ports. At each of the $N^2$ intersections sits an electronic crosspoint switch controlled by fabric scheduling logic:
- When a packet at Input Port $A$ is destined for Output Port $X$, the crosspoint $(A, X)$ is closed, establishing an isolated electrical connection.
- **Non-Blocking Characteristic**: A crossbar is non-blocking. **Multiple packets can be transferred concurrently across the fabric in parallel**, provided that each packet is destined for a **different** output port.
- If Input 1 sends to Output 1, and Input 2 sends to Output 2, both transfers proceed simultaneously at full wire speed!

##### Multistage Switches (Clos and Banyan Networks)
As port count $N$ scales into hundreds or thousands, an $N \times N$ crossbar requires $N^2$ crosspoints, becoming cost-prohibitive. Routers instead construct **multistage switches** (e.g., three-stage Clos networks or multi-stage Banyan networks) using cascading matrices of smaller $k \times k$ crossbar switching elements.

##### Advanced Fabric Scaling: Cell-Based Switching and Parallel Planes
To maximize switching efficiency and prevent large variable-length IP packets (e.g., 1500 bytes) from monopolizing internal switching paths:
1. **Cell Segmentation**: Input ports chop incoming variable-length IP datagrams into small, uniform, fixed-length "cells" (typically 64 bytes).
2. **Cell Scheduling**: Cells are routed through the multistage fabric using fast hardware clocks.
3. **Cell Reassembly**: Output ports collect the cells and reassemble them into original IP datagrams before link transmission.
4. **Parallel Switching Planes**: Massive carrier-scale routers (such as the Cisco Carrier Routing System — CRS) deploy multiple identical switching fabric planes in parallel (e.g., 8 parallel fabric planes, each containing a 3-stage interconnection network). This achieves aggregate switching capacities exceeding hundreds of Terabits per second ($\text{Tbps}$).

---

### 3.5 Input Port Queuing and Head-of-Line (HOL) Blocking

#### Mechanism of Head-of-Line (HOL) Blocking
Even if a switching fabric is completely non-blocking, queuing can still occur at the input ports due to **output port contention**. If packets at the heads of two or more distinct input queues are destined for the **exact same output port**, the fabric can switch only one packet during that time slot. The remaining competing packets must remain queued at their respective input ports.

This leads to a catastrophic phenomenon termed **Head-of-Line (HOL) Blocking**:
> **Definition of Head-of-Line (HOL) Blocking**: A condition in an input-queued switch where a packet waiting at the front (head) of a First-In, First-Out (FIFO) input queue blocks all subsequent packets behind it in that queue, preventing them from being forwarded—even if the output ports requested by those subsequent packets are completely idle.

```
TIME SLOT t:
Input Port 1 Queue:  [Green to Out 3] [Red to Out 1]  ===> Fabric switches Input 1 (Red) to Out 1.
Input Port 2 Queue:  [Green to Out 3] [Red to Out 1]  ===> Input 2 (Red) is BLOCKED (Contention for Out 1).
                                                           (Red packet must wait at head of line).

TIME SLOT t + 1:
Input Port 2 Queue:  [Green to Out 3] [Red to Out 1]  ===> Red packet remains at head of queue.
                                                           Notice that Output Port 3 is IDLE!
                                                           However, the Green packet (destined for Out 3)
                                                           CANNOT MOVE because it is trapped behind Red!
                                                           ===> Green experiences HOL BLOCKING!
```

#### Theoretical Throughput Derivation: The 58.6% Limit
In their seminal 1987 paper (*"Input Versus Output Queueing on a Space-Division Packet Switch"*, IEEE Transactions on Communications), Mark Karol, Michael Hluchyj, and Samuel Morgan mathematically proved the fundamental throughput limit of input-queued crossbar switches operating under FIFO discipline:

##### Assumptions of the Karol-Hluchyj-Morgan Model:
1. An $N \times N$ crossbar switch with infinite input FIFO buffers.
2. In each time slot, packets arrive at each input port according to independent, identical Bernoulli processes with arrival probability $\rho$ (offered traffic load).
3. Packet destinations are uniformly and independently distributed across all $N$ output ports (probability $1/N$ for each output).
4. Synchronous slotted operation; transmission of one packet across the fabric takes one time slot.

##### Mathematical Derivation:
Let $B_m(t)$ denote the number of blocked head-of-line packets destined for output port $m$ at the end of slot $t$. In slot $t+1$, let $A_m(t+1)$ denote the number of freshly arriving head-of-line packets (packets that moved to the front because their predecessors departed) destined for output $m$.

Only one packet can be served by output port $m$ per slot. If $B_m(t) + A_m(t+1) > 0$, exactly one packet departs. The queue of HOL packets competing for output port $m$ evolves identically to a discrete-time $M/D/1$ queue:
$$Q_m(t+1) = \max\Big(0, \, Q_m(t) + A_m(t+1) - 1\Big)$$

In steady state, the total throughput per port is $\rho$. The average number of newly arriving HOL packets per slot across all inputs that find an output available is governed by the saturation condition. As the switch size scales to infinity ($N \to \infty$):
- The arrival process of new HOL packets to any specific output queue converges to a Poisson distribution with parameter $\rho$.
- Applying the Pollaczek-Khinchine formula for the mean queue length of an $M/D/1$ queue:
  $$\overline{Q} = \frac{\rho^2}{2(1 - \rho)}$$
- The total fraction of input ports that have a packet actively being serviced must equal the offered load $\rho$. 
- Equating the probability of an input queue being blocked to the remaining headroom leads to the fundamental balance equation:
  $$\rho + \frac{\rho^2}{2(1 - \rho)} = 1$$
- Multiplying through by $2(1 - \rho)$:
  $$2\rho(1 - \rho) + \rho^2 = 2(1 - \rho)$$
  $$2\rho - 2\rho^2 + \rho^2 = 2 - 2\rho$$
  $$-\rho^2 + 2\rho = 2 - 2\rho$$
- Rearranging into standard quadratic form:
  $$\rho^2 - 4\rho + 2 = 0$$
- Solving for $\rho$ using the quadratic formula $\rho = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$:
  $$\rho = \frac{4 \pm \sqrt{(-4)^2 - 4(1)(2)}}{2(1)} = \frac{4 \pm \sqrt{16 - 8}}{2} = \frac{4 \pm \sqrt{8}}{2} = \frac{4 \pm 2\sqrt{2}}{2} = 2 \pm \sqrt{2}$$
- Since throughput cannot exceed $100\%$ ($\rho \le 1$), the root $2 + \sqrt{2} \approx 3.414$ is physically impossible. The unique valid physical root is:
  $$\rho_{\max} = 2 - \sqrt{2} \approx 2 - 1.41421356 = 0.585786 \dots \approx 58.6\%$$

> [!CAUTION]
> **Key Takeaway**: Due entirely to Head-of-Line (HOL) blocking, an input-queued crossbar switch with FIFO queues cannot exceed a maximum asymptotic throughput of **$58.6\%$** of its theoretical capacity under uniform random traffic. Over $41.4\%$ of switch bandwidth is wasted!

#### Eliminating HOL Blocking: Virtual Output Queuing (VOQ)

To circumvent this limitation, modern switches abandon simple FIFO queues at input ports and implement **Virtual Output Queuing (VOQ)**:
- **Architecture**: Each physical input port maintains $N$ distinct, independent logical FIFO queues, where Queue $VOQ(i, j)$ holds packets arriving at Input Port $i$ that are destined exclusively for Output Port $j$.
- **Operation**: If a packet at the head of $VOQ(1, 1)$ is blocked because Output 1 is busy, packets waiting in $VOQ(1, 2)$ or $VOQ(1, 3)$ are **not blocked**! The fabric arbiter can inspect all $N \times N$ queues.
- **Arbitration Algorithms**: Switches execute bipartite matching algorithms (such as Nick McKeown's **iSLIP** algorithm—iterative Request, Grant, Accept with round-robin pointers) to match inputs to outputs in every time slot.
- **Result**: VOQ completely eliminates Head-of-Line blocking, allowing input-queued switches to achieve **$100\%$ theoretical throughput**!

---

### 3.6 Output Port Queuing and Buffer Sizing

#### Where and Why Output Queuing Occurs
An output port receives packets delivered by the switching fabric and transmits them onto the physical outgoing link. Queuing occurs at the output port whenever the arrival rate of packets from the switching fabric exceeds the transmission capacity (line rate $R$) of the outgoing link.

Even if an $N \times N$ switching fabric operates with an internal speedup of $N$ (capable of transferring $N$ packets to the same output port simultaneously), the output link can only transmit **one packet at a time**. If $k$ packets destined for the same output port arrive in the same time slot, 1 packet begins transmission immediately, and $k - 1$ packets must be queued in the output port's packet buffer.

If packets continue to arrive faster than the link transmission rate, the output buffer eventually fills to capacity. Subsequent arriving packets find no available buffer memory and are dropped, resulting in **packet loss**.

```
+-------------------------------------------------------------------------+
|                              OUTPUT PORT                                |
|                                                                         |
| From Switching ===> +----------------------+ ===> +------------------+  |
| Fabric              | PACKET BUFFER MEMORY |      | LINK PROTOCOL &  | ===> Outgoing
| (Delivered at       | (Queuing delay,      |      | LINE TRANSMISSION|      Link
|  rate up to NR)     |  Drop policies:      |      | (Transmitted at  |     (Rate R)
|                     |  Drop-tail, RED)     |      |  rate R)         |  |
|                     +----------------------+      +------------------+  |
|                         |                                               |
|                         v Packet Scheduler                              |
|                         (FIFO, Priority, Round Robin, WFQ)              |
+-------------------------------------------------------------------------+
```

#### Buffer Sizing: Traditional Rule vs. Stanford/Appenzeller Rule

Determining the optimal buffer size $B$ in router line cards is a critical network engineering problem:
- **Too little buffering**: Brief traffic bursts cause immediate packet drops, triggering TCP window collapse and severe link underutilization.
- **Too much buffering**: Large buffers introduce massive queuing delay, sluggish TCP responsiveness, and wasted hardware expenditure.

##### 1. The Traditional Rule of Thumb (RFC 3439)
Historically, router manufacturers sized output buffers to match the **Bandwidth-Delay Product (BDP)**:
$$B = \text{RTT} \times C$$
where:
- $\text{RTT}$ is the average Round-Trip Time of TCP connections traversing the link (classically assumed to be $250\text{ milliseconds} = 0.25\text{ s}$).
- $C$ is the transmission capacity (bandwidth) of the bottleneck output link.

**Theoretical Basis**: In a network carrying a single long-lived TCP Reno connection, TCP's Additive Increase Multiplicative Decrease (AIMD) algorithm probes for bandwidth until a packet is dropped, at which point it cuts its congestion window ($W$) in half (from $W_{\max}$ to $W_{\max}/2$). To keep the bottleneck link $100\%$ utilized while the TCP sender ramps back up, the router's buffer must hold enough packets to continuously feed the link during that window recovery:
$$B = \frac{W_{\max}}{2} = \text{RTT} \times C$$

*Numerical Example*: For a $10\text{ Gbps}$ link with an average $\text{RTT}$ of $250\text{ ms}$:
$$B = 0.250\text{ s} \times (10 \times 10^9\text{ bps}) = 2.5\times 10^9\text{ bits} = 2.5\text{ Gbits} \approx 312.5\text{ Megabytes (MB)}$$

##### 2. The Stanford / Appenzeller Rule for $N$ Independent TCP Flows
In 2004, Guido Appenzeller, Isaac Keslassy, and Nick McKeown (ACM SIGCOMM 2004) proved that the traditional BDP rule severely over-buffers core Internet routers. 

In a real-world core backbone link, the buffer is shared by thousands of independent, uncoordinated TCP flows. Because these flows have varied round-trip times and start times, their sawtooth congestion cycles are **desynchronized**. When one flow drops its window, dozens of others are increasing theirs.

Applying the Central Limit Theorem, the variance of the sum of $N$ independent random variables scales with $\sqrt{N}$. Appenzeller et al. proved that to maintain nearly $100\%$ link utilization, the required buffer size scales down by $\frac{1}{\sqrt{N}}$:
$$B = \frac{\text{RTT} \times C}{\sqrt{N}}$$
where $N$ is the number of independent, long-lived TCP flows traversing the router.

*Numerical Comparison*: Consider the same $C = 10\text{ Gbps}$ link with $\text{RTT} = 250\text{ ms}$, but carrying $N = 10{,}000$ independent TCP connections:
$$B = \frac{250\text{ ms} \times 10\text{ Gbps}}{\sqrt{10{,}000}} = \frac{2.5\text{ Gbits}}{100} = 25\text{ Mbits} \approx 3.125\text{ Megabytes (MB)}$$

> [!TIP]
> **Engineering Significance of the Stanford Rule**:
> Sizing the buffer according to $B = \frac{\text{RTT} \times C}{\sqrt{N}}$ reduces required memory by **$99\%$** (from $312.5\text{ MB}$ down to $3.125\text{ MB}$). A $3.125\text{ MB}$ buffer can be implemented directly on-chip inside the router's ASIC using ultra-fast **SRAM** (access latency $< 1\text{ ns}$), eliminating the need for slow, power-hungry, off-chip **DRAM** (access latency $20\text{--}50\text{ ns}$).

#### Bufferbloat and Delay Penalties
In home access routers, cable modems, and edge switches, manufacturers frequently embed massive DRAM buffers configured with simple drop-tail policies. When users run heavy upload or download streams (e.g., peer-to-peer file sharing or large cloud backups), these enormous buffers fill up completely, holding seconds worth of packets.

This condition is known as **Bufferbloat**:
- Packets generated by interactive, real-time applications (such as VoIP calls, multiplayer online gaming, or DNS queries) are stuck waiting behind thousands of bulk-transfer packets.
- Latencies spike from normal values of $20\text{ ms}$ up to $1{,}000\text{--}2{,}000\text{ ms}$ (1 to 2 seconds!).
- TCP congestion control signals are severely delayed, defeating delay-based congestion control algorithms.

#### Active Queue Management (AQM): Drop-Tail vs. RED

Buffer management policies govern what happens when buffers experience congestion.

```
Buffer Occupancy:
Empty Buffer                                                              Full Buffer
|---------------------|---------------------------------|-------------------|
0                    min_th                            max_th            Capacity
[  Drop Prob p = 0   ] [   Probabilistic Early Drop    ] [ Drop Prob p = 1 ]
                       [   p increases linearly to p_max] [ (Forced Drops)   ]
```

##### 1. Drop-Tail (Tail Drop)
The simplest, default queue management discipline:
- Incoming packets are enqueued as long as free buffer space exists.
- When the buffer is $100\%$ full, every subsequent arriving packet is dropped until space is freed.
- **Severe Flaws**:
  1. **Global Synchronization**: When the buffer fills, packets from hundreds of TCP flows are dropped simultaneously. All these flows simultaneously detect loss, cut their congestion windows in half at the exact same time, and throttle transmission. The bottleneck link immediately plunges into underutilization. Then, all flows ramp up synchronously, cause buffer overflow again, and repeat the cycle.
  2. **Burst Bias / Lock-Out**: A sudden burst of packets from a single misbehaved flow can fill the remaining buffer, locking out well-behaved, low-rate interactive flows.

##### 2. RED (Random Early Detection)
Introduced by Sally Floyd and Van Jacobson (1993) to eliminate global synchronization and keep average queue sizes low:
1. **Exponentially Weighted Moving Average (EWMA)**: RED tracks an average queue length $\text{AvgQueue}$ rather than instantaneous queue occupancy $q$, smoothing out transient bursts:
   $$\text{AvgQueue} = (1 - w_q) \times \text{AvgQueue} + w_q \times q$$
   where $w_q$ is a small weighting filter constant (typically $w_q \approx 0.002$).
2. **Dual Queue Thresholds**: RED defines a minimum threshold $min_{th}$ and a maximum threshold $max_{th}$.
3. **Drop / Mark Decision Logic**:
   - **Case 1 ($\text{AvgQueue} < min_{th}$)**: No congestion. Arriving packet is **admitted** unconditionally ($p = 0$).
   - **Case 2 ($min_{th} \le \text{AvgQueue} \le max_{th}$)**: Incipient congestion. The packet is dropped (or marked if using ECN) with a probability $p$ that scales linearly with queue occupancy:
     $$p_b = p_{\max} \times \frac{\text{AvgQueue} - min_{th}}{max_{th} - min_{th}}$$
     $$p = \frac{p_b}{1 - \text{count} \times p_b}$$
     where $\text{count}$ is the number of consecutive packets accepted since the last drop, ensuring dropped packets are spaced evenly over time.
   - **Case 3 ($\text{AvgQueue} > max_{th}$)**: Severe congestion. Arriving packet is **dropped** unconditionally ($p = 1.0$), identical to drop-tail.
4. **Explicit Congestion Notification (ECN) Synergy**: If both router and end-hosts support ECN (RFC 3168), a RED router operating in the $[min_{th}, max_{th}]$ zone does not drop the packet; instead, it sets the two **Congestion Experienced (CE)** bits in the IPv4 Type of Service (TOS) or IPv6 Traffic Class header. The receiver echoes this signal back to the sender in TCP ACKs, prompting the sender to reduce its transmission rate **without suffering a packet loss**!

---

### 3.7 Packet Scheduling Policies

When multiple packets reside in an output port queue, the **packet scheduler** selects which packet is transmitted next onto the physical link. Schedulers are evaluated on whether they are **work-conserving**: a scheduler is work-conserving if it never allows the transmission link to remain idle whenever there is at least one packet buffered in any queue.

```
1. FIFO:
   In ===> [ 1 ][ 2 ][ 3 ][ 4 ] ===> Out (Strict arrival order)

2. PRIORITY QUEUING:
   High Priority ===> [ H1 ][ H2 ] ===\
                                       ===> Multiplexer ===> Out (High served first)
   Low Priority  ===> [ L1 ][ L2 ] ===/

3. ROUND ROBIN:
   Class 1 ===> [ C1-1 ][ C1-2 ] ===\
                                     ===> Circular Selector ===> Out (1 from C1, 1 from C2...)
   Class 2 ===> [ C2-1 ][ C2-2 ] ===/

4. WEIGHTED FAIR QUEUING (WFQ):
   Class 1 (Weight w1 = 2) ===> [ P1 ] ===\
                                           ===> Virtual Finish Time ===> Out (Guaranteed ratio
   Class 2 (Weight w2 = 1) ===> [ P2 ] ===/     Scheduler                 w1 / (w1 + w2))
```

#### 1. First-In, First-Out (FIFO)
- **Principle**: Packets are queued in a single buffer and serviced in the exact sequential order of their arrival.
- **Properties**: Work-conserving, computationally trivial ($O(1)$ enqueue and dequeue).
- **Limitation**: Provides zero service differentiation. An interactive VoIP packet arriving behind a massive 1500-byte FTP packet is delayed by the full transmission time of the FTP packet.

#### 2. Priority Queuing
- **Principle**: Arriving packets are inspected (via IP header TOS/DSCP bits, IP addresses, or port numbers) and sorted into discrete priority classes (e.g., High, Medium, Low).
- **Non-Preemptive Serving Rule**: When the transmission link finishes sending a packet, the scheduler selects the packet at the head of the **highest-priority non-empty queue**. A packet already being transmitted is never interrupted (non-preemptive).
- **Advantage**: Guaranteed minimal queuing delay and jitter for mission-critical and real-time traffic (e.g., VoIP and network control signaling).
- **Severe Flaw — Starvation**: If high-priority traffic continuously arrives at a rate equal to or exceeding the link capacity $R$, lower-priority queues will **never be serviced** (complete starvation).

#### 3. Round Robin (RR)
- **Principle**: Arriving packets are sorted into discrete class queues. The scheduler cycles through the classes in round-robin sequence: it serves one packet from Class 1, then one packet from Class 2, ..., and repeats.
- **Work-Conserving Feature**: If the scheduler inspects a class queue that is empty, it immediately skips to the next non-empty class without wasting link time.
- **Advantage**: Absolute protection against starvation; every class receives guaranteed opportunities to transmit.
- **Limitation**: Unfair if packet sizes vary across classes. If Class 1 sends 1500-byte packets and Class 2 sends 64-byte packets, Class 1 consumes $\frac{1500}{1564} \approx 96\%$ of total link bandwidth!

#### 4. Weighted Fair Queuing (WFQ)
Weighted Fair Queuing is the most sophisticated and widely deployed scheduling discipline in modern core routers (democratized by Alan Demers, Srinivasan Keshav, and Scott Shenker in 1989; Parekh and Gallager in 1993).

##### Operational Principle:
WFQ is a packetized approximation of ideal **Generalized Processor Sharing (GPS)**:
- Arriving packets are sorted into $K$ separate class queues.
- Each class $i$ is assigned a positive static weight $w_i$.
- During any interval of time where a set of classes is actively backlogged (have packets in their queues), class $i$ is guaranteed to receive a fraction of the total link bandwidth $R$ equal to:
  $$R_i = R \times \frac{w_i}{\sum_{j \in \text{Active}} w_j}$$

##### Emulating Bit-by-Bit Round Robin via Virtual Finish Time:
In ideal GPS, the link transmits a fraction of a bit from each queue simultaneously. In a real-world network, packets are indivisible and must be transmitted completely as discrete units.

To emulate GPS, WFQ calculates a **Virtual Finish Time** $F_{i,k}$ for each packet $k$ arriving at class $i$:
$$F_{i,k} = \max\Big(V(a_{i,k}), \, F_{i,k-1}\Big) + \frac{L_{i,k}}{w_i}$$
where:
- $a_{i,k}$ is the physical arrival timestamp of packet $k$ into class $i$.
- $V(t)$ is the system **Virtual Time** tracking the progress of work across all currently active queues.
- $F_{i,k-1}$ is the virtual finish time of the preceding packet in class $i$ (ensures packets within the same class maintain FIFO order).
- $L_{i,k}$ is the length of packet $k$ in bits.
- $w_i$ is the weight assigned to class $i$.

The WFQ scheduler operates by always selecting the packet that has the **smallest virtual finish time** among all packets currently at the heads of the backlogged queues.

##### Comprehensive WFQ Numerical Example:
Suppose a $10\text{ Mbps}$ link connects three traffic classes with weights:
- Class 1 (VoIP): $w_1 = 5$
- Class 2 (Video): $w_2 = 3$
- Class 3 (Web/FTP): $w_3 = 2$

Total sum of weights $\sum w_j = 5 + 3 + 2 = 10$.
Guaranteed minimum bandwidth allocations when all three classes are active:
- Class 1: $10\text{ Mbps} \times \frac{5}{10} = 5.0\text{ Mbps}$ ($50\%$)
- Class 2: $10\text{ Mbps} \times \frac{3}{10} = 3.0\text{ Mbps}$ ($30\%$)
- Class 3: $10\text{ Mbps} \times \frac{2}{10} = 2.0\text{ Mbps}$ ($20\%$)

If Class 2 becomes idle (no video packets), its unused bandwidth is automatically and proportionally divided among the active classes (Class 1 and Class 3):
- Class 1 now receives: $10\text{ Mbps} \times \frac{5}{5 + 2} = \frac{25}{7} \approx 7.14\text{ Mbps}$
- Class 3 now receives: $10\text{ Mbps} \times \frac{2}{5 + 2} = \frac{10}{7} \approx 2.86\text{ Mbps}$

WFQ is completely work-conserving, provably starvation-free, provides tight deterministic delay bounds, and isolates well-behaved flows from malicious or bursty flows!

---

## 4. The Internet Protocol (IPv4)

The **Internet Protocol version 4 (IPv4 — RFC 791)** is the foundational network-layer communication protocol of the global Internet. It defines the universal addressing scheme, packet formatting, and fragmentation rules governing host-to-host delivery.

---

### 4.1 IPv4 Datagram Format

Every IPv4 packet consists of a structured **header** followed by a variable-length **payload** (data). The header is organized in 32-bit (4-byte) horizontal words:

#### Full 32-Bit Header Diagram

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version|  IHL  |Type of Service|          Total Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Identification        |Flags|      Fragment Offset    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Time to Live |    Protocol   |        Header Checksum        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source IP Address                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination IP Address                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Options (0 to 40 bytes)    |    Padding    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                            Payload                            |
|                     (Transport-Layer Data)                    |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Comprehensive Field-by-Field Breakdown (All 13 Fields)

##### 1. Version (4 bits)
Specifies the IP protocol version. For IPv4, this field contains the binary value `0100` (decimal 4). If a router receives a datagram whose Version field does not match the protocol software processing it, the datagram is immediately discarded.

##### 2. Internet Header Length (IHL) (4 bits)
Specifies the total length of the IPv4 header measured in **32-bit (4-byte) words**:
- **Minimum Value**: A standard IPv4 header without options is 20 bytes long. Measured in 4-byte words, $\frac{20}{4} = 5$. Thus, the minimum valid value of IHL is `0101` (decimal 5).
- **Maximum Value**: The 4-bit field can hold values up to $2^4 - 1 = 15$ (`1111` in binary). Multiplying by 4 bytes yields $15 \times 4 = 60\text{ bytes}$.
- Therefore, an IPv4 header is bounded between **20 bytes and 60 bytes**.

##### 3. Type of Service (TOS) / Differentiated Services (DiffServ) (8 bits)
Originally defined in RFC 791 to convey packet precedence and trade-offs between delay, throughput, and reliability. Redefined by modern standards (RFC 2474, RFC 3168) into two functional subfields:
- **Differentiated Services Code Point (DSCP)** (first 6 bits): Classifies traffic into Quality of Service (QoS) classes (e.g., Expedited Forwarding for VoIP, Assured Forwarding for video, Best Effort for web).
- **Explicit Congestion Notification (ECN)** (last 2 bits):
  - `00`: Non ECN-Capable Transport (Non-ECT).
  - `01` or `10`: ECN-Capable Transport (ECT(1) or ECT(0)). Set by endpoints to indicate support.
  - `11`: Congestion Experienced (CE). Set by a congested router in transit to signal network bottlenecks back to endpoints without dropping the datagram.

##### 4. Total Length (16 bits)
Specifies the total size of the entire IP datagram (header plus payload) measured in **bytes**:
- With 16 bits, the theoretical maximum datagram size is $2^{16} - 1 = 65{,}535\text{ bytes}$.
- In practice, datagrams rarely exceed 1500 bytes to avoid link-layer fragmentation across standard Ethernet networks.
- The payload size can be computed as:
  $$\text{Payload Length} = \text{Total Length} - (\text{IHL} \times 4)$$

##### 5. Identification (16 bits)
An integer sequence number assigned by the sending host. When a router must fragment an oversized datagram, every fragment copied from that datagram retains the exact same Identification number. This enables the destination host to group arriving fragments together for reassembly.

##### 6. Flags (3 bits)
Three control bits governing fragmentation:
- **Bit 0 (Reserved)**: Must be set to `0`. (Occasionally leveraged in security research as the "Evil Bit" — RFC 3514).
- **Bit 1 — Don't Fragment (DF)**:
  - If $\text{DF} = 0$: Routers are permitted to fragment the datagram if necessary.
  - If $\text{DF} = 1$: Routers are **forbidden** from fragmenting the datagram. If the datagram exceeds the Maximum Transmission Unit (MTU) of an outgoing link, the router drops the datagram and returns an ICMP (Internet Control Message Protocol) Destination Unreachable error message (Type 3, Code 4: *"Fragmentation Needed and DF set"*), reporting the link's MTU. This mechanism is the bedrock of **Path MTU Discovery (PMTUD)**.
- **Bit 2 — More Fragments (MF)**:
  - If $\text{MF} = 1$: Indicates that this packet is a fragment, and **more fragments follow**.
  - If $\text{MF} = 0$: Indicates that this is the **last fragment** of the original datagram (or that the datagram was never fragmented).

##### 7. Fragment Offset (13 bits)
Specifies where the data payload in this fragment belongs relative to the start of the unfragmented original IP payload.
- **Crucial Unit Constraint**: The Fragment Offset is measured in **units of 8-byte (64-bit) chunks**:
  $$\text{Fragment Offset Value} = \frac{\text{Byte Offset of Data in Original Payload}}{8}$$
- *Why 8-byte units?* A 13-bit field can represent integers from 0 up to $2^{13} - 1 = 8{,}191$. If offset were measured in single bytes, the maximum offset would be only 8,191 bytes, leaving the rest of a 65,535-byte datagram unreachable! Multiplying by 8 bytes allows 13 bits to cover:
  $$8{,}191 \times 8 = 65{,}528\text{ bytes}$$
  spanning the entire maximum IPv4 datagram space!
- Consequently, every fragment except the final one **must carry a payload whose byte length is an exact multiple of 8**.

##### 8. Time to Live (TTL) (8 bits)
A hop counter designed to prevent datagrams from circulating infinitely in the event of routing loops:
- Initialized by the source host (typical defaults: 64 in Linux/macOS, 128 in Windows).
- **Decrement Rule**: Every router that processes the datagram **must decrement TTL by at least 1**.
- If a router decrements TTL to 0, it drops the datagram immediately and sends an **ICMP Time Exceeded** message (Type 11, Code 0) back to the source IP address. (This exact mechanism is exploited by the `traceroute` utility).

##### 9. Upper Layer Protocol (8 bits)
Identifies the specific higher-layer transport protocol or network helper protocol to which the decapsulated IP payload must be passed at the destination host:
- `1`: ICMP (Internet Control Message Protocol)
- `2`: IGMP (Internet Group Management Protocol)
- `6`: TCP (Transmission Control Protocol)
- `17`: UDP (User Datagram Protocol)
- `41`: IPv6 Encapsulation (6in4 Tunneling)
- `89`: OSPF (Open Shortest Path First)

##### 10. Header Checksum (16 bits)
Provides error detection for the **IPv4 header only** (the transport-layer payload is not checked, as TCP and UDP contain their own end-to-end checksums):
- **Computation**: The header is treated as a sequence of 16-bit integers. The checksum field is initially set to zero. The 16-bit words are summed using **1's complement arithmetic**, and the 1's complement negation of the sum is placed in the checksum field.
- **Verification**: At the receiving router, the 16-bit words of the header (including the checksum) are summed. If the result equals `1111111111111111` (all 1s), the header is valid; otherwise, an error occurred and the packet is discarded.
- **Hop-by-Hop Recomputation**: Because the TTL field is decremented at every single router hop, **every router along the path must recompute the IPv4 header checksum**!

##### 11. Source IP Address (32 bits)
The permanent or globally unique 32-bit IPv4 address of the originating end system that created the datagram.

##### 12. Destination IP Address (32 bits)
The 32-bit IPv4 address of the ultimate destination end system.

##### 13. Options (0 to 40 bytes) and Padding
Optional parameters used for network diagnostics, security classification, or source-directed routing:
- Examples: Record Route, Timestamp, Strict Source Routing, Loose Source Routing.
- **Padding**: Because options vary in length, variable padding bits (all zeros) are appended to guarantee that the IPv4 header ends on a strict 32-bit (4-byte) boundary, ensuring IHL remains an exact integer.
- *Modern Reality*: IPv4 options are rarely used today. Core routers frequently drop or deprioritize packets with options because they require software "slow-path" exception processing on the routing CPU, breaking hardware ASIC line-speed pipelines.

---

### 4.2 IP Fragmentation and Reassembly

#### Maximum Transmission Unit (MTU) Constraints
The network layer operates above the data link layer. Different physical link technologies have radically different physical constraints on the maximum frame size they can encapsulate:
- **Ethernet (IEEE 802.3)**: Standard MTU = $1500\text{ bytes}$.
- **Wi-Fi (IEEE 802.11)**: MTU up to $2272\text{ bytes}$.
- **FDDI (Fiber Distributed Data Interface)**: MTU = $4352\text{ bytes}$.
- **Point-to-Point Protocol over Ethernet (PPPoE)**: MTU = $1492\text{ bytes}$.

When an IP router receives a datagram from an upstream link with a large MTU (e.g., FDDI with 4352 bytes) and must forward it across a link with a smaller MTU (e.g., Ethernet with 1500 bytes), the datagram cannot fit into a single link-layer frame. The router must divide the datagram into two or more smaller pieces called **fragments**.

```
[ Incoming Datagram: 4000 Bytes ]
              |
              v (Enters Router via High-MTU Interface)
      +---------------+
      |    ROUTER     | ===> MTU of Outgoing Link = 1500 Bytes
      +---------------+
              |
              +---> [ Fragment 1: 1500 B (20B Hdr + 1480B Data) | MF=1 | Offset=0   ]
              |
              +---> [ Fragment 2: 1500 B (20B Hdr + 1480B Data) | MF=1 | Offset=185 ]
              |
              +---> [ Fragment 3: 1040 B (20B Hdr + 1020B Data) | MF=0 | Offset=370 ]
```

#### Why Reassembly Occurs Exclusively at Destination Hosts
A defining architectural decision of IP is that **reassembly of fragments is performed strictly and exclusively at the final destination host**, never at intermediate core routers:

1. **Router Simplicity and Performance**: If intermediate routers were required to reassemble packets, they would have to buffer incoming fragments, maintain complex state tables for every fragmented flow, and run reassembly timers. A missing fragment would tie up router buffer memory, causing queue exhaustion and destroying line-speed performance.
2. **Dynamic Multipath Routing**: In a packet-switched network, different fragments of the exact same original datagram can travel along completely different physical routes due to dynamic routing changes or per-packet load balancing. An intermediate router might never see all the fragments!
3. **The End-to-End Argument**: If even a single fragment is lost or corrupted in transit, the entire datagram cannot be reassembled and is discarded by the destination host. The transport layer (e.g., TCP) must retransmit the entire segment anyway. Performing reassembly anywhere other than the end hosts violates Saltzer, Reed, and Clark's classic **End-to-End Argument in System Design**.

#### Fragmentation Mathematics & The 8-Byte Rule

To perform fragmentation correctly, follow this step-by-step mathematical algorithm:

1. **Header Size ($H$)**: Determine the header length. Unless options are specified, $H = 20\text{ bytes}$.
2. **Total Data Payload ($P$)**: Given original datagram length $L$:
   $$P = L - H$$
3. **Maximum Payload Per Fragment ($D_{\max}$)**:
   Given outgoing link MTU $M$, the maximum raw space for data is $M - H$. However, because Fragment Offset is measured in 8-byte units, the data length of every intermediate fragment **must be a multiple of 8**:
   $$D_{\max} = \left\lfloor \frac{M - H}{8} \right\rfloor \times 8$$
4. **Number of Fragments ($k$)**:
   $$k = \left\lceil \frac{P}{D_{\max}} \right\rceil$$
5. **Field Assignments for Fragment $i$ ($i = 0, 1, \dots, k-1$)**:
   - **Identification**: Identical to original datagram Identification.
   - **Data Payload ($D_i$)**: $D_i = D_{\max}$ for all fragments except the last, which carries the remaining bytes $P - (k - 1)D_{\max}$.
   - **Total Length ($L_i$)**: $L_i = H + D_i$.
   - **More Fragments (MF) Flag**:
     $$\text{MF} = \begin{cases} 1, & \text{if } i < k - 1 \text{ (intermediate fragments)} \\ 0, & \text{if } i = k - 1 \text{ (final fragment)} \end{cases}$$
   - **Fragment Offset**:
     $$\text{Offset}_i = \frac{i \times D_{\max}}{8}$$

---

#### Comprehensive Solved Numerical Examples

##### Solved Problem 1 (Canonical Slide Numerical):
**Scenario**: An IP datagram of $4{,}000\text{ bytes}$ (composed of a $20\text{-byte}$ header and $3{,}980\text{ bytes}$ of data) arrives at a router and must be forwarded over an outgoing link with an $\text{MTU} = 1{,}500\text{ bytes}$. The original datagram has $\text{Identification} = 777$, $\text{DF} = 0$, $\text{MF} = 0$, and $\text{Offset} = 0$. Determine the parameters of all generated fragments.

###### Step-by-Step Calculation:
1. Header length $H = 20\text{ bytes}$.
2. Total data payload $P = 4000 - 20 = 3980\text{ bytes}$.
3. Maximum data space per fragment:
   $$M - H = 1500 - 20 = 1480\text{ bytes}$$
   Check divisibility by 8: $\frac{1480}{8} = 185$ (an exact integer!). Thus, $D_{\max} = 1480\text{ bytes}$.
4. Number of fragments:
   $$k = \left\lceil \frac{3980}{1480} \right\rceil = \lceil 2.689 \rceil = 3\text{ fragments}$$
5. Partitioning data payload:
   - Fragment 1 data: $1480\text{ bytes}$ (Bytes 0 to 1479).
   - Fragment 2 data: $1480\text{ bytes}$ (Bytes 1480 to 2959).
   - Fragment 3 data: Remaining data $= 3980 - (1480 + 1480) = 3980 - 2960 = 1020\text{ bytes}$ (Bytes 2960 to 3979).  
     *(Note: 1020 is not divisible by 8, which is completely valid because MF = 0 for the final fragment!)*

###### Summary Table of Generated Fragments:

| Fragment | Total Length | Header Length | Payload Length | Identification | DF | MF | Fragment Offset | Original Byte Range |
|---|---|---|---|---|---|---|---|---|
| **Original** | $4000\text{ B}$ | $20\text{ B}$ | $3980\text{ B}$ | 777 | 0 | 0 | 0 | Bytes 0 — 3979 |
| **Fragment 1** | $1500\text{ B}$ | $20\text{ B}$ | $1480\text{ B}$ | 777 | 0 | 1 | $\frac{0}{8} = 0$ | Bytes 0 — 1479 |
| **Fragment 2** | $1500\text{ B}$ | $20\text{ B}$ | $1480\text{ B}$ | 777 | 0 | 1 | $\frac{1480}{8} = 185$ | Bytes 1480 — 2959 |
| **Fragment 3** | $1040\text{ B}$ | $20\text{ B}$ | $1020\text{ B}$ | 777 | 0 | 0 | $\frac{2960}{8} = 370$ | Bytes 2960 — 3979 |

---

##### Solved Problem 2 (Non-Multiple of 8 MTU & Sub-Fragmentation):
**Scenario**: An IP datagram of length $2{,}000\text{ bytes}$ ($20\text{-byte}$ header) with $\text{ID} = 999$ encounters a link with $\text{MTU} = 620\text{ bytes}$.
1. Find the fragment parameters.
2. Suppose Fragment 1 then encounters a subsequent downstream link with an $\text{MTU} = 300\text{ bytes}$. Show how Fragment 1 is sub-fragmented!

###### Part 1: Initial Fragmentation ($\text{MTU} = 620\text{ B}$)
- Data length $P = 2000 - 20 = 1980\text{ bytes}$.
- Raw payload limit $= 620 - 20 = 600\text{ bytes}$.
- Check divisibility by 8: $\frac{600}{8} = 75$ (exact multiple of 8). Thus, $D_{\max} = 600\text{ bytes}$.
- Total fragments: $\lceil \frac{1980}{600} \rceil = 4\text{ fragments}$.
  - Fragment 1: Total Length $= 620\text{ B}$ ($600\text{ B}$ data), $\text{MF} = 1$, $\text{Offset} = 0$. (Bytes 0 to 599).
  - Fragment 2: Total Length $= 620\text{ B}$ ($600\text{ B}$ data), $\text{MF} = 1$, $\text{Offset} = \frac{600}{8} = 75$. (Bytes 600 to 1199).
  - Fragment 3: Total Length $= 620\text{ B}$ ($600\text{ B}$ data), $\text{MF} = 1$, $\text{Offset} = \frac{1200}{8} = 150$. (Bytes 1200 to 1799).
  - Fragment 4: Total Length $= 200\text{ B}$ ($180\text{ B}$ data), $\text{MF} = 0$, $\text{Offset} = \frac{1800}{8} = 225$. (Bytes 1800 to 1979).

###### Part 2: Sub-Fragmentation of Fragment 1 ($\text{MTU} = 300\text{ B}$)
- Fragment 1 has Total Length $= 620\text{ B}$, Data $= 600\text{ B}$, $\text{MF} = 1$, $\text{Offset} = 0$.
- Available data space on new link $= 300 - 20 = 280\text{ bytes}$.
- Check divisibility: $\frac{280}{8} = 35$ (exact multiple). Thus $D_{\max}' = 280\text{ bytes}$.
- Sub-fragments generated from Fragment 1 ($600\text{ bytes}$ of data):
  - Sub-fragment 1a: Total Length $= 300\text{ B}$ ($280\text{ B}$ data), $\text{MF} = 1$, $\text{Offset} = 0$. (Carries bytes 0 to 279 of original datagram).
  - Sub-fragment 1b: Total Length $= 300\text{ B}$ ($280\text{ B}$ data), $\text{MF} = 1$, $\text{Offset} = \frac{280}{8} = 35$. (Carries bytes 280 to 559 of original datagram).
  - Sub-fragment 1c: Remaining data $= 600 - (280 + 280) = 40\text{ bytes}$. Total Length $= 60\text{ B}$ ($40\text{ B}$ data).
    - **Crucial Rule on MF**: The original Fragment 1 had $\text{MF} = 1$ because it was not the end of the original datagram! Therefore, Sub-fragment 1c **must inherit $\text{MF} = 1$**!
    - $\text{Offset} = \frac{560}{8} = 70$. (Carries bytes 560 to 599 of original datagram).

---

### 4.3 IPv4 Addressing, Subnets, and CIDR

#### Structure of an IPv4 Address
An IPv4 address is a **32-bit binary number** that uniquely identifies a specific network interface on a host or router. Addresses are written in human-readable **dotted-decimal notation**, where the 32 bits are grouped into four 8-bit octets separated by periods:
$$\underbrace{11000000}_{192} \cdot \underbrace{10101000}_{168} \cdot \underbrace{00000001}_{1} \cdot \underbrace{00001010}_{10} \implies 192.168.1.10$$

Every IP address is hierarchically partitioned into two components:
1. **Network Prefix (Network ID)**: The most significant bits, identifying the logical network to which the interface is attached.
2. **Host Identifier (Host ID)**: The remaining least significant bits, identifying the specific individual interface within that network.

#### Formal Definition of a Subnet
In networking terminology, an isolated physical segment of an IP network is called a **subnet**:
> **Formal Definition of a Subnet**: A collection of host and router interfaces that are interconnected via a physical medium (e.g., an Ethernet switch or hub) such that any interface can physically send a link-layer frame directly to any other interface in the group **without passing through an intervening Layer 3 router**.

```
  +--------------+               +--------------+
  | Host 1       |               | Host 2       |
  | 223.1.1.1/24 |               | 223.1.1.2/24 |
  +-------+------+               +------+-------+
          |                             |
          +--------------+--------------+
                         |
           ============================= (Ethernet Switch)
                         |
                  +------+-------+
                  | Router R1    |
                  | 223.1.1.4/24 |
                  +--------------+
               <--- SUBNET 1 --->
```

#### Subnet Masks, Subnetting, and Host Allocation
A **subnet mask** is a 32-bit binary mask consisting of contiguous `1`s followed by contiguous `0`s:
- The binary `1`s designate the network and subnet bits.
- The binary `0`s designate the host bits.
- Prefix notation: $/x$ indicates that the first $x$ bits are binary `1`s.

##### Formulas for Fixed-Length Subnetting:
Given an address block with prefix $/x$:
- Number of Host Bits: $h = 32 - x$
- Total Addresses in Block: $N_{\text{total}} = 2^h = 2^{32 - x}$
- Total Usable Host Addresses: $N_{\text{usable}} = 2^h - 2$  
  *(Subtract 2 because the all-zeros host address and all-ones host address are reserved)*
- If borrowing $k$ bits from the host part to create subnets:
  - New prefix length: $x' = x + k$
  - Number of created subnets: $N_{\text{subnets}} = 2^k$
  - Remaining host bits per subnet: $h' = h - k$
  - Usable hosts per subnet: $2^{h'} - 2$

#### Special-Use IPv4 Addresses

| Address Type | Binary Structure | Dotted Decimal Representation | Operational Purpose |
|---|---|---|---|
| **Network Address** | Host bits all `0`s | e.g., `192.168.1.0/24` | Identifies the subnet itself; cannot be assigned to any host interface. |
| **Directed Broadcast** | Host bits all `1`s | e.g., `192.168.1.255/24` | Transmits a packet to all hosts located on that specific destination subnet. |
| **Limited Broadcast** | All 32 bits are `1`s | `255.255.255.255` | Broadcasts to all devices on the local physical network. Routers **never forward** limited broadcasts. |
| **Loopback Address** | `01111111` in 1st octet | `127.0.0.0/8` (typically `127.0.0.1` — `localhost`) | Directs traffic internally within the local operating system without touching network hardware. |
| **Default / Unspecified**| All 32 bits are `0`s | `0.0.0.0/0` | Used as a source address by hosts during DHCP initialization; represents the default route in routing tables. |
| **Private IP: Class A** | `10.0.0.0/8` | `10.0.0.0` to `10.255.255.255` | RFC 1918 non-routable private addresses for internal enterprise networks ($16{,}777{,}216$ addresses). |
| **Private IP: Class B** | `172.16.0.0/12` | `172.16.0.0` to `172.31.255.255` | RFC 1918 private addresses (16 contiguous Class B networks: $1{,}048{,}576$ addresses). |
| **Private IP: Class C** | `192.168.0.0/16` | `192.168.0.0` to `192.168.255.255`| RFC 1918 private addresses (256 contiguous Class C networks: $65{,}536$ addresses). |
| **Link-Local (APIPA)** | `169.254.0.0/16` | `169.254.0.0` to `169.254.255.255`| Automatic Private IP Addressing; auto-assigned by operating systems when no DHCP server responds. |

#### Classful Addressing History and Its Collapse
From 1981 until 1993, the global Internet address space was rigidly divided into five classes based on the leading bits of the first octet:

```
Class A:  0 [ 7 bits Net ] [           24 bits Host ID          ]  Range: 1.0.0.0 - 126.255.255.255
Class B: 10 [  14 bits Net   ] [      16 bits Host ID       ]      Range: 128.0.0.0 - 191.255.255.255
Class C: 110 [     21 bits Net        ] [   8 bits Host ID  ]      Range: 192.0.0.0 - 223.255.255.255
Class D: 1110 [           28 bits Multicast Group ID        ]      Range: 224.0.0.0 - 239.255.255.255
Class E: 1111 [          28 bits Reserved / Experimental    ]      Range: 240.0.0.0 - 255.255.255.255
```

| Class | Leading Bits | 1st Octet Range | Default Subnet Mask | Total Networks | Total Usable Hosts per Network |
|---|---|---|---|---|---|
| **Class A** | `0` | $1\text{--}126$ | `255.0.0.0` ($/8$) | 126 | $2^{24} - 2 = 16{,}777{,}214$ |
| **Class B** | `10` | $128\text{--}191$ | `255.255.0.0` ($/16$) | 16,384 | $2^{16} - 2 = 65{,}534$ |
| **Class C** | `110` | $192\text{--}223$ | `255.255.255.0` ($/24$) | 2,097,152 | $2^8 - 2 = 254$ |
| **Class D** | `1110` | $224\text{--}239$ | Undefined (Multicast) | N/A | Multicast Groups (No Host/Net separation) |
| **Class E** | `1111` | $240\text{--}255$ | Undefined (Reserved) | N/A | Reserved for experimental research |

##### Fatal Flaws of Classful Addressing:
1. **Severe Address Granularity Mismatch**: The allocation jump from Class C (254 hosts) to Class B (65,534 hosts) was immense. An enterprise with 500 computers found Class C too small. Consequently, the Internet Assigned Numbers Authority (IANA) was forced to allocate a full Class B network of 65,534 addresses to an organization requiring only 500, wasting over $65{,}000$ globally unique IP addresses per allocation!
2. **Rapid Exhaustion of Class B Space**: By 1992, available Class B address space was on the brink of total depletion.
3. **Routing Table Explosion**: If organizations were given multiple Class C blocks instead of a Class B, global Internet core routers had to store an independent forwarding table entry for every single Class C block, threatening to overwhelm router memory and CPU capacity.

#### Classless Inter-Domain Routing (CIDR — RFC 1519)
Introduced in 1993, **Classless Inter-Domain Routing (CIDR)** abolished class distinctions entirely:
- Network boundaries can be established at any arbitrary bit position ($/0$ to $/32$).
- Network allocations are specified as $a.b.c.d/x$, where $/x$ is the exact length of the network prefix.
- Address space is allocated in exact powers of two ($2^{32-x}$), tailoring allocations precisely to organizational requirements.

---

### Interactive Exercise Problems (PES University Slides & Exams)

The following exercises are transcribed directly from the interactive exercise curriculum and examinations of PES University:

---

#### Problem 1: Subnet Range and Boundary Calculations
**Question**: Find the range of addresses, network address, directed broadcast address, and number of valid hosts for the following three CIDR blocks:
1. `123.56.77.32/29`
2. `180.34.64.64/30`
3. `200.17.21.128/27`

##### Solution to 1.1 (`123.56.77.32/29`):
- **Subnet Mask**: Prefix $/29$ means 29 leading `1`s:
  `11111111.11111111.11111111.11111000` $\implies 255.255.255.248$
- **Host Bits**: $h = 32 - 29 = 3\text{ bits}$.
- **Block Size (Total Addresses)**: $2^3 = 8\text{ addresses}$.
- **Valid Usable Hosts**: $2^3 - 2 = 6\text{ hosts}$.
- **Analysis of 4th Octet**:
  The base decimal value is 32: $00100000_2$.
  Host bits are the 3 lowest bits (`***`):
  - Lowest host value (`000`): $32 + 0 = 32 \implies$ **Network Address**: `123.56.77.32`
  - Highest host value (`111`): $32 + 7 = 39 \implies$ **Broadcast Address**: `123.56.77.39`
- **Total Address Range**: `123.56.77.32` to `123.56.77.39/29`
- **Valid Host Address Range**: `123.56.77.33` to `123.56.77.38/29`

##### Solution to 1.2 (`180.34.64.64/30`):
- **Subnet Mask**: Prefix $/30$ means 30 leading `1`s:
  `11111111.11111111.11111111.11111100` $\implies 255.255.255.252$
- **Host Bits**: $h = 32 - 30 = 2\text{ bits}$.
- **Total Addresses**: $2^2 = 4\text{ addresses}$.
- **Valid Usable Hosts**: $2^2 - 2 = 2\text{ hosts}$ *(The standard size for point-to-point router links!)*.
- **Analysis of 4th Octet**:
  Base value 64: $01000000_2$. Host bits are the 2 lowest bits (`**`):
  - Lowest host value (`00`): $64 + 0 = 64 \implies$ **Network Address**: `180.34.64.64`
  - Highest host value (`11`): $64 + 3 = 67 \implies$ **Broadcast Address**: `180.34.64.67`
- **Total Address Range**: `180.34.64.64` to `180.34.64.67/30`
- **Valid Host Address Range**: `180.34.64.65` to `180.34.64.66/30`

##### Solution to 1.3 (`200.17.21.128/27`):
- **Subnet Mask**: Prefix $/27$ means 27 leading `1`s:
  `11111111.11111111.11111111.11100000` $\implies 255.255.255.224$
- **Host Bits**: $h = 32 - 27 = 5\text{ bits}$.
- **Total Addresses**: $2^5 = 32\text{ addresses}$.
- **Valid Usable Hosts**: $2^5 - 2 = 30\text{ hosts}$.
- **Analysis of 4th Octet**:
  Base value 128: $10000000_2$. Host bits are 5 lowest bits (`*****`):
  - Lowest host value (`00000`): $128 + 0 = 128 \implies$ **Network Address**: `200.17.21.128`
  - Highest host value (`11111`): $128 + 31 = 159 \implies$ **Broadcast Address**: `200.17.21.159`
- **Total Address Range**: `200.17.21.128` to `200.17.21.159/27`
- **Valid Host Address Range**: `200.17.21.129` to `200.17.21.158/27`

---

#### Problem 2: Fixed-Length Subnetting of a Class B Address Block
**Question**: You are given the network address `175.200.0.0/16`. You are required to create exactly **4 subnets**.
1. What is the minimum number of host bits you must borrow into the network prefix?
2. Write down the subnet mask for the network.
3. How many valid hosts can each subnet support?
4. Write down the network addresses of all 4 subnets.

##### Solution:
1. **Bits to Borrow ($k$)**:
   To create 4 subnets: $2^k \ge 4 \implies k = 2\text{ bits}$.
2. **Subnet Mask**:
   Original prefix was $/16$. Borrowing 2 bits yields a new prefix of $16 + 2 = /18$.
   In binary: `11111111.11111111.11000000.00000000`
   In decimal: **`255.255.192.0`**
3. **Valid Hosts per Subnet**:
   Remaining host bits: $h' = 32 - 18 = 14\text{ bits}$.
   Usable hosts $= 2^{14} - 2 = 16{,}384 - 2 = \mathbf{16{,}382\text{ valid hosts per subnet}}$.
4. **Addresses of the 4 Subnets**:
   The two borrowed bits reside in the 3rd octet. Their binary combinations are `00`, `01`, `10`, and `11`:
   - Subnet 0 (`00`): $00000000_2 = 0 \implies \mathbf{175.200.0.0/18}$ (Range: `175.200.0.0` — `175.200.63.255`)
   - Subnet 1 (`01`): $01000000_2 = 64 \implies \mathbf{175.200.64.0/18}$ (Range: `175.200.64.0` — `175.200.127.255`)
   - Subnet 2 (`10`): $10000000_2 = 128 \implies \mathbf{175.200.128.0/18}$ (Range: `175.200.128.0` — `175.200.191.255`)
   - Subnet 3 (`11`): $11000000_2 = 192 \implies \mathbf{175.200.192.0/18}$ (Range: `175.200.192.0` — `175.200.255.255`)

---

#### Problem 3: Determining Network and Broadcast from a Single Host IP
**Question**: In a block of addresses, an administrator knows the IP address of one active host interface is `25.34.12.56/16`. Find the network address, the directed broadcast address, and the valid host range of this block. Show full binary calculation.

##### Solution:
- **Given Host IP**: `25.34.12.56`
- **Subnet Mask**: $/16 \implies 255.255.0.0$
- **Binary Bitwise AND Operation (Network Address)**:
  ```
  Host IP:      00011001 . 00100010 . 00001100 . 00111000  (25.34.12.56)
  Subnet Mask:  11111111 . 11111111 . 00000000 . 00000000  (255.255.0.0)
  --------------------------------------------------------
  Bitwise AND:  00011001 . 00100010 . 00000000 . 00000000  (25.34.0.0)
  ```
  $\implies$ **Network Address**: **`25.34.0.0/16`**
- **Directed Broadcast Address Calculation**:
  Set all 16 host bits to binary `1`s:
  ```
  Network Bits: 00011001 . 00100010 . 
  Host Bits:                          11111111 . 11111111  (.255.255)
  --------------------------------------------------------
  Broadcast:    00011001 . 00100010 . 11111111 . 11111111  (25.34.255.255)
  ```
  $\implies$ **Broadcast Address**: **`25.34.255.255`**
- **Valid Host Range**:
  - First Valid Host: `25.34.0.1`
  - Last Valid Host: `25.34.255.254`

---

#### Problem 4: Creating 500 Subnets from a Class A Block
**Question**: An organization is granted the address block `16.0.0.0/8`. The network administrator needs to create **500 fixed-length subnets**.
1. Find the required subnet mask.
2. Find the total number of addresses in each subnet.
3. Find the first address (network address) and last address (broadcast address) in:
   - The first subnet (Subnet 1 / Index 0).
   - The 500th subnet (Subnet 500 / Index 499).

##### Solution:
1. **Subnet Mask**:
   We require 500 subnets. Find integer $k$ such that:
   $$2^k \ge 500 \implies k = 9\text{ bits} \quad (2^8 = 256 < 500, \text{ and } 2^9 = 512 \ge 500)$$
   We must borrow 9 bits from the host field.
   New prefix length: $8 + 9 = /17$.
   In binary: `11111111.11111111.10000000.00000000`
   Subnet Mask: **`255.255.128.0`**
2. **Total Addresses per Subnet**:
   Remaining host bits: $h' = 32 - 17 = 15\text{ bits}$.
   Total addresses per subnet $= 2^{15} = \mathbf{32{,}768\text{ addresses}}$ (with $32{,}766$ usable hosts).
3. **First Subnet (Index 0)**:
   The 9 borrowed subnet bits are all `0`:
   - Subnet 0: $\mathbf{16.0.0.0/17}$
   - Broadcast Address: $16.0.0.0 + 32{,}767 \implies \mathbf{16.0.127.255}$
   - Address Range: `16.0.0.0` to `16.0.127.255`
4. **500th Subnet (Index 499)**:
   We must express decimal 499 in 9-bit binary:
   $$499 = 256 + 128 + 64 + 32 + 16 + 2 + 1 = 111110011_2$$
   Map these 9 subnet bits into the 2nd octet (8 bits) and 3rd octet (1 bit):
   - 2nd Octet (first 8 bits): $11111001_2 = 249$
   - 3rd Octet (9th bit followed by seven `0`s): $10000000_2 = 128$
   - Network Address of 500th Subnet: **`16.249.128.0/17`**
   - Broadcast Address of 500th Subnet:
     Set all 15 host bits to `1`s (3rd octet becomes $11111111_2 = 255$, 4th octet becomes 255):
     Broadcast Address: **`16.249.255.255`**
   - Address Range: `16.249.128.0` to `16.249.255.255`

---

#### Problem 5: Subnet Analysis of Class B Address with Non-Trivial Mask
**Question**: Find the network address and directed broadcast address of the subnetted IPv4 address `172.25.171.182` configured with a subnet mask of `255.255.224.0`.

##### Solution:
- **Inspect 3rd Octet**:
  Subnet mask 3rd octet $= 224 = 11100000_2$ (3 subnet bits, 5 host bits).
  This corresponds to a $/19$ prefix ($8 + 8 + 3 = 19$).
  Host IP 3rd octet $= 171 = 10101011_2$.
- **Perform Bitwise AND on 3rd Octet**:
  ```
  Host 3rd Octet: 10101011  (171)
  Mask 3rd Octet: 11100000  (224)
  ------------------------
  Result:         10100000  (128 + 32 = 160)
  ```
  4th octet mask is `0`, so 4th octet of network address is `0`.
  $\implies$ **Network Address**: **`172.25.160.0/19`**
- **Directed Broadcast Address**:
  Set the 5 host bits in 3rd octet to `1`:
  $10111111_2 = 160 + 31 = 191$.
  Set all 8 host bits in 4th octet to `1`: $255$.
  $\implies$ **Directed Broadcast Address**: **`172.25.191.255`**
- **Valid Host Range**: `172.25.160.1` to `172.25.191.254`

---

#### Problem 6: Variable Length Subnet Masking (VLSM) Design
**Question**: An organization is granted a CIDR block of addresses with the starting prefix `14.24.74.0/24`. The organization must design three subblocks of addresses for three separate physical subnets:
- Subnet A: Requires **120 addresses**
- Subnet B: Requires **60 addresses**
- Subnet C: Requires **10 addresses**

Design the subnets, specifying for each: subnet mask, allocated address block, network address, usable host range, broadcast address, and unused addresses.

##### Design Strategy (Largest-First Allocation):
When allocating subnets with Variable Length Subnet Masking (VLSM), **always allocate blocks in descending order of size** to ensure proper binary boundary alignment.

###### 1. Allocate Subnet A (120 addresses):
- Required: 120 addresses.
- Find $h_A$ such that $2^{h_A} \ge 120 \implies h_A = 7\text{ host bits}$ ($2^7 = 128\text{ addresses}$).
- Subnet Prefix: $32 - 7 = /25$.
- Subnet Mask: `255.255.255.128`.
- Allocate starting at base: **`14.24.74.0/25`**
  - Network Address: `14.24.74.0`
  - Broadcast Address: `14.24.74.127`
  - Usable Host Range: `14.24.74.1` to `14.24.74.126` (126 usable hosts).

###### 2. Allocate Subnet B (60 addresses):
- Next available address starts at: `14.24.74.128`.
- Required: 60 addresses.
- Find $h_B$ such that $2^{h_B} \ge 60 \implies h_B = 6\text{ host bits}$ ($2^6 = 64\text{ addresses}$).
- Subnet Prefix: $32 - 6 = /26$.
- Subnet Mask: `255.255.255.192`.
- Allocate: **`14.24.74.128/26`**
  - Network Address: `14.24.74.128`
  - Broadcast Address: $128 + 63 = 191 \implies$ `14.24.74.191`
  - Usable Host Range: `14.24.74.129` to `14.24.74.190` (62 usable hosts).

###### 3. Allocate Subnet C (10 addresses):
- Next available address starts at: `14.24.74.192`.
- Required: 10 addresses.
- Find $h_C$ such that $2^{h_C} \ge 10 \implies h_C = 4\text{ host bits}$ ($2^4 = 16\text{ addresses}$, since $2^3 = 8 < 10$).
- Subnet Prefix: $32 - 4 = /28$.
- Subnet Mask: `255.255.255.240`.
- Allocate: **`14.24.74.192/28`**
  - Network Address: `14.24.74.192`
  - Broadcast Address: $192 + 15 = 207 \implies$ `14.24.74.207`
  - Usable Host Range: `14.24.74.193` to `14.24.74.206` (14 usable hosts).

###### 4. Remaining Unallocated Address Space:
- Addresses from `14.24.74.208` to `14.24.74.255` ($48\text{ addresses}$) remain completely free for future corporate growth.

---

### 4.4 Hierarchical Addressing and Route Aggregation (Supernetting)

#### Principles of Route Aggregation
The primary scalability triumph of CIDR is its ability to perform **hierarchical route aggregation** (commonly referred to as **supernetting**):
- An Internet Service Provider (ISP) is assigned a single large CIDR block of contiguous addresses (e.g., a $/20$ prefix).
- The ISP subdivides this block into smaller subnets (e.g., eight $/23$ or sixteen $/24$ subnets) and assigns them to distinct corporate customers.
- When advertising routes to global Tier-1 upstream providers (via BGP), the ISP **does not advertise 16 separate $/24$ routes**. Instead, it advertises a **single aggregated $/20$ prefix**.
- Routers across the worldwide Internet backbone only need a single forwarding table entry for the aggregated prefix, drastically shrinking global routing table size!

```
                    +------------------------------------+
                    |        GLOBAL INTERNET CORE        |
                    | (Sees ONLY single route:           |
                    |  "Send 200.23.16.0/20 to ISP")    |
                    +-----------------+------------------+
                                      |
                           Single Advertised Prefix:
                               200.23.16.0/20
                                      |
                                      v
                    +------------------------------------+
                    |       REGIONAL ISP ROUTER          |
                    | (Maintains internal routes for     |
                    |  individual customer subnets)      |
                    +--+---------+----------+---------+--+
                       |         |          |         |
                       v         v          v         v
                   Org 0     Org 1      Org 2      Org 7
                .16.0/24  .17.0/24   .18.0/24   .23.0/24
```

#### Supernetting Numerical & Binary Alignment

Suppose an ISP supports eight independent customer organizations, allocating each a contiguous $/24$ network:

```
Organization 0: 200.23.16.0/24  ===>  11001000.00010111.00010 000.********
Organization 1: 200.23.17.0/24  ===>  11001000.00010111.00010 001.********
Organization 2: 200.23.18.0/24  ===>  11001000.00010111.00010 010.********
Organization 3: 200.23.19.0/24  ===>  11001000.00010111.00010 011.********
Organization 4: 200.23.20.0/24  ===>  11001000.00010111.00010 100.********
Organization 5: 200.23.21.0/24  ===>  11001000.00010111.00010 101.********
Organization 6: 200.23.22.0/24  ===>  11001000.00010111.00010 110.********
Organization 7: 200.23.23.0/24  ===>  11001000.00010111.00010 111.********
```

##### Aggregation Math:
Notice that across all 8 customer blocks, the first **21 bits are completely identical**:
$$\underbrace{11001000.00010111.00010}_{21\text{ bits}} \, \underbrace{xxx.********}_{11\text{ bits}}$$
- The ISP aggregates these 8 individual $/24$ routes into a single supernet prefix:
  $$\mathbf{200.23.16.0/21}$$
- The global Internet core router stores only **1 forwarding entry** instead of 8.

#### More Specific Routing and Multihoming

What happens if Organization 1 (`200.23.17.0/24`) changes providers and switches from the original ISP (Fly-By-Night ISP) to a new Tier-1 ISP (Telekom)?
- Must Organization 1 renumber all its servers, internal routers, and host computers? **No!**
- **How Longest Prefix Matching Resolves the Conflict**:
  1. Fly-By-Night ISP continues to advertise its aggregated block: `200.23.16.0/21` to the Internet backbone.
  2. Telekom advertises a specific route for Organization 1: `200.23.17.0/24` to the Internet backbone.
  3. When an Internet core router receives a packet destined for `200.23.17.50`:
     - It matches Fly-By-Night ISP's route: `200.23.16.0/21` (21-bit match).
     - It matches Telekom's route: `200.23.17.0/24` (24-bit match).
  4. Under **Longest Prefix Matching**, $24 > 21$. The core router forwards the packet to **Telekom**!
  5. Packets for any of the other 7 organizations (e.g., `200.23.16.1` or `200.23.18.1`) match only the `/21` prefix and are correctly routed to Fly-By-Night ISP!

---

### 4.5 IPv6 Transition and Tunneling

#### IPv6 Architecture Overview
The rapid depletion of the 32-bit IPv4 address pool ($2^{32} \approx 4.29 \times 10^9$ addresses) prompted the IETF to standardize **IPv6 (RFC 8200)**:
- **128-Bit Address Space**: $2^{128} \approx 3.4 \times 10^{38}$ globally unique addresses (over $5 \times 10^{28}$ addresses for every human on Earth).
- **Streamlined 40-Byte Fixed Base Header**: By fixing the header size to exactly 40 bytes, hardware processing in router ASICs is vastly accelerated.
- **Elimination of Router Fragmentation**: Intermediate routers **never fragment** an IPv6 datagram. If a packet exceeds link MTU, the router drops it and returns an ICMPv6 *Packet Too Big* message. The sending host must perform Path MTU Discovery (PMTUD) or fragment at the source.
- **Removal of Header Checksum**: Eliminates the overhead of recomputing checksums at every hop, relying on Layer 2 CRCs and Layer 4 (TCP/UDP) checksums.
- **Next Header Field**: Handles optional network-layer capabilities (hop-by-hop options, routing, security) through modular daisy-chained extension headers.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version| Traffic Class |           Flow Label                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Payload Length        |  Next Header  |   Hop Limit   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                                                               +
|                                                               |
+                     Source IP Address                         +
|                         (128 bits)                            |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                                                               +
|                                                               |
+                  Destination IP Address                       +
|                         (128 bits)                            |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Dual-Stack Transition Architecture
Because it is impossible to shut down the entire global Internet for a single "flag day" upgrade, networks operate in transition:
- **Dual-Stack Nodes**: Hosts and routers run both IPv4 and IPv6 protocol stacks simultaneously. They maintain both IPv4 and IPv6 addresses.
- **DNS Resolution**: When querying a domain name, a dual-stack host asks for both IPv4 address records (DNS A records) and IPv6 address records (DNS AAAA records), preferring IPv6 if available.

#### IPv6 Tunneling Over IPv4 Infrastructure (6in4 / Protocol 41)
When two modern IPv6-enabled networks must communicate across an intermediate transit network composed of legacy IPv4-only routers, the networks employ **IPv6 Tunneling**:
- The tunnel entrance router encapsulates the **entire, complete IPv6 datagram as the payload** inside a standard IPv4 datagram.
- The outer IPv4 header specifies:
  - Source IP: IPv4 address of the tunnel entrance router.
  - Destination IP: IPv4 address of the tunnel exit router.
  - Protocol Field: Set to decimal **41** (which explicitly denotes an encapsulated IPv6 payload).
- Intermediate legacy IPv4 routers process only the outer IPv4 header, completely unaware that the payload contains an IPv6 packet.
- Upon reaching the tunnel exit router, the IPv4 header is stripped off (decapsulation), and the original IPv6 datagram is forwarded into the destination IPv6 network.

```
IPv6 Domain                               Legacy IPv4 Transit Network                              IPv6 Domain
+-----------+       +-------------------+                             +-------------------+       +-----------+
| Host Src  | ====> | Router E          | ==========================> | Router B          | ====> | Host Dest |
|  (IPv6)   |       | (Tunnel Entrance) |   (Encapsulated in IPv4)    | (Tunnel Exit)     |       |  (IPv6)   |
+-----------+       +-------------------+                             +-------------------+       +-----------+
  Native IPv6             Encapsulates:                                     Decapsulates:           Native IPv6
  Datagram                [IPv4 Hdr][IPv6 Datagram]                         Strips IPv4 Hdr         Datagram
```

---

#### Comprehensive Interactive Exercise Walkthrough

The following problem is drawn directly from the PES University CN Interactive Exercise curriculum (Kurose & Ross interactive exercise suite):

##### Scenario Description:
Suppose a host on **Subnet D** (an IPv6 network) wants to send an IPv6 datagram to a host on **Subnet B** (an IPv6 network). The end-to-end path traverses both IPv6-capable and legacy IPv4 routers along the forwarding path:
$$\text{Host on Subnet D} \longrightarrow \text{Router E} \longrightarrow \text{Router d} \longrightarrow \text{Router b} \longrightarrow \text{Router c} \longrightarrow \text{Router B} \longrightarrow \text{Host on Subnet B}$$
- **Subnet D and Router E**: IPv6-capable.
- **Routers d, b, and c**: Legacy IPv4-only transit routers.
- **Router B and Subnet B**: IPv6-capable.

**Network Node Addresses**:
- Source Host (on Subnet D) IPv6 Address: `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8`
- Destination Host (on Subnet B) IPv6 Address: `7361:4A69:9553:D483:D2B6:97FC:E945:33CC`
- Router E (Tunnel Entrance) IPv4 Interface Address: `94.100.4.145`
- Router B (Tunnel Exit) IPv4 Interface Address: `130.73.216.43`

---

##### Detailed Hop-by-Hop Trace and Analysis Table:

| Hop Segment | Physical Path | Datagram Protocol Type | Outer Header Source IP | Outer Header Destination IP | Is Datagram Encapsulated? | Inner Encapsulated Source IP | Inner Encapsulated Destination IP |
|---|---|---|---|---|---|---|---|
| **Hop 1** | Subnet D $\to$ Router E | **IPv6** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` | **No** | N/A (Native Datagram) | N/A (Native Datagram) |
| **Hop 2** | Router E $\to$ Router d | **IPv4** | `94.100.4.145` | `130.73.216.43` | **Yes** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` |
| **Hop 3** | Router d $\to$ Router b | **IPv4** | `94.100.4.145` | `130.73.216.43` | **Yes** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` |
| **Hop 4** | Router b $\to$ Router c | **IPv4** | `94.100.4.145` | `130.73.216.43` | **Yes** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` |
| **Hop 5** | Router c $\to$ Router B | **IPv4** | `94.100.4.145` | `130.73.216.43` | **Yes** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` |
| **Hop 6** | Router B $\to$ Subnet B | **IPv6** | `6CED:EE97:997A:35F2:8EF9:5F8E:DDAF:7DD8` | `7361:4A69:9553:D483:D2B6:97FC:E945:33CC` | **No** | N/A (Decapsulated) | N/A (Decapsulated) |

---

##### Key Conceptual Exam Questions & Explicit Answers:

1. **Which router serves as the 'tunnel entrance'?**  
   **Answer**: **Router E**. Router E detects that the outgoing path traverses an IPv4-only transit network, creates an outer IPv4 header with $\text{Protocol} = 41$, and encapsulates the IPv6 datagram.
2. **Which router serves as the 'tunnel exit'?**  
   **Answer**: **Router B**. Router B receives the IPv4 datagram addressed to its IPv4 interface `130.73.216.43`, inspects $\text{Protocol} = 41$, strips off the IPv4 header, and exposes the original IPv6 datagram.
3. **Which protocol encapsulates the other: IPv4 or IPv6?**  
   **Answer**: **IPv4 encapsulates IPv6**. To ensure compatibility across legacy IPv4 transit networks, the complete IPv6 datagram is placed inside the payload field of an IPv4 datagram.
4. **Do intermediate routers d, b, and c modify the encapsulated IPv6 packet?**  
   **Answer**: **No**. Intermediate routers d, b, and c treat the IPv4 datagram as standard payload. They inspect only the outer IPv4 header, decrement the outer IPv4 TTL, recompute the outer IPv4 header checksum, and forward the packet based on destination `130.73.216.43`. The inner IPv6 header and payload remain entirely untouched until decapsulation at Router B.

---

## 5. Network Address Translation (NAT)

### 5.1 Motivation and Architecture

#### The IPv4 Address Exhaustion Crisis
The Internet Protocol version 4 (IPv4) architecture, formalized in Request for Comments (RFC) 791 in 1981, defines a 32-bit address space. This provides an absolute theoretical limit of:
$$N_{\text{total}} = 2^{32} = 4,294,967,296 \text{ distinct IP addresses}$$

In the formative decades of the Internet, IP addresses were assigned using rigid **classful addressing** (Class A, Class B, and Class C):
- A **Class A** network allocated a `/8` prefix ($2^{24} = 16,777,216$ host addresses) to a single organization.
- A **Class B** network allocated a `/16` prefix ($2^{16} = 65,536$ host addresses).
- A **Class C** network allocated a `/24` prefix ($2^8 = 256$ host addresses).

This allocation strategy was notoriously inefficient. An organization requiring 3,000 addresses had to be granted an entire Class B block, wasting over 62,000 addresses. As personal computers, residential broadband (DSL and cable modems), corporate intranets, mobile smartphones, and Internet of Things (IoT) devices proliferated, the pool of unallocated 32-bit addresses vanished at an exponential rate.

On **3 February 2011**, the Internet Assigned Numbers Authority (IANA) depleted its global unallocated pool, delegating its final five `/8` blocks equally to the five Regional Internet Registries (RIRs):
1. **ARIN** (American Registry for Internet Numbers) — North America
2. **RIPE NCC** (Réseaux IP Européens Network Coordination Centre) — Europe, the Middle East, and parts of Central Asia
3. **APNIC** (Asia-Pacific Network Information Centre) — Asia-Pacific region
4. **LACNIC** (Latin America and Caribbean Network Information Centre) — Latin America and the Caribbean
5. **AFRINIC** (African Network Information Centre) — Africa

While Internet Protocol version 6 (IPv6) was engineered as the definitive, permanent replacement—boasting a 128-bit address space ($2^{128} \approx 3.4 \times 10^{38}$ unique addresses)—the massive installed base of legacy IPv4 infrastructure made global transition slow, costly, and complex. **Network Address Translation (NAT)**, standardized in **RFC 1631** and revised in **RFC 3022**, emerged as the single most critical bridge technology that extended the functional lifetime of IPv4 by decades.

---

#### Private IP Address Blocks (RFC 1918)
To prevent organizations from burning globally routable public IP addresses for internal nodes that do not require direct public addressability, the Internet Engineering Task Force (IETF) reserved three dedicated IPv4 address blocks in **RFC 1918 (Address Allocation for Private Internets)**:

| Class Equivalent | CIDR Prefix Block | IPv4 Address Range | Total Addresses ($2^N$) | Intended Scope & Deployment Scale |
| :---: | :---: | :---: | :---: | :--- |
| **Class A** | `10.0.0.0/8` | `10.0.0.0` – `10.255.255.255` | $2^{24} = 16,777,216$ | Massive enterprise networks, global data centers, cloud Virtual Private Clouds (VPCs) |
| **Class B** | `172.16.0.0/12` | `172.16.0.0` – `172.31.255.255` | $2^{20} = 1,048,576$ | Large university campuses, corporate division campuses |
| **Class C** | `192.168.0.0/16` | `192.168.0.0` – `192.168.255.255` | $2^{16} = 65,536$ | Small Office / Home Office (SOHO), residential Wi-Fi local networks |

##### Core Properties of RFC 1918 Private Addresses:
1. **Non-Routable Across the Global Internet Backbone**: Core Internet transit routers are configured by default to immediately drop any packet with a destination or source address inside these private ranges.
2. **Infinite Global Reusability**: Because these addresses possess meaning only within their respective local networks, millions of homes and businesses across the planet simultaneously use `192.168.1.1` or `10.0.0.1` concurrently without address collisions.
3. **Internal Privacy**: Internal network topology and internal node counts are completely hidden from the public Internet.

---

#### Unified NAT Edge Gateway Architecture
A NAT-enabled edge router sits at the physical boundary between a private Local Area Network (LAN) and the public Internet (Wide Area Network - WAN).

```
   +-----------------------------------------------------------+
   |             PRIVATE LOCAL NETWORK (10.0.0.0/24)           |
   |                                                           |
   |  Host A: 10.0.0.1 -------+                                |
   |                          |                                |
   |  Host B: 10.0.0.2 -------+--- [ LAN Switch ]              |
   |                          |         |                      |
   |  Host C: 10.0.0.3 -------+         |                      |
   +------------------------------------|----------------------+
                                        | LAN Interface (10.0.0.4)
                              +--------------------+
                              |     NAT ROUTER     |
                              +--------------------+
                                        | WAN Interface (138.76.29.7)
   +------------------------------------|----------------------+
   |                                    |                      |
   |                              [ ISP Router ]               |
   |                                    |                      |
   |                             PUBLIC INTERNET               |
   |                                    |                      |
   |                     Web Server: 128.119.40.186:80         |
   +-----------------------------------------------------------+
```

##### Architectural Semantics (Kurose & Ross Model):
- **Inside the Private Network**: Every device receives a 32-bit IP address from the private address space (typically via the Dynamic Host Configuration Protocol - DHCP running on the LAN side of the router). Packets traveling strictly between internal hosts use ordinary `10.0.0.0/24` routing.
- **To the Outside World**: The NAT-enabled router hides the entire local network behind a **single, globally unique, publicly routable IPv4 address** (`138.76.29.7`) assigned to its WAN interface by the Internet Service Provider (ISP). To external servers, the entire internal network behaves as if it were a single computer.
- **Operational Benefits**:
  - **Single IP Requirement**: An enterprise with 50,000 internal computers requires only one public IP address from its ISP.
  - **Local Subnet Independence**: Internal network administrators can change internal IP assignments, add new subnets, or redesign network topology arbitrarily without notifying the ISP or registering with IANA.
  - **Zero-Disruption ISP Migration**: Changing ISPs requires reconfiguring only the router's single WAN-facing IP address; none of the internal host configurations need to change.
  - **Inherent Perimeter Ingress Protection**: Because private addresses are unreachable from the outside, external attackers cannot directly scan or send unsolicited packets to internal hosts.

---

### 5.2 NAT Router Operation & The NAT Translation Table

#### NAPT (Network Address Port Translation / PAT)
Basic NAT provides a one-to-one translation between private and public IP addresses from a finite pool. However, residential broadband and modern enterprises typically possess only **one** public IP address. To multiplex thousands of simultaneous outbound connections over a single IP, the router implements **Network Address Port Translation (NAPT)**, often termed **Port Address Translation (PAT)** or **IP Masquerading**.

NAPT achieves this by leveraging the 16-bit source port numbers located in the Layer 4 (Transport Layer) Transmission Control Protocol (TCP) and User Datagram Protocol (UDP) headers to track and demultiplex returning traffic.

##### 16-Bit Transport Port Multiplexing Capacity:
The port number field in TCP and UDP headers is 16 bits wide:
$$\text{Port Space} = 2^{16} = 65,536 \text{ ports (Range: } 0 \text{ to } 65,535\text{)}$$

- Ports $0$ to $1023$ are reserved well-known server ports (e.g., HTTP 80, HTTPS 443, DNS 53, SSH 22).
- Ports $1024$ to $65,535$ ($64,512$ ports) are dynamic ephemeral ports available for outbound NAT allocation.
- Because TCP and UDP maintain distinct protocol spaces, a single public IPv4 address can theoretically support:
  $$N_{\text{sessions}} = 64,512 \times 2 = 129,024 \text{ concurrent bidirectional flows!}$$

---

#### Deep-Dive Packet Rewriting Engine

##### 1. Outgoing Packet Rewriting (LAN $\to$ WAN):
When internal host `10.0.0.1` initiates an HTTP request to web server `128.119.40.186:80` using local ephemeral port `3345`:
1. **Packet Capture**: The NAT router intercepts the outbound datagram on its internal LAN interface (`eth1`).
2. **Flow Identification**: The router extracts the 5-tuple: `(Protocol: TCP, Src IP: 10.0.0.1, Src Port: 3345, Dst IP: 128.119.40.186, Dst Port: 80)`.
3. **Table Allocation**: If no mapping exists, the router selects an unused port from its public WAN interface (e.g., `5001`) and creates an entry in its stateful **NAT Translation Table**:
   $$\text{WAN Side: } (138.76.29.7, 5001) \longleftrightarrow \text{LAN Side: } (10.0.0.1, 3345)$$
4. **Header Rewriting**:
   - In the **IPv4 Header**: The Source IP is rewritten from `10.0.0.1` to `138.76.29.7`.
   - In the **TCP Header**: The Source Port is rewritten from `3345` to `5001`.
5. **Checksum Recalculation (Critical Step)**:
   - **IPv4 Header Checksum**: Because the Source IP address was altered, the router recomputes the 16-bit one's complement IPv4 header checksum.
   - **TCP/UDP Checksum**: The Layer 4 TCP/UDP checksum algorithm includes an **IPv4 Pseudo-Header** composed of:
     - Source IPv4 Address (32 bits)
     - Destination IPv4 Address (32 bits)
     - Reserved Zero byte (8 bits)
     - Protocol Number (8 bits: 6 for TCP, 17 for UDP)
     - TCP/UDP Segment Length (16 bits)
   - Because the Source IP was modified, the original TCP checksum is invalidated! The NAT router must calculate a new 16-bit one's complement sum across the modified pseudo-header, TCP header, and TCP payload before transmitting the datagram onto the WAN link (`eth0`).

##### 2. Incoming Packet Rewriting (WAN $\to$ LAN):
When external web server `128.119.40.186` responds:
1. **Arrival**: The datagram arrives at the router's WAN interface with Destination IP `138.76.29.7` and Destination Port `5001`.
2. **Table Indexing**: The router extracts Destination Port `5001` and queries its NAT Translation Table.
3. **Mapping Resolution**: Port `5001` matches the active session for internal host `10.0.0.1` on port `3345`.
4. **Header Rewriting**:
   - In the **IPv4 Header**: The Destination IP is rewritten from `138.76.29.7` to `10.0.0.1`.
   - In the **TCP Header**: The Destination Port is rewritten from `5001` to `3345`.
5. **Checksum Recalculation**: Both the IPv4 header checksum and the TCP checksum (with the updated pseudo-header) are recomputed.
6. **Forwarding**: The datagram is forwarded through the LAN interface toward host `10.0.0.1`.

---

#### Comprehensive ASCII Diagram & Step-by-Step Packet Trace
*(Reflecting Kurose & Ross Figure 4.25)*

```
                     +---------------------------------------+
                     |         NAT TRANSLATION TABLE         |
                     +-------------------+-------------------+
                     | WAN Side Address  | LAN Side Address  |
                     +-------------------+-------------------+
                     | 138.76.29.7, 5001 | 10.0.0.1, 3345    |
                     | 138.76.29.7, 5002 | 10.0.0.2, 4891    |
                     +-------------------+-------------------+

    +-------------------+                             +--------------------+
    |   Internal Host   |                             |   External Server  |
    |     10.0.0.1      |                             |   128.119.40.186   |
    +---------+---------+                             +----------+---------+
              |                                                  |
              | [Step 1: Outbound Datagram to NAT Router]        |
              | Src: 10.0.0.1, 3345                              |
              | Dst: 128.119.40.186, 80                          |
              |                                                  |
              v                                                  |
     +-----------------+                                         |
     |   NAT Router    |                                         |
     |  (138.76.29.7)  |                                         |
     +--------+--------+                                         |
              |                                                  |
              | [Step 2: Rewritten Datagram onto Public WAN]     |
              | Src: 138.76.29.7, 5001                           |
              | Dst: 128.119.40.186, 80                          |
              +------------------------------------------------->|
                                                                 |
                                                                 | Web Server
                                                                 | Processes
                                                                 | HTTP Request
                                                                 |
              | [Step 3: Response Arrives at NAT Router]         |
              | Src: 128.119.40.186, 80                          |
              | Dst: 138.76.29.7, 5001                           |
              |<-------------------------------------------------+
              v                                                  |
     +-----------------+                                         |
     |   NAT Router    |                                         |
     |  (138.76.29.7)  |                                         |
     +--------+--------+                                         |
              |                                                  |
              | [Step 4: Rewritten Datagram Delivered to Host]   |
              | Src: 128.119.40.186, 80                          |
              | Dst: 10.0.0.1, 3345                              |
              v                                                  |
    +-------------------+                                        |
    |   Internal Host   |                                        |
    |     10.0.0.1      |                                        |
    +-------------------+                                        v
```

##### End-to-End Packet Trace Table:

| Trace Step | Flow Direction | Interface Traversed | Layer 3 Source IP | Layer 4 Source Port | Layer 3 Destination IP | Layer 4 Destination Port | Router Internal Logic & State Changes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 1** | Host $\to$ Router | LAN (`eth1`) | `10.0.0.1` | `3345` | `128.119.40.186` | `80` | Intercepts datagram; allocates unused public port `5001`; creates mapping `(138.76.29.7:5001 <-> 10.0.0.1:3345)` in NAT table. |
| **Step 2** | Router $\to$ Internet | WAN (`eth0`) | `138.76.29.7` | `5001` | `128.119.40.186` | `80` | Overwrites source IP and port; recomputes IPv4 checksum and TCP checksum with updated pseudo-header; sends to ISP. |
| **Step 3** | Server $\to$ Router | WAN (`eth0`) | `128.119.40.186` | `80` | `138.76.29.7` | `5001` | Receives reply; indexes NAT table on destination port `5001`; locates target entry: `10.0.0.1` on port `3345`. |
| **Step 4** | Router $\to$ Host | LAN (`eth1`) | `128.119.40.186` | `80` | `10.0.0.1` | `3345` | Overwrites destination IP and port; recomputes IPv4 checksum and TCP checksum; forwards packet onto the private LAN. |

---

### 5.3 NAT Traversal for Peer-to-Peer Applications

#### The Inbound Connection Problem
NAT table entries are populated **reactively by outbound traffic initiated from within the private LAN**.
If an external client wishes to initiate a connection to a server or peer located behind a NAT router:
1. The external client transmits a TCP SYN or UDP datagram to `138.76.29.7:Port_X`.
2. The NAT router inspects its translation table.
3. Because no internal host has recently transmitted an outbound packet to that external destination on `Port_X`, **no matching state entry exists**.
4. The router silently drops the packet or returns an Internet Control Message Protocol (ICMP) Port Unreachable message.

This asymmetric property breaks **Peer-to-Peer (P2P)** applications (e.g., BitTorrent file sharing, Voice over IP - VoIP using SIP, online multiplayer gaming, and Web Real-Time Communication - WebRTC), where arbitrary peers must be able to establish direct connections to exchange media and files.

---

#### Traversal Solutions

##### 1. Static Port Forwarding:
- **Operation**: The network administrator manually configures static forwarding rules within the router firmware:
  $$\text{Incoming Traffic on WAN Port } 25000 \longrightarrow \text{Forward to Private IP } 10.0.0.1:\text{Port } 25000$$
- **Trade-offs**: Inflexible, fragile, requires static IP addressing for internal hosts, and cannot scale to arbitrary consumer applications operated by non-technical end users.

##### 2. Universal Plug and Play (UPnP) / Internet Gateway Device (IGD) Protocol:
- **Operation**: A software protocol enabling client applications to automate port forwarding. A host inside the LAN uses the Simple Service Discovery Protocol (SSDP) over multicast (`239.255.255.250:1900`) to detect a UPnP/IGD-compliant NAT gateway.
- Once discovered, the application uses Extensible Markup Language (XML) and Simple Object Access Protocol (SOAP) remote procedure calls to dynamically lease incoming pinholes:
  > *"Map incoming WAN port 55555 to my private IP 10.0.0.2:55555 for the next 3600 seconds."*
- **Trade-offs**: Zero manual configuration; however, it presents severe security vulnerabilities because rogue software or malware inside the LAN can open incoming firewall ports without administrative knowledge. For this reason, UPnP is disabled by default in enterprise networks.

##### 3. Relay Servers (TURN — Traversal Using Relays around NAT, RFC 5766 / RFC 8656):
- **Operation**: When direct hole punching fails (e.g., when one or both peers are behind Symmetric NAT), both peers connect outbound to an intermediate, publicly accessible **TURN relay server**.
- **Data Flow**:
  $$\text{Peer A} \underset{\text{Outbound}}{\xrightarrow{\hspace{1cm}}} \text{TURN Server} \underset{\text{Outbound}}{\xleftarrow{\hspace{1cm}}} \text{Peer B}$$
  The TURN server acts as an application-level relay: Peer A transmits media to the TURN server, which forwards it over the established outbound channel to Peer B, and vice-versa.
- **Trade-offs**: Guarantees a 100% connection success rate across all NAT topologies. However, it incurs severe latency penalties (due to triangular routing) and requires massive server bandwidth, placing financial and computational burdens on the service operator.

##### 4. Hole Punching (STUN — Session Traversal Utilities for NAT, RFC 5389 & ICE):

```
       [ Private Peer A ]                             [ Private Peer B ]
         10.0.0.1:4000                                  192.168.1.5:8000
               |                                               |
       [ NAT Router A ]                               [ NAT Router B ]
        WAN: 1.1.1.1:50000                             WAN: 2.2.2.2:60000
               \                                               /
                \                                             /
                 v                                           v
             [ STUN Server: Discovers Public IP & Ports ]
                             (Reflexive Bindings)
                                      |
         +----------------------------+----------------------------+
         | Peers exchange bindings via out-of-band Signaling Channel |
         +---------------------------------------------------------+
                                      |
               +----------------------+----------------------+
               |                                             |
               v                                             v
     [ Peer A sends packet to ]                    [ Peer B sends packet to ]
          2.2.2.2:60000                                 1.1.1.1:50000
               |                                             |
   (Punches hole in NAT A)                       (Punches hole in NAT B)
               \                                             /
                \====== DIRECT P2P COMMUNICATION ===========/
```

- **Step A: STUN Binding Discovery**:
  - Peer A (`10.0.0.1:4000`) sends an outbound UDP request to a public STUN server (`74.125.250.1`).
  - As the packet traverses NAT Router A, the router binds `10.0.0.1:4000` to public endpoint `1.1.1.1:50000`.
  - The STUN server inspects the Layer 3 source IP and Layer 4 source port of the incoming packet and embeds these values (`1.1.1.1:50000`) inside the STUN response payload.
  - Peer A parses the response and learns its own **Server Reflexive Address** (its public-facing IP and port).
- **Step B: Out-of-Band Signaling**:
  - Peers A and B exchange their server-reflexive addresses via an independent signaling server (using WebSockets, SIP, or XMPP). Peer A learns that Peer B is reachable at `2.2.2.2:60000`, and Peer B learns Peer A is at `1.1.1.1:50000`.
- **Step C: Simultaneous Hole Punching**:
  - Peer A transmits a UDP datagram to `2.2.2.2:60000`. NAT Router A notes this outbound packet and opens an internal filter rule: *"Permit incoming packets from 2.2.2.2:60000 to reach 10.0.0.1:4000."*
  - Peer B transmits a UDP datagram to `1.1.1.1:50000`. NAT Router B opens a corresponding reciprocal filter rule.
  - Initial packets crossing in flight may be dropped by the opposite NAT if they arrive before the local outbound packet creates the state, but subsequent retransmissions succeed. A direct, bi-directional P2P media conduit is established without proxying!
- **NAT Classification and Hole Punching Viability**:
  - **Full Cone NAT**: Maps all requests from the same internal IP/port to the same public IP/port. Any external host can send packets to the public port. Easiest for P2P.
  - **Restricted Cone NAT**: Incoming packets accepted only from external IP addresses to which the internal host has previously sent a packet.
  - **Port-Restricted Cone NAT**: Incoming packets accepted only from external (IP, Port) tuples to which the internal host has previously sent a packet. Hole punching works reliably.
  - **Symmetric NAT**: Assigns a *different* public port for every distinct destination IP/port combination contacted by the internal host. STUN hole punching fails; requires TURN relay fallback.
- **Interactive Connectivity Establishment (ICE, RFC 8445)**:
  - An overarching protocol framework used by WebRTC and VoIP. ICE gathers all possible connectivity candidates for each endpoint:
    1. **Host Candidates**: Local physical/virtual IP addresses on the machine.
    2. **Server Reflexive Candidates**: Public IP/port discovered via STUN.
    3. **Relayed Candidates**: Relayed IP/port allocated on a TURN server.
  - ICE performs systematic connectivity checks across all candidate pairs using prioritized STUN binding requests, selects the lowest-latency direct viable path, and gracefully falls back to TURN only if direct hole punching fails.

---

### 5.4 Architectural Controversies and Trade-offs

| Architectural Dimension | Principle / Axiom Challenged | Real-World Consequence & Technical Impact |
| :--- | :--- | :--- |
| **Layering Violation** | **Strict OSI / TCP/IP Layer Separation**: Routers are Layer 3 devices; they must only inspect and manipulate network layer headers. | NAT routers actively parse, modify, and track Layer 4 (Transport) TCP/UDP port fields. If transport headers are encrypted (e.g., IPsec ESP), standard NAPT fails entirely. |
| **End-to-End Principle** | **End-to-End Argument (Saltzer, Reed, Clark 1984)**: The network core should remain simple, dumb, and stateless; intelligence and state must reside at endpoints. | NAT introduces stateful middleboxes. If an edge router reboots or crashes, its volatile NAT table is lost, instantly terminating all active transport sessions across the organization. |
| **Asymmetric Addressability** | **Universal Reachability**: Every Internet node should be capable of initiating contact with any other node. | Hosts behind NAT become "second-class citizens." They cannot act as native servers or accept unsolicited inbound sessions without specialized traversal protocols. |
| **Application Layer Gateways (ALGs)** | **Payload Transparency**: Routers must never inspect or modify Layer 7 application data. | Legacy protocols embedding IP addresses inside payloads (e.g., FTP `PORT` command, SIP VoIP headers) break across NAT. Routers must deploy ALGs to rewrite L7 payloads and adjust TCP sequence numbers on the fly, creating major security and complexity vulnerabilities. |
| **IPsec Incompatibility** | **End-to-End Cryptographic Integrity**: Packet headers must not be altered in flight. | IPsec Authentication Header (AH) hashes immutable fields, including the Source IP. NAT rewrites the Source IP, causing cryptographic validation failure at the receiver. This forced the development of complex NAT-Traversal (NAT-T, RFC 3948) UDP encapsulation wrappers. |

---

## 6. Software-Defined Networking (SDN)

### 6.1 Evolution and Motivation

#### Legacy Networking: The Monolithic Era
Historically, computer networks operated on a **distributed, per-router control plane** paradigm:
- **Vertical Integration**: Every router and switch was a closed, proprietary appliance. Specialized silicon (Application-Specific Integrated Circuits - ASICs), low-level hardware control drivers, network operating systems (e.g., Cisco IOS, Juniper Junos), and routing applications were tightly coupled and sold by a single vendor.
- **Autonomous Distributed Logic**: Each router executed its own distributed routing protocols (e.g., Open Shortest Path First - OSPF, Border Gateway Protocol - BGP, Intermediate System to Intermediate System - IS-IS). Routers computed forwarding paths locally through distributed state exchange.
- **Operational Sclerosis**: Modifying network behavior or rolling out new protocols required years of standards deliberation at the IETF, followed by vendor firmware development cycles, and manual, error-prone box-by-box Command Line Interface (CLI) configuration changes by network engineers.
- **Middlebox Proliferation**: Specialized physical middleboxes (firewalls, NAT gateways, load balancers, intrusion detection systems) had to be manually spliced into physical cabling topologies, preventing flexible traffic re-routing.

#### The Computing Paradigm Analogy: Mainframes to Personal Computers

```
   TRADITIONAL MONOLITHIC COMPUTING              MODERN OPEN PC ECOSYSTEM
   +------------------------------+             +--------------------------+
   | Applications (Proprietary)   |             | Specialized User Apps    |
   +------------------------------+             +--------------------------+
   | Monolithic OS (Vendor-Locked)|             | Open OS (Linux, Windows) |
   +------------------------------+             +--------------------------+
   | Proprietary Hardware / CPU   |             | Open x86 / ARM Hardware  |
   +------------------------------+             +--------------------------+
         [ IBM Mainframe Era ]                      [ Commodity Computing ]

                 ||                                          ||
                 \/                                          \/

   TRADITIONAL LEGACY ROUTING                    SOFTWARE-DEFINED NETWORKING
   +------------------------------+             +--------------------------+
   | Routing Logic (OSPF, BGP)    |             | SDN Applications (Prog)  |
   +------------------------------+  ======>    +--------------------------+
   | Proprietary OS (Cisco IOS)   |             | SDN Controller (NOS)     |
   +------------------------------+             +--------------------------+
   | Closed ASIC / Custom Chassis |             | Commodity White-Box ASICs|
   +------------------------------+             +--------------------------+
```

SDN replicates the computing revolution in networking: it breaks vertical integration by disaggregating proprietary hardware from the control logic, commoditizing switching silicon, and introducing open, standard software interfaces.

---

### 6.2 Key Characteristics of SDN

1. **Separation of Data Plane and Control Plane**:
   - **Data Plane (Forwarding Plane)**: Operates entirely in fast hardware at line rate (nanosecond-to-microsecond scale). Its sole responsibility is processing arriving packets: matching header bits against hardware tables and executing forwarding actions.
   - **Control Plane (Control Logic)**: Operates in software (millisecond-to-second scale). Its responsibility is computing routing state, enforcing access control policies, determining traffic engineering paths, and programming the data plane switches.
2. **Logically Centralized Network Operating System (SDN Controller)**:
   - Control logic is extracted from individual switches and centralized within an SDN Controller.
   - *Logically centralized* does not imply a single physical machine (which would present a single point of failure); it is implemented as a fault-tolerant, horizontally scalable distributed cluster (e.g., using Raft consensus).
   - The controller maintains a unified, real-time global view of the entire network topology, traffic matrices, and link states.
3. **Flow-Based Forwarding (Generalized Match-Plus-Action)**:
   - Traditional routers perform destination-based forwarding: forwarding decisions are governed strictly by the destination IPv4 address using longest-prefix matching.
   - SDN implements generalized match-plus-action: forwarding rules operate on "flows"—sequences of packets matching arbitrary combinations of Layer 2 (MAC), Layer 3 (IP), and Layer 4 (Port) header fields.
   - Actions extend beyond forwarding to dropping, duplicating, header modification, and redirection.
4. **Network Programmability via Software Applications**:
   - Network policies are expressed not via static router scripts, but as executable software programs running on top of the controller. Routing algorithms, load balancing, firewall enforcement, and Quality of Service (QoS) are modular software components.

---

### 6.3 SDN Architectural Layers

```
  +=========================================================================+
  |                        NETWORK APPLICATION LAYER                        |
  |   +------------------+  +------------------+  +---------------------+   |
  |   | Shortest-Path    |  | Dynamic Firewall |  | Server Load         |   |
  |   | Routing App      |  | & ACL Engine     |  | Balancer Application|   |
  |   +------------------+  +------------------+  +---------------------+   |
  |   +-----------------------------------------------------------------+   |
  |   | Traffic Engineering & Quality of Service (QoS) Management       |   |
  |   +-----------------------------------------------------------------+   |
  +=========================================================================+
                                      |
                 NORTHBOUND INTERFACE | (RESTful APIs, gRPC, Java/Python SDKs)
                                      v
  +=========================================================================+
  |                  CONTROL PLANE: SDN CONTROLLER (NOS)                   |
  |  +-------------------------------------------------------------------+  |
  |  | Network State & Global Topology Database (Graph Representation)   |  |
  |  +-------------------------------------------------------------------+  |
  |  | Flow Table Computation & Rule Generation Engine                   |  |
  |  +-------------------------------------------------------------------+  |
  |  | Device Discovery (LLDP), Link-State Monitors, Statistics Aggregator|  |
  |  +-------------------------------------------------------------------+  |
  +=========================================================================+
                                      |
                 SOUTHBOUND INTERFACE | (OpenFlow, P4Runtime, NETCONF/YANG)
                                      v
  +=========================================================================+
  |                 DATA PLANE / INFRASTRUCTURE LAYER                       |
  |                                                                         |
  |    [ Commodity White-Box Switch 1 ]       [ White-Box Switch 2 ]        |
  |    +-----------------------------+       +---------------------+        |
  |    | Hardware TCAM Flow Tables   |       | Hardware TCAM Tables|        |
  |    +-----------------------------+       +---------------------+        |
  |    | Ports | Match-Action Silicon|       | Ports | Fast Silicon|        |
  |    +-----------------------------+       +---------------------+        |
  +=========================================================================+
```

##### Layer Details:
- **Data Plane / Infrastructure Layer**: Consists of cost-effective "white-box" switches built with commercial merchant silicon (e.g., Broadcom Tomahawk/Trident, Intel Tofino). Switches maintain hardware flow tables implemented using **Ternary Content Addressable Memory (TCAM)**, enabling $O(1)$ constant-time parallel matching across multi-field headers at terabit line rates.
- **Southbound Interface**: The open protocol connecting the controller to the switches. It permits the controller to inspect switch ports, read counters, push flow table rules, and receive event notifications (e.g., link failures). Examples: **OpenFlow**, **P4Runtime**, **NETCONF/YANG**, **OVSDB** (Open vSwitch Database Management Protocol).
- **Control Plane / SDN Controller**: The Network Operating System (NOS). Maintains the network graph, tracks host attachments, monitors link metrics via the Link Layer Discovery Protocol (LLDP), and exposes abstractions to applications. Major open-source platforms include:
  - **ONOS** (Open Network Operating System) — engineered for high availability and performance in telecommunication and service provider backbones.
  - **OpenDaylight (ODL)** — modular, enterprise-focused Java-based controller framework backed by the Linux Foundation.
  - **Ryu** — lightweight, event-driven Python controller popular in research and rapid prototyping.
- **Northbound Interface**: Application Programming Interfaces (APIs) exposing high-level network abstractions to user applications. Unlike the Southbound interface, Northbound APIs are typically programmatic **RESTful APIs** or gRPC endpoints allowing applications to specify operational intents without managing low-level hardware table indices.
- **Network Application Layer**: Independent software programs implementing business and operational logic: adaptive routing, policy-based access control, distributed firewalls, and traffic engineering.

---

### 6.4 The OpenFlow Protocol

Maintained by the Open Networking Foundation (ONF), OpenFlow was the first standardized Southbound communications interface. It operates over **TCP** (standard port `6653`, historically `6633`), optionally secured via Transport Layer Security (TLS).

#### OpenFlow Message Taxonomy

##### 1. Controller-to-Switch Messages (Initiated by Controller):
- **Features Request**: The controller queries the switch to discover its identity, number of hardware flow tables, supported actions, and port configurations. The switch responds with a *Features Reply*.
- **Configuration (Get-Config / Set-Config)**: Controller inspects or programs operational configuration parameters (e.g., handling of IP fragments, maximum packet buffer lengths).
- **Modify-State (`FlowMod`)**: The primary operational message. Used by the controller to **Add**, **Modify**, or **Delete** flow entries inside the switch’s flow tables.
- **Read-State**: Gathers performance metrics and telemetry (per-table statistics, per-flow match counters, per-port error rates).
- **Packet-Out**: Instructs the switch to emit a specific packet (generated by the controller, or buffered during a prior table miss) out of designated physical or virtual ports.

##### 2. Asynchronous Messages (Initiated by Switch without Controller Request):
- **`Packet-In`**: When an arriving packet encounters a **table miss** (no matching flow rule exists in the flow table), or when an explicit action specifies `Output: CONTROLLER`, the switch encapsulates the packet (or its header bytes) and transmits it to the controller for inspection.
- **Flow-Removed**: Informs the controller that a specific flow entry was evicted from the table due to timeout expiration (idle or hard timeout) or manual deletion.
- **Port-Status**: Alerts the controller that a physical link/port state has altered (e.g., link transitioned from UP to DOWN), triggering dynamic topology recomputation.

##### 3. Symmetric Messages (Initiated Bidirectionally without Solicitation):
- **Hello**: Exchanged during the initial TCP handshake to negotiate the highest mutually supported OpenFlow protocol version.
- **Echo Request / Echo Reply**: Heartbeat/keepalive mechanism verifying connection liveness and measuring control-channel latency.
- **Error**: Alerts the peer to unsupported requests, permission violations, or malformed parameters.

---

#### Flow Table Entry Anatomy
Each entry within an OpenFlow flow table consists of five discrete structures:

```
+--------------------------------------------------------------------------------------+
|                                   FLOW TABLE ENTRY                                   |
+---------------------+----------+--------------------+-------------+------------------+
|    MATCH FIELDS     | PRIORITY |      COUNTERS      | INSTRUCTIONS|     TIMEOUTS     |
| (Header & Port Bits)| (Integer)| (Packets, Bytes, s)|  (Actions)  | (Idle, Hard Sec) |
+---------------------+----------+--------------------+-------------+------------------+
```

##### 1. Match Fields (12+ Header Tuples):
An OpenFlow match rule can inspect fields spanning Layers 1 through 4:

```
+-----------+--------------------+-------------------+--------------------+
| INGRESS   | LAYER 2 (Ethernet) | LAYER 3 (IP)      | LAYER 4 (Transport)|
+-----------+--------------------+-------------------+--------------------+
| Switch    | Source MAC         | Source IPv4/IPv6  | TCP/UDP Src Port   |
| Port      | Destination MAC    | Dest IPv4/IPv6    | TCP/UDP Dst Port   |
|           | Ethernet Type      | IP Protocol (6/17)| ICMP Type & Code   |
|           | VLAN ID & Priority | IP ToS / DSCP Bits|                    |
+-----------+--------------------+-------------------+--------------------+
```
Any field may contain an exact value, a subnet mask, or a wildcard (`*`).

##### 2. Counters (Telemetry):
Updated in hardware at line rate whenever a packet matches the entry:
- Received packet count ($\sum \text{packets}$)
- Cumulative byte count ($\sum \text{bytes}$)
- Flow duration (active time in seconds and nanoseconds)

##### 3. Actions / Instructions:
Dictate packet processing upon a successful match:
- **`Forward`**: Directs packet to physical port $N$, or virtual ports:
  - `ALL`: Broadcast out all interfaces except ingress.
  - `FLOOD`: Forward along the spanning tree to avoid loops.
  - `CONTROLLER`: Encapsulate in `Packet-In` message and transmit to controller.
  - `IN_PORT`: Reflect packet back out the arrival interface.
- **`Drop`**: Instantly discard the packet (specified by an empty action set).
- **`Modify-Field`**: Rewrite header bits (e.g., rewrite Destination IP for NAT, decrement TTL, rewrite MAC address for Layer 3 hop transit, push/pop 802.1Q VLAN tags or MPLS labels).
- **`Goto-Table`**: Direct packet to Table $N+1$ in multi-table pipelined architectures.

##### 4. Timeouts:
- **Idle Timeout**: Entry is automatically purged if no matching packets arrive for $T_{\text{idle}}$ consecutive seconds.
- **Hard Timeout**: Entry is unconditionally purged $T_{\text{hard}}$ seconds after installation, regardless of traffic volume.

---

#### Practical Flow Table Configuration Examples

##### Example 1: Destination-Based Layer 3 Routing
Forward all traffic destined for subnet `51.6.0.0/16` out physical Switch Port 6:

| Switch Port | Src MAC | Dst MAC | Eth Type | VLAN ID | IP Src | IP Dst | IP Prot | TCP Src | TCP Dst | Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `*` | `*` | `*` | `0x0800` | `*` | `*` | `51.6.0.0/16` | `*` | `*` | `*` | **Forward(Port 6)** |

##### Example 2: Transport-Layer Firewall
Block all incoming external SSH (Secure Shell) traffic on TCP destination port 22:

| Switch Port | Src MAC | Dst MAC | Eth Type | VLAN ID | IP Src | IP Dst | IP Prot | TCP Src | TCP Dst | Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `*` | `*` | `*` | `0x0800` | `*` | `*` | `*` | `6` (TCP) | `*` | `22` | **Drop** |

##### Example 3: Layer 2 Destination-Based Switching
Forward frames addressed to destination MAC `00:1A:2B:3C:4D:5E` out physical Switch Port 3:

| Switch Port | Src MAC | Dst MAC | Eth Type | VLAN ID | IP Src | IP Dst | IP Prot | TCP Src | TCP Dst | Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `*` | `*` | `00:1A:2B:3C:4D:5E` | `*` | `*` | `*` | `*` | `*` | `*` | `*` | **Forward(Port 3)** |

---

#### Control/Data Plane Interaction Scenario: Link Failure & Dynamic Rerouting

```
   [ Routing Application: Dijkstra Link-State Algorithm ]
                        ^                  |
        3. Event Fired: |                  | 5. Recomputed Path
           Invoke App   |                  |    Pushed
                        v                  v
   +----------------------------------------------------+
   |               SDN CONTROLLER CORE                  |
   |   - Network Graph Engine: Updates Topology State   |
   |   - Flow Table Generation: Constructs FlowMods     |
   +----------------------------------------------------+
             ^                               |
             | 2. Link Down Notification     | 6. FlowMod
             |    (OpenFlow Port-Status)     |    (Modify-State)
             |                               v
       +-----------+                   +-----------+
       | Switch S1 |                   | Switch S2 |
       +-----+-----+                   +-----+-----+
             |                               |
             |       LINK SEVERED (X)        |
             +-------------------------------+
```

1. **Failure Inception**: Physical link between Switch S1 and Switch S2 fails.
2. **Asynchronous Notification**: Switch S1’s local hardware driver detects loss of carrier light and generates an OpenFlow **`Port-Status`** message (`port_no: 2, state: DOWN`) transmitted over TCP to the controller.
3. **Controller Ingestion**: The controller receives the event, updates its internal network graph data structure, and triggers registered event-listener applications.
4. **Algorithmic Path Recomputation**: The Shortest-Path Routing Application executes Dijkstra’s algorithm over the updated network graph, discovering an alternate detour via Switch S3.
5. **Rule Generation**: The controller constructs new `FlowMod` instructions containing updated output actions for the impacted traffic flows.
6. **Hardware Reprogramming**: The controller dispatches OpenFlow `FlowMod` messages to the affected switches, updating their TCAM hardware flow tables. Traffic resumes over the alternate path without manual intervention.

---

### 6.5 P4 (Programming Protocol-Independent Packet Processors)

#### Beyond OpenFlow: The Limits of Bottom-Up Protocol Standardization
OpenFlow revolutionized networking but was constrained by its **bottom-up design**:
- **Fixed Header Schema**: OpenFlow explicitly bakes standard protocol headers (Ethernet, IPv4, IPv6, TCP, UDP) into its specification.
- **Specification Bloat**: As modern networking introduced new encapsulations (e.g., Virtual Extensible LAN - VXLAN, Network Virtualization using Generic Routing Encapsulation - NVGRE, Segment Routing over IPv6 - SRv6, Geneve), the OpenFlow standard expanded from 12 match fields in version 1.0 to over **40 match fields** in version 1.4.
- **Hardware Rigidity**: If a cloud data center wanted to deploy a novel telemetry or custom encapsulation header, they had to wait years for the ONF to standardize the header and for ASIC vendors to fabricate new silicon.

#### The P4 Philosophy: Top-Down Software-Driven Packet Processing
Developed by Nick McKeown et al. (2014), **P4 (Programming Protocol-Independent Packet Processors)** fundamentally redefines packet processing:
> **Core Axiom**: The physical switch silicon should have **zero built-in knowledge of networking protocols**. It should not inherently understand what an IP packet or an Ethernet frame is. Instead, the programmer defines the packet parsing logic, header schemas, and processing pipelines using the high-level P4 language, compiling the software directly down to target hardware.

---

#### The P4 Abstract Switch Architecture (PISA Model)
The Protocol Independent Switch Architecture (PISA) structures packet processing into three programmable stages:

```
  +-------------------------------------------------------------------------------+
  |                        P4 PROGRAMMABLE PIPELINE (PISA)                        |
  |                                                                               |
  |  +-------------------+     +-------------------------+     +---------------+  |
  |  |   PROGRAMMABLE    |     |      PROGRAMMABLE       |     |  PROGRAMMABLE |  |
  |  |      PARSER       | ==> |   MATCH-ACTION STAGES   | ==> |   DEPARSER    |  |
  |  |  (Finite State    |     |  (Tables, Arithmetic    |     |  (Re-assemble |  |
  |  |   Machine - FSM)  |     |   Bitwise ALUs, Regs)   |     |   Wire Bytes) |  |
  |  +-------------------+     +-------------------------+     +---------------+  |
  |           ^                             ^                          |          |
  +-----------|-----------------------------|--------------------------|----------+
              |                             |                          |
    Raw Ingress Bytes               Control Plane (P4Runtime)    Egress Packet to Wire
```

1. **Programmable Parser (Finite State Machine - FSM)**:
   - Evaluates raw byte streams arriving from physical ports.
   - The programmer specifies an explicit state machine that extracts bits into typed header formats (e.g., parsing Ethernet, inspecting `etherType`, transitioning to parse IPv4, IPv6, or custom telemetry tags).
2. **Programmable Match-Action Pipelines (Ingress & Egress)**:
   - Parsed headers traverse a sequence of user-defined match-action tables.
   - P4 allows exact, ternary, longest-prefix match (LPM), and range matching.
   - Unlike OpenFlow’s fixed actions, P4 actions are programmed algorithms constructed from fundamental arithmetic, bitwise logic, and stateful register updates (e.g., computing custom hashes, adding custom encapsulation, or measuring queue residency).
3. **Programmable Deparser**:
   - Reassembles structured, modified headers back into a contiguous sequence of wire bytes, appends the unmodified payload, and hands the finalized frame to the physical MAC transmitter.

---

#### Detailed Technical Comparison: OpenFlow vs. P4

| Architectural Dimension | OpenFlow Protocol | P4 (Programming Protocol-Independent Packet Processors) |
| :--- | :--- | :--- |
| **Design Philosophy** | **Bottom-Up**: Standardizes control-channel access to fixed-function hardware. | **Top-Down**: Standardizes language to program hardware forwarding behavior from scratch. |
| **Protocol Independence** | **Protocol-Dependent**: Switch silicon hardcodes explicit headers (Ethernet, IP, TCP/UDP). | **Fully Protocol-Independent**: Switch knows no protocols natively; parser is fully programmed. |
| **Header Extensibility** | **Rigid**: Adding novel headers requires multi-year IETF/ONF spec updates and new silicon. | **Instantaneous**: Programmers define custom headers in software and compile to existing silicon. |
| **Forwarding Pipeline** | **Fixed**: Pre-defined tables with fixed match capabilities. | **Fully Programmable**: Dynamic sequence of custom tables, custom actions, and stateful registers. |
| **Target Hardware** | Switches with OpenFlow firmware agents. | Any P4-compliant target: ASICs (Intel Tofino), FPGAs, SmartNICs, software switches (BMv2). |
| **Control Plane API** | OpenFlow Protocol (messages like `FlowMod`, `Packet-In`). | **P4Runtime**: An autogenerated, silicon-independent API derived directly from the P4 program. |

---

## 7. Data Center Networking

### 7.1 Architecture Requirements and Traffic Dynamics

#### Scale of Modern Hyperscale Data Centers
Modern cloud data centers operated by hyper-scalers (e.g., Google, Amazon Web Services - AWS, Microsoft Azure) house between **100,000 and 500,000+ servers** spread across football-field-sized facilities. Networking architectures must interconnect these servers with high throughput, ultra-low latency, and high resilience.

#### Traffic Dynamics: North-South vs. East-West Traffic

```
                           [ EXTERNAL INTERNET ]
                                     |
                                     | North-South Traffic (<20%)
                                     v
                        +-------------------------+
                        | Data Center Edge Router |
                        +-------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
  +--------------+                                        +--------------+
  | Server Rack  | <======== East-West Traffic ========> | Server Rack  |
  |   (Rack A)   |              (> 80%)                   |   (Rack B)   |
  +--------------+                                        +--------------+
```

##### 1. North-South Traffic:
- **Definition**: Traffic entering or exiting the data center boundary (client-to-server traffic, such as a user loading an external web page or downloading an asset).
- **Historical Context**: In early corporate enterprise hosting, North-South traffic accounted for the vast majority of bandwidth demand.

##### 2. East-West Traffic:
- **Definition**: Traffic that originates and terminates entirely within the data center boundary (server-to-server, compute-to-storage, and container-to-container traffic).
- **Catalysts**:
  - **Distributed Big Data Computing**: Frameworks like Apache Hadoop, MapReduce, and Apache Spark continuously shuffle gigabytes of intermediate data across racks.
  - **Microservice Architectures**: A single external user request to a web front-end triggers hundreds of internal Remote Procedure Calls (RPCs) across authentication, billing, database, recommendation, and cache services.
  - **Distributed AI/ML Training**: Training Large Language Models (LLMs) requires continuous, line-rate gradient synchronization across thousands of GPUs using AllReduce paradigms.
- **Empirical Reality**: **East-West traffic accounts for more than 80% to 90% of all data center traffic today.**

---

#### Bisection Bandwidth and Non-Blocking Fabrics
- **Bisection Bandwidth**: The transmission capacity across the narrowest bottleneck cut that divides the network into two equal halves with equal numbers of hosts:

```
        Partition 1 (N/2 Hosts)               Partition 2 (N/2 Hosts)
      +-------------------------+           +-------------------------+
      |  [H1] [H2] ... [H_N/2]  |           | [H_(N/2+1)] ... [H_N]   |
      +-------------------------+           +-------------------------+
                   \                             /
                    \=== Cut Links (Bandwidth) ==/
```

$$\text{Bisection Bandwidth} = \sum_{\text{link } i \in \text{Cut}} \text{Capacity}(i)$$

- **Non-Blocking Fabric**: A network topology whose bisection bandwidth is large enough that the aggregate communication rate between any arbitrary split of servers is limited *only* by the network interface cards (NICs) of the servers themselves, rather than any intermediate core links. 
  $$\text{Oversubscription Ratio} = 1:1$$

---

### 7.2 Traditional 3-Tier Architecture

```
                       +-----------------------+
                       |       CORE TIER       |
                       |  (Chassis Routers)    |
                       +-----------+-----------+
                                  / \
                        Blocked  /   \  Active Links
                       by STP   /     \
                               v       v
                       +-----------------------+
                       |   AGGREGATION TIER    |
                       | (Distribution Switches|
                       +-----------+-----------+
                                  / \
                                 /   \
                                v     v
                       +-----------------------+
                       |      ACCESS TIER      |
                       | (Top-of-Rack Switches)|
                       +-----------+-----------+
                                   |
                         [ Servers in Racks ]
```

#### Structural Flaws and Fatal Bottlenecks

##### 1. Massive Oversubscription:
- The oversubscription ratio represents the ratio of total server access bandwidth to uplink core bandwidth:
  $$\text{Oversubscription} = \frac{\text{Total Downlink Bandwidth}}{\text{Total Uplink Bandwidth}}$$
- In traditional 3-tier designs, Top-of-Rack (ToR) switches exhibited $4:1$ or $10:1$ oversubscription, while Aggregation-to-Core exhibited $10:1$ to $20:1$. The cumulative core oversubscription ranged between **$20:1$ and $200:1$**.
- While acceptable for low-volume North-South traffic, this choke point caused severe packet drops, high latency, and buffer bloat under high East-West distributed server workloads.

##### 2. Spanning Tree Protocol (STP) Path Blocking:
- The access and aggregation layers historically ran Layer 2 Ethernet to support Virtual Machine (VM) live migration within the same subnet.
- Because meshed Layer 2 topologies suffer from fatal broadcast storms if any loops exist, the **Spanning Tree Protocol (IEEE 802.1D STP)** intentionally disabled and blocked redundant links.
- Consequently, **50% or more of expensive, provisioned physical infrastructure sat completely idle**, unable to forward traffic.

##### 3. Failure Blast Radius and Scalability Limits:
- Scaling required buying increasingly massive, proprietary, multi-million-dollar core chassis routers ("scale-up" approach).
- A hardware failure, power glitch, or firmware crash on a core chassis disrupted network connectivity for thousands of servers simultaneously.

---

### 7.3 Modern Clos / Fat-Tree (Spine-Leaf) Architecture

Adapted from Charles Clos’s 1953 telephone switching network theory and formalized for computing by Charles Leiserson (1985) and Al-Fares et al. (SIGCOMM 2008), the modern **Clos / Fat-Tree (Spine-Leaf)** architecture is the industry standard for cloud data center design.

#### The 2-Tier Spine-Leaf Topology & Interconnection Invariants

```
                            SPINE SWITCHES (Fabric Core)
                             +--------+    +--------+
                             | Spine1 |    | Spine2 |
                             +--------+    +--------+
                              /   \        /   \
                             /     \      /     \
                            /       \    /       \
                           /         \  /         \
                 +--------+        +--------+   +--------+
                 | Leaf 1 |        | Leaf 2 |   | Leaf 3 |
                 +--------+        +--------+   +--------+
                     |                 |            |
                 [Rack 1]          [Rack 2]     [Rack 3]
                 Servers           Servers      Servers
```

##### Strict Wiring Invariants:
1. **Every Leaf switch connects to EVERY Spine switch.**
2. **Spine switches NEVER connect to other Spine switches.**
3. **Leaf switches NEVER connect directly to other Leaf switches.**
4. **Deterministic Two-Hop Latency**: Every server in the data center is separated from every other inter-rack server by exactly **three switch hops (two fabric hops)**:
   $$\text{Source Server} \longrightarrow \text{Leaf}_{\text{src}} \longrightarrow \text{Spine}_k \longrightarrow \text{Leaf}_{\text{dst}} \longrightarrow \text{Destination Server}$$

---

#### The $k$-Port Switch Fat-Tree Parameterization
A $k$-port Fat-Tree constructed entirely from uniform, commodity switches with $k$ physical ports yields a symmetric, non-blocking fabric:

```
+------------------------------------+------------------------------------------+
| Architectural Property             | Mathematical Formulation                 |
+------------------------------------+------------------------------------------+
| Number of Pods                     | $k$ pods                                 |
| Switches per Pod                   | $k$ switches ($(k/2)$ Leaf, $(k/2)$ Agg) |
| Total Core (Spine) Switches        | $(k/2)^2 = \frac{k^2}{4}$                |
| Total Leaf (Edge) Switches         | $k \times (k/2) = \frac{k^2}{2}$         |
| Total Non-Blocking Supported Hosts | $\frac{k^3}{4}$ servers                  |
+------------------------------------+------------------------------------------+
```

##### Example Calculation ($k = 48$ Port Switches):
- Core Spine Switches $= (48/2)^2 = 24^2 = \mathbf{576 \text{ switches}}$
- Leaf Edge Switches $= 48 \times 24 = \mathbf{1,152 \text{ switches}}$
- Total Supported Servers $= \frac{48^3}{4} = \frac{110,592}{4} = \mathbf{27,648 \text{ bare-metal servers}}$
- **Oversubscription**: Exactly **$1:1$ non-blocking bisection bandwidth** achieved using only commodity white-box merchant silicon!

---

#### Equal-Cost Multi-Path (ECMP) Routing at Layer 3
The modern Spine-Leaf fabric eliminates Layer 2 Spanning Tree Protocol by routing **Layer 3 (IP)** directly to the Top-of-Rack Leaf switches (commonly using external BGP - eBGP, or OSPF):

```
       [ Packet Arrives at Leaf 1 destined for Rack 3 ]
                             |
                  5-Tuple Hash Calculation:
       H = Hash(SrcIP, DstIP, Protocol, SrcPort, DstPort)
                             |
                 Spine_Index = H mod N_spines
                             |
         +-------------------+-------------------+
         | (If Index = 0)                        | (If Index = 1)
         v                                       v
     [ Spine 1 ]                             [ Spine 2 ]
         \                                       /
          +------------------+------------------+
                             |
                             v
                         [ Leaf 3 ]
                             |
                     [ Target Server ]
```

##### ECMP Mechanics:
1. **Active-Active Utilization**: When Leaf 1 routes an outbound packet, it has $N$ equal-cost shortest paths through all $N$ Spine switches. All physical links actively forward traffic simultaneously.
2. **5-Tuple Flow Hashing**: To prevent packet reordering within a single TCP connection (which degrades TCP window performance), the switch computes a cyclic redundancy or Jenkins/Murmur hash over the packet’s 5-tuple:
   $$\text{Hash Value} = \text{CRC32}(\text{Src IP}, \text{Dst IP}, \text{Protocol}, \text{Src Port}, \text{Dst Port})$$
   $$\text{Selected Link} = \text{Hash Value} \pmod{N_{\text{Spines}}}$$
3. **Flow Consistency**: All packets belonging to the same transport flow produce an identical hash and traverse the exact same spine path, preserving strict in-order packet delivery, while tens of thousands of concurrent flows are uniformly distributed across all available spines.

---

#### Comprehensive Side-by-Side Comparison: Traditional 3-Tier vs. Clos / Spine-Leaf

| Dimension | Traditional 3-Tier Architecture | Modern Clos / Spine-Leaf Architecture |
| :--- | :--- | :--- |
| **Topology Structure** | Hierarchical: Core $\to$ Aggregation $\to$ Access (ToR). | 2-Tier Full Bipartite Mesh: Spine $\longleftrightarrow$ Leaf. |
| **Scaling Philosophy** | **Scale-Up**: Buy larger, specialized, proprietary core chassis. | **Scale-Out**: Add more low-cost, commodity white-box spine switches. |
| **Oversubscription Ratio**| Highly oversubscribed ($20:1$ to $200:1$ at Core). | **$1:1$ Non-blocking** line-rate bisection bandwidth. |
| **East-West Optimization**| Extremely poor; traffic chokes at aggregation/core bottleneck. | **Optimal**: 80%+ East-West traffic flows across parallel spines. |
| **Link Utilization** | Poor: Spanning Tree Protocol (STP) blocks redundant links. | **100% Active-Active**: Equal-Cost Multi-Path (ECMP) utilizes all paths. |
| **Failure Blast Radius** | Catastrophic: Core router failure impacts thousands of hosts. | Minimal: Loss of 1 spine switch in 32 degrades capacity by only $\approx 3.1\%$. |
| **Latency Characteristics**| Highly variable, non-deterministic (depends on hops and STP).| **Deterministic & Uniform**: Exactly 3 switch hops between any servers. |
| **Hardware Economics** | Proprietary ASICs, expensive vendor lock-in (high Capex). | Open merchant silicon white-box switches (low Capex). |

---

## 8. Multicast Routing (Case Study)

### 8.1 Principles and Addressing

#### Transmission Paradigms: Unicast vs. Broadcast vs. Multicast

```
      UNICAST (1-to-1)             BROADCAST (1-to-All)           MULTICAST (1-to-Group)
     [S]                  [S]                           [S]
      |                    |                             |
     [R]                  [R]                           [R]
    /   \                /   \                         /   \
  [R1]  [R2]           [R1]  [R2]                    [R1]  [R2]
  (Separate copy       (Single copy flooded           (Single copy replicated
   per receiver)        to everyone on LAN)            ONLY at branch points)
```

| Transmission Type | Delivery Semantics | Network Efficiency & Resource Utilization | Impact on Uninterested Receivers |
| :--- | :--- | :--- | :--- |
| **Unicast** | One-to-One ($1 \to 1$) | Inefficient for $N$ receivers: sender injects $N$ duplicate packets, saturating access links. | None; packets addressed to a unique host IP. |
| **Broadcast** | One-to-All ($1 \to \text{All}$) | Transmits 1 packet per subnet, but confined to local Layer 2 broadcast domain (routers drop broadcasts). | Severe; forces every host on the subnet to interrupt CPU to inspect headers. |
| **Multicast** | One-to-Many ($1 \to G$) | **Maximally Efficient**: sender emits a single packet. Routers replicate packets *only at diverging branch points*. | None; processed only by hosts whose network stack has explicitly joined Group $G$. |

---

#### Class D IPv4 Multicast Addressing
Multicast addresses occupy the former Class D address block, designated by the high-order binary prefix `1110`:
$$\text{Class D Range} = 224.0.0.0 \text{ to } 239.255.255.255 \quad (\text{Prefix: } 224.0.0.0/4)$$
This pool provides $2^{28} \approx 268,435,456$ unique multicast group addresses.

##### Key Standard Address Ranges:
- **`224.0.0.0/24` (Local Network Control Block: `224.0.0.0` – `224.0.0.255`)**:
  - Reserved for link-local routing and infrastructure protocols. Packets are transmitted with $\text{Time-to-Live (TTL)} = 1$ and are **never forwarded by multicast routers**.
  - `224.0.0.1`: **All Systems / Hosts** on this subnet.
  - `224.0.0.2`: **All Multicast Routers** on this subnet.
  - `224.0.0.5` / `224.0.0.6`: OSPF Routers / OSPF Designated Routers.
  - `224.0.0.13`: Protocol Independent Multicast (PIM) Routers.
- **`239.0.0.0/8` (Administratively Scoped IPv4 Multicast Space)**:
  - Equivalent to RFC 1918 private addressing for multicast. Reserved for private enterprise networks; dropped at organizational perimeter routers.

---

#### Mapping Multicast IP to Ethernet Multicast MAC Addresses (RFC 1112)
Because multicast packets are transmitted over standard Ethernet physical networks, network cards must filter multicast frames in hardware to prevent CPU interruption of uninterested hosts.

IANA was assigned the Ethernet Organizationally Unique Identifier (OUI) block `01:00:5E`.
In **RFC 1112**, Steve Deering established that the lower half of this OUI would be dedicated to IPv4 multicast:
- The first **25 bits** of the 48-bit MAC address are fixed to:
  $$\text{Hex: } 01:00:5E:00:00:00 \quad \text{with 25th bit } = 0$$
- This leaves **23 bits** of the Ethernet MAC address available for mapping.

```
  IPv4 Multicast Address (32 bits):
  +------+-------+----------------------------------------------------+
  | 1110 | 5 bits|               Lower 23 Bits of IP                  |
  +------+-------+----------------------------------------------------+
     ^       |                             |
  Class D    | 5 Lost Bits                 | Copied directly
  (Fixed)    | (Not Mapped)                |
             v                             v
  +-------------------------+---+-------------------------------------+
  | 00000001:00000000:01011110: 0 |        Lower 23 Bits of MAC       |
  +-------------------------+---+-------------------------------------+
  Ethernet Multicast MAC Address (48 bits, OUI = 01:00:5E)
```

##### The 5-Bit Ambiguity Problem:
An IPv4 multicast address has $28$ variable bits (after excluding the fixed prefix `1110`). However, the Ethernet mapping accommodates only $23$ bits:
$$\text{Unmapped Bits} = 28 - 23 = \mathbf{5 \text{ bits}}$$
$$2^5 = \mathbf{32 \text{ distinct IPv4 multicast IP addresses map to the exact same Ethernet MAC address!}}$$

##### Example:
The following distinct IP multicast addresses produce the identical Ethernet MAC address `01:00:5E:00:01:01`:
1. `224.0.1.1`
2. `224.128.1.1`
3. `225.0.1.1`
4. ... through all 32 combinations!

##### Resolution at the Host:
1. **NIC Filtering (Hardware)**: The Network Interface Card (NIC) inspects the destination MAC. If it matches a hash of joined multicast MACs, it transfers the frame to main memory via Direct Memory Access (DMA).
2. **IP Stack Demultiplexing (Software)**: The Operating System’s Layer 3 IP stack examines the full 32-bit IPv4 destination address. If the host joined `224.0.1.1`, but the frame contains `225.0.1.1`, the IP layer silently discards the packet.

---

### 8.2 Local Group Management: IGMP (Internet Group Management Protocol)

#### Scope and Role of IGMP
**IGMP operates locally between a host and its immediately attached first-hop multicast router.** 
- IGMP is **NOT** a routing protocol across the Internet backbone.
- Its sole function is allowing local hosts to signal their dynamic membership in specific multicast groups to the local designated router, and allowing the router to discover which multicast groups have active listeners on its connected subnets.
- Packets are encapsulated directly in IPv4 datagrams with **IP Protocol Number 2**.

```
    +----------------------------------------------------+
    |                     LOCAL SUBNET                   |
    |                                                    |
    |    Host A                Host B           Host C   |
    |  (Joined 239.1.1.1)    (No Group)   (Joined 239.1.1.1)
    |       \                     |             /        |
    +--------\--------------------|------------/---------+
              \                   |           /
               v                  v          v
          +--------------------------------------+
          |           ETHERNET SWITCH            |
          +-------------------+------------------+
                              | IGMP Signaling (Local to LAN)
                              v
                   +---------------------+
                   | FIRST-HOP MULTICAST |
                   |       ROUTER        |
                   +----------+----------+
                              | PIM / Multicast Routing Protocols
                              v (Inter-Router WAN Fabric)
```

---

#### IGMP Protocol Mechanics

```
       FIRST-HOP ROUTER                             HOST A (Group G)           HOST C (Group G)
              |                                            |                          |
              | 1. General Query (224.0.0.1)               |                          |
              |------------------------------------------->|                          |
              |---------------------------------------------------------------------->|
              |                                            |                          |
              |                                     Starts Timer (T=3s)        Starts Timer (T=7s)
              |                                            |                          |
              |                                      Timer Expires!                   |
              | 2. Membership Report: Group G              |                          |
              |<-------------------------------------------|                          |
              |                                            | 3. Hears Report on LAN   |
              |                                            + - - - - - - - - - - - - >|
              |                                                                Suppresses Report!
              |                                                                       |
              | [Later: Host A departs group]                                         |
              | 4. Leave Group Message (224.0.0.2)                                    |
              |<-------------------------------------------|                          |
              |                                                                       |
              | 5. Group-Specific Query (Group G)                                     |
              |---------------------------------------------------------------------->|
              |                                                                       |
              | 6. Membership Report: Group G                                         |
              |<----------------------------------------------------------------------|
              |                                                                       |
              | [Router maintains active state for Group G]                           |
```

##### 1. General Membership Query:
- The multicast router periodically transmits an IGMP General Query addressed to `224.0.0.1` (All Systems) with $\text{TTL} = 1$.
- The query asks: *"Are there any active multicast group listeners on this subnet?"*

##### 2. Membership Report & Randomized Countdown Suppression:
- Any host belonging to group $G$ must reply with an IGMP Membership Report addressed to group $G$.
- **The Report Storm Problem**: If 1,000 hosts on a subnet simultaneously reply, the LAN would suffer severe packet collisions and broadcast saturation.
- **The Suppression Mechanism**:
  1. Upon receiving a General Query, each host starts a randomized countdown timer:
     $$T_{\text{wait}} \in [0, \text{Max Response Time}], \quad \text{where default Max Response Time} = 10 \text{ seconds}$$
  2. The host whose timer expires earliest (e.g., Host A at $3\text{ s}$) transmits its Membership Report to group address $G$.
  3. Because the report is addressed to the multicast group $G$, all other listening hosts on the same shared subnet receive Host A’s report.
  4. Hosts whose timers have not yet expired (e.g., Host C at $7\text{ s}$) cancel their pending timers and **suppress their duplicate reports**. A single report per group suffices to keep the router’s port active.

##### 3. Explicit Leave Group Message (IGMPv2):
- In IGMPv1, hosts departed silently, forcing routers to wait up to 3 minutes ($3 \times \text{query intervals}$) of unacknowledged queries before stopping unwanted multicast streams.
- In IGMPv2, when a host leaves group $G$, it transmits an explicit **Leave Group message** addressed to `224.0.0.2` (All Multicast Routers).
- The router immediately transmits a **Group-Specific Query** with a short timeout ($1\text{ s}$). If no other host responds with a Membership Report, the router purges group $G$ from the interface, saving WAN bandwidth.

##### 4. Soft-State Timers:
- Routers maintain multicast forwarding state on a **soft-state** timer. If no Membership Report is received for a group within the group membership timeout period ($[Query\ Count \times Query\ Interval] + Response\ Time$), the router evicts the group state automatically.

---

### 8.3 Multicast Routing Tree Strategies

#### Why Trees?
In an arbitrary network mesh containing cycles, flooding packets causes infinite forwarding loops and explosive packet multiplication (multicast broadcast storms). Multicast routing protocols construct **directed acyclic spanning trees** connecting the source(s) to all subscribed receivers, guaranteeing that every receiver gets exactly one copy of each packet without loops.

---

#### Strategy 1: Source-Based Trees (Shortest Path Trees - SPT)
A separate, unique delivery tree is constructed for **each individual source** transmitting to group $G$. The tree is designated by the notation **$(S, G)$**, where $S$ is the source IP address and $G$ is the multicast group address.

```
       Source S1 (Root of SPT 1)                  Source S2 (Root of SPT 2)
                 |                                          |
             [Router A]                                 [Router B]
            /          \                               /          \
      [Router C]    [Router D]                   [Router E]    [Router F]
          |              |                           |              |
     Receiver R1    Receiver R2                 Receiver R1    Receiver R2

     (S1, G) Shortest Path Tree                 (S2, G) Shortest Path Tree
```

##### 1. Reverse Path Forwarding (RPF):
The foundational algorithmic primitive for loop-free source-based forwarding. RPF leverages the router’s existing **unicast routing table** without running a new routing algorithm.

```
                              [ Multicast Source S ]
                                        |
                             (Shortest Unicast Path)
                                        |
                                        v
                               +-----------------+
             Arriving Packet   |                 |   Arriving Packet
             from Interface 1  |   ROUTER R      |   from Interface 2
          ===================> |                 | <===================
                               +--------+--------+
                                        |
                              Lookup Unicast Table:
                  "What is next hop interface to reach Source S?"
                                        |
         +------------------------------+------------------------------+
         |                                                             |
   Interface 1 == RPF Interface                                  Interface 2 != RPF Interface
         |                                                             |
         v                                                             v
    [ RPF CHECK SUCCEEDS ]                                       [ RPF CHECK FAILS ]
  Forward packet copies out all                                  DROP packet immediately!
  outgoing multicast tree ports                                  (Loop Prevention)
```

##### RPF Algorithm:
```python
def process_multicast_packet(packet, incoming_interface):
    source_ip = packet.src
    group_ip = packet.dst
    
    # Query standard unicast routing table
    expected_rpf_interface = unicast_routing_table.get_next_hop_interface(source_ip)
    
    if incoming_interface == expected_rpf_interface:
        # RPF Check Succeeded
        for out_intf in multicast_tree.get_outgoing_interfaces(source_ip, group_ip):
            if out_intf != incoming_interface:
                forward_packet(packet, out_intf)
    else:
        # RPF Check Failed: packet arrived on a non-optimal detour
        drop_packet(packet)
```

##### 2. Distance Vector Multicast Routing Protocol (DVMRP, RFC 1075):
- Implements **Flood-and-Prune**: When a source first transmits, the packet is broadcast across the entire network using RPF.
- Leaf routers that have no downstream hosts subscribed to group $G$ generate upstream **`Prune`** messages.
- Routers receiving Prune messages prune that branch from their forwarding tree.
- Prune states have soft-state timeouts (typically 1 minute). When they expire, traffic is reflooded, creating periodic bandwidth spikes.

##### 3. Protocol Independent Multicast - Dense Mode (PIM-DM, RFC 3973):
- "Dense Mode" assumes that group members are densely distributed across almost every subnet in the enterprise.
- Employs RPF with Flood-and-Prune. Highly inefficient if subscribers are sparse across the Internet.

---

#### Strategy 2: Shared Trees (Core-Based Trees / CBT)
Instead of building $S$ individual trees for $S$ separate sources, a **single shared delivery tree** is constructed for the entire group $G$, designated by the notation **$(*, G)$**.
- The root of the shared tree is a designated, centralized router known as the **Rendezvous Point (RP)** or **Core Router**.
- All sources send their packets to the RP, which replicates and forwards packets down the shared tree to all receivers.

```
       Source S1                  Source S2
           \                          /
            \                        /  (Unicast PIM Registers)
             v                      v
        +--------------------------------+
        |     RENDEZVOUS POINT (RP)      |  <== ROOT OF SHARED TREE (*, G)
        +---------------+----------------+
                       / \
                      /   \
                     v     v
              +----------+ +----------+
              | Router 1 | | Router 2 |
              +----+-----+ +----+-----+
                   |            |
                   v            v
               Receiver 1   Receiver 2
```

##### Protocol Independent Multicast - Sparse Mode (PIM-SM, RFC 7761):
- "Sparse Mode" assumes that receivers are widely scattered across the network, and bandwidth must be conserved by avoiding flooding.
- **Explicit Join Architecture**: Routers do not receive multicast traffic unless they explicitly transmit an upstream **PIM Join** message.
- **Receiver Registration**: When a host joins group $G$ via IGMP, its local router sends an explicit `(*, G) Join` toward the Rendezvous Point (RP), grafting an active branch onto the shared tree.
- **Source Registration**: When Source $S$ begins transmitting, its first-hop router encapsulates the multicast data into a unicast **PIM Register** packet sent directly to the RP. The RP decapsulates the packet and distributes it down the $(*, G)$ shared tree.
- **Dynamic Switchover to Shortest Path Tree (SPT)**:
  - Shared trees introduce latency because traffic must take a triangular detour through the RP.
  - In PIM-SM, if traffic volume exceeds a configured bandwidth threshold, the receiver’s first-hop router initiates an explicit `(S, G) Join` directly toward the Source $S$ and sends a `Prune` toward the RP.
  - The flow transitions seamlessly from the $(*, G)$ shared tree to an $(S, G)$ Shortest Path Tree, minimizing latency for high-volume streams.

---

#### Comprehensive Technical Comparison: Source-Based Trees vs. Shared Trees

| Architectural Feature | Source-Based Trees (Shortest Path Trees - SPT) | Shared Trees (Core-Based Trees - CBT) |
| :--- | :--- | :--- |
| **Tree Root** | The individual sending **Source host ($S$)**. | A centralized **Rendezvous Point (RP) / Core**. |
| **Table Notation** | **$(S, G)$** state entry. | **$(*, G)$** state entry. |
| **Forwarding Paths** | **Optimal**: packets follow the mathematical shortest path from source to receiver. | **Sub-Optimal**: packets take a triangular detour through the RP (higher latency). |
| **Router State Scalability** | **Poor ($O(S \times G)$)**: A network with 50 sources and 100 groups requires 5,000 distinct tree states across core routers. | **Excellent ($O(G)$)**: Core routers store only 1 tree state per group, regardless of source count. |
| **Traffic Initial Flooding** | **Yes** (in dense mode): Floods whole network, then prunes unneeded branches. | **No**: Explicit join model; zero packets forwarded without prior subscription. |
| **Protocol Implementations**| **DVMRP**, **PIM-DM** (Protocol Independent Multicast - Dense Mode). | **PIM-SM** (Protocol Independent Multicast - Sparse Mode), **CBT**. |
| **Best Application Match** | High-bandwidth, small groups with few sources (e.g., enterprise video broadcast). | Multi-sender, interactive low-bandwidth applications (e.g., videoconferencing, distributed gaming). |

---

## 9. Comprehensive Solved Numerical Problems (Step-by-Step with Diagrams)

### 9.1 TCP Congestion Control Round-by-Round Evolution: TCP Tahoe vs. TCP Reno

**Problem Statement**:
Consider a TCP connection transmitting data across a network with a fixed Round-Trip Time (RTT).
- The Maximum Segment Size is $\text{MSS} = 1\text{ KB} = 1024\text{ bytes}$.
- Initial Congestion Window: $\text{cwnd} = 1\text{ MSS}$.
- Initial Slow Start Threshold: $\text{ssthresh} = 16\text{ MSS}$.
- The connection encounters two loss events:
  1. At **Round 8**, after transmitting with $\text{cwnd} = 32\text{ MSS}$, a packet loss is detected via **Triple Duplicate ACKs** (3 duplicate ACKs = 4 identical ACKs).
  2. At **Round 16**, a packet loss is detected via a **Retransmission Timeout**.

Trace the values of $\text{cwnd}$ and $\text{ssthresh}$ for both **TCP Tahoe** and **TCP Reno** for transmission rounds 1 through 20. Present the results in a detailed round-by-round comparative table and explain the behavioral differences.

---

#### Step-by-Step Analytical Trace:

#### Initial Phase (Rounds 1 to 8):
- Both Tahoe and Reno start in **Slow Start** with $\text{cwnd} = 1$ and $\text{ssthresh} = 16$.
- In Slow Start, $\text{cwnd}$ doubles every RTT ($1 \to 2 \to 4 \to 8 \to 16$).
- At Round 5, $\text{cwnd} = 16 = \text{ssthresh}$. Both protocols transition to **Congestion Avoidance (Additive Increase)**.
- In Congestion Avoidance, $\text{cwnd}$ increments by $+1\text{ MSS}$ per RTT ($16 \to 17 \to 18 \dots$).
- At Round 8, $\text{cwnd} = 19\text{ MSS}$. Suppose in this example transmission test, the loss occurs when $\text{cwnd} = 32\text{ MSS}$ (or let us trace standard textbook rounds starting at $\text{ssthresh}=16$).

Let us follow the canonical exam test trace:
- Round 1: $\text{cwnd} = 1$ (Slow Start)
- Round 2: $\text{cwnd} = 2$ (Slow Start)
- Round 3: $\text{cwnd} = 4$ (Slow Start)
- Round 4: $\text{cwnd} = 8$ (Slow Start)
- Round 5: $\text{cwnd} = 16 = \text{ssthresh}$ (Reaches threshold; transitions to Congestion Avoidance)
- Round 6: $\text{cwnd} = 17$ (Congestion Avoidance: $+1$)
- Round 7: $\text{cwnd} = 18$ (Congestion Avoidance: $+1$)
- Round 8: $\text{cwnd} = 19$ (Triple Duplicate ACKs detected!)

#### Reaction to Loss Event 1 (Triple Duplicate ACKs at Round 8):
- **For TCP Tahoe**:
  - Treats 3 duplicate ACKs identically to a timeout.
  - Updates $\text{ssthresh} = \lfloor \text{cwnd} / 2 \rfloor = \lfloor 19 / 2 \rfloor = \mathbf{9\text{ MSS}}$.
  - Drops $\text{cwnd}$ down to $\mathbf{1\text{ MSS}}$.
  - Re-enters **Slow Start**.
- **For TCP Reno (Fast Recovery)**:
  - Updates $\text{ssthresh} = \lfloor \text{cwnd} / 2 \rfloor = \lfloor 19 / 2 \rfloor = \mathbf{9\text{ MSS}}$.
  - Sets $\text{cwnd} = \text{ssthresh} + 3\text{ MSS} = 9 + 3 = 12\text{ MSS}$ (Fast Recovery window inflation).
  - Upon receiving the acknowledgment for the retransmitted packet, it deflates $\text{cwnd} = \text{ssthresh} = \mathbf{9\text{ MSS}}$ and re-enters **Congestion Avoidance** directly!

#### Progression (Rounds 9 to 15):
- **Tahoe**:
  - Round 9: $\text{cwnd} = 1$ (Slow Start)
  - Round 10: $\text{cwnd} = 2$ (Slow Start)
  - Round 11: $\text{cwnd} = 4$ (Slow Start)
  - Round 12: $\text{cwnd} = 8$ (Slow Start)
  - Round 13: $\text{cwnd} = 9 = \text{ssthresh}$ (Enters Congestion Avoidance)
  - Round 14: $\text{cwnd} = 10$ (Congestion Avoidance)
  - Round 15: $\text{cwnd} = 11$
  - Round 16: $\text{cwnd} = 12$ (Timeout occurs!)
- **Reno**:
  - Round 9: $\text{cwnd} = 10$ (Congestion Avoidance: $+1$)
  - Round 10: $\text{cwnd} = 11$
  - Round 11: $\text{cwnd} = 12$
  - Round 12: $\text{cwnd} = 13$
  - Round 13: $\text{cwnd} = 14$
  - Round 14: $\text{cwnd} = 15$
  - Round 15: $\text{cwnd} = 16$
  - Round 16: $\text{cwnd} = 17$ (Timeout occurs!)

#### Reaction to Loss Event 2 (Timeout at Round 16):
- **Both TCP Tahoe and TCP Reno**:
  - A timeout represents complete network stall (no packets getting through).
  - Both protocols set $\text{ssthresh} = \lfloor \text{cwnd} / 2 \rfloor$.
    - Tahoe: $\text{ssthresh} = \lfloor 12 / 2 \rfloor = \mathbf{6\text{ MSS}}$.
    - Reno: $\text{ssthresh} = \lfloor 17 / 2 \rfloor = \mathbf{8\text{ MSS}}$.
  - Both reset $\text{cwnd} = \mathbf{1\text{ MSS}}$ and enter **Slow Start**.

---

#### Complete Comparative Trace Table:

| Transmission Round (RTT) | TCP Tahoe `cwnd` (MSS) | TCP Tahoe `ssthresh` | TCP Tahoe State | TCP Reno `cwnd` (MSS) | TCP Reno `ssthresh` | TCP Reno State |
| :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **1** | 1 | 16 | Slow Start | 1 | 16 | Slow Start |
| **2** | 2 | 16 | Slow Start | 2 | 16 | Slow Start |
| **3** | 4 | 16 | Slow Start | 4 | 16 | Slow Start |
| **4** | 8 | 16 | Slow Start | 8 | 16 | Slow Start |
| **5** | 16 | 16 | Congestion Avoidance | 16 | 16 | Congestion Avoidance |
| **6** | 17 | 16 | Congestion Avoidance | 17 | 16 | Congestion Avoidance |
| **7** | 18 | 16 | Congestion Avoidance | 18 | 16 | Congestion Avoidance |
| **8** | 19 | 16 | **3 Dup ACKs Detected!** | 19 | 16 | **3 Dup ACKs Detected!** |
| **9** | **1** | **9** | **Slow Start** | **10** | **9** | **Congestion Avoidance** |
| **10** | 2 | 9 | Slow Start | 11 | 9 | Congestion Avoidance |
| **11** | 4 | 9 | Slow Start | 12 | 9 | Congestion Avoidance |
| **12** | 8 | 9 | Slow Start | 13 | 9 | Congestion Avoidance |
| **13** | 9 | 9 | Congestion Avoidance | 14 | 9 | Congestion Avoidance |
| **14** | 10 | 9 | Congestion Avoidance | 15 | 9 | Congestion Avoidance |
| **15** | 11 | 9 | Congestion Avoidance | 16 | 9 | Congestion Avoidance |
| **16** | 12 | 9 | **Timeout Detected!** | 17 | 9 | **Timeout Detected!** |
| **17** | **1** | **6** | **Slow Start** | **1** | **8** | **Slow Start** |
| **18** | 2 | 6 | Slow Start | 2 | 8 | Slow Start |
| **19** | 4 | 6 | Slow Start | 4 | 8 | Slow Start |
| **20** | 6 | 6 | Congestion Avoidance | 8 | 8 | Congestion Avoidance |

```
TCP CONGESTION WINDOW EVOLUTION GRAPH:
cwnd (MSS)
  ▲
20│                /---\ (Round 8: Loss)
  │               /     \                   /---\ (Round 16: Timeout)
15│       /------/       \  Reno --------  /     \
  │      /                \-------/       /       \
10│     /                          \     /         \
  │    /                            \   /           \
 5│   /                              \ /             \
  │  /    Tahoe drops to 1 MSS ------>X               \-----> Both drop to 1 MSS
 0└──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───► Round
     1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
```

**Key Takeaway**: TCP Reno achieves substantially higher aggregate throughput than TCP Tahoe because it skips the slow start penalty after mild losses (triple duplicate ACKs), cutting window in half instead of resetting to 1 MSS.

---

### 9.2 TCP Average Throughput & Loss Rate Calculation

**Problem Statement**:
A long-distance high-bandwidth research connection utilizes TCP Reno over an optical path:
- Round-Trip Time: $\text{RTT} = 100\text{ ms} = 0.1\text{ seconds}$.
- Maximum Segment Size: $\text{MSS} = 1460\text{ bytes} = 11{,}680\text{ bits}$.
- Observed Packet Loss Rate: $L = 10^{-4} = 0.0001$ ($0.01\%$).

1. Calculate the average throughput achieved by the TCP Reno connection.
2. Determine the average congestion window size ($\text{cwnd}$) in segments required to sustain this throughput.
3. If the user desires to achieve a throughput of $\mathbf{1\text{ Gbps}}$ across this same path, calculate the maximum tolerable packet loss rate $L$.

#### Step-by-Step Solution:

**1. Calculate Average Throughput**:
Using the canonical **Mathis et al. Macroscopic TCP Throughput Formula**:
$$\text{Average Throughput} \approx \frac{1.22 \times \text{MSS}}{\text{RTT} \times \sqrt{L}}$$

Substitute the given parameters:
$$\text{Throughput} \approx \frac{1.22 \times 11{,}680\text{ bits}}{0.1\text{ s} \times \sqrt{0.0001}} = \frac{14{,}249.6}{0.1 \times 0.01} = \frac{14{,}249.6}{0.001} = \mathbf{14{,}249{,}600\text{ bps}} \approx \mathbf{14.25\text{ Mbps}}$$

**2. Calculate Average Congestion Window ($\text{cwnd}_{avg}$)**:
$$\text{Throughput} = \frac{\text{cwnd}_{avg} \times \text{MSS}}{\text{RTT}} \implies \text{cwnd}_{avg} = \frac{\text{Throughput} \times \text{RTT}}{\text{MSS}}$$
$$\text{cwnd}_{avg} = \frac{14{,}249{,}600\text{ bps} \times 0.1\text{ s}}{11{,}680\text{ bits}} \approx \mathbf{122\text{ segments}}$$

**3. Maximum Tolerable Packet Loss Rate for 1 Gbps**:
Set $\text{Throughput} = 1\text{ Gbps} = 10^9\text{ bps}$:
$$10^9 = \frac{1.22 \times 11{,}680}{0.1 \times \sqrt{L}} = \frac{142{,}496}{\sqrt{L}}$$
$$\sqrt{L} = \frac{142{,}496}{10^9} = 1.42496 \times 10^{-4}$$
Squaring both sides:
$$L = (1.42496 \times 10^{-4})^2 \approx \mathbf{2.03 \times 10^{-8}}$$

**Engineering Insight**: To achieve $1\text{ Gbps}$ over a $100\text{ ms}$ RTT link, standard TCP Reno can tolerate no more than **one lost packet for every 50 million packets transmitted**! This extreme fragility led directly to modern high-speed congestion algorithms like **TCP CUBIC** and **BBR (Bottleneck Bandwidth and RTT)**.

---

### 9.3 IPv4 Datagram Fragmentation Across Multiple MTU Links

**Problem Statement**:
*(Adapted directly from Course Lecture Slide Deck `UE22CS252B_60c878f2...`)*

A host creates an IPv4 datagram of total length $L = 4000\text{ bytes}$ (containing a $20\text{-byte}$ header and $3980\text{ bytes}$ of application data). The datagram is assigned Identification number $\text{ID} = 777$, $\text{DF} = 0$, $\text{MF} = 0$, and $\text{Offset} = 0$.

The datagram is routed across a path made of two consecutive physical links:
- **Link 1**: Ethernet with $\text{MTU} = 1500\text{ bytes}$.
- **Link 2**: WAN link with $\text{MTU} = 800\text{ bytes}$.

1. Calculate the fragmentation parameters when traversing Link 1.
2. Suppose Fragment 1 from Link 1 now traverses Link 2 ($\text{MTU} = 800\text{ bytes}$). Show how Fragment 1 is further fragmented.

---

#### Step-by-Step Solution:

#### Step 1: Fragmentation on Link 1 ($\text{MTU} = 1500\text{ bytes}$)
- Header size = $20\text{ bytes}$.
- Maximum allowable payload per fragment $= 1500 - 20 = 1480\text{ bytes}$.
- Check 8-byte boundary constraint: $\frac{1480}{8} = 185$ (an exact integer, valid!).
- Total payload to send: $3980\text{ bytes}$.
  - Fragment 1 payload $= 1480\text{ bytes}$ (leaves $3980 - 1480 = 2500\text{ bytes}$).
  - Fragment 2 payload $= 1480\text{ bytes}$ (leaves $2500 - 1480 = 1020\text{ bytes}$).
  - Fragment 3 payload $= 1020\text{ bytes}$.

**Fragment Parameters for Link 1**:

| Fragment # | Total Length | Header | Payload Bytes | Data Range | Identification | DF | MF | Fragment Offset (units of 8B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frag 1** | $1500\text{ B}$ | $20\text{ B}$ | $1480\text{ B}$ | Bytes $0 - 1479$ | 777 | 0 | **1** | $0 / 8 = \mathbf{0}$ |
| **Frag 2** | $1500\text{ B}$ | $20\text{ B}$ | $1480\text{ B}$ | Bytes $1480 - 2959$ | 777 | 0 | **1** | $1480 / 8 = \mathbf{185}$ |
| **Frag 3** | $1040\text{ B}$ | $20\text{ B}$ | $1020\text{ B}$ | Bytes $2960 - 3979$ | 777 | 0 | **0** | $2960 / 8 = \mathbf{370}$ |

---

#### Step 2: Sub-Fragmentation of Fragment 1 on Link 2 ($\text{MTU} = 800\text{ bytes}$)
- Fragment 1 has total length $1500\text{ bytes}$ ($20\text{ B}$ header, $1480\text{ B}$ data), which exceeds Link 2's $\text{MTU} = 800\text{ bytes}$.
- Maximum payload per sub-fragment $= 800 - 20 = 780\text{ bytes}$.
- Must be a multiple of 8: $\lfloor 780 / 8 \rfloor \times 8 = 97 \times 8 = \mathbf{776\text{ bytes}}$.
- Payload breakdown of Fragment 1 ($1480\text{ bytes}$):
  - Sub-frag 1a: $776\text{ bytes}$ (leaves $1480 - 776 = 704\text{ bytes}$).
  - Sub-frag 1b: $704\text{ bytes}$. Check multiple of 8: $704 / 8 = 88$ (exact!).

**Sub-Fragment Parameters for Fragment 1 on Link 2**:

| Sub-Fragment | Total Length | Header | Payload | Original Data Range | ID | DF | MF | Fragment Offset |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frag 1a** | $796\text{ B}$ | $20\text{ B}$ | $776\text{ B}$ | Bytes $0 - 775$ | 777 | 0 | **1** | $0 / 8 = \mathbf{0}$ |
| **Frag 1b** | $724\text{ B}$ | $20\text{ B}$ | $704\text{ B}$ | Bytes $776 - 1479$ | 777 | 0 | **1** | $776 / 8 = \mathbf{97}$ |

*(Note: Frag 1b has $\text{MF} = 1$ because the original datagram still had more data following Fragment 1!)*

---

### 9.4 IPv4 Subnetting & CIDR Block Analysis

*(Directly derived from Lecture Slide Deck `UE24CS252B_257584a3...`)*

#### Problem 1: Block Analysis for `/29`, `/30`, and `/27`

**Case A: Block `123.56.77.32/29`**
1. **Subnet Mask**: $/29$ prefix means 29 ones and 3 zeros in binary:
   $$11111111.11111111.11111111.11111000_2 = \mathbf{255.255.255.248}$$
2. **Total Addresses**: Host bits $h = 32 - 29 = 3 \implies 2^3 = \mathbf{8\text{ total addresses}}$.
3. **Usable Hosts**: $2^3 - 2 = \mathbf{6\text{ usable host addresses}}$.
4. **Network Address**: $\mathbf{123.56.77.32}$
5. **Usable Host Range**: $\mathbf{123.56.77.33 \text{ to } 123.56.77.38}$
6. **Directed Broadcast Address**: $32 + 8 - 1 = \mathbf{123.56.77.39}$

**Case B: Block `180.34.64.64/30`**
1. **Subnet Mask**: $/30 \implies 11111100_2 \implies \mathbf{255.255.255.252}$
2. **Total Addresses**: $h = 32 - 30 = 2 \implies 2^2 = \mathbf{4\text{ total addresses}}$.
3. **Usable Hosts**: $2^2 - 2 = \mathbf{2\text{ usable hosts}}$ (ideal for point-to-point router links).
4. **Network Address**: $\mathbf{180.34.64.64}$
5. **Usable Host Range**: $\mathbf{180.34.64.65 \text{ to } 180.34.64.66}$
6. **Directed Broadcast Address**: $\mathbf{180.34.64.67}$

**Case C: Block `200.17.21.128/27`**
1. **Subnet Mask**: $/27 \implies 11100000_2 \implies \mathbf{255.255.255.224}$
2. **Total Addresses**: $h = 32 - 27 = 5 \implies 2^5 = \mathbf{32\text{ total addresses}}$.
3. **Usable Hosts**: $2^5 - 2 = \mathbf{30\text{ usable hosts}}$.
4. **Network Address**: $\mathbf{200.17.21.128}$
5. **Usable Host Range**: $\mathbf{200.17.21.129 \text{ to } 200.17.21.158}$
6. **Directed Broadcast Address**: $\mathbf{200.17.21.159}$

---

#### Problem 2: Partitioning `175.200.0.0/16` into 4 Equal Subnets
- Given network address: `175.200.0.0/16`.
- Required number of subnets: $4 = 2^2 \implies$ borrow **$2\text{ bits}$** from host portion.
- New prefix length: $16 + 2 = \mathbf{/18}$.
- New Subnet Mask: $\mathbf{255.255.192.0}$ (since $11000000_2 = 192$).
- Block size in 3rd octet: $256 / 4 = 64$.
- The 4 subnets are:
  1. **Subnet 0**: `175.200.0.0/18` (Range: `175.200.0.1` to `175.200.63.254`, Broadcast: `175.200.63.255`)
  2. **Subnet 1**: `175.200.64.0/18` (Range: `175.200.64.1` to `175.200.127.254`, Broadcast: `175.200.127.255`)
  3. **Subnet 2**: `175.200.128.0/18` (Range: `175.200.128.1` to `175.200.191.254`, Broadcast: `175.200.191.255`)
  4. **Subnet 3**: `175.200.192.0/18` (Range: `175.200.192.1` to `175.200.255.254`, Broadcast: `175.200.255.255`)

---

#### Problem 3: Class B Address Subnet Identification
Find the Network Address and Directed Broadcast Address for host IP `172.25.171.182` with subnet mask `255.255.224.0` (/19).

**Solution**:
1. Convert the 3rd octet of IP and mask to binary:
   - IP 3rd octet: $171 = 10101011_2$
   - Mask 3rd octet: $224 = 11100000_2$
2. Perform bitwise AND to find the Network Address 3rd octet:
   $$10101011_2 \text{ AND } 11100000_2 = 10100000_2 = \mathbf{160}$$
   $$\mathbf{\text{Network Address} = 172.25.160.0/19}$$
3. Find the Directed Broadcast Address:
   - Invert the mask: $255.255.224.0 \implies 0.0.31.255$
   - Bitwise OR:
     $$\text{Broadcast 3rd Octet} = 160 + 31 = \mathbf{191}$$
   $$\mathbf{\text{Directed Broadcast Address} = 172.25.191.255}$$

---

### 9.5 Variable-Length Subnet Masking (VLSM) Enterprise Design

**Problem Statement**:
*(Adapted directly from Slide 6 of `UE24CS252B_257584a3...`)*

An organization is allocated the block `14.24.74.0/24`. It needs to create three distinct subnets:
- **Subnet A**: Requires $120\text{ host addresses}$
- **Subnet B**: Requires $60\text{ host addresses}$
- **Subnet C**: Requires $10\text{ host addresses}$

Design the VLSM subnets efficiently, showing network address, subnet mask, usable host range, broadcast address, and unused address space.

#### Golden Rule of VLSM Design:
> **Always allocate subnets in descending order of size (largest first)** to prevent fragmentation of the address space!

#### 1. Allocate Subnet A (120 addresses):
- Address requirement: $120 + 2\text{ (network + broadcast)} = 122\text{ addresses}$.
- Smallest power of 2: $2^7 = 128 \implies h = 7\text{ host bits}$.
- Prefix length: $32 - 7 = \mathbf{/25}$.
- Subnet Mask: $\mathbf{255.255.255.128}$.
- **Network Address**: $\mathbf{14.24.74.0/25}$
- **Usable Host Range**: `14.24.74.1` to `14.24.74.126`
- **Broadcast Address**: `14.24.74.127`

#### 2. Allocate Subnet B (60 addresses):
- Address requirement: $60 + 2 = 62\text{ addresses}$.
- Smallest power of 2: $2^6 = 64 \implies h = 6\text{ host bits}$.
- Prefix length: $32 - 6 = \mathbf{/26}$.
- Subnet Mask: $\mathbf{255.255.255.192}$.
- Beginning address after Subnet A: $\mathbf{14.24.74.128/26}$
- **Usable Host Range**: `14.24.74.129` to `14.24.74.190`
- **Broadcast Address**: `14.24.74.191`

#### 3. Allocate Subnet C (10 addresses):
- Address requirement: $10 + 2 = 12\text{ addresses}$.
- Smallest power of 2: $2^4 = 16 \implies h = 4\text{ host bits}$.
- Prefix length: $32 - 4 = \mathbf{/28}$.
- Subnet Mask: $\mathbf{255.255.255.240}$.
- Beginning address after Subnet B: $\mathbf{14.24.74.192/28}$
- **Usable Host Range**: `14.24.74.193` to `14.24.74.206`
- **Broadcast Address**: `14.24.74.207`

#### 4. Unused Reserve Block:
- Addresses from `14.24.74.208` to `14.24.74.255` ($48\text{ addresses}$) remain completely free for future corporate growth.

---

### 9.6 Longest Prefix Matching (LPM) Forwarding Table Lookup

**Problem Statement**:
A core router maintains the following forwarding table using CIDR prefix notation:

| Entry # | Prefix Representation | Binary Prefix | Outgoing Interface |
| :---: | :--- | :--- | :---: |
| **1** | `200.23.16.0/21` | `11001000 00010111 00010*** ********` (21 bits) | **Interface 0** |
| **2** | `200.23.24.0/21` | `11001000 00010111 00011*** ********` (21 bits) | **Interface 1** |
| **3** | `200.23.24.0/24` | `11001000 00010111 00011000 ********` (24 bits) | **Interface 2** |
| **4** | `Default / 0` | All other addresses | **Interface 3** |

Determine the outgoing interface for each of the following destination IP addresses:
1. Destination A: `200.23.24.5`
2. Destination B: `200.23.16.1`
3. Destination C: `200.23.25.12`
4. Destination D: `128.9.176.4`

#### Step-by-Step Solution:

1. **Destination A (`200.23.24.5`)**:
   - Binary representation: `11001000.00010111.00011000.00000101`
   - Matches Entry 2 (`/21`): First 21 bits match `11001000 00010111 00011` (Match length: 21).
   - Matches Entry 3 (`/24`): First 24 bits match `11001000 00010111 00011000` (Match length: 24).
   - **Longest Prefix Match Rule**: Entry 3 has 24 matching bits vs 21 matching bits.
   - **Forwarded Out: Interface 2**.
2. **Destination B (`200.23.16.1`)**:
   - Binary representation: `11001000.00010111.00010000.00000001`
   - Matches Entry 1 (`/21`): First 21 bits match `11001000 00010111 00010` (Match length: 21).
   - Does not match Entry 2 or 3.
   - **Forwarded Out: Interface 0**.
3. **Destination C (`200.23.25.12`)**:
   - Binary representation: `11001000.00010111.00011001.00001100`
   - Check Entry 3 (`/24`): The 3rd octet $25 = 00011001_2$ does not match Entry 3's required $00011000_2$ (24th bit is 1 instead of 0). Does not match!
   - Check Entry 2 (`/21`): First 21 bits are `11001000 00010111 00011`, which matches!
   - **Forwarded Out: Interface 1**.
4. **Destination D (`128.9.176.4`)**:
   - Matches none of the specific prefixes.
   - Falls back to Default route.
   - **Forwarded Out: Interface 3**.

---

### 9.7 NAT Translation Table & Packet Rewriting Trace

**Problem Statement**:
A small office network has private subnet `192.168.1.0/24`. The NAT-enabled gateway router has public IP address `128.119.40.86`.
- Host A (`192.168.1.10`) initiates an HTTP connection from port `3345` to Web Server `142.250.190.46:80`.
- Host B (`192.168.1.20`) initiates an HTTPS connection from port `3345` (identical port number!) to Secure Server `151.101.1.69:443`.
- The NAT router assigns WAN port `5001` to Host A and WAN port `5002` to Host B.

Trace the headers of the outbound requests, NAT table entries, and inbound replies.

#### Step-by-Step Trace:

```
HOST A (192.168.1.10) ──── (1) Src: 192.168.1.10:3345 ────► [ NAT ROUTER ] ──── (2) Src: 128.119.40.86:5001 ───► WEB SERVER
                           (Dst: 142.250.190.46:80)         [ (Public IP:  ]         (Dst: 142.250.190.46:80)
                                                            [ 128.119.40.86]
HOST B (192.168.1.20) ──── (3) Src: 192.168.1.20:3345 ────► [              ] ──── (4) Src: 128.119.40.86:5002 ───► SECURE SERVER
                           (Dst: 151.101.1.69:443)                                    (Dst: 151.101.1.69:443)
```

#### 1. NAT Translation Table State:

| WAN Side Address (Public IP : Port) | LAN Side Address (Private IP : Port) | Protocol |
| :---: | :---: | :---: |
| `128.119.40.86 : 5001` | `192.168.1.10 : 3345` | TCP |
| `128.119.40.86 : 5002` | `192.168.1.20 : 3345` | TCP |

#### 2. Inbound Packet Demultiplexing Trace:
- When Web Server replies:
  - Arrives at NAT: `Src: 142.250.190.46:80`, `Dst: 128.119.40.86:5001`.
  - Router looks up Port `5001` in NAT Table $\implies$ maps to `192.168.1.10:3345`.
  - Rewrites packet: `Dst: 192.168.1.10:3345`, recalculates TCP & IP checksums, delivers to Host A.
- When Secure Server replies:
  - Arrives at NAT: `Src: 151.101.1.69:443`, `Dst: 128.119.40.86:5002`.
  - Router looks up Port `5002` in NAT Table $\implies$ maps to `192.168.1.20:3345`.
  - Rewrites packet: `Dst: 192.168.1.20:3345`, recalculates checksums, delivers to Host B.
- **Port Disambiguation**: Notice that both internal hosts used source port `3345`. NAT avoided collision by allocating distinct public ports (`5001` and `5002`).

---

### 9.8 Router Buffer Sizing Calculations

**Problem Statement**:
A core Internet router connects a $C = 10\text{ Gbps} = 10 \times 10^9\text{ bps}$ backbone fiber link. The average Round-Trip Time of TCP flows traversing this router is $\text{RTT} = 200\text{ ms} = 0.2\text{ seconds}$.

1. Calculate the required buffer capacity $B$ in Megabytes using the traditional **Bandwidth-Delay Product (BDP)** rule of thumb.
2. Suppose measurement reveals that $N = 400$ independent, desynchronized TCP flows share this link. Calculate the revised buffer capacity using the **Stanford (Appenzeller et al.) rule**.
3. What percentage reduction in buffer memory is achieved, and what is the impact on maximum queuing delay?

#### Step-by-Step Solution:

**1. Traditional Rule of Thumb**:
$$B = \text{RTT} \times C = 0.2\text{ s} \times 10 \times 10^9\text{ bps} = 2 \times 10^9\text{ bits}$$
Convert to Bytes:
$$B = \frac{2 \times 10^9\text{ bits}}{8\text{ bits/byte}} = 250{,}000{,}000\text{ bytes} = \mathbf{250\text{ MB}}$$
Maximum Queuing Delay:
$$d_{queue, max} = \frac{B}{C} = \frac{2 \times 10^9\text{ bits}}{10 \times 10^9\text{ bps}} = \mathbf{0.2\text{ seconds}} = 200\text{ ms}$$

**2. Stanford Rule for $N = 400$ Desynchronized TCP Flows**:
$$B_{Stanford} = \frac{\text{RTT} \times C}{\sqrt{N}} = \frac{250\text{ MB}}{\sqrt{400}} = \frac{250\text{ MB}}{20} = \mathbf{12.5\text{ MB}}$$
Maximum Queuing Delay under Stanford Sizing:
$$d_{queue, Stanford} = \frac{0.2\text{ s}}{20} = \mathbf{0.010\text{ seconds}} = 10\text{ ms}$$

**3. Comparative Benefit**:
- Memory Reduction: $\frac{250 - 12.5}{250} = \mathbf{95\%\text{ reduction in required buffer size}}$!
- **Engineering Advantage**: Allows router line cards to use fast on-chip **SRAM (Static RAM)** instead of slower off-chip **DRAM (Dynamic RAM)**, drastically reducing hardware manufacturing costs and slashing worst-case packet queuing latency from $200\text{ ms}$ down to $10\text{ ms}$!

---

### 9.9 Clos / Fat-Tree Data Center Architecture Calculations

**Problem Statement**:
A cloud provider designs a modern data center using a $k\text{-ary}$ Fat-Tree (Clos) network with $k = 4$ port switches:
- Every switch has $k = 4$ physical ports.

1. Calculate the total number of pods.
2. Calculate the number of Core switches, Aggregation switches, and Edge (ToR) switches.
3. Calculate the total number of physical servers (hosts) supported.
4. Calculate the bisection bandwidth and oversubscription ratio.

#### Step-by-Step Solution:

**1. Number of Pods**:
$$\text{Pods} = k = \mathbf{4\text{ pods}}$$

**2. Switch Count Calculations**:
- **Core Switches**:
  $$\text{Core Switches} = \left(\frac{k}{2}\right)^2 = \left(\frac{4}{2}\right)^2 = 2^2 = \mathbf{4\text{ Core Switches}}$$
- **Switches per Pod**:
  - Each pod contains $k/2 = 2$ Aggregation switches and $k/2 = 2$ Edge switches.
- **Total Aggregation Switches**:
  $$\text{Agg Switches} = k \times \left(\frac{k}{2}\right) = 4 \times 2 = \mathbf{8\text{ Aggregation Switches}}$$
- **Total Edge (ToR) Switches**:
  $$\text{Edge Switches} = k \times \left(\frac{k}{2}\right) = 4 \times 2 = \mathbf{8\text{ Edge Switches}}$$
- **Total Switches in Fabric**:
  $$\text{Total Switches} = 4\text{ (Core)} + 8\text{ (Agg)} + 8\text{ (Edge)} = \mathbf{20\text{ Switches}}$$

**3. Total Supported Servers**:
- Each Edge switch connects $k/2 = 2$ servers.
- Total Servers:
  $$\text{Total Servers} = (\text{Total Edge Switches}) \times \left(\frac{k}{2}\right) = 8 \times 2 = \mathbf{16\text{ Servers}}$$
- *(General Formula for $k$-ary Fat-Tree: $\text{Servers} = \frac{k^3}{4} = \frac{4^3}{4} = \frac{64}{4} = \mathbf{16\text{ Servers}}$)*.
- *(For a production $k = 48$ port switch deployment: $\frac{48^3}{4} = \mathbf{27{,}648\text{ servers supported}}$ at line-rate!)*

**4. Oversubscription Ratio**:
- Each Edge switch has 2 downlinks to servers ($2 \times 10\text{ Gbps} = 20\text{ Gbps}$) and 2 uplinks to Aggregation switches ($2 \times 10\text{ Gbps} = 20\text{ Gbps}$).
- Aggregate Uplink Capacity = Aggregate Downlink Capacity.
- **Oversubscription Ratio** $= \frac{20\text{ Gbps}}{20\text{ Gbps}} = \mathbf{1 : 1\text{ (Non-Blocking Fabric)}}$!
- Any server can communicate with any other server in the data center at full bidirectional line rate, completely eliminating the east-west throughput bottlenecks of traditional 3-tier networks.

---

*End of Unit 3 Comprehensive Study Notes.*
