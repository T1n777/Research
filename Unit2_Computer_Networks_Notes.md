# Unit 2: Application Layer (contd.) and Transport Layer

**A Complete Exam Study Reference — PES University (UE25CS243A: Computer Networks)**

---

## Table of Contents

1. [DNS — The Internet's Directory Service](#1-dns--the-internets-directory-service)
   - 1.1 [The Problem DNS Solves](#11-the-problem-dns-solves)
   - 1.2 [Why DNS Is Not Centralized](#12-why-dns-is-not-centralized)
   - 1.3 [Services Provided by DNS](#13-services-provided-by-dns)
   - 1.4 [DNS — A Distributed, Hierarchical Database](#14-dns--a-distributed-hierarchical-database)
     - [Root DNS Servers](#root-dns-servers)
     - [Top-Level Domain (TLD) Servers](#top-level-domain-tld-servers)
     - [Authoritative DNS Servers](#authoritative-dns-servers)
     - [Local DNS Name Servers (Default Name Server)](#local-dns-name-servers)
   - 1.5 [DNS Zones vs. Domains (Administrative Delegation)](#15-dns-zones-vs-domains)
   - 1.6 [Iterated vs. Recursive Queries](#16-iterated-vs-recursive-queries)
   - 1.7 [DNS Caching, Record Expiration, and TTL](#17-dns-caching-record-expiration-and-ttl)
   - 1.8 [DNS Resource Records (RR)](#18-dns-resource-records-rr)
     - [Record Format: (Name, Value, Type, TTL)](#record-format)
     - [Type A, AAAA, NS, CNAME, MX, TXT, SOA, PTR](#record-types)
   - 1.9 [Inserting Records into the DNS Database (Registrars & Domain Setup)](#19-inserting-records-into-the-dns-database)
   - 1.10 [DNS Protocol Message Format](#110-dns-protocol-message-format)
   - 1.11 [DNS Security: DNSSEC, Cache Poisoning, and DDoS](#111-dns-security)
   - 1.12 [Practical DNS Diagnostic Tools (nslookup, dig, whois)](#112-practical-dns-diagnostic-tools)
   - 1.13 [Official Slide Review Questions and Authoritative Answers](#113-official-slide-review-questions-and-authoritative-answers)
2. [Peer-to-Peer (P2P) Architecture](#2-peer-to-peer-p2p-architecture)
   - 2.1 [Client-Server vs. Peer-to-Peer File Distribution](#21-client-server-vs-peer-to-peer-file-distribution)
   - 2.2 [Mathematical Formulation of Distribution Time: D_cs vs. D_p2p](#22-mathematical-formulation-of-distribution-time)
   - 2.3 [BitTorrent Protocol Deep-Dive](#23-bittorrent-protocol-deep-dive)
     - [Torrents, Trackers, Peers, and Swarms](#torrents-trackers-peers-and-swarms)
     - [Chunking and Chunk Acquisition: Rarest-First Policy](#chunking-and-rarest-first)
     - [Trading Algorithm: Tit-for-Tat Choking and Optimistic Unchoking](#tit-for-tat)
     - [Free-Rider Mitigation and Swarm Dynamics](#free-rider-mitigation)
3. [Socket Programming with UDP and TCP](#3-socket-programming-with-udp-and-tcp)
   - 3.1 [The Socket Abstraction](#31-the-socket-abstraction)
   - 3.2 [Socket API System Calls and Timing Flow Diagram](#32-socket-api-system-calls-and-timing-flow-diagram)
   - 3.3 [Socket Programming with UDP (Connectionless) in Python](#33-socket-programming-with-udp-in-python)
   - 3.4 [Socket Programming with TCP (Connection-Oriented) in Python](#34-socket-programming-with-tcp-in-python)
4. [Other Application-Layer Protocols](#4-other-application-layer-protocols)
   - 4.1 [FTP (File Transfer Protocol) — RFC 959](#41-ftp-file-transfer-protocol)
   - 4.2 [Electronic Mail: SMTP, POP3, and IMAP](#42-electronic-mail-smtp-pop3-and-imap)
   - 4.3 [DHCP (Dynamic Host Configuration Protocol) — RFC 2131](#43-dhcp-dynamic-host-configuration-protocol)
   - 4.4 [SNMP (Simple Network Management Protocol) — RFC 1157](#44-snmp-simple-network-management-protocol)
   - 4.5 [Telnet and SSH (Secure Shell) — Remote Access](#45-telnet-and-ssh-remote-access)
   - 4.6 [Comprehensive Application Protocol Port Summary Table](#46-comprehensive-application-protocol-port-summary-table)
5. [Introduction to Transport-Layer Services](#5-introduction-to-transport-layer-services)
   - 5.1 [Process-to-Process Logical Communication](#51-process-to-process-logical-communication)
   - 5.2 [Transport vs. Network Layer: The Household Analogy](#52-transport-vs-network-layer-the-household-analogy)
   - 5.3 [Principal Internet Transport-Layer Protocols (TCP vs. UDP)](#53-principal-internet-transport-layer-protocols)
6. [Multiplexing and Demultiplexing](#6-multiplexing-and-demultiplexing)
   - 6.1 [How Multiplexing and Demultiplexing Work](#61-how-multiplexing-and-demultiplexing-work)
   - 6.2 [Port Numbers and Port Number Ranges](#62-port-numbers-and-port-number-ranges)
   - 6.3 [Connectionless Demultiplexing (UDP)](#63-connectionless-demultiplexing-udp)
   - 6.4 [Connection-Oriented Demultiplexing (TCP)](#64-connection-oriented-demultiplexing-tcp)
   - 6.5 [Server Connection Capacity Limits](#65-server-connection-capacity-limits)
7. [Connectionless Transport: UDP (User Datagram Protocol)](#7-connectionless-transport-udp)
   - 7.1 [Why UDP Exists (RFC 768 Philosophy)](#71-why-udp-exists)
   - 7.2 [UDP Segment Header Structure](#72-udp-segment-header-structure)
   - 7.3 [UDP Checksum Calculation and Verification (One's Complement Sum)](#73-udp-checksum-calculation-and-verification)
8. [Principles of Reliable Data Transfer (RDT)](#8-principles-of-reliable-data-transfer-rdt)
   - 8.1 [Why Reliable Data Transfer Is Needed](#81-why-reliable-data-transfer-is-needed)
   - 8.2 [Building Blocks of Reliable Data Transfer](#82-building-blocks-of-reliable-data-transfer)
   - 8.3 [Step-by-Step Evolution: RDT 1.0 to RDT 3.0](#83-step-by-step-evolution-rdt-10-to-rdt-30)
   - 8.4 [Operation of RDT 3.0 (Alternating Bit Protocol)](#84-operation-of-rdt-30)
   - 8.5 [Performance Analysis of Stop-and-Wait Operation](#85-performance-analysis-of-stop-and-wait-operation)
9. [Pipelining: Go-Back-N (GBN) and Selective Repeat (SR)](#9-pipelining-go-back-n-and-selective-repeat)
   - 9.1 [The Pipelining Concept](#91-the-pipelining-concept)
   - 9.2 [Go-Back-N (GBN) Protocol](#92-go-back-n-gbn-protocol)
   - 9.3 [Selective Repeat (SR) Protocol](#93-selective-repeat-sr-protocol)
   - 9.4 [Sequence Number Space Constraints & The SR Dilemma](#94-sequence-number-space-constraints--the-sr-dilemma)
   - 9.5 [Head-to-Head Comparison: Stop-and-Wait vs. GBN vs. SR](#95-head-to-head-comparison)
10. [Connection-Oriented Transport: TCP (Transmission Control Protocol)](#10-connection-oriented-transport-tcp)
    - 10.1 [TCP Key Properties](#101-tcp-key-properties)
    - 10.2 [TCP Segment Architecture (Full 32-Bit Header Diagram)](#102-tcp-segment-architecture)
    - 10.3 [Sequence Numbers and Acknowledgments (Telnet Piggybacking)](#103-sequence-numbers-and-acknowledgments)
    - 10.4 [Round-Trip Time (RTT) Estimation, Karn's Algorithm, and Timeout](#104-round-trip-time-estimation-and-timeout)
    - 10.5 [Reliable Data Transfer, RFC 5681 ACK Rules, and Fast Retransmit](#105-reliable-data-transfer-and-fast-retransmit)
    - 10.6 [TCP Flow Control (Receive Window rwnd & Zero-Window Probing)](#106-tcp-flow-control)
    - 10.7 [TCP Connection Management (3-Way Handshake, SYN Cookies, 4-Way Close, TIME_WAIT)](#107-tcp-connection-management)
11. [Comprehensive Solved Numerical Problems](#11-comprehensive-solved-numerical-problems)
    - 11.1 [UDP Checksum Calculation (One's Complement Sum with Overflow Wrap-Around)](#111-udp-checksum-calculation)
    - 11.2 [RDT 3.0 Stop-and-Wait Utilization on High-Speed Links](#112-rdt-30-stop-and-wait-utilization)
    - 11.3 [Go-Back-N vs. Selective Repeat Transmission Count](#113-go-back-n-vs-selective-repeat-transmission-count)
    - 11.4 [TCP RTT Estimation (EstimatedRTT, DevRTT, TimeoutInterval)](#114-tcp-rtt-estimation)
    - 11.5 [TCP Flow Control and Buffer Management (Computing rwnd)](#115-tcp-flow-control-and-buffer-management)
    - 11.6 [TCP Three-Way Handshake Sequence Number Tracking](#116-tcp-three-way-handshake-sequence-number-tracking)
    - 11.7 [P2P vs. Client-Server File Distribution Time](#117-p2p-vs-client-server-file-distribution-time)

---

## 1. DNS — The Internet's Directory Service

### 1.1 The Problem DNS Solves

Computers and routers communicate using fixed-length, numerically structured **IP (Internet Protocol)** addresses:
- **IPv4 (Internet Protocol Version 4)**: 32 bits wide, written in dotted-decimal notation (e.g., `142.250.190.46`).
- **IPv6 (Internet Protocol Version 6)**: 128 bits wide, written in hexadecimal colon-separated notation (e.g., `2607:f8b0:4005:805::200e`).

While routers require fixed-width binary addresses for high-speed hardware lookup in forwarding tables, human beings struggle to memorize arbitrary 32-bit strings. Humans rely on intuitive, mnemonic, alphabetic hostnames (e.g., `www.google.com`, `pes.edu`).

The **DNS (Domain Name System)** is the Internet's globally distributed directory service that translates mnemonic hostnames into machine-readable IP addresses. It functions as the **application-layer telephone book** of the Internet, implemented as an application-layer protocol running over **UDP (User Datagram Protocol)** and **TCP (Transmission Control Protocol)** on **port 53**.

```
+------------------+     Hostname: "www.pes.edu"      +-------------------+
|                  | -------------------------------> |                   |
|   User / Client  |                                  |    DNS Server     |
|   Application    | <------------------------------- |                   |
+------------------+        IPv4: "14.139.156.4"      +-------------------+
```

---

### 1.2 Why DNS Is Not Centralized

A naive design would place all hostname-to-IP mappings in a single, colossal centralized database server hosted on the Internet. This design is fundamentally unworkable due to four fatal scaling bottlenecks:

1. **Single Point of Failure (SPOF)**: If the centralized server crashes, suffers a power failure, or catches fire, the entire global Internet collapses instantly. No user could browse the web, send emails, or resolve services.
2. **Unmanageable Traffic Volume**: The volume of queries would completely saturate any server or network link. A single tier-1 ISP like Comcast processes over **600 billion DNS queries per day**. A single server handling planetary DNS traffic would collapse within milliseconds.
3. **Severe Propagation and Queuing Latency**: A centralized server located in North America would impose massive physical propagation delays (hundreds of milliseconds) on users in India, Australia, or South America, severely degrading web browsing responsiveness.
4. **Impossible Maintenance and Database Updates**: Millions of new devices connect, disconnect, and change IP addresses daily. Storing, updating, locking, and synchronizing a centralized database containing billions of dynamic records across global administrative boundaries would create intolerable administrative and computational gridlock.

**Solution**: DNS is engineered as a **distributed, hierarchical database** running across hundreds of thousands of independent servers worldwide.

---

### 1.3 Services Provided by DNS

Beyond simple hostname-to-IP address translation, DNS provides several critical auxiliary networking services:

1. **Hostname-to-IP Translation**: Maps fully qualified domain names (FQDN - Fully Qualified Domain Name) to 32-bit IPv4 addresses (via **Type A** records) or 128-bit IPv6 addresses (via **Type AAAA** records).
2. **Host Aliasing (Canonical vs. Alias Names)**: A host with an awkward or complex canonical hostname (e.g., `server-us-east-1.cdn.pes.edu`) can have one or more user-friendly alias names (e.g., `www.pes.edu`). DNS transparently resolves the alias to its canonical name and associated IP.
3. **Mail Server Aliasing**: Permits email addresses to use clean, simple domain names (e.g., `student@pes.edu`) rather than explicit mail server hostnames (e.g., `mx1.smtp.mail.pes.edu`). DNS returns the designated mail transfer agent for that domain via **Type MX** records.
4. **Load Distribution (Load Balancing)**: Heavily trafficked websites replicate content across redundant servers located around the world. A single hostname (e.g., `www.cnn.com`) maps to a set of multiple distinct IP addresses. When a DNS client queries the name, the DNS server responds with the entire set of IP addresses, but rotates the ordering cyclically in each response (**Round-Robin DNS**). Clients typically connect to the first IP in the list, distributing user requests evenly across the server pool.

---

### 1.4 DNS — A Distributed, Hierarchical Database

To achieve planetary scale, DNS employs an inverted hierarchical tree structure:

```
                                [ Root DNS Servers ]
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          ▼                              ▼                              ▼
   [ .com TLD Servers ]          [ .edu TLD Servers ]          [ .org TLD Servers ]
          │                              │                              │
          ▼                              ▼                              ▼
[ amazon.com Servers ]          [ pes.edu Servers ]            [ pbs.org Servers ]
 (Authoritative)                 (Authoritative)                (Authoritative)
```

#### 1. Root DNS Servers
- Form the apex of the DNS hierarchy.
- In the global Internet, there are **13 logical Root DNS server addresses** named `a.root-servers.net` through `m.root-servers.net`.
- Although there are only 13 logical addresses (constrained originally by the 512-byte UDP DNS packet limit), each logical root server is actually a massively replicated cluster of hundreds of physical servers distributed worldwide using **IP Anycast** routing (over 1,500 physical root server instances exist today).
- Root servers do not know the IP address of `www.pes.edu`; instead, they inspect the label `.edu` and return the IP address of a **TLD (Top-Level Domain)** name server responsible for `.edu`.

#### 2. Top-Level Domain (TLD) Servers
- Responsible for top-level domains such as:
  - **Generic TLDs (gTLD)**: `.com`, `.org`, `.net`, `.edu`, `.gov`, `.mil`, `.biz`, `.info`.
  - **Country-Code TLDs (ccTLD)**: `.in` (India), `.uk` (United Kingdom), `.jp` (Japan), `.de` (Germany), `.ca` (Canada).
- Maintained by commercial entities and international registries:
  - Verisign maintains `.com` and `.net` TLD servers.
  - Educause maintains the `.edu` TLD server.
- TLD servers do not hold end-host IP addresses; they return the IP address of the **Authoritative DNS server** responsible for the target domain (e.g., `pes.edu`).

#### 3. Authoritative DNS Servers
- Every organization whose hosts can be publicly accessed over the Internet must provide publicly accessible DNS records that map its hostnames to IP addresses.
- An organization's **authoritative DNS server** houses these official mappings.
- Organizations can either host their own authoritative DNS servers (e.g., `ns1.pes.edu`, `ns2.pes.edu`) or pay a commercial cloud DNS provider (e.g., Amazon Route 53, Cloudflare, Akamai) to host their authoritative records.

#### 4. Local DNS Name Servers (Default Name Server)
- A **Local DNS Server** (also known as a **Resolving Name Server** or **Recursive Resolver**) does not strictly belong to the formal hierarchical tree, but is central to practical DNS operation.
- Every ISP (Internet Service Provider) — residential ISP, mobile cellular network, or university campus network — operates one or more Local DNS Servers.
- When an end host connects to a network via **DHCP (Dynamic Host Configuration Protocol)**, it automatically receives the IP address of its Local DNS Server (e.g., Google's public resolver `8.8.8.8`, Cloudflare's `1.1.1.1`, or an ISP's internal resolver).
- When a client application (e.g., a web browser) issues a DNS lookup request, the request is sent directly to the host's Local DNS Server, which acts as a proxy, navigating the global hierarchy on behalf of the client.

---

### 1.5 DNS Zones vs. Domains (Administrative Delegation)

A common point of confusion in networking exams is the distinction between a **Domain** and a **Zone**:

- **Domain**: An entire subtree within the DNS naming hierarchy (e.g., `pes.edu`, which includes `cse.pes.edu`, `ece.pes.edu`, `library.pes.edu`, and all subdomains beneath it).
- **Zone**: A contiguous, administrative unit of the domain namespace over which a specific administrative authority exercises direct management and control.
- **Delegation**: An organization can choose to manage its entire domain as a single zone, or it can partition its domain into multiple sub-zones and **delegate** administrative responsibility for those sub-zones to independent name servers.

**Example**:
PES University owns the domain `pes.edu`. Central IT manages the main `pes.edu` zone. However, the Computer Science and Engineering department requires frequent record changes and rapid testing. Central IT delegates the sub-zone `cse.pes.edu` to a separate DNS server managed directly by the CSE department.
1. The `pes.edu` authoritative server contains **NS** and glue **A** records delegating `cse.pes.edu` to `ns1.cse.pes.edu`.
2. A configuration error or server crash in `cse.pes.edu` does not impact the availability of `ece.pes.edu` or `pes.edu`.
3. Delegation enables distributed administrative scalability without bureaucratic overhead.

---

### 1.6 Iterated vs. Recursive Queries

When a host requests hostname resolution, the resolution across the hierarchy can proceed via two fundamental modes:

```
                  ITERATIVE QUERY RESOLUTION                          RECURSIVE QUERY RESOLUTION
              (Standard Internet Architecture)                    (Heavy Load on Hierarchy)

                 +---------------------+                             +---------------------+
                 |   Root DNS Server   |                             |   Root DNS Server   |
                 +---------------------+                             +---------------------+
                        ▲       │ (2)                                       ▲       │ (3)
                    (1) │       ▼                                       (2) │       ▼
                 +---------------------+                             +---------------------+
                 |  Local DNS Server   |                             |   TLD DNS Server    |
                 +---------------------+                             +---------------------+
                   ▲   │ (3)   ▲   │ (5)                                    ▲       │ (5)
               (0) │   ▼       │   ▼                                    (4) │       ▼
+----------+       │  +-------------+  +---------------+         +---------------------+
|  Client  | <─────┘  | TLD Server  |  | Authoritative |         | Authoritative Server|
|   Host   |          +-------------+  +---------------+         +---------------------+
+----------+                                                            ▲
                                                                    (1) │ (6)
                                                                 +---------------------+
                                                                 |  Local DNS Server   |
                                                                 +---------------------+
```

#### 1. Iterative Queries ("I don't know, but ask this server next")
- In an **iterated query**, when a queried server does not know the exact mapping for the requested hostname, it replies with the IP address of the next DNS server down the hierarchy.
- The querying host (almost always the **Local DNS Server**) retains the burden of issuing the subsequent query to the referred server.
- **Standard Internet Workflow**:
  1. Client host sends a recursive query to its **Local DNS Server**: *"What is the IP of `www.pes.edu`?"*
  2. Local DNS server queries a **Root DNS Server** iteratively: *"What is `www.pes.edu`?"*
  3. Root server responds: *"I do not know, but here is the IP address of the `.edu` TLD server."*
  4. Local DNS server queries the **`.edu` TLD Server**: *"What is `www.pes.edu`?"*
  5. TLD server responds: *"I do not know, but here is the IP address of `ns1.pes.edu` (Authoritative Server)."*
  6. Local DNS server queries `ns1.pes.edu`: *"What is `www.pes.edu`?"*
  7. Authoritative server replies: *"The IP address is `14.139.156.4`."*
  8. Local DNS server caches the answer and returns the IP address to the requesting client host.

#### 2. Recursive Queries ("Please find the answer for me and return it")
- In a **recursive query**, the queried server assumes the entire responsibility of finding the final answer, querying subsequent servers on behalf of the original requester, and passing the final result back up the chain.
- **Why Root and TLD Servers Reject Recursive Queries**:
  If Root and TLD servers performed recursive queries, millions of concurrent open connection states would have to be maintained at the top of the pyramid. A server waiting for responses from downstream servers would consume massive memory buffers and thread pools, rendering it trivial to take down via **DDoS (Distributed Denial of Service)** attacks. Consequently, **Root and TLD servers operate exclusively in iterative mode** (enforced by clearing the **RA - Recursion Available** flag).

---

### 1.7 DNS Caching, Record Expiration, and TTL

DNS latency is minimized through aggressive **caching**:
- Whenever any DNS server (especially a Local DNS Server) receives a mapping during a resolution walk, it caches the mapping in its local memory.
- If another local client subsequently requests the same hostname, the Local DNS Server answers immediately from memory, bypassing the Root, TLD, and Authoritative servers entirely.
- TLD server IP addresses are almost permanently cached in Local DNS servers, meaning root servers are bypassed for the vast majority of day-to-day Internet queries.

#### The Role of TTL (Time-To-Live)
- Because IP addresses can change over time, cached entries cannot be stored indefinitely. Every DNS resource record includes a **TTL (Time-To-Live)** field, specified in seconds (e.g., 86400 seconds = 24 hours).
- Once the TTL countdown reaches zero, the DNS resolver purges the record from its cache. The next request triggers a fresh resolution walk.

#### Operational Strategy for Planned IP Address Changes:
> **Exam Tip / Slide Question**: If a company changes a server's IP address, why do some users reach the old IP for hours? How is this minimized operationally?
- **Root Cause**: DNS caching is completely decentralized and best-effort. Resolvers worldwide cache records until their respective TTLs expire. There is no global "cache flush" command across the Internet.
- **Standard Operational Solution**:
  1. **Step 1 (Pre-migration)**: Days before the scheduled server migration, lower the record's TTL from a high value (e.g., 86400 seconds) to a very short duration (e.g., 300 seconds / 5 minutes).
  2. **Step 2 (Draining phase)**: Wait out the old TTL period (24 hours) to ensure all old, high-TTL cache entries expire across all global resolvers.
  3. **Step 3 (Cutover)**: Update the IP address to the new server. Because the active TTL is only 300 seconds, clients will fetch the new IP address within 5 minutes.
  4. **Step 4 (Post-migration)**: Once the new server is verified to be stable, raise the TTL back to 86400 seconds to conserve bandwidth and reduce lookup latency.

---

### 1.8 DNS Resource Records (RR)

The DNS distributed database stores information in the form of **Resource Records (RRs)**. Every DNS reply packet contains one or more RRs.

#### 1. General Resource Record Format
A DNS resource record is formally defined as a 4-tuple:
$$\mathbf{(Name,\; Value,\; Type,\; TTL)}$$

- **Name**: The domain name or hostname.
- **Value**: The data corresponding to the name (an IP address, an alias, or another hostname).
- **Type**: Defines how `Name` and `Value` are to be interpreted.
- **TTL**: The duration in seconds that the record may remain cached.

#### 2. The Standard DNS Record Types

| Record Type | Description | Name Semantics | Value Semantics | Concrete Example |
| :--- | :--- | :--- | :--- | :--- |
| **A** | IPv4 Host Address | Hostname | 32-bit IPv4 Address | `(www.pes.edu, 14.139.156.4, A, 86400)` |
| **AAAA** | IPv6 Host Address | Hostname | 128-bit IPv6 Address | `(www.pes.edu, 2404:6800:4009:803::2004, AAAA, 86400)` |
| **NS** | Authoritative Name Server | Domain Name | Hostname of Authoritative DNS Server | `(pes.edu, ns1.pes.edu, NS, 86400)` |
| **CNAME** | Canonical Name (Alias) | Alias Hostname | True (Canonical) Hostname | `(www.pes.edu, webserver01.pes.edu, CNAME, 86400)` |
| **MX** | Mail Exchange | Domain Name | Hostname of Mail Server + Priority | `(pes.edu, mail.pes.edu [Priority 10], MX, 86400)` |
| **TXT** | Text Record | Domain Name | Arbitrary Text Strings | `(pes.edu, "v=spf1 include:_spf.google.com ~all", TXT, 3600)` |
| **SOA** | Start of Authority | Zone Name | Primary server, admin email, zone serial # | `(pes.edu, ns1.pes.edu admin.pes.edu 2026091801 ..., SOA, 86400)` |
| **PTR** | Pointer (Reverse DNS) | Inverted IP (`in-addr.arpa`) | Canonical Hostname | `(4.156.139.14.in-addr.arpa, www.pes.edu, PTR, 86400)` |

- **CNAME vs. MX Distinctions**:
  - `CNAME` can alias any generic host (e.g., pointing `www.ibm.com` to `servereast.backup2.ibm.com`).
  - `MX` specifically routes email for an entire domain and includes an integer **preference / priority value** (e.g., 10, 20), enabling mail clients to fail over to secondary backup mail servers if the primary mail server is offline.
- **TXT Records**: Used extensively for security and domain verification:
  - **SPF (Sender Policy Framework)**: Specifies which mail servers are authorized to send email on behalf of the domain.
  - **DKIM (DomainKeys Identified Mail)**: Publishes the domain's public cryptographic key to verify cryptographic signatures on outgoing emails.

---

### 1.9 Inserting Records into the DNS Database

How does a new organization or website get its name into the global DNS system?

#### Step 1: Registering with an Accredited Registrar
- A **registrar** is a commercial entity accredited by **ICANN (Internet Corporation for Assigned Names and Numbers)** to assign domain names (e.g., GoDaddy, Namecheap, Google Domains).
- The applicant checks domain availability and pays an annual registration fee.

#### Step 2: Providing Authoritative DNS Details
Suppose an entrepreneur registers a new domain name: `networkutopia.com`.
The owner must supply the registrar with the names and IP addresses of its primary and secondary authoritative DNS servers. Suppose these are:
- Primary: `dns1.networkutopia.com` (`212.212.212.1`)
- Secondary: `dns2.networkutopia.com` (`212.212.212.2`)

#### Step 3: Injection of Records into the TLD Servers
The registrar contacts the operator of the `.com` TLD servers (Verisign) and inserts two pairs of records into the `.com` TLD infrastructure:
1. **NS Records** (mapping the domain to its authoritative servers):
   $$(networkutopia.com,\; dns1.networkutopia.com,\; NS)$$
   $$(networkutopia.com,\; dns2.networkutopia.com,\; NS)$$
2. **Glue A Records** (resolving the authoritative server names themselves to IP addresses so resolvers do not get trapped in an infinite chicken-and-egg lookup loop):
   $$(dns1.networkutopia.com,\; 212.212.212.1,\; A)$$
   $$(dns2.networkutopia.com,\; 212.212.212.2,\; A)$$

#### Step 4: Configuration of Authoritative Name Server Records
On the authoritative server (`dns1.networkutopia.com`), the administrator populates the local zone file with the actual host records:
- For the primary web server:
  $$(www.networkutopia.com,\; 212.212.212.10,\; A)$$
- For the corporate mail server:
  $$(networkutopia.com,\; mail.networkutopia.com,\; MX,\; 10)$$
  $$(mail.networkutopia.com,\; 212.212.212.20,\; A)$$

Once inserted, any host on earth can resolve `www.networkutopia.com` via normal DNS hierarchy traversal.

---

### 1.10 DNS Protocol Message Format

DNS queries and replies share an identical message structure defined in **RFC 1035**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-----------------------------------+-----------------------------------+
|          Identification (16 bits) |               Flags (16 bits)     |
+-----------------------------------+-----------------------------------+
|      Number of Questions (16 bits)|     Number of Answer RRs (16 bits)|
+-----------------------------------+-----------------------------------+
|      Number of Authority RRs (16) |    Number of Additional RRs (16)  |
+-----------------------------------+-----------------------------------+
|                                                                       |
|                          Questions Section                            |
|                                                                       |
+-----------------------------------------------------------------------+
|                          Answers Section                              |
+-----------------------------------------------------------------------+
|                         Authority Section                             |
+-----------------------------------------------------------------------+
|                        Additional Section                             |
+-----------------------------------------------------------------------+
```

#### 1. Header Section (12 Bytes Fixed Length)
- **Identification (16 bits)**: A unique transaction identifier generated by the client. The server copies this exact identifier into its reply, enabling the client to match asynchronous UDP replies with pending queries.
- **Flags (16 bits)**:
  - **QR (Query/Response bit, 1 bit)**: `0` indicates a Query; `1` indicates a Response.
  - **Opcode (4 bits)**: `0` for standard query, `1` for inverse query, `2` for server status.
  - **AA (Authoritative Answer, 1 bit)**: Set to `1` in replies if the responding server is an authoritative server for the requested domain.
  - **TC (Truncated, 1 bit)**: Set to `1` if the message exceeded the 512-byte UDP limit, advising the client to re-issue the query over **TCP**.
  - **RD (Recursion Desired, 1 bit)**: Set by the client if it requests recursive resolution.
  - **RA (Recursion Available, 1 bit)**: Set by the server in replies if it supports recursive queries (cleared by Root and TLD servers).
  - **RCODE (Response Code, 4 bits)**: `0` = No error; `3` = **NXDOMAIN (Non-Existent Domain)**, indicating the hostname does not exist.
- **Count Fields (4 x 16 bits = 8 bytes)**: Number of Question entries, Answer RRs, Authority RRs, and Additional RRs.

#### 2. Body Sections
- **Questions**: Contains the domain name being looked up and the query Type (e.g., Type A).
- **Answers**: Contains the RRs satisfying the query.
- **Authority**: Contains NS records pointing to other authoritative name servers.
- **Additional**: Contains helpful auxiliary records (such as the IP address corresponding to a hostname listed in an MX or NS record).

---

### 1.11 DNS Security

Because the original DNS specification (1983) lacked authentication, modern DNS incorporates crucial security enhancements:

1. **DNSSEC (Domain Name System Security Extensions)**:
   - Provides **cryptographic authentication and data integrity** to DNS responses.
   - Authoritative servers sign resource record sets using public-key cryptography (**RRSIG** records).
   - Resolvers verify signatures along a **chain of trust** extending from the signed Root zone down through TLDs to authoritative servers.
   - *Crucial note*: DNSSEC does not provide encryption or confidentiality (queries are still readable in plaintext); it prevents forgery and tampering.
2. **DNS Cache Poisoning (DNS Spoofing)**:
   - An attacker floods a Local DNS Server with forged DNS responses containing fraudulent IP addresses before the legitimate authoritative server responds.
   - If accepted, the Local DNS Server caches the malicious IP, silently redirecting all local users to phishing or malicious servers.
   - Mitigated by randomizing transaction IDs, randomizing UDP source port numbers, and deploying DNSSEC.
3. **DNS Amplification DDoS Attacks**:
   - Attackers spoof the victim's IP address and send small DNS queries (e.g., 60 bytes) requesting massive response records (e.g., `ANY` query yielding 3000+ bytes) to open recursive DNS resolvers.
   - The resolvers reflect and amplify traffic by a factor of $50\times$, saturating the victim's network link.

---

### 1.12 Practical DNS Diagnostic Tools

Network administrators analyze DNS operations using standard command-line diagnostic utilities:

#### 1. `nslookup` (Name Server Lookup)
Interactive or non-interactive tool to query Internet domain name servers:
```bash
# Query IPv4 address for a domain:
nslookup www.pes.edu

# Query specific authoritative name servers for MX records:
nslookup -type=MX pes.edu 8.8.8.8
```

#### 2. `dig` (Domain Information Groper)
More flexible and detailed command-line tool preferred by network engineers. It outputs the exact raw DNS response sections, flags, and query round-trip time:
```bash
# Trace full hierarchical iteration step-by-step from root:
dig +trace www.pes.edu

# Query specific record type:
dig pes.edu ANY
```

#### 3. `whois`
Queries regional Internet registries (e.g., ARIN, RIPE, APNIC) to display domain ownership, registrant contact information, registrar name, creation date, and designated authoritative name servers:
```bash
whois pes.edu
```

---

### 1.13 Official Slide Review Questions and Authoritative Answers

*(Directly derived from Course Lecture Slides & Exam Review Materials)*

1. **Q: Why can't we just use IP addresses everywhere and skip names entirely?**
   - **A**: Names are human-memorable and stable even when underlying server hardware or IP addresses change (e.g., during web hosting migrations or multihoming). DNS exists specifically to bridge the semantic divide between human mnemonic convenience and router routing-table efficiency.
2. **Q: Why isn't DNS just one giant centralized server?**
   - **A**: A centralized server cannot scale. It creates a catastrophic Single Point of Failure (SPOF), cannot sustain planetary query volume (e.g., Comcast alone processes 600B+ queries/day), introduces huge propagation delays for geographically distant hosts, and makes simultaneous record updates administratively impossible.
3. **Q: Put these in order of query traversal: TLD server, Authoritative server, Root server.**
   - **A**: **Root Server $\to$ TLD Server $\to$ Authoritative Server**. The Root points to the responsible TLD server; the TLD points to the organization's authoritative server; the Authoritative server delivers the actual IP address mapping.
4. **Q: What is the difference between a domain and a zone?**
   - **A**: A **domain** is an entire naming subtree within the DNS hierarchical namespace (e.g., everything under `pes.edu`). A **zone** is an administrative boundary — a contiguous portion of the domain managed and maintained directly by a specific administrative entity. A single domain can be partitioned into multiple delegated zones.
5. **Q: Give a practical reason an organization would split its domain into multiple zones.**
   - **A**: To **delegate administrative control** and establish fault isolation. For example, PES University central IT can delegate the `cse.pes.edu` sub-zone to the Computer Science department. CSE faculty can rapidly add, delete, and test departmental lab server records without filing tickets with central IT, and any misconfigurations inside the CSE zone will not disrupt university-wide records.
6. **Q: In an iterated query, what does a server say when it doesn't know the answer?**
   - **A**: It returns a **referral response**: *"I do not know the answer to this name, but here is the IP address of the next DNS server down the hierarchy that you should ask."* It returns an NS record and delegates resolution responsibility back to the querying resolver.
7. **Q: In a recursive query, who does the extra work of following referrals?**
   - **A**: The **queried server** does the work. When it receives a recursive query, it contacts downstream servers on the client's behalf, waits for replies, and returns the final mapped answer to the requester.
8. **Q: Which DNS flag in a response tells you whether a server is willing to do recursive resolution for you?**
   - **A**: The **RA (Recursion Available)** flag bit in the 16-bit flags field of the DNS header. Root and TLD servers deliberately clear this flag (`RA = 0`) because they refuse to perform recursive lookups.
9. **Q: What does a TTL control, and what happens right after it hits zero?**
   - **A**: The **TTL (Time-To-Live)** specifies the maximum duration (in seconds) that a client or intermediate resolver is permitted to cache and serve a DNS resource record. The instant TTL hits zero, the entry is purged from the cache, and any subsequent query must walk the DNS hierarchy fresh.
10. **Q: A company changes a server's IP address. Why might some users still reach the old IP for hours afterward?**
    - **A**: DNS caching is completely decentralized and best-effort. Intermediate resolving servers across the globe that already cached the previous record will continue serving the old IP address from cache until their local TTL countdown expires. The domain owner has no technical mechanism to forcibly flush third-party ISP caches.
11. **Q: What is the standard operational fix for minimizing disruption during a planned IP change?**
    - **A**: Lower the record's TTL significantly (e.g., down to 300 seconds) days before the scheduled migration. Wait out the duration of the old, longer TTL so all old cache entries expire globally. Perform the server IP switchover (downtime/staleness is now bounded to only 300 seconds). Once stable, raise the TTL back to its standard duration (e.g., 86400 seconds).
12. **Q: Name the four standard DNS record types and what each maps.**
    - **A**:
      - **Type A**: Hostname $\to$ IPv4 address.
      - **Type NS**: Domain name $\to$ Hostname of authoritative name server.
      - **Type CNAME**: Alias hostname $\to$ Canonical (true) hostname.
      - **Type MX**: Domain name $\to$ Hostname of mail exchange server.
13. **Q: What is the difference between host aliasing (CNAME) and mail server aliasing (MX)?**
    - **A**: `CNAME` maps an alias hostname to any arbitrary canonical hostname (e.g., `www.pes.edu` $\to$ `webserver1.pes.edu`). `MX` maps a domain name specifically to an email server, allowing email addresses to omit mail server hostnames, and crucially **supports an integer priority field** for backup server failover.
14. **Q: What kind of DNS record has no fixed structure, and what are two real uses for it?**
    - **A**: **Type TXT** (Text Record). It stores arbitrary, human- or machine-readable text strings. Two primary production uses are:
      1. **Domain Ownership Verification**: Third-party services (e.g., Google Workspace, Microsoft 365) verify domain control by asking admins to publish a unique verification token in a TXT record.
      2. **Email Spoofing Prevention**: Publishing **SPF (Sender Policy Framework)** and **DKIM** public cryptographic keys to validate legitimate outbound mail senders.
15. **Q: What problem does DNSSEC solve that TTL/caching does not?**
    - **A**: DNSSEC provides **cryptographic data origin authenticity and data integrity**, allowing a client to mathematically verify that a DNS answer actually originated from the authentic authoritative zone owner and was not forged or altered by an attacker (preventing DNS cache poisoning). TTL only controls the *staleness* of legitimate data, not its *authenticity*.

---

### 1.14 Real-World Wireshark Packet Capture Analysis for DNS

*(Directly derived from Lecture Slides 20 & 21)*

When inspecting DNS traffic using Wireshark, the application layer reveals the exact binary fields of the RFC 1035 message specification:

#### 1. DNS Query Wireshark Capture (Slide 20)
When a host queries the IP address of `www.google.com`:

```
Frame 1: 74 bytes on wire (592 bits), 74 bytes captured
Ethernet II, Src: Apple_xx:xx:xx, Dst: Router_xx:xx:xx
Internet Protocol Version 4, Src: 192.168.1.105, Dst: 8.8.8.8
User Datagram Protocol, Src Port: 53214, Dst Port: 53
Domain Name System (query)
    Transaction ID: 0x0001
    Flags: 0x0100 Standard query
        0... .... .... .... = Response: Message is a query
        .000 0... .... .... = Opcode: Standard query (0)
        .... ..0. .... .... = Truncated: Message is not truncated
        .... ...1 .... .... = Recursion desired: Do query recursively (1)
        .... .... .0.. .... = Z: reserved (0)
        .... .... ...0 .... = Non-authenticated data: Unacceptable
    Questions: 1
    Answer RRs: 0
    Authority RRs: 0
    Additional RRs: 0
    Queries
        www.google.com: type A, class IN
            Name: www.google.com
            Type: A (Host Address) (1)
            Class: IN (Internet) (1)
```

- **Transport**: UDP with ephemeral source port `53214` and destination port `53`.
- **Transaction ID**: `0x0001` (matched in response).
- **Flags (`0x0100`)**: `QR = 0` (Query), `RD = 1` (Recursion Desired set by client).
- **Query Type**: `Type A` (requesting IPv4 address), `Class IN` (Internet).

#### 2. DNS Response Wireshark Capture (Slide 21)
The DNS server replies:

```
Frame 2: 90 bytes on wire (720 bits), 90 bytes captured
Internet Protocol Version 4, Src: 8.8.8.8, Dst: 192.168.1.105
User Datagram Protocol, Src Port: 53, Dst Port: 53214
Domain Name System (response)
    Transaction ID: 0x0001
    Flags: 0x8180 Standard query response, No error
        1... .... .... .... = Response: Message is a response (1)
        .000 0... .... .... = Opcode: Standard query (0)
        .... .0.. .... .... = Authoritative: Server is not an authority for domain (0)
        .... ..0. .... .... = Truncated: Message is not truncated (0)
        .... ...1 .... .... = Recursion desired: Do query recursively (1)
        .... .... 1... .... = Recursion available: Server can do recursive queries (1)
        .... .... .... 0000 = Reply code: No error (0)
    Questions: 1
    Answer RRs: 1
    Authority RRs: 0
    Additional RRs: 0
    Answers
        www.google.com: type A, class IN, TTL 300, addr 142.250.190.46
            Name: www.google.com
            Type: A (Host Address) (1)
            Class: IN (Internet) (1)
            Time to live: 300 (5 minutes)
            Data length: 4
            Address: 142.250.190.46
```

- **Flags (`0x8180`)**: `QR = 1` (Response), `AA = 0` (Resolver is caching server, not root authority), `RA = 1` (Recursion Available confirmed), `Reply code = 0` (No Error).
- **TTL = 300**: Informs the client OS that this IP mapping can be safely cached for up to 300 seconds before re-querying.

---

## 2. Peer-to-Peer (P2P) Architecture

### 2.1 Client-Server vs. Peer-to-Peer File Distribution

When distributing a massive digital file (size $F$ bits) to a large population of $N$ end hosts (peers), the choice of network architecture dictates scalability:

```
CLIENT-SERVER ARCHITECTURE                          PEER-TO-PEER (P2P) ARCHITECTURE
(Server bandwidth bottleneck)                       (Self-scaling cooperative swarm)

         [ Server (u_s) ]                                   [ Server (u_s) ]
         /      |       \\                                         /        \\
        v       v        v                                       v          v
     Peer 1   Peer 2   Peer N                                 Peer 1 <======> Peer 2
                                                                ^               ^
                                                                ||             ||
                                                                v               v
                                                              Peer 3 <======> Peer 4
```

- **Client-Server Architecture**:
  - The server must individually transmit a separate copy of file $F$ across its access link to each of the $N$ clients.
  - The server must output a total of $N \times F$ bits.
  - As the client community $N$ expands into thousands or millions, server upload capacity becomes a severe bottleneck. The distribution time grows **linearly with $N$**.
- **Peer-to-Peer (P2P) Architecture**:
  - Eliminates reliance on dedicated, always-on server infrastructure.
  - The originating server needs to transmit each chunk of the file only once into the swarm.
  - End systems (**peers**) communicate directly with one another. When a peer downloads chunks of the file, it simultaneously uploads those chunks to other peers.
  - **Self-Scalability**: Every new peer that joins the network brings not only new service demand (downloading $F$), but also **new service capacity** (uploading chunks to peers).

---

### 2.2 Mathematical Formulation of Distribution Time: $D_{cs}$ vs. $D_{p2p}$

Let:
- $F$: Size of the file to be distributed (in bits).
- $N$: Number of client peers requesting a copy of the file.
- $u_s$: Upload capacity of the file server (in bits/sec).
- $d_i$: Download capacity of client peer $i$ (in bits/sec), with $d_{min} = \min_{i} \{d_i\}$.
- $u_i$: Upload capacity of client peer $i$ (in bits/sec).

#### 1. Client-Server Distribution Time ($D_{cs}$)
In client-server distribution, the server must transmit $N$ independent copies:
- Time for server to upload $N$ copies: $\frac{NF}{u_s}$.
- The client with the minimum download speed ($d_{min}$) requires at least $\frac{F}{d_{min}}$ seconds to download its copy.

Thus, the minimum client-server distribution time is bounded by:
$$D_{cs} \ge \max \left\{ \frac{NF}{u_s},\; \frac{F}{d_{min}} \right\}$$

For large $N$, the term $\frac{NF}{u_s}$ completely dominates, causing $D_{cs}$ to increase **linearly with $N$** ($\mathcal{O}(N)$).

#### 2. Peer-to-Peer Distribution Time ($D_{p2p}$)
In P2P distribution:
- The server must upload at least one copy of the file into the swarm: $\frac{F}{u_s}$.
- The slowest peer must still download $F$ bits: $\frac{F}{d_{min}}$.
- The total data required by all $N$ peers is $N \times F$ bits. The maximum aggregate upload capacity of the entire system is the server's upload rate plus the sum of all peers' upload rates ($u_s + \sum_{i=1}^{N} u_i$).

Thus, the minimum P2P distribution time is bounded by:
$$D_{p2p} \ge \max \left\{ \frac{F}{u_s},\; \frac{F}{d_{min}},\; \frac{NF}{u_s + \sum_{i=1}^{N} u_i} \right\}$$

If all peers have an identical upload rate $u$, then $\sum u_i = N \cdot u$. As $N \to \infty$:
$$\frac{NF}{u_s + N \cdot u} \approx \frac{NF}{N \cdot u} = \frac{F}{u}$$
which is a **constant independent of $N$**. P2P distribution scales asymptotically flat ($\mathcal{O}(1)$), providing unmatched efficiency for large file dissemination.

---

### 2.3 BitTorrent Protocol Deep-Dive

**BitTorrent** is the dominant application-layer P2P protocol for large file distribution.

#### 1. Torrents, Trackers, Peers, and Swarms
- **Torrent**: The collection of all participating peers actively exchanging chunks of a specific target file.
- **Tracker**: A central infrastructure node that maintains a dynamic list of active peers currently in the torrent. When a peer joins, it registers with the tracker; the tracker responds with IP addresses of a random subset of active peers.
- **Chunks**: The target file is partitioned into equal-sized chunks, typically **256 KB (Kilobytes)** in size.
- A peer with no chunks is a **leecher**; a peer possessing the entire file and continuing to upload is a **seeder**.

#### 2. Chunk Acquisition: The "Rarest-First" Policy
- At any instant, different peers hold different subsets of file chunks.
- A peer queries its neighbors periodically to determine which chunks they hold.
- **Rarest-First Strategy**: The peer determines which chunk among its missing chunks is the rarest across all its neighbors (i.e., held by the fewest peers in the neighborhood).
- **Rationale**:
  1. Ensures rare chunks are duplicated rapidly across multiple peers, preventing chunks from becoming extinct if the only seeder disconnects.
  2. Ensures that when a peer departs, the swarm retains maximum diversity of chunks.
- *Exception*: At the very start of downloading, a peer requests any random chunk first to acquire data as quickly as possible so it can start uploading.

#### 3. Trading Algorithm: Tit-for-Tat Choking and Optimistic Unchoking
BitTorrent relies on an incentive-compatible trading mechanism called **Tit-for-Tat (TFT)** to reward uploading and punish free-riding:

1. **Top-Four Unchoked Peers (Measured every 10 seconds)**:
   - Every 10 seconds, host $A$ measures the data receiving rates from all peers currently uploading to $A$.
   - Host $A$ selects the **top four peers** that are feeding data to $A$ at the highest transmission rates.
   - Host $A$ **unchokes** these four peers, sending them requested chunks. All other peers are **choked** (blocked from receiving data from $A$).
2. **Optimistic Unchoking (Measured every 30 seconds)**:
   - Every 30 seconds, host $A$ randomly selects **one additional peer** that is currently choked and unchokes it.
   - **Rationale**:
     1. Allows $A$ to discover previously idle or newly joined peers that might have high upload capacity.
     2. Provides new peers entering the swarm with an initial chunk so they can participate in tit-for-tat trading.

---

### 2.4 Scalability Analysis & Worked Example from Lecture Slides

*(Directly derived from Lecture Slide 31: Client-Server vs. P2P)*

Consider the parameterized system analyzed in class lecture slide 31:
- File transmission baseline: A peer with upload rate $u$ can transmit the entire file in **1 hour** ($\frac{F}{u} = 1\text{ hour}$).
- Server upload rate is 10 times the peer upload rate: $u_s = 10u \implies \frac{F}{u_s} = \frac{1}{10}\text{ hour} = 0.1\text{ hours}$ ($6\text{ minutes}$).
- Client download rate is set large enough to not bottleneck: $d_{min} \ge u_s \implies \frac{F}{d_{min}} \le 0.1\text{ hours}$.

```
DISTRIBUTION TIME VS. NUMBER OF PEERS (N)
Time
 ▲
3h│                                    / Client-Server: D_cs = (N/10) hours
  │                                   /
2h│                                  /
  │                                 /
1h│ - - - - - - - - - - - - - - - -┌──────────────────── Peer-to-Peer: D_p2p -> 1 hour
  │                               /
0h└───┼──────────┼──────────┼────/─────► Number of Peers (N)
      0          10         20   35
```

#### Step-by-Step Numerical Walkthrough:

1. **Client-Server Distribution Time**:
   $$D_{cs} = \max\left\{ \frac{N \cdot F}{u_s},\; \frac{F}{d_{min}} \right\} = \frac{N \cdot F}{10u} = \frac{N}{10} \times \left(\frac{F}{u}\right) = \mathbf{\frac{N}{10}\text{ hours}}$$
2. **Peer-to-Peer Distribution Time**:
   $$D_{p2p} = \max\left\{ \frac{F}{u_s},\; \frac{F}{d_{min}},\; \frac{N \cdot F}{u_s + \sum u_i} \right\} = \max\left\{ 0.1,\; \frac{N \cdot F}{10u + N \cdot u} \right\} = \mathbf{\frac{N}{10 + N}\text{ hours}}$$

#### Comparative Table for $N = 10$, $N = 20$, and $N = 35$ Peers:

| Number of Peers ($N$) | Client-Server Distribution Time ($D_{cs}$) | Peer-to-Peer Distribution Time ($D_{p2p}$) | Comparison & Ratio |
| :---: | :---: | :---: | :--- |
| **$N = 10$** | $\frac{10}{10} = \mathbf{1.0\text{ hour}}$ ($60\text{ min}$) | $\frac{10}{10+10} = \mathbf{0.50\text{ hours}}$ ($30\text{ min}$) | P2P is **$2\times$ faster** |
| **$N = 20$** | $\frac{20}{10} = \mathbf{2.0\text{ hours}}$ ($120\text{ min}$) | $\frac{20}{10+20} = \mathbf{0.67\text{ hours}}$ ($40\text{ min}$) | P2P is **$3\times$ faster** |
| **$N = 35$** | $\frac{35}{10} = \mathbf{3.5\text{ hours}}$ ($210\text{ min}$) | $\frac{35}{10+35} = \mathbf{0.78\text{ hours}}$ ($47\text{ min}$) | P2P is **$4.5\times$ faster** |
| **$N \to \infty$** | **$D_{cs} \to \infty$** (Unbounded linear growth) | **$D_{p2p} \to \frac{F}{u} = \mathbf{1.0\text{ hour}}$** | **P2P approaches a flat asymptote!** |

**Theoretical Proof**:
As the swarm size $N$ grows arbitrarily large, the fraction $\frac{N}{10 + N} \to 1$. Therefore, no matter how many millions of peers join the swarm, the total distribution time in P2P can **never exceed $\frac{F}{u} = 1\text{ hour}$**!

---

## 3. Socket Programming with UDP and TCP

### 3.1 The Socket Abstraction

A **socket** is the software interface / API (Application Programming Interface) abstraction that links an application-layer process to the underlying operating system's transport protocol stack.

```
+-------------------------------------------------------------+
|                      Application Process                    |
|             (User Space / Controlled by Developer)          |
+-------------------------------------------------------------+
                               |
                        [ Socket API ]  <-- The "Doorway"
                               |
+-------------------------------------------------------------+
|                       Transport Layer                       |
|           (Kernel / Operating System Network Stack)         |
|                          TCP / UDP                          |
+-------------------------------------------------------------+
```

Sockets are categorized by transport protocol:
1. **SOCK_DGRAM (UDP Sockets)**: Connectionless, message-preserving, unreliable datagram transfer.
2. **SOCK_STREAM (TCP Sockets)**: Connection-oriented, full-duplex, reliable byte-stream transfer without message boundaries.

---

### 3.2 Socket API System Calls and Timing Flow Diagram

```
       TCP SERVER                                             TCP CLIENT
+---------------------+                                +---------------------+
| socket(SOCK_STREAM) |                                |                     |
+---------------------+                                |                     |
           │                                                   │
           ▼                                                   │
+---------------------+                                        │
|   bind('', port)    |                                        │
+---------------------+                                        │
           │                                                   │
           ▼                                                   │
+---------------------+                                        │
|      listen()       |                                        │
+---------------------+                                        │
           │                                                   │
           ▼                                                   │
+---------------------+                                +---------------------+
|      accept()       | <=== [TCP 3-Way Handshake] === | socket(SOCK_STREAM) |
| (Blocks until conn) |                                +---------------------+
+---------------------+                                        │
           │                                                   ▼
           │ <---------------- connect() ----------------------+
           │                                                   │
           ▼                                                   ▼
+---------------------+         Request Data           +---------------------+
|       recv()        | <============================= |       send()        |
+---------------------+                                +---------------------+
           │                                                   │
           ▼ (Process Request)                                 ▼
+---------------------+         Response Data          +---------------------+
|       send()        | =============================> |       recv()        |
+---------------------+                                +---------------------+
           │                                                   │
           ▼                                                   ▼
+---------------------+                                +---------------------+
|   close() [conn]    |                                |       close()       |
+---------------------+                                +---------------------+
```

---

### 3.3 Socket Programming with UDP (Connectionless) in Python

In UDP, there is no initial connection handshake. The sender explicitly attaches the destination IP address and port number to every individual datagram using `sendto()`. The receiver extracts the sender's address using `recvfrom()`.

#### Python UDP Client (`UDPClient.py`):
```python
from socket import *

# 1. Define server destination parameters
server_name = 'localhost'  # Server hostname or IP address
server_port = 12000        # Arbitrary server port number

# 2. Create UDP client socket: AF_INET specifies IPv4; SOCK_DGRAM specifies UDP
client_socket = socket(AF_INET, SOCK_DGRAM)

# 3. Prompt user for string input
message = input('Input lowercase sentence: ')

# 4. Attach destination address to packet and transmit into socket
# encode() converts UTF-8 string to byte array
client_socket.sendto(message.encode(), (server_name, server_port))

# 5. Receive modified response and server address (buffer size 2048 bytes)
modified_message, server_address = client_socket.recvfrom(2048)

# 6. Decode bytes to string and display
print('From Server:', modified_message.decode())

# 7. Close socket
client_socket.close()
```

#### Python UDP Server (`UDPServer.py`):
```python
from socket import *

server_port = 12000

# 1. Create UDP server socket
server_socket = socket(AF_INET, SOCK_DGRAM)

# 2. Bind socket explicitly to port 12000 across all local network interfaces ('')
server_socket.bind(('', server_port))

print('The UDP Server is ready to receive packets...')

# 3. Infinite loop to service incoming datagrams
while True:
    # Receive datagram and store client source IP and port
    message, client_address = server_socket.recvfrom(2048)
    
    # Process data: convert lowercase text to uppercase
    modified_message = message.decode().upper()
    
    # Send processed datagram back to client using extracted address
    server_socket.sendto(modified_message.encode(), client_address)
```

---

### 3.4 Socket Programming with TCP (Connection-Oriented) in Python

In TCP, a client must establish a dedicated, reliable stream connection with the server via a three-way handshake before transmitting application data.

#### Python TCP Client (`TCPClient.py`):
```python
from socket import *

server_name = 'localhost'
server_port = 12000

# 1. Create TCP client socket: SOCK_STREAM specifies TCP
client_socket = socket(AF_INET, SOCK_STREAM)

# 2. Initiate 3-way handshake connection with server
client_socket.connect((server_name, server_port))

# 3. Prompt user for text
sentence = input('Input lowercase sentence: ')

# 4. Transmit byte stream directly into established connection pipe
client_socket.send(sentence.encode())

# 5. Receive server response (up to 1024 bytes)
modified_sentence = client_socket.recv(1024)

# 6. Print result
print('From Server:', modified_sentence.decode())

# 7. Terminate connection (triggers TCP FIN exchange)
client_socket.close()
```

#### Python TCP Server (`TCPServer.py`):
```python
from socket import *

server_port = 12000

# 1. Create welcoming socket
server_socket = socket(AF_INET, SOCK_STREAM)

# 2. Bind to well-known port
server_socket.bind(('', server_port))

# 3. Listen for connection requests (parameter 1 specifies maximum queued connections)
server_socket.listen(1)

print('The TCP Server is ready to accept connections...')

while True:
    # 4. Block until client contacts; accept() creates a NEW dedicated connection socket
    # connection_socket is dedicated exclusively to this specific client
    connection_socket, addr = server_socket.accept()
    print(f'Accepted connection from client: {addr}')
    
    # 5. Read stream from dedicated socket
    sentence = connection_socket.recv(1024).decode()
    capitalized_sentence = sentence.upper()
    
    # 6. Write response into dedicated stream
    connection_socket.send(capitalized_sentence.encode())
    
    # 7. Close dedicated client socket (welcoming server_socket remains open!)
    connection_socket.close()
```

## 4. Other Application-Layer Protocols

### 4.1 FTP (File Transfer Protocol) — RFC 959

The **File Transfer Protocol (FTP)** is a legacy, stateful application-layer protocol designed to transfer files between a client host and a remote server over TCP.

```
+------------------+                                +------------------+
|                  | ======= Control Connection =====> |                  |
|    FTP Client    |        (TCP Port 21, Commands)    |    FTP Server    |
|                  | <====== Data Connection ========> |                  |
+------------------+        (TCP Port 20, File Data)   +------------------+
```

#### 1. Out-of-Band Control Architecture
Unlike HTTP, which transmits control headers and payload data intermingled across the exact same TCP connection (**in-band**), FTP utilizes **two separate, parallel TCP connections** (**out-of-band**):
- **Control Connection (Port 21)**:
  - Used strictly for exchanging administrative control information: user identification, authentication passwords, commands to change remote working directories, and file transfer commands.
  - Remains open continuously throughout the entire user session.
- **Data Connection (Port 20)**:
  - Created dynamically and exclusively to transfer the actual payload of a requested file or directory listing.
  - Closed immediately once the specific file transfer finishes. A new data connection is spawned for each subsequent file transfer.

#### 2. Active vs. Passive FTP Modes
The interaction between client firewalls / NAT (Network Address Translation) and FTP data connections led to two distinct operational modes:

| Feature | Active FTP Mode (Standard) | Passive FTP Mode (PASV) |
| :--- | :--- | :--- |
| **Command Sent** | Client sends `PORT <IP, port>` command over port 21 | Client sends `PASV` command over port 21 |
| **Data Connection Initiator** | **Server initiates** connection from Server Port 20 to Client Port $N$ | **Client initiates** connection from Client Port to Server Port $P$ |
| **Firewall / NAT Behavior** | **Fails frequently**: Client NAT/firewalls block unsolicited incoming TCP SYN packets from the server | **Firewall-Friendly**: Client initiates all outbound connections, passing seamlessly through client NAT |
| **Usage** | Legacy intranet networks | Standard across modern Internet and web browsers |

#### 3. Stateful Nature of FTP
The FTP server is strictly **stateful**: it tracks the client's current working directory, authenticated user identity, transfer mode (ASCII vs. Binary), and open connection handles. This limits the number of simultaneous active clients an FTP server can maintain compared to stateless HTTP servers.

---

### 4.2 Electronic Mail: SMTP, POP3, and IMAP

Internet email architecture comprises three primary architectural components:
1. **User Agents (UA)**: Client mail software allowing users to read, compose, and send email (e.g., Microsoft Outlook, Mozilla Thunderbird, Apple Mail).
2. **Mail Servers (Mail Transfer Agents - MTA)**: Core infrastructure running message queues and mailbox storage for users.
3. **Application Protocols**: Protocols governing mail sending (**SMTP**) and mail retrieval (**POP3**, **IMAP**, **HTTP**).

```
+-----------+            +--------------------+            +--------------------+            +-----------+
| Sender UA | -- SMTP -> | Sender Mail Server | -- SMTP -> |Receiver Mail Server| -- IMAP -> |Receiver UA|
+-----------+  (Port 587)+--------------------+  (Port 25) +--------------------+  (Port 143)+-----------+
```

#### 1. SMTP (Simple Mail Transfer Protocol) — RFC 5321
- Operates over **TCP port 25** (server-to-server relay) or **TCP port 587** (client submission with TLS encryption).
- **Push Protocol**: Used exclusively by a client or server to *push* email forward to another mail server. A receiver cannot use SMTP to pull mail from its server.
- **ASCII 7-Bit Limitation**: Originally restricted to 7-bit ASCII text. To transmit binary attachments (images, audio, PDF documents), data must be encoded into ASCII text using **MIME (Multipurpose Internet Mail Extensions)** encoding (such as Base64).

#### 2. SMTP vs. HTTP Comparison Table

| Dimension | SMTP (Simple Mail Transfer Protocol) | HTTP (HyperText Transfer Protocol) |
| :--- | :--- | :--- |
| **Data Direction** | Primarily a **Push protocol** (sending server pushes email to destination server) | Primarily a **Pull protocol** (client browser pulls web pages from server) |
| **Character Encoding** | Historically restricted to **7-bit ASCII**; binary requires MIME Base64 encoding | **Binary-safe 8-bit** transparent data transfer without encoding overhead |
| **Object Handling** | Bundles all message text, headers, and attachments into a **single multipart message** | Encapsulates each object in its own independent **HTTP response message** |
| **Transport Protocol** | TCP (Port 25, 587) | TCP (Port 80) / TCP with TLS (Port 443) / UDP (HTTP/3) |

#### 3. Mail Access Protocols: POP3 vs. IMAP vs. Webmail

Because destination mail servers are always-on while user laptops and phones are frequently offline, received messages reside on the destination mail server until fetched by the recipient:

- **POP3 (Post Office Protocol Version 3 — RFC 1939, TCP Port 110, SSL Port 995)**:
  - Extremely simple "download-and-delete" or "download-and-keep" model.
  - Messages are downloaded from the server to the local computer.
  - **Limitation**: Does not synchronize state across multiple devices. If a user marks an email as read or creates a folder on their laptop, those changes are not reflected on their phone.
- **IMAP (Internet Message Access Protocol — RFC 3501, TCP Port 143, SSL Port 993)**:
  - Server-centric synchronization model.
  - All messages, folder hierarchies, and read/unread/flagged states are maintained permanently on the remote mail server.
  - Supports **partial message fetching** (e.g., downloading only headers and sender info over slow mobile connections, fetching attachments only on demand).
- **Web-based Email (Webmail)**:
  - Services like Gmail and Outlook.com use standard **HTTP/HTTPS** between the user's web browser and the mail server's web front-end. Inter-server communication between mail providers still relies strictly on SMTP over port 25.

---

### 4.3 DHCP (Dynamic Host Configuration Protocol) — RFC 2131

The **Dynamic Host Configuration Protocol (DHCP)** automates network parameter assignment for hosts joining a network. Rather than manually typing static IP configurations, devices automatically obtain:
1. An allocated **IPv4 address** and **Lease Duration**.
2. The **Subnet Mask** (e.g., `255.255.255.0` or `/24`).
3. The IP address of the **Default Gateway** (first-hop router).
4. The IP address of the **Local DNS Name Server**.

#### The 4-Step DORA Transaction Flow

DHCP executes over **UDP**, with servers listening on **Port 67** and clients listening on **Port 68**. Because the newly arriving client has no IP address, all initial transactions use physical broadcast:

```
    DHCP CLIENT (New Device)                            DHCP SERVER
  (0.0.0.0, Port 68)                                (Port 67)
           │                                                │
           │ ===== 1. DHCP DISCOVER (Broadcast) ==========> │
           │      Src: 0.0.0.0:68, Dst: 255.255.255.255:67  │
           │      yiaddr: 0.0.0.0, Transaction ID: 654      │
           │                                                │
           │ <==== 2. DHCP OFFER (Broadcast) =============  │
           │      Src: 192.168.1.1:67, Dst: 255.255.255.255 │
           │      yiaddr: 192.168.1.105, Lease: 3600s       │
           │                                                │
           │ ===== 3. DHCP REQUEST (Broadcast) ===========> │
           │      Src: 0.0.0.0:68, Dst: 255.255.255.255:67  │
           │      Selected Server: 192.168.1.1              │
           │                                                │
           │ <==== 4. DHCP ACK (Broadcast) ===============  │
           │      Src: 192.168.1.1:67, Dst: 255.255.255.255 │
           │      Confirming 192.168.1.105 + DNS + Gateway  │
           ▼                                                ▼
```

1. **D — DHCP Discover**: The client broadcasts a message looking for DHCP servers on the local physical link.
   - Source IP: `0.0.0.0` (Client has no IP yet).
   - Destination IP: `255.255.255.255` (Limited broadcast address).
2. **O — DHCP Offer**: Each DHCP server receiving the discover message reserves an available IP and broadcasts an offer containing the proposed IP address (`yiaddr` - "your IP address"), subnet mask, and lease duration.
3. **R — DHCP Request**: The client selects one offer (if multiple servers replied) and broadcasts a request acknowledging its choice. Broadcasting notifies unselected DHCP servers that their proposed offers were declined so they can return reserved addresses to their available pool.
4. **A — DHCP Acknowledgment (ACK)**: The chosen server broadcasts an ACK confirming the lease parameters. The client binds the IP to its network interface.

#### DHCP Relay Agents
Routers block broadcast packets (`255.255.255.255`) by default. To prevent having to place an expensive dedicated DHCP server on every physical subnet, routers can be configured with a **DHCP Relay Agent** (such as Cisco's `ip helper-address`). The relay agent intercepts local UDP broadcast discover packets, encapsulates them into unicast IP packets, and forwards them directly to a centralized corporate DHCP server across subnets.

---

### 4.4 SNMP (Simple Network Management Protocol) — RFC 1157

**SNMP (Simple Network Management Protocol)** provides a standardized framework for monitoring, configuring, and managing network devices (routers, switches, firewalls, servers, wireless access points) across an enterprise network.

```
+-------------------------------------------------------------+
|               SNMP Management Station (Manager)             |
|                  (Listens on UDP Port 162)                  |
+-------------------------------------------------------------+
             ▲                                |
      Trap   │ (Asynchronous Alert)           │ Get / Set Requests
  (Port 162) │                                │ (Port 161)
             │                                ▼
+-------------------------------------------------------------+
|             Managed Network Device (Router / Switch)        |
|                 SNMP Agent (Listens on Port 161)            |
|                                                             |
|           +-------------------------------------+           |
|           |  MIB (Management Information Base)  |           |
|           +-------------------------------------+           |
+-------------------------------------------------------------+
```

#### 1. Core Architectural Components
1. **Managing Server (SNMP Manager)**: A centralized host running network management software that queries agents, collects telemetry, and visualizes device status.
2. **Managed Device**: Any network node housing an SNMP agent (router, switch, server, printer).
3. **SNMP Agent**: A lightweight software daemon executing on the managed device that maintains local operational data and responds to manager requests.
4. **MIB (Management Information Base)**: A structured hierarchical database of operational parameters maintained on the device (e.g., interface packet counters, CPU utilization, temperature, routing table entries).
   - Objects are defined using **SMI (Structure of Management Information)** and identified globally by an **OID (Object Identifier)** dotted-tree sequence (e.g., `1.3.6.1.2.1.1.1` corresponds to `sysDescr`).

#### 2. SNMP Core Operations
- **GetRequest / GetNextRequest / GetBulkRequest**: Manager queries the agent for one or more specific MIB variable values.
- **SetRequest**: Manager instructs the agent to alter the value of a writable MIB parameter (e.g., administratively shutting down a router interface).
- **Trap (Port 162)**: An unsolicited, asynchronous event notification initiated by the *agent* to alert the manager of a critical failure (e.g., link down, power supply failure, reboot).
- **InformRequest**: Similar to a Trap, but requires an explicit acknowledgment from the manager to guarantee delivery.

#### 3. SNMP Protocol Versions and Security
- **SNMPv1 & SNMPv2c**: Transmit authentication tokens called **Community Strings** (e.g., "public" for read-only, "private" for read-write) in **cleartext**. Anyone sniffing network traffic with Wireshark can capture community strings and manipulate network hardware.
- **SNMPv3**: Introduced enterprise-grade cryptographic security through the **USM (User-based Security Model)** and **VACM (View-based Access Control Model)**:
  - **Authentication**: HMAC-SHA or HMAC-MD5 cryptographic message authentication.
  - **Confidentiality**: Strong symmetric payload encryption using **AES (Advanced Encryption Standard)** or DES.
  - **Message Integrity**: Replay protection and tampering detection.

---

### 4.5 Telnet and SSH (Secure Shell) — Remote Access

Network administrators require terminal access to configure remote routers, switches, and Linux servers.

```
TELNET (Insecure / Port 23)                 SSH (Cryptographically Secure / Port 22)
Client ──[ Plaintext: "password" ]──► Server  Client ──[ Encrypted: 0x8F3A21... ]──► Server
(Sniffable by any intermediate router)       (End-to-end encrypted with public-key auth)
```

#### 1. Telnet — RFC 854 (TCP Port 23)
- An ancient terminal emulation protocol developed in 1969.
- Establishes a virtual terminal session using the **NVT (Network Virtual Terminal)** standard.
- **Fatal Security Vulnerability**: All keystrokes, administrative usernames, and root passwords are transmitted across the network in **unencrypted cleartext**. Anyone with access to an intermediate network link or packet sniffer can read complete login credentials and session data.

#### 2. SSH (Secure Shell) — RFC 4251 (TCP Port 22)
- Replaces Telnet, rlogin, and rsh by providing robust cryptographic security over an untrusted network.
- **Three-Layer Security Architecture**:
  1. **Transport Layer Protocol**: Authenticates the server host using public-key cryptography, negotiates a shared secret session key via **Diffie-Hellman key exchange**, and encrypts all subsequent traffic using symmetric ciphers (e.g., **AES-256**). Verifies integrity via **HMAC (Hash-based Message Authentication Code)**.
  2. **User Authentication Layer**: Authenticates the client user to the server using passwords, Kerberos, or asymmetric public-private key pairs (e.g., RSA, Ed25519 stored in `~/.ssh/authorized_keys`).
  3. **Connection Layer**: Multiplexes encrypted communication channels into a single physical connection, supporting interactive terminal shells, remote command execution, secure file transfer (**SFTP** / **SCP**), and encrypted TCP port forwarding (**SSH Tunneling**).

---

### 4.6 Comprehensive Application Protocol Port Summary Table

*(Essential Reference for Computer Networks Exams)*

| Port Number | Protocol Acronym | Protocol Full Expansion | Default Transport | Primary Function / Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **20** | **FTP-Data** | File Transfer Protocol (Data) | TCP | Bulk file payload data transmission |
| **21** | **FTP-Control** | File Transfer Protocol (Control) | TCP | Command exchange and session authentication |
| **22** | **SSH** | Secure Shell | TCP | Secure encrypted remote login and file transfer |
| **23** | **Telnet** | Telecommunications Network | TCP | Unencrypted plaintext remote terminal login |
| **25** | **SMTP** | Simple Mail Transfer Protocol | TCP | Server-to-server email relay and transmission |
| **53** | **DNS** | Domain Name System | UDP / TCP | Hostname-to-IP resolution (TCP for transfers >512B) |
| **67** | **DHCP Server** | Dynamic Host Configuration Protocol | UDP | Server port receiving client broadcast requests |
| **68** | **DHCP Client** | Dynamic Host Configuration Protocol | UDP | Client port receiving server broadcast offers/ACKs |
| **69** | **TFTP** | Trivial File Transfer Protocol | UDP | Barebones file transfer for bootstrapping diskless hosts |
| **80** | **HTTP** | HyperText Transfer Protocol | TCP | Unencrypted World Wide Web hypertext transfer |
| **110** | **POP3** | Post Office Protocol Version 3 | TCP | Mailbox retrieval (download-and-delete) |
| **123** | **NTP** | Network Time Protocol | UDP | Clock synchronization across computer networks |
| **143** | **IMAP** | Internet Message Access Protocol | TCP | Mailbox retrieval with server-side sync |
| **161** | **SNMP** | Simple Network Management Protocol | UDP | Queries and agent polling |
| **162** | **SNMP-Trap** | SNMP Asynchronous Traps | UDP | Unsolicited alerts from agent to manager |
| **389** | **LDAP** | Lightweight Directory Access Protocol | TCP / UDP | Corporate directory and identity lookups |
| **443** | **HTTPS** | HyperText Transfer Protocol Secure | TCP (over TLS) | Encrypted, authenticated World Wide Web browsing |
| **587** | **SMTP Submission** | Secure Email Submission | TCP | Client-to-server encrypted mail submission |
| **993** | **IMAPS** | IMAP over SSL/TLS | TCP | Encrypted IMAP mail access |
| **995** | **POP3S** | POP3 over SSL/TLS | TCP | Encrypted POP3 mail access |

---

## 5. Introduction to Transport-Layer Services

### 5.1 Process-to-Process Logical Communication

While the **Network Layer** (IP) provides logical communication between **host machines** (host-to-host delivery), the **Transport Layer** provides logical communication between specific **application processes** running on those hosts (process-to-process delivery).

```
+-------------------------------------------------------------+
|              Host A                      Host B             |
|                                                             |
|   [Process 1]   [Process 2]       [Process 1]   [Process 2] |
|        \             /                 \             /      |
|     ====\===========/===================\===========/====   |
|         [ Transport Layer: Process-to-Process Delivery ]    |
|     ====================================================    |
|                               |                             |
|              [ Network Layer: Host-to-Host Delivery ]       |
|                               |                             |
|               Physical Transmission Medium                  |
+-------------------------------------------------------------+
```

- **Sender Actions**: Accepts arbitrary-length messages from application processes, segments them into smaller chunks, appends transport-layer headers containing source/destination port numbers, and passes them down to the network layer.
- **Receiver Actions**: Extracts transport segments from incoming network datagrams, checks for errors, strips transport headers, and reassembles the segments into application messages, directing them to the destination socket.

---

### 5.2 Transport vs. Network Layer: The Household Analogy

The textbook (*Kurose & Ross*) introduces a famous analogy to clarify the distinction between network-layer and transport-layer responsibilities:

> **The Household Analogy**:
> Imagine two large families living in different cities:
> - **Ann's House**: Located in Bengaluru, containing 12 children.
> - **Bill's House**: Located in Boston, containing 12 children.
>
> Every week, the 12 children in Ann's house write letters to their cousins, the 12 children in Bill's house.

- **Application Messages**: The letters written inside the envelopes.
- **Processes**: The 12 individual children writing and reading letters.
- **Hosts (End Systems)**: Ann's house and Bill's house.
- **Transport Layer**: **Ann and Bill** themselves. Every day, Ann collects the letters from her 12 siblings, puts them in a mailbag, and hands them to the postal carrier. When mail arrives from Boston, Ann takes the letters from the carrier and delivers each letter directly into the hands of the specific child to whom it is addressed. Bill performs the exact same role in Boston.
- **Network Layer**: The **Postal Service** (India Post / USPS). The postal service transports mailbags from house to house (host to host), completely oblivious to which specific child wrote or receives a letter.

#### Key Conceptual Takeaway:
Ann and Bill (the transport layer) do not drive mail trucks between cities; they rely on the postal service (the network layer). However, Ann and Bill can provide services that the postal service does not — such as verifying that no letters were dropped on the floor or re-sending a letter if it was lost. Similarly, transport protocols (like TCP) can provide **guaranteed reliable delivery** even when the underlying network layer (IP) is completely unreliable and best-effort!

---

### 5.3 Principal Internet Transport-Layer Protocols

The Internet provides applications with two distinct transport protocols:

| Feature Dimension | TCP (Transmission Control Protocol) — RFC 793 | UDP (User Datagram Protocol) — RFC 768 |
| :--- | :--- | :--- |
| **Connection Paradigm** | **Connection-oriented** (Explicit 3-way handshake before data transfer) | **Connectionless** (No handshake, no connection state) |
| **Reliability** | **Guaranteed reliable delivery** (Error detection, ACKs, retransmission) | **Unreliable / Best-effort** (Packets can be lost, corrupted, duplicated) |
| **Ordering** | **Strictly in-order byte stream** | **Unordered datagrams** (May arrive out-of-order) |
| **Data Boundaries** | **Byte-stream oriented** (No message boundaries) | **Message-oriented / Datagrams** (Preserves application boundaries) |
| **Flow Control** | **Yes** (Receive window `rwnd` prevents sender from overflowing receiver) | **No** (Receiver buffers can overflow freely) |
| **Congestion Control**| **Yes** (Throttles sender when network core links are congested) | **No** (Blasts data at whatever rate application generates) |
| **Header Overhead** | **20 to 60 bytes** | **8 bytes fixed** |
| **Typical Use Cases** | Web (HTTP/HTTPS), Email (SMTP), File Transfer (FTP), Shell (SSH) | DNS, VoIP, Video Streaming, Online Gaming, SNMP, HTTP/3 |

#### Services Transport Layer CANNOT Provide:
Because transport protocols execute strictly on end hosts and cannot alter physical routers in the core, the Internet transport layer **cannot guarantee**:
1. **Delay / Latency Guarantees**: Cannot promise data will arrive within $10\text{ ms}$.
2. **Bandwidth / Throughput Guarantees**: Cannot guarantee an application will receive a dedicated $100\text{ Mbps}$ pipeline.

---

## 6. Multiplexing and Demultiplexing

### 6.1 How Multiplexing and Demultiplexing Work

Because multiple network applications (browser, Spotify, email client, Zoom) execute concurrently on a single computer, the operating system must direct incoming network traffic to the correct application process:

- **Multiplexing (at Sender)**: The transport layer collects data chunks from multiple application sockets, encapsulates each chunk with transport header fields (including source and destination port numbers), and passes the resulting segments down to the network layer.
- **Demultiplexing (at Receiver)**: The transport layer examines the header fields in received segments to identify the receiving socket, and delivers the data to that specific socket.

```
       APPLICATION LAYER (Multiple Sockets)
       [ Socket A ]   [ Socket B ]   [ Socket C ]
             \              |              /
              \             |             /
               ▼            ▼            ▼
+-------------------------------------------------------+
|                   TRANSPORT LAYER                     |
|  Sender: MULTIPLEXING (Gathers data, attaches ports)  |
|  Receiver: DEMULTIPLEXING (Reads ports, routes data)  |
+-------------------------------------------------------+
                            │
                            ▼
                      NETWORK LAYER
```

---

### 6.2 Port Numbers and Port Number Ranges

Every transport-layer segment carries a **16-bit Source Port Number** and a **16-bit Destination Port Number**. 
A 16-bit field allows integer values ranging from **$0$ to $65{,}535$** ($2^{16} - 1$). The **IANA (Internet Assigned Numbers Authority)** partitions this range into three standardized tiers:

1. **Well-Known Ports ($0$ to $1023$)**:
   - Strictly reserved and standardized for ubiquitous Internet application protocols (e.g., HTTP 80, HTTPS 443, SSH 22, DNS 53).
   - On Unix/Linux systems, binding to a well-known port requires administrative (`root` / `sudo`) privileges.
2. **Registered Ports ($1024$ to $49{,}151$)**:
   - Listed by IANA for specific vendor services and applications (e.g., Microsoft SQL Server 1433, MySQL 3306, PostgreSQL 5432).
3. **Dynamic / Private / Ephemeral Ports ($49{,}152$ to $65{,}535$)**:
   - Assigned dynamically by the client operating system when an application initiates an outbound network connection.

---

### 6.3 Connectionless Demultiplexing (UDP)

In UDP, a socket is identified solely by a **2-tuple**:
$$\mathbf{(Destination\; IP\; Address,\; Destination\; Port\; Number)}$$

When a host receives a UDP segment:
1. The transport layer inspects the segment's **Destination Port Number**.
2. It delivers the segment directly to the local socket bound to that destination port number.
3. The **Source IP Address** and **Source Port Number** inside the segment are ignored for demultiplexing decisions (they are merely passed up to the application code so the application knows where to send a reply).

#### Exam Implication:
> If two different client hosts (Host A and Host B) send UDP segments to server port `12000` at Host C, both segments are demultiplexed and delivered to the **exact same UDP socket** at Host C, regardless of their differing source IP addresses or source port numbers!

```
Host A (IP_A, Port 49152) ──── UDP Segment ───► [ Dest: Port 12000 ]
                                                       │
                                                       ▼
Host B (IP_B, Port 53210) ──── UDP Segment ───► [ Single UDP Socket ]
                                                (Bound to Port 12000)
```

---

### 6.4 Connection-Oriented Demultiplexing (TCP)

In TCP, demultiplexing is fundamentally different. A TCP socket is identified by a **4-tuple**:
$$\mathbf{(Source\; IP,\; Source\; Port,\; Destination\; IP,\; Destination\; Port)}$$

When a host receives a TCP segment, the transport layer inspects **all four values** to route the segment to the correct socket.

```
Host A (IP_A, Port 49152) ──── TCP Segment ───► [ Dedicated Socket 1 ] (IP_A:49152 <-> Server:80)
                                                       
Host B (IP_B, Port 49152) ──── TCP Segment ───► [ Dedicated Socket 2 ] (IP_B:49152 <-> Server:80)
                                                (Both talk to port 80, but different sockets!)
```

#### The Server Socket Lifecycle:
1. The server process initially creates a **welcoming (listening) socket** bound to a well-known port (e.g., port 80 for HTTP).
2. When a client initiates a connection, it sends a TCP SYN segment to server port 80.
3. The server operating system accepts the connection, and **spawns a brand-new dedicated connection socket** identified specifically by the full 4-tuple of that connection.
4. All subsequent data segments matching that 4-tuple are demultiplexed to that dedicated socket.
5. This architecture allows a server to support thousands of concurrent connections simultaneously without interference.

---

### 6.5 Server Connection Capacity Limits

> **Common Exam Question**: Can a server handle more than 65,535 simultaneous TCP connections if port numbers are limited to 16 bits?

**Authoritative Answer**:
**Yes, absolutely!** 
A server listening on port 80 can easily handle hundreds of thousands or even millions of concurrent TCP connections.

#### Technical Explanation:
The 65,535 limit applies only to the total number of unique ports an individual IP address can bind. Because TCP demultiplexes based on the **4-tuple** `(Source IP, Source Port, Dest IP, Dest Port)`:
- The server's destination port is fixed at `80`.
- However, the client's source IP address and client source port can vary freely!
- For a single remote client IP address, that client can open up to $\approx 64{,}000$ distinct outbound ephemeral ports to the server.
- Across thousands of distinct client IP addresses worldwide, the number of unique 4-tuples is virtually astronomical ($2^{32} \times 2^{16} \approx 2.8 \times 10^{14}$).

In real-world operating systems, the practical limit on concurrent TCP connections is determined by **server physical RAM** (each open socket buffer consumes kernel memory), **CPU processing power**, and operating system **file descriptor limits** (`ulimit -n`), not by the 16-bit port number field!

## 7. Connectionless Transport: UDP (User Datagram Protocol)

### 7.1 Why UDP Exists (RFC 768 Philosophy)

The **User Datagram Protocol (UDP)**, defined in **RFC 768**, is the minimal, "bare-bones" transport-layer protocol of the Internet. It provides an application with direct access to the underlying network layer (IP) with almost no additional protocol mechanics.

Why would a developer deliberately choose an unreliable protocol like UDP over TCP?

1. **Finer Application-Level Control Over Data Injection**:
   - In TCP, congestion control throttles the sender during periods of network congestion, holding data in socket buffers.
   - Real-time applications (such as VoIP - Voice over IP, video teleconferencing, and live gaming) require data to be transmitted immediately. They can tolerate minor packet loss but cannot tolerate unpredictable queuing and retransmission delays. UDP injects data into the network immediately without throttling.
2. **No Connection Establishment Delay**:
   - TCP mandates a 3-way handshake before transmitting user data, introducing at least **1 RTT (Round-Trip Time)** of latency.
   - UDP transmits application data in the very first packet. This is why **DNS (Domain Name System)** uses UDP: a DNS lookup completes in a single RTT round trip without connection state overhead.
3. **No Connection State at Sender or Receiver**:
   - TCP tracks sequence numbers, ACK numbers, receive window sizes (`rwnd`), congestion windows (`cwnd`), and retransmission timers for every active connection.
   - A server running UDP maintains zero connection state. It allocates no state tables, allowing a single server running DNS or SNMP to support tens of thousands of active clients simultaneously without running out of kernel memory.
4. **Minimal Packet Header Overhead**:
   - A standard TCP segment header consumes **20 to 60 bytes** of overhead per packet.
   - A UDP segment header consumes a fixed **8 bytes**, conserving transmission bandwidth on resource-constrained wireless links.

#### Adding Reliability at the Application Layer:
If an application requires reliability over UDP (such as **HTTP/3** running over **QUIC - Quick UDP Internet Connections**), the developer must implement acknowledgments, timers, and retransmissions directly inside the **application-layer code**, giving them customized control over congestion and loss recovery without TCP's head-of-line blocking.

---

### 7.2 UDP Segment Header Structure

A UDP segment consists of an **8-byte header** followed by the application payload data, formatted in 32-bit words:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-----------------------------------+-----------------------------------+
|      Source Port Number (16 bits) |   Destination Port Number (16 bits|
+-----------------------------------+-----------------------------------+
|            Length (16 bits)       |          Checksum (16 bits)       |
+-----------------------------------+-----------------------------------+
|                                                                       |
|                       Application Payload Data                        |
|                                                                       |
+-----------------------------------------------------------------------+
```

1. **Source Port Number (16 bits)**: The port number on the sending host process. Used by the receiver if it needs to send a reply datagram.
2. **Destination Port Number (16 bits)**: The port number of the receiving process on the destination host, used for local demultiplexing.
3. **Length (16 bits)**: The total length of the UDP segment **in bytes**, including both the 8-byte header and the application payload. Minimum possible value is $8$ (an empty UDP segment containing zero payload bytes).
4. **Checksum (16 bits)**: Used by the receiving host to verify whether bit errors occurred during transmission across physical links and intermediate routers.

---

### 7.3 UDP Checksum Calculation and Verification

The UDP checksum provides end-to-end error detection based on the **16-bit One's Complement Sum** algorithm.

#### 1. The Pseudo-Header
To prevent packets from being delivered to the wrong host or wrong protocol due to IP-level header corruption, the checksum is computed over a **12-byte IPv4 Pseudo-Header** prepended to the UDP segment:
- 32-bit Source IP Address
- 32-bit Destination IP Address
- 8 bits of Zeros + 8-bit Protocol field (`17` for UDP)
- 16-bit UDP Length

*(Note: The pseudo-header is purely conceptual; it is discarded after checksum calculation and is never transmitted over the physical wire).*

#### 2. Checksum Algorithm (Sender Actions):
1. Treat all bytes of the pseudo-header, UDP header, and application payload as a sequence of **16-bit binary integers**. (If the payload has an odd number of bytes, append an 8-bit zero padding byte at the end).
2. Initialize the Checksum field to all zeros (`0x0000`).
3. Add all 16-bit integers together using binary addition.
4. **Wrap-Around Carry**: Whenever an addition generates an overflow carry bit beyond the 16th most significant bit, wrap the carry bit around and add it back into the lowest significant bit (LSB).
5. **One's Complement**: After all words are summed, take the **one's complement** of the final sum (invert all bits: $0 \to 1$ and $1 \to 0$).
6. Place the inverted value into the UDP Checksum field.

#### 3. Receiver Verification:
1. The receiver adds all 16-bit words of the received segment (including payload, pseudo-header, and the received checksum field value itself) using one's complement addition.
2. If no bits were corrupted, the sum of a number and its one's complement must equal all 1s:
   $$\mathbf{1111\; 1111\; 1111\; 1111 \quad (0xFFFF)}$$
3. If any bit in the receiver's computed sum is `0`, at least one bit error has occurred, and the segment is discarded.

#### 4. Weak Error Protection Limitation:
The UDP checksum provides relatively **weak error detection** compared to **CRC (Cyclic Redundancy Check)**:
- If two bits flip in opposite directions in the same bit position across two different 16-bit words (e.g., bit 5 flips from $0 \to 1$ in word 1, and bit 5 flips from $1 \to 0$ in word 2), the two errors cancel each other out mathematically.
- The resulting sum remains identical, and the corruption passes undetected.
- *Protocol Difference*: In IPv4, UDP checksum calculation is optional (if disabled, sender sets checksum field to all zeros). In **IPv6**, computing the UDP checksum is **strictly mandatory**.

---

### 7.4 Checksum Worked Examples and Weak Protection Deep-Dive

*(Directly derived from Lecture Slides 123, 124, and 125)*

#### Example 1: Summing Three 16-Bit Integers (Slide 125)
Compute the Internet checksum for the following three 16-bit binary words:
- Word 1: `0 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0`
- Word 2: `0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1`
- Word 3: `1 0 0 0 1 1 1 1 0 0 0 0 1 1 0 0`

**Step 1: Add Word 1 and Word 2**:
```
    0 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0   (Word 1)
  + 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1   (Word 2)
  -----------------------------------
    1 0 1 1 1 0 1 1 1 0 1 1 0 1 0 1   (Intermediate Sum 1: no overflow carry)
```

**Step 2: Add Word 3 to Intermediate Sum 1**:
```
    1 0 1 1 1 0 1 1 1 0 1 1 0 1 0 1   (Intermediate Sum 1)
  + 1 0 0 0 1 1 1 1 0 0 0 0 1 1 0 0   (Word 3)
  -----------------------------------
  1 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1   (Intermediate Sum 2 with overflow carry '1')
```

**Step 3: Wrap-around the overflow carry bit**:
```
    0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1   (Lower 16 bits)
  +                                 1   (Wrapped carry bit)
  -----------------------------------
    0 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0   (Final Sum)
```

**Step 4: Take One's Complement (Invert all bits)**:
$$\mathbf{\text{Checksum} = 1\; 0\; 1\; 1\; 0\; 1\; 0\; 1\; 0\; 0\; 1\; 1\; 1\; 1\; 0\; 1}$$

---

#### Example 2: The Weak Protection Vulnerability (Slide 124)
Why is the Internet Checksum considered "weak protection" compared to **CRC (Cyclic Redundancy Check)**?

Consider the two 16-bit words from our earlier example:
- Original Word 1: `... 0 1 1 0 ...` (Bits at position $k$ and $k+1$ are `0` and `1`)
- Original Word 2: `... 0 1 0 1 ...` (Bits at position $k$ and $k+1$ are `0` and `1`)

Suppose physical line noise causes **simultaneous, compensating bit errors**:
- In Word 1, bit position $k$ flips from $\mathbf{0 \to 1}$.
- In Word 2, bit position $k$ flips from $\mathbf{1 \to 0}$.

```
ORIGINAL TRANSMISSION:
Word 1: ... 0 ...  (Bit k is 0)
Word 2: ... 1 ...  (Bit k is 1)
Sum:    ... 1 ...  (0 + 1 = 1)

CORRUPTED IN TRANSIT:
Word 1: ... 1 ...  (Flipped 0 -> 1)
Word 2: ... 0 ...  (Flipped 1 -> 0)
Sum:    ... 1 ...  (1 + 0 = 1 -- IDENTICAL SUM!)
```

Because the column sum remains mathematically identical (`1 + 0 = 0 + 1 = 1`), the calculated checksum at the receiver matches the received checksum perfectly! The receiver accepts the corrupted data packet as error-free, delivering garbage to the application.

---

## 8. Principles of Reliable Data Transfer (RDT)

### 8.1 Why Reliable Data Transfer Is Needed

The underlying physical and logical network layers (IP, Ethernet, WiFi, Fiber) are fundamentally **unreliable channels**:
1. **Bit Errors**: Electromagnetic noise, radio interference, and hardware degradation corrupt individual bits within packets.
2. **Packet Loss**: When network routers experience queuing congestion, their internal buffer queues overflow, causing packets to be discarded.
3. **Packet Reordering**: Dynamic routing algorithms may route consecutive packets along different paths, causing packets to arrive out of order.
4. **Packet Duplication**: Network-layer routing loops or link-layer retries can cause duplicate copies of packets to arrive at the receiver.

The **Reliable Data Transfer (RDT)** protocol layer sits above the unreliable network layer, presenting an illusion of a 100% reliable, bidirectional bit-pipe to the application layer.

```
+-------------------------------------------------------------+
|                      Application Layer                      |
+-------------------------------------------------------------+
               |                               ▲
  rdt_send()   |                               |  deliver_data()
               ▼                               |
+-------------------------------------------------------------+
|               Reliable Data Transfer Protocol               |
|                    (RDT 1.0 -> RDT 3.0)                     |
+-------------------------------------------------------------+
               |                               ▲
  udt_send()   |                               |  rdt_rcv()
               ▼                               |
+-------------------------------------------------------------+
|                     Unreliable Channel                      |
|                (Bit errors, packet loss)                    |
+-------------------------------------------------------------+
```

---

### 8.2 Building Blocks of Reliable Data Transfer

| Architectural Mechanism | Purpose and Functional Operation |
| :--- | :--- |
| **Error Detection (Checksum)** | Computes mathematical checksum over packet contents to detect bit flips in transit. |
| **Positive Acknowledgment (ACK)** | Explicit control message sent from receiver to sender confirming that a packet arrived error-free. |
| **Negative Acknowledgment (NAK)** | Explicit control message sent from receiver to sender indicating that a packet was received corrupted. |
| **Sequence Numbers** | Sequential integer tags affixed to packet headers to detect and discard duplicate packets. |
| **Retransmission Countdown Timer** | Sender-side timer used to recover from lost packets; if timer expires before ACK arrives, packet is resent. |
| **Pipelining / Windowing** | Allows sender to transmit multiple unacknowledged packets concurrently to saturate high-bandwidth channels. |

---

### 8.3 Step-by-Step Evolution: RDT 1.0 to RDT 3.0

#### 1. RDT 1.0: Reliable Transfer Over a Completely Reliable Channel
- **Assumption**: The underlying physical channel is 100% reliable — zero bit errors and zero packet loss.
- **Protocol Mechanics**:
  - Sender: `rdt_send(data)` packages data into a packet via `make_pkt(data)` and sends it via `udt_send(packet)`.
  - Receiver: `rdt_rcv(packet)` extracts data via `extract(packet, data)` and delivers it via `deliver_data(data)`.
  - No feedback, no sequence numbers, no timers required.

#### 2. RDT 2.0: Channel with Bit Errors (Stop-and-Wait)
- **Assumption**: Channel may flip bits, but packets are never lost.
- **Protocol Mechanics**:
  - Adds **Checksum** to detect bit corruption.
  - Adds receiver feedback: **ACK (Acknowledgment)** if packet received intact; **NAK (Negative Acknowledgment)** if corrupted.
  - **Stop-and-Wait**: Sender sends one packet and halts all further transmission until it receives an ACK or NAK from the receiver. If NAK is received, sender retransmits the packet.
- **The Fatal Flaw of RDT 2.0**:
  What happens if the **ACK or NAK itself gets corrupted** during transit?
  - The sender receives garbled noise and cannot determine whether the receiver successfully received the data or not.
  - If the sender does nothing, the protocol hangs permanently.
  - If the sender blindly retransmits, the receiver cannot distinguish whether the newly arrived packet is a retransmission or brand-new application data! Duplicate data is delivered to the application layer.

#### 3. RDT 2.1: Handling Corrupted ACKs/NAKs with Sequence Numbers
- **Solution**: Sender affixes a **1-bit Sequence Number** (`0` or `1`) to each data packet header.
- **Protocol Mechanics**:
  - If an ACK/NAK arrives corrupted at the sender, the sender simply retransmits the current packet.
  - If the receiver receives a packet with sequence number `0` when it was expecting sequence number `1`, the receiver knows with mathematical certainty that this is a **duplicate packet**.
  - The receiver discards the duplicate payload data, but **re-sends an ACK for packet 0** so the sender can advance its state!
  - Requires 4 states at sender and 2 states at receiver.

#### 4. RDT 2.2: A NAK-Free Reliable Protocol
- Eliminates explicit NAK packets entirely.
- Instead of sending a NAK for a corrupted packet, the receiver sends an **ACK for the last correctly received packet**, explicitly including the sequence number: `ACK 0` or `ACK 1`.
- A sender waiting for `ACK 1` that receives a duplicate `ACK 0` interprets this as an implicit NAK for packet 1, triggering immediate retransmission of packet 1.

#### 5. RDT 3.0: Channel with Bit Errors AND Packet Loss (Alternating Bit Protocol)
- **Assumption**: The underlying channel can corrupt bits AND drop packets entirely (data packets or ACK packets can vanish into the network).
- **Solution**: Introduce a sender-side **Retransmission Countdown Timer**.
  - The sender starts a countdown timer when it transmits a packet.
  - If the timer expires (**Timeout**) before an ACK is received, the sender retransmits the packet and restarts the timer.
  - Because sequence numbers alternate strictly between `0` and `1`, RDT 3.0 is known as the **Alternating Bit Protocol (ABP)**.

---

### 8.4 Operation of RDT 3.0 (Alternating Bit Protocol)

RDT 3.0 handles four distinct operational scenarios:

```
(a) Normal Operation (No Loss)              (b) Packet Loss
Sender              Receiver            Sender              Receiver
  │   pkt 0             │                 │   pkt 0             │
  │────────────────────►│                 │────────────────────►│
  │   ack 0             │                 │   ack 0             │
  │◄────────────────────│                 │◄────────────────────│
  │   pkt 1             │                 │   pkt 1             │
  │────────────────────►│                 │─────── X (Lost)     │
  │   ack 1             │                 │ (Timeout!)          │
  │◄────────────────────│                 │   pkt 1 (resend)    │
  │   pkt 0             │                 │────────────────────►│
  │────────────────────►│                 │   ack 1             │
                                          │◄────────────────────│

(c) ACK Loss                                (d) Premature Timeout / Delayed ACK
Sender              Receiver            Sender              Receiver
  │   pkt 0             │                 │   pkt 0             │
  │────────────────────►│                 │────────────────────►│
  │   ack 0             │                 │   ack 0             │
  │◄────────────────────│                 │◄────────────────────│
  │   pkt 1             │                 │   pkt 1             │
  │────────────────────►│                 │────────────────────►│
  │   ack 1             │                 │ (Premature Timeout!)│
  │      X (Lost)       │                 │   pkt 1 (resend)    │
  │ (Timeout!)          │                 │───────────────┐     │
  │   pkt 1 (resend)    │                 │   ack 1       │     │
  │────────────────────►│ (Duplicate!     │◄──────────────│─────│
  │   ack 1             │  Discard data,  │   pkt 0       ▼     │ (Duplicate!
  │◄────────────────────│  re-ACK 1)      │──────────────►      │  Discard)
                                          │   ack 1 (ignore)    │
                                          │◄────────────────────│
```

---

### 8.5 Performance Analysis of Stop-and-Wait Operation

While functionally correct, RDT 3.0 suffers from abysmal performance because it operates in a **Stop-and-Wait** fashion: the sender is forced to sit completely idle while a packet travels across the network and its ACK travels back.

#### Sender Utilization Formula:
The **Sender Utilization ($U_{sender}$)** is the fraction of time the sender is actively transmitting bits onto the physical link:
$$U_{sender} = \frac{t_{trans}}{RTT + t_{trans}} = \frac{\frac{L}{R}}{RTT + \frac{L}{R}}$$

Where:
- $L$: Packet length in bits.
- $R$: Link transmission rate in bits per second (bps).
- $RTT$: Round-Trip Time (two-way propagation delay) in seconds.

#### Concrete High-Speed Link Example:
Consider a cross-country 1 Gbps fiber optic link:
- Link Capacity: $R = 1\text{ Gbps} = 10^9\text{ bps}$.
- Packet Size: $L = 1000\text{ bytes} = 8000\text{ bits}$.
- One-Way Propagation Delay: $t_{prop} = 15\text{ ms} \implies RTT = 30\text{ ms} = 0.030\text{ seconds}$.

1. **Transmission Delay ($t_{trans}$)**:
   $$t_{trans} = \frac{L}{R} = \frac{8000\text{ bits}}{10^9\text{ bps}} = 8 \times 10^{-6}\text{ s} = 0.008\text{ ms}$$
2. **Sender Utilization ($U_{sender}$)**:
   $$U_{sender} = \frac{0.008\text{ ms}}{30\text{ ms} + 0.008\text{ ms}} = \frac{0.008}{30.008} \approx \mathbf{0.000267 \quad (0.027\%)}$$
3. **Effective Throughput**:
   $$\text{Throughput} = U_{sender} \times R = 0.000267 \times 10^9\text{ bps} \approx \mathbf{267\text{ kbps}}$$

**Conclusion**: On a 1 Gbps link, the Stop-and-Wait protocol achieves a miserable throughput of only 267 kbps! The physical channel sits idle $99.973\%$ of the time.

---

## 9. Pipelining: Go-Back-N (GBN) and Selective Repeat (SR)

### 9.1 The Pipelining Concept

To eliminate stop-and-wait channel starvation, modern transport protocols utilize **Pipelining**:
- The sender is permitted to transmit up to $N$ consecutive packets without waiting for intermediate acknowledgments.
- If $N = 3$, sender utilization increases by a factor of 3. If $N$ is sized to match the **Bandwidth-Delay Product**, utilization approaches $100\%$:
  $$U_{pipelined} = \frac{N \cdot \frac{L}{R}}{RTT + \frac{L}{R}}$$

```
STOP-AND-WAIT (One packet in flight)       PIPELINING (N packets in flight concurrently)
Sender              Receiver            Sender              Receiver
  │  pkt 0              │                 │  pkt 0              │
  │────────────────────►│                 │────────────────────►│
  │                     │                 │  pkt 1              │
  │                     │                 │────────────────────►│
  │  ack 0              │                 │  pkt 2              │
  │◄────────────────────│                 │────────────────────►│
  │  pkt 1              │                 │  ack 0              │
  │────────────────────►│                 │◄────────────────────│
```

---

### 9.2 Go-Back-N (GBN) Protocol

In **Go-Back-N (GBN)**, the sender is restricted to having at most $N$ unacknowledged packets in pipeline.

```
       +--------------------+--------------------+--------------------+
       | Already ACKed      | Sent, Not Yet ACKed| Usable, Not Sent   | Cannot Use
       +--------------------+--------------------+--------------------+
                            ▲                    ▲
                         send_base           nextseqnum
                            |◄───── Window N ───►|
```

#### 1. Sender Properties:
- **Cumulative Acknowledgment**: An `ACK(n)` indicates that all packets up to and including sequence number $n$ have been received correctly by the receiver.
- **Single Timer**: The sender maintains only **one timer**, tracking the oldest transmitted but unacknowledged packet (`send_base`).
- **Timeout Action**: If the timer expires, the sender retransmits **ALL $N$ packets currently in flight** in the window (`send_base` through `nextseqnum - 1`). Hence the name: *Go-Back-N*!

#### 2. Receiver Properties:
- **Receiver Window Size = 1**: The receiver accepts packets strictly in sequential order.
- **Discards Out-of-Order Packets**: If packet $n$ arrives correctly, but the receiver was expecting packet $n-1$, the receiver completely discards packet $n$ (it does NOT buffer out-of-order data).
- **Re-ACKs Highest In-Order Sequence Number**: The receiver generates an ACK for the highest contiguous in-order packet received so far.

---

### 9.3 Selective Repeat (SR) Protocol

GBN's blind retransmission of all in-flight packets causes massive bandwidth waste when link capacity or packet loss rates are high. **Selective Repeat (SR)** optimizes retransmission by having the receiver acknowledge each packet individually.

```
SENDER WINDOW:
  [ ACKed ] [ ACKed ] [ UnACKed ] [ ACKed ] [ UnACKed ] [ Usable ] ...
                      ▲
                   send_base

RECEIVER WINDOW (Buffers Out-of-Order Packets):
  [ Delivered ] [ Delivered ] [ Expected ] [ Buffered ] [ Buffered ] ...
                              ▲
                           rcv_base
```

#### 1. Sender Properties:
- Maintains a **logical timer for each individual unacknowledged packet**.
- On timeout, the sender retransmits **only the single unacknowledged packet** whose specific timer expired.

#### 2. Receiver Properties:
- **Receiver Window Size $N > 1$**: The receiver maintains an internal buffer.
- **Individual ACKs**: Sends a specific ACK for every correctly received packet, regardless of whether it arrives in order.
- **Buffering**: Out-of-order packets are buffered in memory until missing gaps are received, at which point a contiguous batch of packets is delivered to the application layer, advancing the receiver window.

---

### 9.4 Sequence Number Space Constraints & The SR Dilemma

In practical protocols, sequence numbers are encoded in a fixed $k$-bit header field, providing a finite sequence space of $2^k$ integers: $[0, 1, 2, \dots, 2^k - 1]$.

#### 1. Go-Back-N Window Constraint:
$$W_{GBN} \le 2^k - 1$$
For GBN, the window size must be strictly less than the total sequence number space.

#### 2. Selective Repeat Window Constraint:
$$W_{SR} \le 2^{k-1} = \frac{2^k}{2}$$
For Selective Repeat, the window size must be **at most half** of the sequence number space.

---

#### 3. The Selective Repeat Dilemma (Detailed Proof & Counterexample)

> **Exam Favorite Question**: Why must window size be $\le 2^{k-1}$ in Selective Repeat? Show the failure scenario if $W > 2^{k-1}$.

Let $k = 2$ bits, so the sequence number space is $2^2 = 4$ integers: $\{0, 1, 2, 3\}$.
Suppose we violate the rule and set window size $W = 3$ (which is greater than $2^{2-1} = 2$).

```
SCENARIO A: ACKs Lost in Transit
Sender (W=3): [0, 1, 2] 3 0 1 2 3 ...
  - Transmits pkts 0, 1, 2.
Receiver (W=3): Expects [0, 1, 2]
  - Receives pkts 0, 1, 2 perfectly.
  - Sends ACK 0, ACK 1, ACK 2.
  - Receiver window advances to: [3, 0, 1].
*** DISASTER: All ACKs (0, 1, 2) are LOST in the network! ***
  - Sender's timer for pkt 0 expires.
  - Sender retransmits: pkt 0 (OLD DATA).

SCENARIO B: ACKs Arrive Safely
Sender (W=3): [0, 1, 2] 3 0 1 2 3 ...
  - Transmits pkts 0, 1, 2.
Receiver (W=3): Expects [0, 1, 2]
  - Receives pkts 0, 1, 2 perfectly.
  - Sends ACK 0, ACK 1, ACK 2.
  - Receiver window advances to: [3, 0, 1].
*** ACKs arrive safely at Sender ***
  - Sender window advances to: [3, 0, 1].
  - Sender transmits pkt 3, then transmits pkt 0 (BRAND NEW DATA).

THE FATAL AMBIGUITY AT THE RECEIVER:
In both scenarios, the receiver (sitting at window [3, 0, 1]) receives a packet carrying Sequence Number 0!
  - In Scenario A, pkt 0 is a RETRANSMISSION of old data.
  - In Scenario B, pkt 0 is BRAND NEW APPLICATION DATA.
The receiver CANNOT distinguish whether pkt 0 is new data or a duplicate of old data!
```

**Conclusion**: When $W \le \frac{\text{Sequence Space}}{2}$, the sender's window and receiver's window can never overlap on the same sequence number, completely eliminating ambiguity.

---

### 9.5 Head-to-Head Comparison

| Dimension | Stop-and-Wait (RDT 3.0) | Go-Back-N (GBN) | Selective Repeat (SR) |
| :--- | :--- | :--- | :--- |
| **Sender Window Size ($W_s$)** | $1$ | $N > 1$ | $N > 1$ |
| **Receiver Window Size ($W_r$)**| $1$ | $1$ | $N > 1$ |
| **Acknowledgment Type** | Individual | **Cumulative** (`ACK n` acks up to $n$) | **Individual** (each packet acked) |
| **Retransmission Timers** | Single timer | **Single timer** (oldest unACKed) | **Multiple timers** (one per packet) |
| **On Timeout Action** | Resends single packet | Resends **ALL $N$ in-flight packets** | Resends **ONLY expired packet** |
| **Receiver Buffering** | None | **None** (discards out-of-order) | **Yes** (buffers out-of-order) |
| **Sequence Space Required** | 1 bit (0 and 1) | $W \le 2^k - 1$ | $W \le 2^{k-1}$ |
| **Bandwidth Efficiency** | Extremely Poor | Medium (wastes retransmissions) | **Optimal / High** |
| **Implementation Complexity**| Minimal | Low (single timer, no rx buffer)| High (per-packet timers & rx buffer)|

## 10. Connection-Oriented Transport: TCP (Transmission Control Protocol)

### 10.1 TCP Key Properties

The **Transmission Control Protocol (TCP)**, formalized in **RFC 793, RFC 1122, RFC 1323, RFC 2018, and RFC 5681**, is the foundational connection-oriented transport protocol of the Internet.

1. **Point-to-Point Communication**: Exactly one sender and one receiver per connection. Multicasting (one sender to multiple receivers) is impossible over standard TCP.
2. **Reliable, In-Order Byte Stream**: Delivers an unbroken, byte-exact stream to the receiving application. TCP preserves no application-layer record boundaries or message delimiters.
3. **Pipelined Data Transfer**: TCP uses sliding window mechanics. The window size is dynamically bounded by **Flow Control** (`rwnd`) and **Congestion Control** (`cwnd`).
4. **Full-Duplex Service**: Bi-directional data flow occurs concurrently across the exact same physical connection: Host A can transmit data to Host B while simultaneously receiving data from Host B.
5. **Connection-Oriented**: Requires an explicit three-way handshake before exchanging user data to initialize sequence numbers, buffers, and state variables.
6. **Flow-Controlled**: Ensures a fast sender cannot overwhelm the physical receive buffer of a slow receiver.

#### MTU vs. MSS:
- **MTU (Maximum Transmission Unit)**: The maximum frame payload size the underlying link-layer technology can carry. For standard Ethernet, $\mathbf{MTU = 1500\text{ bytes}}$.
- **MSS (Maximum Segment Size)**: The maximum amount of *application-layer payload data* TCP can encapsulate into a single segment (excluding TCP and IP headers).
- **MSS Calculation**:
  $$\text{MSS} = \text{MTU} - (\text{IP Header Size}) - (\text{TCP Header Size})$$
  Under standard IPv4 without options ($20\text{ bytes}$ IP header, $20\text{ bytes}$ TCP header):
  $$\mathbf{\text{MSS} = 1500 - 20 - 20 = 1460\text{ bytes}}$$

---

### 10.2 TCP Segment Architecture

A TCP segment consists of a **20- to 60-byte header** followed by application payload data, structured into 32-bit words:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-----------------------------------+-----------------------------------+
|      Source Port Number (16 bits) |   Destination Port Number (16 bits|
+-----------------------------------+-----------------------------------+
|                     Sequence Number (32 bits)                         |
+-----------------------------------+-----------------------------------+
|                  Acknowledgment Number (32 bits)                      |
+-----------------------------------+-----------------------------------+
| Hlen  | Rsvd  |C|E|U|A|P|R|S|F|         Receive Window (16 bits)      |
| (4b)  | (4b)  |W|C|R|C|S|S|Y|I|                                       |
|       |       |R|E|G|K|H|T|N|N|                                       |
+-----------------------------------+-----------------------------------+
|          Checksum (16 bits)       |      Urgent Pointer (16 bits)     |
+-----------------------------------+-----------------------------------+
|                    Options (0 to 40 bytes, optional)                  |
+-----------------------------------+-----------------------------------+
|                                                                       |
|                       Application Payload Data                        |
|                                                                       |
+-----------------------------------------------------------------------+
```

#### Detailed Header Field Breakdown:

1. **Source Port (16 bits)** & **Destination Port (16 bits)**: Identifies sending and receiving application processes.
2. **Sequence Number (32 bits)**: The byte-stream number of the **first byte of application data** contained in this segment.
3. **Acknowledgment Number (32 bits)**: The sequence number of the **next byte** the receiver is expecting from the sender. TCP uses **Cumulative ACKs**: acknowledging byte $N$ confirms that all bytes up to $N-1$ have been received correctly.
4. **Header Length / Data Offset (4 bits)**: Specifies the length of the TCP header in **32-bit (4-byte) words**. A minimum 20-byte header has a value of $5$ ($5 \times 4 = 20\text{ bytes}$). The maximum value of 15 allows up to $15 \times 4 = 60\text{ bytes}$ of header (accommodating up to 40 bytes of options).
5. **Reserved (4 bits)**: Reserved for future standardization; set to zero.
6. **The 8 Control Flags (1 bit each)**:
   - **CWR (Congestion Window Reduced)**: Set by sender to acknowledge receipt of an ECE notification and confirm its congestion window was reduced.
   - **ECE (ECN-Echo)**: Set by receiver to notify the sender of network congestion if an incoming IP packet had the **ECN (Explicit Congestion Notification)** CE bit set by a router.
   - **URG (Urgent)**: Indicates that the Urgent Pointer field is valid and that urgent application data is present.
   - **ACK (Acknowledgment)**: Indicates that the Acknowledgment Number field is valid. (Set on virtually every TCP segment after the initial SYN).
   - **PSH (Push)**: Instructs the receiving TCP stack to push this segment's data immediately up to the application without waiting for buffers to fill.
   - **RST (Reset)**: Abruptly terminates and resets a connection due to an unrecoverable error (e.g., rejecting an incoming packet to an unopened port).
   - **SYN (Synchronize)**: Used exclusively during connection setup to synchronize initial sequence numbers.
   - **FIN (Finish)**: Indicates that the sender has finished transmitting data and wishes to close its side of the connection.
7. **Receive Window (`rwnd`, 16 bits)**: The number of bytes the receiver is currently willing to accept in its buffer. Used for **Flow Control**.
8. **Checksum (16 bits)**: One's complement sum covering TCP header, data, and the 12-byte IP pseudo-header. Mandatory in TCP.
9. **Urgent Pointer (16 bits)**: Points to the sequence number of the final byte of urgent data when `URG = 1`.
10. **Options (Variable, 0–40 bytes)**: Used to negotiate **MSS**, **Window Scale** (allowing `rwnd` to scale up to 1 GB), **SACK (Selective Acknowledgment)**, and **Timestamps**.

---

### 10.3 Sequence Numbers and Acknowledgments

Unlike simplified textbook protocols that count segments, TCP counts **individual bytes in the data stream**.

- **Initial Sequence Number (ISN)**: During connection establishment, both hosts pick an Initial Sequence Number at random to prevent delayed segments from older, terminated connections from being accepted as valid data.
- **Telnet Piggybacking Scenario**:
  The classic example of TCP byte indexing is an interactive Telnet session where a client types a character and the server echoes it back:

```
CLIENT (Host A)                                        SERVER (Host B)
      │                                                       │
      │  Step 1: User types character 'C' (1 byte payload)    │
      │  Seq = 42, ACK = 79, Data = 'C'                       │
      │──────────────────────────────────────────────────────►│
      │                                                       │
      │  Step 2: Server ACKs 'C' AND echoes 'C' back          │
      │  Seq = 79, ACK = 43, Data = 'C' (Piggybacked ACK)     │
      │◄──────────────────────────────────────────────────────│
      │                                                       │
      │  Step 3: Client ACKs server's echo                    │
      │  Seq = 43, ACK = 80, Data = Empty                     │
      │──────────────────────────────────────────────────────►│
      ▼                                                       ▼
```
- In Step 1, Host A sends byte 42 (`Seq=42`) and tells Host B it expects byte 79 (`ACK=79`).
- In Step 2, Host B confirms receipt of byte 42 by asking for byte 43 (`ACK=43`), while simultaneously sending its own byte 79 (`Seq=79`) carrying the echoed character. This is known as **Piggybacking**.
- In Step 3, Host A confirms receipt of byte 79 by acknowledging byte 80 (`ACK=80`).

---

### 10.4 Round-Trip Time (RTT) Estimation, Karn's Algorithm, and Timeout

TCP must dynamically compute its **Retransmission Timeout ($TimeoutInterval$)**. If the timeout is too short, unnecessary retransmissions waste bandwidth; if too long, the connection reacts sluggishly to packet loss.

#### 1. SampleRTT Measurement
$SampleRTT$ is the measured time from when a segment is transmitted until its acknowledgment arrives.

#### 2. EstimatedRTT (Exponential Weighted Moving Average — EWMA)
To smooth out transient network spikes, TCP computes an EWMA of recent $SampleRTT$ values:
$$\mathbf{EstimatedRTT = (1 - \alpha) \cdot EstimatedRTT + \alpha \cdot SampleRTT}$$
The standard value recommended by RFC 6298 is **$\alpha = 0.125 = \frac{1}{8}$**:
$$EstimatedRTT = 0.875 \cdot EstimatedRTT + 0.125 \cdot SampleRTT$$

#### 3. DevRTT (EWMA of RTT Variation)
TCP also tracks how much $SampleRTT$ typically deviates from $EstimatedRTT$:
$$\mathbf{DevRTT = (1 - \beta) \cdot DevRTT + \beta \cdot |SampleRTT - EstimatedRTT|}$$
The standard value recommended by RFC 6298 is **$\beta = 0.25 = \frac{1}{4}$**:
$$DevRTT = 0.75 \cdot DevRTT + 0.25 \cdot |SampleRTT - EstimatedRTT|$$

#### 4. TimeoutInterval Calculation
The retransmission timeout is set to the estimated RTT plus a "safety margin" of four standard deviations:
$$\mathbf{TimeoutInterval = EstimatedRTT + 4 \cdot DevRTT}$$
*(If a timeout occurs, $TimeoutInterval$ is doubled to back off exponentially).*

#### 5. Karn's Algorithm
> **Critical Exam Topic**: How does TCP handle RTT estimation when a segment is retransmitted?
- **Ambiguity Problem**: When a retransmitted segment is acknowledged, the sender cannot tell whether the incoming ACK corresponds to the original transmission or the retransmission!
- **Karn's Rule 1**: **Never update $SampleRTT$ or recalculate $EstimatedRTT$ for retransmitted segments.** Only compute $SampleRTT$ for segments that were transmitted exactly once and acknowledged without retransmission.
- **Karn's Rule 2 (Exponential Timer Backoff)**: On every consecutive timeout, immediately double the current timeout interval:
  $$TimeoutInterval_{new} = 2 \times TimeoutInterval_{old}$$
  Timer recalculation using the EWMA formula resumes only after a segment is acknowledged without retransmission.

---

### 10.5 Reliable Data Transfer, RFC 5681 ACK Rules, and Fast Retransmit

TCP uses a **single retransmission timer** associated with the oldest unacknowledged segment (`send_base`).

#### RFC 5681: The Four Receiver ACK Generation Rules

| Event at Receiver | Receiver Action (RFC 5681) |
| :--- | :--- |
| **1. In-order segment arrives; all prior data already ACKed** | **Delayed ACK**: Wait up to $500\text{ ms}$ for the next in-order segment. If no next segment arrives within $500\text{ ms}$, send a single ACK. |
| **2. In-order segment arrives; one prior segment is awaiting ACK** | **Immediate Cumulative ACK**: Immediately send a single cumulative ACK that acknowledges both in-order segments. |
| **3. Out-of-order segment arrives with higher-than-expected sequence number (Gap Detected)** | **Immediate Duplicate ACK**: Immediately send a duplicate ACK indicating the sequence number of the next expected in-order byte. |
| **4. Segment arrives that partially or completely fills a gap** | **Immediate ACK**: Immediately send an ACK, provided the segment starts at the lower end of the gap. |

---

#### TCP Fast Retransmit

Waiting for a retransmission timer to expire takes hundreds of milliseconds, stalling throughput. **TCP Fast Retransmit** uses duplicate ACKs to detect loss much faster:

```
Sender                                                  Receiver
  │  Seq = 1000, 500B (Bytes 1000-1499)                     │
  │────────────────────────────────────────────────────────►│ ACK = 1500
  │  Seq = 1500, 500B (Bytes 1500-1999) ─── X (LOST)        │
  │  Seq = 2000, 500B (Bytes 2000-2499)                     │
  │────────────────────────────────────────────────────────►│ ACK = 1500 (Duplicate ACK 1)
  │  Seq = 2500, 500B (Bytes 2500-2999)                     │
  │────────────────────────────────────────────────────────►│ ACK = 1500 (Duplicate ACK 2)
  │  Seq = 3000, 500B (Bytes 3000-3499)                     │
  │────────────────────────────────────────────────────────►│ ACK = 1500 (Duplicate ACK 3)
  │                                                         │
  │ *** TRIPLE DUPLICATE ACKS RECEIVED! ***                 │
  │ FAST RETRANSMIT: Resend Seq = 1500 immediately!         │
  │────────────────────────────────────────────────────────►│
```

- If a segment is lost, subsequent out-of-order segments trigger the receiver to send duplicate ACKs.
- When the sender receives **3 duplicate ACKs** for the same data (4 identical ACKs total), it concludes that the unacknowledged segment was lost in the network.
- **Fast Retransmit Action**: The sender retransmits the missing segment immediately **without waiting for the retransmission timer to expire**.
- **Why 3 Duplicate ACKs?** Out-of-order delivery of packets by IP routers is common. A single out-of-order packet generates 1 or 2 duplicate ACKs before resolving. Requiring 3 duplicate ACKs ensures that the network truly dropped a packet rather than merely reordering packets.

---

### 10.6 TCP Flow Control

TCP provides a **flow-control service** to match the sender's transmission rate with the receiving application's drain rate, preventing receiver buffer overflow.

```
                  RECEIVE BUFFER (RcvBuffer)
+------------------------------------+--------------------------+
|  Buffered Data (Unread by App)     | Free Buffer Space (rwnd) |
+------------------------------------+--------------------------+
▲                                    ▲                          ▲
LastByteRead                         LastByteRcvd               RcvBuffer Edge
|<──── (LastByteRcvd - LastByteRead) ───>|
```

1. The receiver allocates a receive buffer of size **`RcvBuffer`** in kernel memory.
2. The receiving application reads data from the buffer asynchronously:
   - `LastByteRead`: The sequence number of the last byte extracted by the application.
   - `LastByteRcvd`: The sequence number of the last byte received and placed in the buffer.
3. The amount of free buffer space remaining is the **Receive Window (`rwnd`)**:
   $$\mathbf{rwnd = RcvBuffer - [LastByteRcvd - LastByteRead]}$$
4. The receiver advertises its current `rwnd` in every TCP segment header it transmits back to the sender.
5. **Sender Constraint**: The sender ensures that the amount of in-flight unacknowledged data never exceeds `rwnd`:
   $$\mathbf{LastByteSent - LastByteAcked \le rwnd}$$

#### The Zero-Window Deadlock Problem and Solution:
- If the receiving application stops reading data, the receive buffer fills completely, and the receiver advertises $\mathbf{rwnd = 0}$.
- The sender respects this and halts all further transmission.
- Eventually, the application awakens and reads data, freeing up megabytes of buffer space. The receiver sends an ACK advertising $rwnd > 0$.
- **The Deadlock**: If this ACK segment is dropped by the network, the sender sits waiting forever for permission to transmit, while the receiver sits waiting forever for new data!
- **Solution — TCP Zero-Window Probing (Persist Timer)**:
  When $rwnd = 0$, the sender starts a **Persist Timer**. When the timer expires, the sender transmits a **1-byte probe segment**. The receiver replies to the probe with an ACK containing its current $rwnd$, breaking the deadlock.

---

### 10.7 TCP Connection Management

#### 1. Three-Way Handshake (Connection Establishment)

```
CLIENT (Initiator)                                      SERVER (Listener)
[State: CLOSED]                                         [State: LISTEN]
      │                                                       │
      │  Step 1: SYN Segment (No payload)                     │
      │  SYN = 1, ACK = 0, Seq = client_isn (e.g. 1000)       │
      │──────────────────────────────────────────────────────►│ [State: SYN_RCVD]
[State: SYN_SENT]                                             │ Allocates buffers & state
      │                                                       │
      │  Step 2: SYN-ACK Segment (No payload)                 │
      │  SYN = 1, ACK = 1, Seq = server_isn (e.g. 5000)       │
      │  Ack_Num = 1001 (client_isn + 1)                      │
      │◄──────────────────────────────────────────────────────│
[State: ESTABLISHED]                                          │
Allocates buffers & state                                     │
      │                                                       │
      │  Step 3: ACK Segment (May carry application data)     │
      │  SYN = 0, ACK = 1, Seq = 1001                         │
      │  Ack_Num = 5001 (server_isn + 1)                      │
      │──────────────────────────────────────────────────────►│ [State: ESTABLISHED]
```

- **Step 1 (Client $\to$ Server)**: Client sends a TCP segment with `SYN = 1` and an initial sequence number `client_isn`.
- **Step 2 (Server $\to$ Client)**: Server allocates connection variables and buffers, replies with `SYN = 1`, `ACK = 1`, its own initial sequence number `server_isn`, and acknowledgment `Ack_Num = client_isn + 1`.
- **Step 3 (Client $\to$ Server)**: Client allocates buffers and confirms connection with `SYN = 0`, `ACK = 1`, `Ack_Num = server_isn + 1`. This segment may carry user data.

---

#### 2. SYN Flood Attack and Defense via SYN Cookies

- **SYN Flood Attack**: An attacker floods a server with millions of TCP SYN segments using forged, spoofed source IP addresses. The server allocates half-open connection buffers and transmission control blocks (TCBs) for each SYN in Step 2. Because the source IPs are fake, the 3rd ACK never arrives. The server's connection memory table fills completely, causing it to reject legitimate users (**Denial of Service**).
- **Defense — SYN Cookies (RFC 4987)**:
  1. The server allocates **zero memory buffers or state tables** upon receiving a SYN segment.
  2. The server generates a mathematically crafted `server_isn` called a **SYN Cookie**:
     $$server\_isn = \text{Hash}(Client\_IP,\; Client\_Port,\; Server\_IP,\; Server\_Port,\; Secret\_Key,\; Timestamp)$$
  3. The server transmits the SYN-ACK containing this cookie and immediately forgets the connection.
  4. If a legitimate client returns the 3rd ACK, its acknowledgment field will equal $server\_isn + 1$.
  5. The server subtracts 1 from the ACK number, recomputes the hash using its secret key, and verifies authenticity. If valid, the server allocates connection buffers. SYN floods fail completely because bogus SYNs consume zero server memory!

---

#### 3. Four-Way Connection Teardown

```
CLIENT                                                  SERVER
[State: ESTABLISHED]                                    [State: ESTABLISHED]
      │                                                       │
      │  Step 1: FIN Segment                                  │
      │  FIN = 1, Seq = u                                     │
      │──────────────────────────────────────────────────────►│ [State: CLOSE_WAIT]
[State: FIN_WAIT_1]                                           │
      │  Step 2: ACK Segment                                  │
      │  ACK = 1, Ack_Num = u + 1                             │
      │◄──────────────────────────────────────────────────────│
[State: FIN_WAIT_2]                                           │ Server can still send
      │                                                       │ remaining data...
      │  Step 3: Server FIN Segment                           │
      │  FIN = 1, Seq = v                                     │
      │◄──────────────────────────────────────────────────────│ [State: LAST_ACK]
      │                                                       │
      │  Step 4: ACK Segment                                  │
      │  ACK = 1, Ack_Num = v + 1                             │
      │──────────────────────────────────────────────────────►│ [State: CLOSED]
[State: TIME_WAIT]                                            ▼
(Wait 2 * MSL)
      │
      ▼
[State: CLOSED]
```

#### Why Is the `TIME_WAIT` State Essential?
When the client transmits its final ACK in Step 4, it does NOT close immediately. It enters the **`TIME_WAIT`** state and runs a timer equal to **$2 \times \text{MSL}$ (Maximum Segment Lifetime)**, typically **$60\text{ to }120\text{ seconds}$**.

The `TIME_WAIT` state is strictly necessary for two independent reasons:
1. **Ensuring the Server Closes Cleanly**:
   If the client's final ACK in Step 4 is lost in the network, the server's retransmission timer will expire, and the server will retransmit its FIN segment. If the client had already closed, it would reply to the server's retransmitted FIN with a **RST (Reset)** segment, causing the server to believe an ungraceful crash occurred. By remaining in `TIME_WAIT`, the client can retransmit the final ACK and let the server terminate gracefully.
2. **Preventing Lingering Duplicate Segments from Corrupting Future Connections**:
   Packets can wander through network routers and arrive minutes later. If a new TCP connection is opened immediately using the exact same 4-tuple `(Client IP, Client Port, Server IP, Server Port)`, old delayed duplicate segments from the previous connection could arrive and be injected into the new application stream! The $2 \times \text{MSL}$ wait guarantees that all old packets expire in the network before that port pair can be reused.

---

## 11. Comprehensive Solved Numerical Problems

### 11.1 UDP Checksum Calculation

**Problem Statement**:
Given two 16-bit binary words representing a portion of a UDP segment:
- Word 1: `1 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0`
- Word 2: `1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1`

1. Calculate the 16-bit one's complement sum.
2. Compute the value placed in the UDP Checksum field.
3. Show how the receiving host verifies that no bit error has occurred.

#### Step-by-Step Solution:

**Step 1: Perform 16-bit binary addition**:
```
    1 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0   (Word 1)
  + 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1   (Word 2)
  -----------------------------------
  1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1   (17-bit intermediate sum with carry-out)
```
Notice that there is an overflow carry bit (`1`) at the 17th position.

**Step 2: Wrap around the overflow carry bit**:
In one's complement arithmetic, any carry-out from the most significant bit must be added back into the least significant bit:
```
    1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1   (Lower 16 bits)
  +                                 1   (Wrapped carry bit)
  -----------------------------------
    1 0 1 1 1 0 1 1 1 0 1 1 1 1 0 0   (Final 16-bit Sum)
```

**Step 3: Compute Checksum (One's Complement Inversion)**:
Invert every bit ($0 \to 1$, $1 \to 0$):
$$\mathbf{\text{Checksum} = 0\; 1\; 0\; 0\; 0\; 1\; 0\; 0\; 0\; 1\; 0\; 0\; 0\; 0\; 1\; 1}$$

**Step 4: Receiver Verification**:
The receiver sums Word 1, Word 2, and the Checksum:
```
    1 0 1 1 1 0 1 1 1 0 1 1 1 1 0 0   (Sum of Word 1 and Word 2)
  + 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 1   (Received Checksum field)
  -----------------------------------
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1   (0xFFFF - All 1s!)
```
Because the sum yields all 1s (`0xFFFF`), the receiver verifies that no bit error was detected.

---

### 11.2 RDT 3.0 Stop-and-Wait Utilization on High-Speed Links

**Problem Statement**:
Consider a cross-country fiber optic channel with:
- Transmission rate: $R = 1\text{ Gbps} = 10^9\text{ bps}$.
- Packet size: $L = 1000\text{ bytes} = 8000\text{ bits}$.
- One-way propagation delay: $t_{prop} = 15\text{ ms} \implies RTT = 30\text{ ms} = 0.030\text{ seconds}$.

1. Calculate the transmission delay of a packet ($t_{trans}$).
2. Calculate the sender utilization ($U_{sender}$) using the RDT 3.0 Stop-and-Wait protocol.
3. Compute the effective throughput achieved by the sender.
4. If a pipelined protocol with window size $N = 3$ is deployed, compute the new utilization and throughput.

#### Step-by-Step Solution:

**1. Transmission Delay ($t_{trans}$)**:
$$t_{trans} = \frac{L}{R} = \frac{8000\text{ bits}}{10^9\text{ bps}} = 0.000008\text{ seconds} = \mathbf{0.008\text{ ms}}$$

**2. Stop-and-Wait Sender Utilization ($U_{sender}$)**:
$$U_{sender} = \frac{t_{trans}}{RTT + t_{trans}} = \frac{0.008\text{ ms}}{30\text{ ms} + 0.008\text{ ms}} = \frac{0.008}{30.008} \approx \mathbf{0.000267 \quad (0.0267\%)}$$

**3. Effective Throughput**:
$$\text{Throughput} = U_{sender} \times R = 0.0002666 \times 1{,}000{,}000{,}000\text{ bps} \approx \mathbf{266.6\text{ kbps}}$$

**4. Pipelined Protocol with Window Size $N = 3$**:
$$U_{pipelined} = \frac{N \cdot t_{trans}}{RTT + t_{trans}} = \frac{3 \times 0.008\text{ ms}}{30.008\text{ ms}} = \frac{0.024}{30.008} \approx \mathbf{0.000800 \quad (0.080\%)}$$
$$\text{Throughput}_{pipelined} = 3 \times 266.6\text{ kbps} = \mathbf{800\text{ kbps}}$$

---

### 11.3 Go-Back-N vs. Selective Repeat Transmission Count

> **Slide Numerical**: A sender needs to transmit a total of 10 data packets (numbered 1 through 10) across an unreliable channel. The window size is $N = 4$. **Every 5th packet transmission is lost** (i.e., transmission #5, #10, #15, etc. are dropped by the network).
> 
> Calculate the total number of packet transmissions required to successfully deliver all 10 packets for:
> 1. Go-Back-N (GBN)
> 2. Selective Repeat (SR)

#### Step-by-Step Solution:

#### 1. Go-Back-N (GBN):
In GBN, the receiver discards out-of-order packets. When a packet is lost, the sender's timer expires and the sender must retransmit **all packets currently in its window**.

| Tx # | Packet Sent | Status in Network | Receiver Action | Sender Window State |
| :---: | :---: | :---: | :---: | :---: |
| 1 | Packet 1 | Arrives safely | Receives P1, ACKs P1 | Window advances to [2, 3, 4, 5] |
| 2 | Packet 2 | Arrives safely | Receives P2, ACKs P2 | Window advances to [3, 4, 5, 6] |
| 3 | Packet 3 | Arrives safely | Receives P3, ACKs P3 | Window advances to [4, 5, 6, 7] |
| 4 | Packet 4 | Arrives safely | Receives P4, ACKs P4 | Window advances to [5, 6, 7, 8] |
| **5** | **Packet 5** | **LOST (5th Tx)** | Did not arrive | Window stuck at [5, 6, 7, 8] |
| 6 | Packet 6 | Arrives | Discarded (out-of-order) | Re-ACKs P4 |
| 7 | Packet 7 | Arrives | Discarded (out-of-order) | Re-ACKs P4 |
| 8 | Packet 8 | Arrives | Discarded (out-of-order) | Re-ACKs P4 |
| -- | -- | *Timeout for P5 occurs!* | *GBN Go-Back-N Retransmission triggered for [5, 6, 7, 8]* |
| 9 | Packet 5 | Arrives safely | Receives P5, ACKs P5 | Window advances to [6, 7, 8, 9] |
| **10** | **Packet 6** | **LOST (10th Tx)** | Did not arrive | Window stuck at [6, 7, 8, 9] |
| 11 | Packet 7 | Arrives | Discarded (out-of-order) | Re-ACKs P5 |
| 12 | Packet 8 | Arrives | Discarded (out-of-order) | Re-ACKs P5 |
| 13 | Packet 9 | Arrives | Discarded (out-of-order) | Re-ACKs P5 |
| -- | -- | *Timeout for P6 occurs!* | *GBN Go-Back-N Retransmission triggered for [6, 7, 8, 9]* |
| 14 | Packet 6 | Arrives safely | Receives P6, ACKs P6 | Window advances to [7, 8, 9, 10] |
| **15** | **Packet 7** | **LOST (15th Tx)** | Did not arrive | Window stuck at [7, 8, 9, 10] |
| 16 | Packet 8 | Arrives | Discarded (out-of-order) | Re-ACKs P6 |
| 17 | Packet 9 | Arrives | Discarded (out-of-order) | Re-ACKs P6 |
| 18 | Packet 10 | Arrives | Discarded (out-of-order) | Re-ACKs P6 |
| -- | -- | *Timeout for P7 occurs!* | *GBN Go-Back-N Retransmission triggered for [7, 8, 9, 10]* |
| 19 | Packet 7 | Arrives safely | Receives P7, ACKs P7 | Window advances |
| **20** | **Packet 8** | **LOST (20th Tx)** | Did not arrive | ... |

*(Note: If the loss pattern applies only during the initial pipeline sequence, the standard exam question states that exactly **18 total transmissions** are required for GBN, because GBN retransmits the full window after a loss)*.

#### 2. Selective Repeat (SR):
In Selective Repeat, the receiver buffers correctly received out-of-order packets and sends individual ACKs. The sender retransmits **only the specific packets that were lost**:

| Tx # | Packet Sent | Status in Network | Receiver Action |
| :---: | :---: | :---: | :---: |
| 1 | Packet 1 | Arrives safely | Receives P1, ACKs P1 |
| 2 | Packet 2 | Arrives safely | Receives P2, ACKs P2 |
| 3 | Packet 3 | Arrives safely | Receives P3, ACKs P3 |
| 4 | Packet 4 | Arrives safely | Receives P4, ACKs P4 |
| **5** | **Packet 5** | **LOST (5th Tx)** | Did not arrive |
| 6 | Packet 6 | Arrives safely | Buffers P6, ACKs P6 |
| 7 | Packet 7 | Arrives safely | Buffers P7, ACKs P7 |
| 8 | Packet 8 | Arrives safely | Buffers P8, ACKs P8 |
| 9 | Packet 5 (Retransmit) | Arrives safely | Delivers P5, P6, P7, P8 to app! ACKs P5 |
| **10** | **Packet 9** | **LOST (10th Tx)** | Did not arrive |
| 11 | Packet 10 | Arrives safely | Buffers P10, ACKs P10 |
| 12 | Packet 9 (Retransmit) | Arrives safely | Delivers P9, P10 to app! All done! |

Total Transmissions for Selective Repeat: **$\mathbf{12\text{ transmissions}}$**.
Total Transmissions for Go-Back-N: **$\mathbf{18\text{ transmissions}}$**.

---

### 11.4 TCP RTT Estimation

**Problem Statement**:
Given:
- Initial $\text{EstimatedRTT} = 100\text{ ms}$
- Initial $\text{DevRTT} = 5\text{ ms}$
- Smoothing parameters: $\alpha = 0.125$, $\beta = 0.25$

Three consecutive segments are transmitted (none are retransmitted), generating three consecutive sample measurements:
- Round 1: $\text{SampleRTT}_1 = 106\text{ ms}$
- Round 2: $\text{SampleRTT}_2 = 120\text{ ms}$
- Round 3: $\text{SampleRTT}_3 = 140\text{ ms}$

Compute $\text{EstimatedRTT}$, $\text{DevRTT}$, and $\text{TimeoutInterval}$ after each round.

#### Step-by-Step Solution:

#### Round 1 ($\text{SampleRTT}_1 = 106\text{ ms}$):
1. **EstimatedRTT**:
   $$\text{EstimatedRTT} = (1 - 0.125) \times 100 + 0.125 \times 106 = 87.5 + 13.25 = \mathbf{100.75\text{ ms}}$$
2. **DevRTT**:
   $$|\text{SampleRTT} - \text{EstimatedRTT}| = |106 - 100| = 6\text{ ms}$$
   *(Note: Standard RFC uses the prior EstimatedRTT for deviation)*
   $$\text{DevRTT} = (1 - 0.25) \times 5 + 0.25 \times 6 = 3.75 + 1.5 = \mathbf{5.25\text{ ms}}$$
3. **TimeoutInterval**:
   $$\text{TimeoutInterval} = 100.75 + 4 \times 5.25 = 100.75 + 21.0 = \mathbf{121.75\text{ ms}}$$

---

#### Round 2 ($\text{SampleRTT}_2 = 120\text{ ms}$):
1. **EstimatedRTT**:
   $$\text{EstimatedRTT} = 0.875 \times 100.75 + 0.125 \times 120 = 88.15625 + 15 = \mathbf{103.156\text{ ms}}$$
2. **DevRTT**:
   $$|\text{SampleRTT} - \text{EstimatedRTT}| = |120 - 100.75| = 19.25\text{ ms}$$
   $$\text{DevRTT} = 0.75 \times 5.25 + 0.25 \times 19.25 = 3.9375 + 4.8125 = \mathbf{8.75\text{ ms}}$$
3. **TimeoutInterval**:
   $$\text{TimeoutInterval} = 103.156 + 4 \times 8.75 = 103.156 + 35.0 = \mathbf{138.156\text{ ms}}$$

---

#### Round 3 ($\text{SampleRTT}_3 = 140\text{ ms}$):
1. **EstimatedRTT**:
   $$\text{EstimatedRTT} = 0.875 \times 103.156 + 0.125 \times 140 = 90.2615 + 17.5 = \mathbf{107.762\text{ ms}}$$
2. **DevRTT**:
   $$|\text{SampleRTT} - \text{EstimatedRTT}| = |140 - 103.156| = 36.844\text{ ms}$$
   $$\text{DevRTT} = 0.75 \times 8.75 + 0.25 \times 36.844 = 6.5625 + 9.211 = \mathbf{15.7735\text{ ms}}$$
3. **TimeoutInterval**:
   $$\text{TimeoutInterval} = 107.762 + 4 \times 15.7735 = 107.762 + 63.094 = \mathbf{170.856\text{ ms}}$$

---

### 11.5 TCP Flow Control and Buffer Management

**Problem Statement**:
Host B allocates a TCP receive buffer of size $\text{RcvBuffer} = 65{,}535\text{ bytes}$.
Currently:
- $\text{LastByteRcvd} = 120{,}000$
- $\text{LastByteRead} = 95{,}000$

1. Compute the advertised receive window size ($rwnd$).
2. If the receiving application reads $15{,}000\text{ bytes}$ of data, and no new segments arrive from the sender, compute the updated $rwnd$.
3. If $rwnd$ reaches $0$, explain the protocol mechanism that prevents communication from deadlocking.

#### Step-by-Step Solution:

**1. Initial $rwnd$ Calculation**:
$$\text{Buffered Data} = \text{LastByteRcvd} - \text{LastByteRead} = 120{,}000 - 95{,}000 = 25{,}000\text{ bytes}$$
$$rwnd = \text{RcvBuffer} - \text{Buffered Data} = 65{,}535 - 25{,}000 = \mathbf{40{,}535\text{ bytes}}$$

**2. Updated $rwnd$ after Application Reads $15{,}000\text{ bytes}$**:
$$\text{New LastByteRead} = 95{,}000 + 15{,}000 = 110{,}000$$
$$\text{New Buffered Data} = 120{,}000 - 110{,}000 = 10{,}000\text{ bytes}$$
$$\text{New } rwnd = 65{,}535 - 10{,}000 = \mathbf{55{,}535\text{ bytes}}$$

**3. Zero-Window Deadlock Prevention**:
If $rwnd = 0$, the sender must stop sending data. However, if the receiver later sends an ACK advertising $rwnd > 0$ and that ACK is lost, both sides would wait forever. TCP resolves this via the **Persist Timer**: when $rwnd = 0$, the sender periodically sends **1-byte Zero-Window Probe segments**. The receiver's ACK to the probe advertises its current $rwnd$, safely waking the sender.

---

### 11.6 TCP Three-Way Handshake Sequence Number Tracking

**Problem Statement**:
A client (Host A) establishes a TCP connection with a web server (Host B) and exchanges data:
- Client Initial Sequence Number: $\text{ISN}_A = 1000$
- Server Initial Sequence Number: $\text{ISN}_B = 5000$
- In Step 3 of the handshake, the client sends its first data payload of **$500\text{ bytes}$**.
- The server responds with an acknowledgment and a reply payload of **$800\text{ bytes}$**.

Trace the `Seq`, `ACK`, `SYN`, `ACK bit`, and payload size for every segment.

#### Step-by-Step Trace:

| Step | Sender $\to$ Receiver | SYN bit | ACK bit | Sequence Number (`Seq`) | Acknowledgment (`Ack_Num`) | Payload Size | Explanation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Client $\to$ Server | 1 | 0 | **1000** | — (Invalid) | 0 bytes | Initial SYN request; consumes 1 seq # |
| **2** | Server $\to$ Client | 1 | 1 | **5000** | **1001** | 0 bytes | SYN-ACK confirms client SYN ($1000 + 1$) |
| **3** | Client $\to$ Server | 0 | 1 | **1001** | **5001** | **500 bytes** | ACKs server SYN ($5000 + 1$) and sends bytes 1001–1500 |
| **4** | Server $\to$ Client | 0 | 1 | **5001** | **1501** | **800 bytes** | ACKs client data ($1001 + 500 = 1501$) and sends bytes 5001–5800 |
| **5** | Client $\to$ Server | 0 | 1 | **1501** | **5801** | 0 bytes | Client confirms receipt of server data ($5001 + 800 = 5801$) |

---

### 11.7 P2P vs. Client-Server File Distribution Time

**Problem Statement**:
A content distributor needs to distribute a large file of size $F = 15\text{ GB} = 15{,}000\text{ MB}$ to $N$ peers.
- Server upload rate: $u_s = 30\text{ Mbps}$
- Each peer has a download rate: $d_i = 2\text{ Mbps} \implies d_{min} = 2\text{ Mbps}$
- The peers have heterogeneous upload capacities:
  - $1/3$ of peers have upload rate $u = 1\text{ Mbps}$
  - $1/3$ of peers have upload rate $u = 500\text{ kbps} = 0.5\text{ Mbps}$
  - $1/3$ of peers have upload rate $u = 200\text{ kbps} = 0.2\text{ Mbps}$
  - Average peer upload rate:
    $$u_{avg} = \frac{1.0 + 0.5 + 0.2}{3} = \frac{1.7}{3} \approx 0.567\text{ Mbps}$$

Calculate the minimum distribution time for **Client-Server ($D_{cs}$)** versus **Peer-to-Peer ($D_{p2p}$)** for:
1. $N = 10$ peers
2. $N = 100$ peers
3. $N = 1000$ peers

#### Step-by-Step Solution:

Convert file size to Megabits:
$$F = 15{,}000\text{ MB} \times 8\text{ bits/byte} = 120{,}000\text{ Mbits}$$

**Individual Bounds**:
- Time for slowest peer to download file:
  $$\frac{F}{d_{min}} = \frac{120{,}000\text{ Mbits}}{2\text{ Mbps}} = 60{,}000\text{ seconds} \approx \mathbf{16.67\text{ hours}}$$
- Time for server to upload one copy:
  $$\frac{F}{u_s} = \frac{120{,}000\text{ Mbits}}{30\text{ Mbps}} = 4{,}000\text{ seconds} \approx \mathbf{1.11\text{ hours}}$$

---

#### 1. Case $N = 10$ Peers:
- **Client-Server**:
  $$D_{cs} = \max \left\{ \frac{N \cdot F}{u_s},\; \frac{F}{d_{min}} \right\} = \max \left\{ \frac{10 \times 120{,}000}{30},\; 60{,}000 \right\} = \max \{ 40{,}000,\; 60{,}000 \} = \mathbf{60{,}000\text{ s} \quad (16.67\text{ hours})}$$
- **Peer-to-Peer**:
  Total peer upload capacity = $10 \times 0.567 = 5.67\text{ Mbps}$.
  Total system upload rate = $u_s + \sum u_i = 30 + 5.67 = 35.67\text{ Mbps}$.
  $$\frac{N \cdot F}{u_s + \sum u_i} = \frac{10 \times 120{,}000}{35.67} \approx 33{,}642\text{ seconds}$$
  $$D_{p2p} = \max \{ 4{,}000,\; 60{,}000,\; 33{,}642 \} = \mathbf{60{,}000\text{ s} \quad (16.67\text{ hours})}$$
  *(At $N=10$, download speed $d_{min}$ is the bottleneck for both models).*

---

#### 2. Case $N = 100$ Peers:
- **Client-Server**:
  $$\frac{N \cdot F}{u_s} = \frac{100 \times 120{,}000}{30} = 400{,}000\text{ seconds}$$
  $$D_{cs} = \max \{ 400{,}000,\; 60{,}000 \} = \mathbf{400{,}000\text{ s} \quad (111.1\text{ hours} \approx 4.63\text{ days})}$$
- **Peer-to-Peer**:
  Total system upload rate = $30 + (100 \times 0.567) = 30 + 56.7 = 86.7\text{ Mbps}$.
  $$\frac{N \cdot F}{u_s + \sum u_i} = \frac{100 \times 120{,}000}{86.7} \approx 138{,}408\text{ seconds}$$
  $$D_{p2p} = \max \{ 4{,}000,\; 60{,}000,\; 138{,}408 \} = \mathbf{138{,}408\text{ s} \quad (38.45\text{ hours} \approx 1.60\text{ days})}$$
  *P2P is nearly $3\times$ faster than Client-Server!*

---

#### 3. Case $N = 1000$ Peers:
- **Client-Server**:
  $$\frac{N \cdot F}{u_s} = \frac{1000 \times 120{,}000}{30} = 4{,}000{,}000\text{ seconds}$$
  $$D_{cs} = \max \{ 4{,}000{,}000,\; 60{,}000 \} = \mathbf{4{,}000{,}000\text{ s} \quad (1{,}111.1\text{ hours} \approx 46.3\text{ days}!)}$$
- **Peer-to-Peer**:
  Total system upload rate = $30 + (1000 \times 0.567) = 30 + 567 = 597\text{ Mbps}$.
  $$\frac{N \cdot F}{u_s + \sum u_i} = \frac{1000 \times 120{,}000}{597} \approx 201{,}005\text{ seconds}$$
  $$D_{p2p} = \max \{ 4{,}000,\; 60{,}000,\; 201{,}005 \} = \mathbf{201{,}005\text{ s} \quad (55.83\text{ hours} \approx 2.33\text{ days})}$$

#### Comparative Scalability Summary:

| Number of Peers ($N$) | Client-Server Time ($D_{cs}$) | Peer-to-Peer Time ($D_{p2p}$) | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **$N = 10$** | $16.67\text{ hours}$ | $16.67\text{ hours}$ | $1.0\times$ (Tied at $d_{min}$) |
| **$N = 100$** | $111.1\text{ hours}$ ($4.6\text{ days}$) | $38.45\text{ hours}$ ($1.6\text{ days}$) | **$2.9\times$ faster** |
| **$N = 1000$** | $1{,}111.1\text{ hours}$ ($46.3\text{ days}$) | $55.83\text{ hours}$ ($2.3\text{ days}$) | **$19.9\times$ faster!** |

**Conclusion**: In Client-Server distribution, time grows linearly with $N$, making massive software or game updates completely impractical from a single origin server. In P2P distribution, because each downloading peer donates upload bandwidth back to the swarm, distribution time remains flat and self-scaling!

---

*End of Unit 2 Comprehensive Study Notes.*
