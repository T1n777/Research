# Unit 1: Computer Networks and the Internet, Application Layer

**A Complete Exam Study Reference — PES University (UE25CS243A: Computer Networks)**

---

## Table of Contents

1. [Introduction to Computer Networks and the Internet](#1-introduction-to-computer-networks-and-the-internet)
   - 1.1 [What is a Computer Network?](#11-what-is-a-computer-network)
   - 1.2 [The "Nuts-and-Bolts" View of the Internet](#12-the-nuts-and-bolts-view-of-the-internet)
   - 1.3 [The "Services" View of the Internet](#13-the-services-view-of-the-internet)
   - 1.4 [What is a Protocol? (Formal Definition)](#14-what-is-a-protocol-formal-definition)
   - 1.5 [Network Classification by Scale: PAN, LAN, MAN, WAN](#15-network-classification-by-scale-pan-lan-man-wan)
2. [The Network Edge](#2-the-network-edge)
   - 2.1 [End Systems, Clients, and Servers](#21-end-systems-clients-and-servers)
   - 2.2 [Access Networks](#22-access-networks)
     - [Home Access Networks (DSL, Cable HFC, FTTH, 5G Fixed Wireless)](#home-access-networks)
     - [Enterprise (Campus/Corporate) Networks](#enterprise-campuscorporate-networks)
     - [Wireless Access Networks (WLAN / WiFi, Cellular 4G/5G)](#wireless-access-networks)
   - 2.3 [Physical Media](#23-physical-media)
     - [Guided Media (Twisted Pair, Coaxial Cable, Fiber Optic Cable)](#guided-media)
     - [Unguided Media (Terrestrial Radio/Microwave, Satellite Microwave, Infrared)](#unguided-media)
   - 2.4 [Physical-Layer Devices (Repeater, Hub, Modem)](#24-physical-layer-devices)
3. [The Network Core](#3-the-network-core)
   - 3.1 [Packet Switching](#31-packet-switching)
     - [Store-and-Forward Transmission](#store-and-forward-transmission)
     - [Statistical Multiplexing and Bursty Traffic](#statistical-multiplexing-and-bursty-traffic)
   - 3.2 [Circuit Switching](#32-circuit-switching)
     - [Frequency Division Multiplexing (FDM) vs. Time Division Multiplexing (TDM)](#frequency-division-multiplexing-fdm-vs-time-division-multiplexing-tdm)
   - 3.3 [Packet Switching vs. Circuit Switching — Comprehensive Comparison](#33-packet-switching-vs-circuit-switching--comprehensive-comparison)
   - 3.4 [Core Functions: Routing vs. Forwarding](#34-core-functions-routing-vs-forwarding)
   - 3.5 [A Network of Networks — Internet Structure and Hierarchy](#35-a-network-of-networks--internet-structure-and-hierarchy)
4. [Delay, Loss, and Throughput in Packet-Switched Networks](#4-delay-loss-and-throughput-in-packet-switched-networks)
   - 4.1 [The Four Sources of Nodal Delay](#41-the-four-sources-of-nodal-delay)
   - 4.2 [End-to-End Delay Across Multiple Links](#42-end-to-end-delay-across-multiple-links)
   - 4.3 [Queuing Delay, Traffic Intensity, and Buffer Loss](#43-queuing-delay-traffic-intensity-and-buffer-loss)
   - 4.4 [Real-World Network Diagnostic Tools: Ping and Traceroute](#44-real-world-network-diagnostic-tools-ping-and-traceroute)
   - 4.5 [Throughput and the Bottleneck Link](#45-throughput-and-the-bottleneck-link)
5. [Protocol Layers and Network Devices](#5-protocol-layers-and-network-devices)
   - 5.1 [Why Layering Exists (The Layered Architecture)](#51-why-layering-exists-the-layered-architecture)
   - 5.2 [The OSI 7-Layer Reference Model](#52-the-osi-7-layer-reference-model)
   - 5.3 [The TCP/IP 5-Layer Protocol Suite](#53-the-tcpip-5-layer-protocol-suite)
   - 5.4 [OSI vs. TCP/IP — Head-to-Head Comparison](#54-osi-vs-tcpip--head-to-head-comparison)
   - 5.5 [Encapsulation and Decapsulation (Protocol Data Units)](#55-encapsulation-and-decapsulation-protocol-data-units)
   - 5.6 [Full Taxonomy of Network Devices and Layer Mapping](#56-full-taxonomy-of-network-devices-and-layer-mapping)
   - 5.7 [Collision Domains vs. Broadcast Domains](#57-collision-domains-vs-broadcast-domains)
6. [Network Application Principles](#6-network-application-principles)
   - 6.1 [Network Application Architectures (Client-Server, P2P, Hybrid)](#61-network-application-architectures-client-server-p2p-hybrid)
   - 6.2 [Processes and Inter-Process Communication (IPC)](#62-processes-and-inter-process-communication-ipc)
   - 6.3 [Sockets — The Application Programming Interface (API)](#63-sockets--the-application-programming-interface-api)
   - 6.4 [Addressing a Process: IP Addresses and Port Numbers](#64-addressing-a-process-ip-addresses-and-port-numbers)
   - 6.5 [Transport Service Requirements of Applications](#65-transport-service-requirements-of-applications)
   - 6.6 [Transport Services Provided by the Internet: TCP vs. UDP](#66-transport-services-provided-by-the-internet-tcp-vs-udp)
7. [The Web, HTTP, and HTTPS](#7-the-web-http-and-https)
   - 7.1 [Overview of HTTP (HyperText Transfer Protocol)](#71-overview-of-http-hypertext-transfer-protocol)
   - 7.2 [Statelessness — Architectural Rationale and Impact](#72-statelessness--architectural-rationale-and-impact)
   - 7.3 [Non-Persistent vs. Persistent HTTP Connections](#73-non-persistent-vs-persistent-http-connections)
   - 7.4 [HTTP Message Format (Request and Response Messages)](#74-http-message-format-request-and-response-messages)
   - 7.5 [HTTP Request Methods and Status Codes](#75-http-request-methods-and-status-codes)
   - 7.6 [Protocol Evolution: HTTP/1.1, HTTP/2, and HTTP/3](#76-protocol-evolution-http11-http2-and-http3)
   - 7.7 [HTTPS and Transport Layer Security (TLS 1.3)](#77-https-and-transport-layer-security-tls-13)
8. [User-Server Interaction: Cookies and Web Caching](#8-user-server-interaction-cookies-and-web-caching)
   - 8.1 [Cookies — Maintaining State on a Stateless Web](#81-cookies--maintaining-state-on-a-stateless-web)
   - 8.2 [Web Caching (Proxy Servers)](#82-web-caching-proxy-servers)
   - 8.3 [The Conditional GET Protocol](#83-the-conditional-get-protocol)
9. [Solved Numericals (Comprehensive Exam-Style Problems)](#9-solved-numericals-comprehensive-exam-style-problems)
   - 9.1 [End-to-End Nodal Delay Across Multiple Links](#91-end-to-end-nodal-delay-across-multiple-links)
   - 9.2 [Queuing Delay Modeling via Traffic Intensity (Non-Linear Growth)](#92-queuing-delay-modeling-via-traffic-intensity-non-linear-growth)
   - 9.3 [Throughput and Bottleneck Links (Shared Backbone & File Download)](#93-throughput-and-bottleneck-links-shared-backbone--file-download)
   - 9.4 [Circuit Switching (TDM) — Transmission Time on Homogeneous & Heterogeneous Links](#94-circuit-switching-tdm--transmission-time-on-homogeneous--heterogeneous-links)
   - 9.5 [Non-Persistent vs. Persistent HTTP — Round-Trip Time (RTT) Accounting](#95-non-persistent-vs-persistent-http--round-trip-time-rtt-accounting)
   - 9.6 [Web Caching — Access Link Utilization and End-to-End Delay Reduction](#96-web-caching--access-link-utilization-and-end-to-end-delay-reduction)
   - 9.7 [Message Segmentation and Pipelining Benefit in Packet Switching](#97-message-segmentation-and-pipelining-benefit-in-packet-switching)

---

## 1. Introduction to Computer Networks and the Internet

### 1.1 What is a Computer Network?

A **computer network** is defined as an interconnected collection of two or more autonomous computing devices that communicate with one another and share hardware, software, or data resources using standardized communication protocols.

The **Internet** (capitalized to designate the specific worldwide public network) is a massive **"network of networks"** that interconnects billions of computing devices globally. Computer scientists analyze the Internet through two complementary frameworks:
1. The **Nuts-and-Bolts View**: Describes the physical hardware and software components that physically constitute the network infrastructure.
2. The **Services View**: Describes the network as an enabling platform that provides communication capabilities to distributed applications.

---

### 1.2 The "Nuts-and-Bolts" View of the Internet

From a hardware and engineering perspective, the Internet consists of the following basic building blocks:

| Component | Formal Definition | Technical Details & Real-World Examples |
|---|---|---|
| **End Systems (Hosts)** | Any computing device connected to the network that runs network application programs at the network edge. | Traditional systems: Desktop Personal Computers (PCs), laptops, workstations, enterprise server blades.<br>Modern Internet of Things (IoT) devices: Smartphones, tablets, smart televisions, security webcams, smart thermostats, automobile telemetry units, industrial sensors. |
| **Packet Switches** | Devices that receive an incoming packet of data on one of their communication links and forward that packet onto one of their outgoing communication links. | **Routers**: Operate at Layer 3 (Network Layer); forward packets based on logical destination IP addresses.<br>**Link-Layer Switches**: Operate at Layer 2 (Data Link Layer); forward frames based on physical destination MAC addresses. |
| **Communication Links** | The physical transmission media that interconnect packet switches and end systems, propagating signals as electrical voltages, light pulses, or electromagnetic waves. | Guided media: Twisted-pair copper wire, coaxial cable, optical fiber.<br>Unguided media: Terrestrial radio, cellular radio spectrum, satellite microwave. |
| **Transmission Rate (Bandwidth)** | The rate at which a link can physically push data bits into the transmission medium, measured in bits per second. | Metric units: bits per second (bps), Kilobits per second ($1\text{ Kbps} = 10^3\text{ bps}$), Megabits per second ($1\text{ Mbps} = 10^6\text{ bps}$), Gigabits per second ($1\text{ Gbps} = 10^9\text{ bps}$), Terabits per second ($1\text{ Tbps} = 10^{12}\text{ bps}$). |
| **Networks** | A collection of interconnected hosts, packet switches, and links administered under a single coherent administrative domain. | Home networks, Enterprise Networks (Universities and Corporations), Mobile Cellular Networks, and commercial **Internet Service Providers (ISP (Internet Service Provider)s)**. |
| **Protocols** | Standardized rules establishing the syntax, semantics, and synchronization of network communication. | **TCP** (Transmission Control Protocol), **IP** (Internet Protocol), **HTTP** (HyperText Transfer Protocol), **Ethernet** (IEEE 802.3), **WiFi** (IEEE 802.11). |

Standards for Internet protocols are developed openly and collaboratively by the **IETF (Internet Engineering Task Force)**. These technical specifications are published as formal documents called **RFCs (Request for Comments)**. Examples: RFC 793 defines TCP; RFC 2616 defines HTTP/1.1; RFC 8446 defines TLS 1.3.

---

### 1.3 The "Services" View of the Internet

From the perspective of software development and end users, the Internet is an **infrastructure that provides services to distributed applications**:

1. **Platform for Distributed Applications**:
   The Internet provides the foundational communication framework that allows programs running on separate end systems to exchange data. Examples include the World Wide Web, electronic mail (email), video streaming platforms (YouTube, Netflix), peer-to-peer file sharing (BitTorrent), voice and video calls (Voice over Internet Protocol — VoIP), cloud storage, multiplayer gaming, and distributed databases.
2. **The Application Programming Interface (API)**:
   The Internet provides a programming interface to distributed applications, exposing a set of software instructions and rules that a program running on one host uses to instruct the Internet infrastructure to deliver data to a specific destination program running on another host.
   - **The Postal Analogy**: Sending data into the Internet API is directly comparable to mailing a paper letter through the postal service. You write the message, place it in an envelope, write the destination address on the outside, and drop it into a collection box. You do not manage the postal delivery trucks, the sorting hubs, or the transport planes — the postal infrastructure delivers the letter according to predefined operational rules. In the exact same way, an application developer writes data to a **socket interface**, trusting the underlying operating system and transport infrastructure to deliver the payload.

---

### 1.4 What is a Protocol? (Formal Definition)

All communication activity on the Internet is controlled by protocols. 

> [!IMPORTANT]
> **Formal Definition of a Protocol**:
> A **protocol** defines the **format** and the **order** of messages exchanged between two or more communicating entities, as well as the **actions taken** on the transmission and/or receipt of a message or other event.

A human analogy illustrates this definition:
- When you greet someone: You say "Hello" (defined format and order). You wait for them to reply "Hello" (expected response). If they reply "Hello", you can ask "What time is it?". If they do not respond or yell angrily, you take a different action (error handling / timeout).
- In a computer network: When a web browser connects to a web server, the computer first sends a TCP Connection Request packet. The server returns a TCP Connection Response packet. Only after receiving this acknowledgement does the browser send an `HTTP GET` request message. The server responds with the requested HTML file. If a packet is lost or corrupted, timers expire and retransmissions occur.

---

### 1.5 Network Classification by Scale: PAN, LAN, MAN, WAN

Computer networks are classically categorized according to their physical and geographical span:

| Classification | Full Name | Geographic Scope | Typical Coverage | Typical Technologies |
|---|---|---|---|---|
| **PAN** | Personal Area Network | Immediate space around an individual | Within a radius of 1 to 10 meters | Bluetooth (IEEE 802.15.1), Zigbee, Ultra-Wideband (UWB), USB |
| **LAN** | Local Area Network | A single room, office, home, school building, or campus | Tens of meters to a few kilometers | Ethernet (IEEE 802.3, twisted-pair/fiber), WiFi (IEEE 802.11) |
| **MAN** | Metropolitan Area Network | An entire city or metropolitan municipality | 5 to 50 kilometers | Cable TV distribution networks, Metro Ethernet, FDDI |
| **WAN** | Wide Area Network | A state, country, continent, or the entire globe | Hundreds to thousands of kilometers | Fiber optic undersea cables, satellite microwave links, leased telco lines |

The Internet is the largest, most pervasive example of a **WAN (Wide Area Network)**.

---

## 2. The Network Edge

### 2.1 End Systems, Clients, and Servers

The **network edge** consists of the devices physically connected to the boundary of the communication network. These devices are exclusively **end systems (hosts)**.

End systems are fundamentally partitioned into two operational roles:
1. **Clients**: Devices that initiate communication sessions to request services. Clients typically include desktop computers, laptops, smartphones, and IoT devices. Clients connect intermittently and are frequently assigned dynamic, temporary IP addresses.
2. **Servers**: Powerful, always-on computing systems with permanent, well-known IP addresses that listen for incoming client requests and provide requested resources (web pages, video files, email mailboxes).
   - In modern network architecture, enterprise servers are rarely standalone machines; they are aggregated inside massive **data centers**. A single commercial data center (operated by cloud providers such as Google, Microsoft, or Amazon) houses hundreds of thousands of server blades interconnected by high-performance internal switching fabrics operating at 10 Gbps, 40 Gbps, 100 Gbps, or 400 Gbps.

---

### 2.2 Access Networks

An **access network** is the physical network that connects an end system (host) to the first router on the path to any other distant end system. This first router is called the **edge router**.

Access networks are evaluated using two critical metrics:
1. **Transmission Rate (Bandwidth)**: How many bits per second can be pushed onto the link.
2. **Shared vs. Dedicated Medium**:
   - **Dedicated Access**: The entire link bandwidth is exclusively reserved for a single subscriber (e.g., DSL).
   - **Shared Access**: Multiple subscriber households share the same transmission channel and compete for aggregate bandwidth; if multiple neighbors download large files simultaneously, each user experiences reduced throughput (e.g., Cable Internet).

#### Home Access Networks

```
Home Network:
[ PC ] ──┐
         ├─► [ Home Router / AP / Switch / NAT / Firewall ] ──► [ Access Link ] ──► [ ISP Edge Router ]
[ Phone ]┘
```

Modern homes connect to the Internet using one of four primary technologies:

| Access Technology | Physical Medium | Key Network Equipment & Architecture | Access Model & Sharing | Typical Transmission Rates | Operating Principle & Key Characteristics |
|---|---|---|---|---|---|
| **DSL (Digital Subscriber Line)** | Existing twisted-pair copper telephone wire to Central Office (CO) | • **Home**: Splitter & DSL modem<br>• **Central Office**: DSLAM (DSL Access Multiplexer) | **Dedicated access** (dedicated copper wire pair to individual household) | • Up to 24–52 Mbps downstream<br>• Up to 3.5 Mbps upstream<br>*(Asymmetric)* | • Uses **Frequency Division Multiplexing (FDM)**:<br>&nbsp;&nbsp;- $0\text{–}4\text{ kHz}$: Two-way analog telephone voice<br>&nbsp;&nbsp;- $4\text{–}50\text{ kHz}$: Upstream data channel<br>&nbsp;&nbsp;- $50\text{ kHz–}1\text{ MHz}$: Downstream data channel<br>• Speed attenuates sharply with distance from Central Office (< 5–10 km). |
| **Cable Internet (HFC — Hybrid Fiber-Coax)** | Hybrid Fiber-Coax: Optical fiber from headend to neighborhood nodes; coaxial cable to homes | • **Home**: Cable modem<br>• **Cable Headend**: CMTS (Cable Modem Termination System) | **Shared broadcast medium** (multiple households share neighborhood coax segment) | • Up to 1 Gbps downstream<br>• Up to 35–50 Mbps upstream<br>*(Asymmetric)* | • Operates over FDM across distinct TV, upstream, and downstream channels governed by the **DOCSIS** standard.<br>• Packets sent downstream reach every home on the cable segment; modems filter out packets not addressed to them.<br>• Shared bandwidth: heavy simultaneous usage by neighbors reduces individual throughput. |
| **FTTH (Fiber to the Home)** | Optical fiber cable pulled directly into the subscriber's residence | • **Central Office**: OLT (Optical Line Terminal)<br>• **Neighborhood**: Passive Optical Splitters (1:16 to 1:64)<br>• **Home**: ONT (Optical Network Terminal) | **Dedicated strand to home**, sharing an optical feeder trunk via passive splitters (**PON**) | • 100 Mbps up to 10 Gbps<br>*(Capable of **Symmetric** rates)* | • Employs **Passive Optical Networks (PON)** with unpowered, purely passive optical splitters.<br>• ONT converts incoming light pulses into standard electrical Ethernet signals.<br>• Near-zero signal attenuation over kilometers and complete immunity to electromagnetic interference (EMI). |
| **5G Fixed Wireless Access (FWA)** | High-frequency radio spectrum (mid-band / mmWave cellular frequencies) | • **Home**: Outdoor antenna or indoor cellular gateway<br>• **Tower**: Cellular Base Station (gNodeB) | **Shared wireless spectrum** (competes with local mobile cellular traffic) | • 100 Mbps to 1+ Gbps downstream<br>• Tens of Mbps upstream<br>*(Asymmetric)* | • Connects homes to a nearby cell tower over radio waves without trenching physical copper or fiber cables.<br>• Rapid deployment; throughput depends on base-station distance, physical line-of-sight obstructions, and weather. |

#### Enterprise (Campus/Corporate) Networks

Universities and corporate campuses deploy high-speed enterprise access networks:
- Workstations, lab PCs, and office machines connect via **twisted-pair copper Ethernet cables** into departmental **Ethernet switches** operating at 100 Mbps, 1 Gbps, or 10 Gbps.
- Institutional servers (web portals, mail servers, research computing clusters) connect directly to high-capacity switches.
- Multiple departmental switches aggregate upward into an **institutional core router** (the enterprise edge router), which links the entire campus to commercial upstream ISPs via high-speed optical fiber links operating at 10 Gbps, 40 Gbps, or 100 Gbps.

#### Wireless Access Networks

Wireless access networks allow untethered mobile end systems to transmit and receive data through an intermediate base station:

| Network Type | Standard | Coverage Range | Typical Transmission Rates | Operational Context |
|---|---|---|---|---|
| **WLAN (Wireless Local Area Network)** | **WiFi** (IEEE 802.11 b/g/n/ac/ax/be) | Tens of meters (within an apartment, office suite, or coffee shop) | Up to hundreds of Mbps or several Gbps | End systems connect to a wireless **Access Point (AP)**, which is physically wired into the local Ethernet infrastructure. |
| **Wide-Area Cellular Access Network** | **4G LTE** (Long Term Evolution) & **5G NR** (New Radio) | Several kilometers to tens of kilometers | Tens of Mbps (4G) to over 1 Gbps (5G millimeter wave) | Mobile devices communicate with a cellular **base station** (cell tower) operated by a commercial telecommunications carrier. |

---

### 2.3 Physical Media

Physical media represent the physical pathways through which bits propagate between transmitters and receivers. They are formally divided into **guided media** and **unguided media**.

#### Guided Media
In guided media, electromagnetic waves are physically constrained and directed along a solid transmission medium (wires or optical glass).

1. **Twisted Pair (TP)**:
   - **Structure**: Consists of two insulated copper wires arranged in a regular spiral pattern. Twisting reduces electromagnetic radiation and shields against electromagnetic interference (crosstalk) from neighboring wire pairs.
   - **Variants**:
     - **UTP (Unshielded Twisted Pair)**: The most common wiring in enterprise LANs and home connections. Cheap and flexible.
     - **STP (Shielded Twisted Pair)**: Adds an external protective foil/braided mesh around the pairs to offer heavy noise immunity in industrial environments.
   - **Categories**:
     - **Category 5 (Cat 5)**: Supports speeds from 10 Mbps up to 100 Mbps Ethernet (100Base-TX).
     - **Category 5e (Cat 5e)**: Enhanced specifications; supports up to 1 Gbps Gigabit Ethernet (1000Base-T).
     - **Category 6 (Cat 6)**: Tighter twists, central plastic spline separator; supports up to 10 Gbps over distances up to 55 meters.
     - **Category 6a / 7**: Heavily shielded; reliably supports 10 Gbps over full 100-meter channel runs.
   - **Connector**: Standard **RJ-45 (Registered Jack 45)** modular connector.

2. **Coaxial Cable**:
   - **Structure**: Constructed from two **concentric** (co-axial) copper conductors sharing a single central axis:
     - An inner solid copper conducting core.
     - An insulating dielectric layer.
     - An outer woven cylindrical conducting metal braid (serves as electrical ground and shielding).
     - A protective outer plastic jacket.
   - **Operating Modes**:
     - **Baseband Coaxial Cable**: Carries a single digital signal at a time (e.g., legacy 10Base2 "Thinnet" and 10Base5 "Thicknet" Ethernet).
     - **Broadband Coaxial Cable**: Uses analog transmission and FDM to carry multiple independent frequency channels simultaneously across hundreds of Megahertz (e.g., Cable TV and DOCSIS cable internet).
   - **Characteristic**: High bandwidth, good noise immunity, serves as a physical **shared medium** where multiple devices tap into the same line. Connector type: **BNC (Bayonet Neill-Concelman)** or threaded **F-type** connector.

3. **Fiber Optic Cable**:
   - **Structure**: Consists of an ultra-pure, ultra-thin flexible glass (silica) core surrounded by a glass **cladding** layer with a lower refractive index, all encased within a protective polymer buffer coating and outer jacket.
   - **Physics**: Operates on the physical principle of **Total Internal Reflection**. Light pulses injected into the core strike the core-cladding boundary at an angle greater than the critical angle, trapping the light within the core and guiding it along the cable with negligible leakage.
   - **Types**:
     - **Single-Mode Fiber (SMF)**: Extremely thin core diameter (~8 to 10 micrometers). Permits only a single ray (mode) of light to propagate directly down the center. Driven by expensive semiconductor **laser diodes**. Exhibits almost zero modal dispersion, allowing signals to travel tens to hundreds of kilometers without repeaters. Used in long-haul telecommunication backbones and transoceanic undersea cables.
     - **Multi-Mode Fiber (MMF)**: Wider core diameter (~50 to 62.5 micrometers). Allows multiple rays (modes) of light to bounce along the core at varying angles. Driven by inexpensive **LEDs (Light Emitting Diodes)**. Suffers from modal dispersion (different light rays arrive at slightly different times), limiting transmission distances to under 500 meters. Used for high-speed interconnects within enterprise data centers.
   - **Key Advantages**: Massive transmission capacity (tens to hundreds of Gbps per strand; multi-Terabits using **WDM — Wavelength Division Multiplexing**), total immunity to electromagnetic interference and radio noise, zero electrical spark risk, and exceptionally low signal attenuation over vast distances. Standard transmission rates are classified by **OC-n (Optical Carrier level n)** standards, where the base rate OC-1 is 51.84 Mbps (e.g., OC-768 operates at ~40 Gbps).

#### Unguided Media
In unguided media, electromagnetic signals propagate freely through the air, vacuum, or open atmosphere without a physical guiding channel. All unguided transmissions are susceptible to environmental attenuation, atmospheric absorption, physical reflections, multipath fading, and interference.

1. **Terrestrial Radio Waves and Microwaves**:
   - Radio waves propagate omnidirectionally (in all directions from the transmitting antenna), easily penetrating walls but susceptible to multipath interference. Used in WiFi, cellular, and FM radio.
   - Terrestrial microwaves operate at higher frequencies (Gigahertz range) and travel in strictly focused, directional **line-of-sight** paths between parabolic dish antennas mounted atop towers or hills. They cannot penetrate physical buildings or mountains. Used for point-to-point telecom trunking carrying data rates up to 45 Mbps or several Gbps.
2. **Satellite Microwave**:
   - A communications satellite in space acts as a microwave relay station. An earth ground station transmits an uplink signal on one frequency band; the satellite's transponder receives it, amplifies it, translates the frequency, and broadcasts it back to Earth on a downlink frequency band.
   - **Geostationary Satellites (GEO)**: Orbit at an altitude of approximately **36,000 kilometers** directly above the Earth's equator. Because their orbital period matches Earth's rotation (24 hours), they appear permanently stationary relative to a fixed point on the ground.
     - **Propagation Delay Penalty**: The immense round-trip distance ($2 \times 36{,}000\text{ km} = 72{,}000\text{ km}$) introduces an inescapable physical propagation delay:
    $$d_{prop} = \frac{2 \times 36{,}000{,}000\text{ m}}{3 \times 10^8\text{ m/s}} = 0.24\text{ seconds} = 240\text{ ms}$$
       When coupled with processing and ground-terminal buffering overhead, end-to-end satellite delays typically reach **~280 ms** one-way (and over 500 ms for a full round-trip handshake), making GEO satellite links poorly suited for interactive gaming or voice conversations.
   - **Low Earth Orbit (LEO) Satellites**: Orbit at altitudes of 500 to 1,500 kilometers (e.g., Starlink). Because they are much closer to Earth, propagation delay drops to 20–40 ms, but continuous global coverage requires constellations of thousands of coordinated satellites.
3. **Infrared**:
   - Operates at frequencies just below visible light. Strictly directional and **cannot penetrate solid walls**. Used in short-range line-of-sight devices (TV remote controls, wireless peripherals), offering natural physical security because signals cannot leak outside the room.

---

### 2.4 Physical-Layer Devices

Physical-layer devices operate purely at **Layer 1** of the network reference model. They deal solely with raw, unformatted binary digits (bits) converted into electrical voltages, optical pulses, or radio frequencies. They have **zero awareness** of packet headers, MAC addresses, IP addresses, or payload content.

| Device | Primary Function | Internal Operation | Limitations & Collision Behavior |
|---|---|---|---|
| **Repeater** | Signal regeneration and amplification | Receives an incoming electrical or optical signal that has suffered attenuation over a long cable run, cleans out noise, regenerates the signal to its original strength and shape, and retransmits it onto the next cable segment. | Extends physical network distance. Does not look at or filter any data bits. Forwards bit errors and noise directly. |
| **Hub** | Multi-port physical repeater | Serves as a central connection point for multiple twisted-pair Ethernet cables in a star topology. Whenever a bit arrives on any single input port, the hub **blindly broadcasts that bit to every other connected port**. | Creates a **single collision domain**: if two connected devices transmit data simultaneously, the electrical signals collide, corrupting both transmissions. Operates strictly in **half-duplex** mode. Possesses no memory, no filtering logic, and cannot read MAC addresses. |
| **Modem (Modulator-Demodulator)** | Digital-to-analog and analog-to-digital signal conversion | **Modulation**: Converts digital square-wave pulses from a computer into continuous analog waveforms suitable for transmission over frequency-limited telephone wires or cable TV coax.<br>**Demodulation**: Converts received analog waveforms back into clean digital 0s and 1s. | Acts as the physical translation bridge between digital end systems and analog access networks. |

---

## 3. The Network Core

The **network core** is the vast, interconnected mesh of packet switches and communication links that transports data between access networks across the globe.

```
Access Network A ──► [ Edge Router ] ──► [ Core Router ] ──► [ Core Router ] ──► [ Edge Router ] ──► Access Network B
                                                │                    ▲
                                                └──► [ Core Router ] ┘
```

The fundamental technical challenge of the network core is: *how is data moved through this mesh of switching devices and communication links?* Two fundamentally distinct architectural paradigms exist: **packet switching** and **circuit switching**.

---

### 3.1 Packet Switching

In **packet-switched networks**, end systems exchange application data by partitioning long messages into smaller, discrete chunks of data called **packets**. Each packet consists of a header (carrying destination address and control flags) and an entity payload.

#### Store-and-Forward Transmission
Almost all packet switches (routers and link-layer switches) use **store-and-forward transmission** at their input links:
- A packet switch must receive the **entire packet completely** (store all $L$ bits of the packet in its buffer) before it can begin transmitting the first bit of that packet onto the outbound communication link.
- **Why is store-and-forward necessary?**
  1. The router must inspect the packet's header to verify checksum integrity (confirming no bit errors occurred in transit).
  2. The router must extract the destination IP address from the header to look up the appropriate outgoing interface in its forwarding table.
  3. The outgoing link might be operating at a different transmission rate than the incoming link (e.g., arriving on 100 Mbps, departing on 10 Mbps).

**Mathematical Formulation of Store-and-Forward Transmission:**
For a single packet of length $L$ bits transmitted over a link with transmission rate $R$ bits per second, the time required to push the entire packet onto the link is:
$$d_{trans} = \frac{L}{R}$$

If a packet must traverse $N$ sequential, identical links (each of transmission capacity $R$), separated by $N-1$ packet switches, the total store-and-forward delay from source to destination (ignoring propagation, queuing, and processing delays) is:
$$d_{end-to-end} = N \times \frac{L}{R}$$

#### Statistical Multiplexing and Bursty Traffic
Packet switching employs **statistical multiplexing**:
- Communication link capacity is shared dynamically among all users on an as-needed basis.
- End-user traffic is naturally **"bursty"** — a user reads a web page for 30 seconds (zero data generated), clicks a link (short burst of packets), and reads again.
- Under packet switching, link capacity is never dedicated to an idle user. Multiple users can statistically share a link whose total capacity is less than the sum of the users' peak demands, because it is statistically improbable that all bursty users will transmit at their peak rates simultaneously.
- **Trade-off**: When demand temporarily exceeds the transmission capacity of an outgoing link, arriving packets cannot be transmitted immediately. They are held in an **output buffer (queue)**, causing **queuing delay**. If the buffer becomes completely full, newly arriving packets are discarded, resulting in **packet loss**.

---

### 3.2 Circuit Switching

In **circuit-switched networks**, the communication resources (buffers, link transmission bandwidth) required along a path between communicating end systems are **strictly reserved and dedicated** for the entire duration of the communication session. This is the traditional paradigm of the public switched telephone network (PSTN).

- **Call Setup Phase**: Before any data or voice can be transmitted, an end-to-end signaling exchange occurs to locate a continuous path and reserve dedicated bandwidth along every link on that path.
- **Guaranteed Performance**: Because resources are reserved exclusively, the connection enjoys constant, guaranteed bandwidth with zero queuing delay and zero packet loss caused by other network users.
- **Wasted Capacity**: If a caller pauses or goes silent during a phone call, the reserved bandwidth remains completely idle; it cannot be utilized by any other user.

#### Frequency Division Multiplexing (FDM) vs. Time Division Multiplexing (TDM)

To allow a single high-capacity physical link to support multiple simultaneous circuits, circuit switching employs multiplexing:

```
FDM: Frequency Spectrum Divided Continuously
Freq ^
     │ [ Circuit 1 ] (Frequency Band 1, all time)
     │ [ Circuit 2 ] (Frequency Band 2, all time)
     │ [ Circuit 3 ] (Frequency Band 3, all time)
     └───────────────────────────────────────────► Time

TDM: Time Divided into Recurring Frames and Slots
Freq ^
     │ [ Slot 1 ] [ Slot 2 ] [ Slot 3 ] [ Slot 1 ] [ Slot 2 ] [ Slot 3 ] ...
     │ (Entire frequency band used, but only during assigned time slot)
     └───────────────────────────────────────────────────────────────────► Time
       ◄────── One Frame ──────►
```

| Dimension | Frequency Division Multiplexing (FDM) | Time Division Multiplexing (TDM) |
|---|---|---|
| **Underlying Mechanism** | The link's total electromagnetic frequency spectrum is partitioned into narrow, dedicated frequency bands. | Time is partitioned into periodic, recurring time slices called **frames**, and each frame is partitioned into a fixed number of **time slots**. |
| **Allocation** | Each connection receives an exclusive frequency band continuously for the entire duration of the call. | Each connection is assigned one dedicated time slot within every recurring frame, utilizing the entire frequency band during its brief slot. |
| **Real-World Example** | Traditional FM/AM radio broadcasting, analog telephone carrier systems, cable TV channels. | Digital telephony (T1/E1 carrier lines), SONET/SDH optical links. |

---

### 3.3 Packet Switching vs. Circuit Switching — Comprehensive Comparison

| Evaluation Metric | Packet Switching | Circuit Switching |
|---|---|---|
| **Resource Allocation** | On-demand (statistical multiplexing); resources are consumed only when packets are physically transmitted. | Pre-allocated and strictly dedicated along the entire path for the call duration. |
| **Setup Phase** | **No connection setup** required at the network layer; packets are transmitted immediately. | **Mandatory call setup** signaling delay required before user data can begin flowing. |
| **Efficiency with Bursty Data** | **Extremely high**; idle users consume zero bandwidth, allowing more users to share the network. | **Low**; idle periods (silence) waste reserved channel capacity. |
| **Delay Characteristics** | Variable delay; packets experience unpredictable queuing delay dependent on current network congestion. | Predictable, constant transmission delay; zero queuing delay once the circuit is established. |
| **Congestion Behavior** | Buffers fill during load spikes, leading to queuing delays and packet loss when buffers overflow. | Call blocking: If all circuits on a link are busy, new connection requests are rejected ("busy signal"). |
| **State Information** | Core routers maintain no per-connection state; each packet is handled independently based on its header. | Core switches must maintain continuous, stateful tracking of active circuits and time/frequency slot allocations. |
| **Primary Modern Application** | The global Internet, computer LANs, modern 4G/5G cellular IP data networks. | Traditional voice telephony, dedicated optical private lines. |

---

### 3.4 Core Functions: Routing vs. Forwarding

Every router in the network core performs two fundamentally distinct, cooperating functions:

```
[ Control Plane: Routing Algorithms ]
                │
                ▼ (Computes and writes)
        [ Forwarding Table ]
                ▲
                │ (Reads header, directs packet to output port)
[ Data Plane: Arriving Packet ] ──► [ Switch Fabric ] ──► [ Selected Output Link ]
```

| Attribute | Routing (The Control Plane) | Forwarding (The Data Plane) |
|---|---|---|
| **Definition** | The global, network-wide process that plans and computes the end-to-end paths that packets take from source to destination. | The router-local, hardware-level action of transferring an arriving packet from an input link interface to the appropriate output link interface. |
| **Scope** | **Network-wide**; involves coordination among all routers across the network using routing protocols (e.g., OSPF, BGP). | **Local to a single router**; executes independently within each router's physical chassis in nanoseconds. |
| **Mechanism** | Routing algorithms compute and populate the router's internal **forwarding table**. | The router reads the destination IP address in the packet's header, searches its local forwarding table, and switches the packet to the corresponding output port. |
| **Analogy** | Using a roadmap or GPS navigation system to plan your entire driving route from Bangalore to Mumbai before starting. | Driving through an intersection and reading a single physical highway signpost that tells you to turn right toward Highway 48. |

---

### 3.5 A Network of Networks — Internet Structure and Hierarchy

Connecting millions of independent access networks across the globe directly to one another via point-to-point physical links is mathematically impossible — connecting $N$ access networks directly would require $\frac{N(N-1)}{2} = \mathcal{O}(N^2)$ separate communication links.

Instead, the global Internet organizes its network core into a **hierarchical network of networks**:

```
                          ┌─────────────────────────────┐
                          │   Tier-1 ISP A ◄────────►   │
                          │        ▲       (Peering)    │
                          │        │                    │
                          │   Tier-1 ISP B              │
                          └────────┬────────────────────┘
                                   ▲
                                   │ (Provider-Customer Transit)
                          ┌────────┴─────────┐
                          │   Regional ISP   │ ◄──── IXP (Internet Exchange Point)
                          └────────┬─────────┘        ▲
                                   ▲                  │
                                   │                  │
                    ┌──────────────┴──────────────┐   │
                    │                             │   │
           [ Access ISP 1 ]              [ Access ISP 2 ]
                  ▲                             ▲
                  │                             │
             [ End Hosts ]                 [ End Hosts ]
```

1. **Tier-1 Commercial ISPs**:
   - At the apex of the hierarchy sits a small group of roughly a dozen massive commercial backbone providers (e.g., Lumen/Level 3, AT&T, NTT, Sprint, Telia).
   - Tier-1 ISPs possess national and international fiber networks.
   - **Settlement-Free Peering**: Tier-1 ISPs connect directly to every other Tier-1 ISP. Because they exchange comparable volumes of traffic, they peer with one another on a **settlement-free basis** — neither pays the other for carrying traffic between their customer networks.
2. **Regional and Intermediate ISPs**:
   - Cover smaller geographic areas (a state or mid-sized country).
   - A regional ISP is a **customer** to one or more upstream Tier-1 ISPs, paying the Tier-1 provider for Internet transit. In turn, regional ISPs act as providers to local access ISPs.
3. **Access ISPs**:
   - The edge networks that provide direct connectivity to residential homes, universities, and businesses.
   - Access ISPs pay upstream regional or Tier-1 ISPs to transport their traffic to the rest of the world.
4. **Physical Interconnection Facilities**:
   - **PoP (Point of Presence)**: A physical group of routers within an ISP's network where customer networks (lower-tier ISPs or enterprise networks) can physically connect into the provider's infrastructure.
   - **Multi-Homing**: An access or regional ISP connects to two or more upstream providers simultaneously. If one upstream provider experiences a link failure, traffic automatically fails over to the other provider, preventing complete network outages.
   - **Peering**: Two ISPs at the same commercial tier establish a direct physical link to exchange traffic directly between their respective customers, avoiding the cost of sending that traffic through an expensive upstream Tier-1 provider.
   - **IXP (Internet Exchange Point)**: A specialized, third-party physical data center facility containing high-speed switching fabrics where dozens or hundreds of different ISPs, content delivery networks, and enterprise networks meet to establish mutual peering connections.
5. **Major Content Provider Networks (CPNs)**:
   - Tech giants such as Google, Microsoft, Meta, and Netflix build their own private global fiber optic networks.
   - These private networks connect their distributed data centers directly to one another, peering with regional and access ISPs directly at IXPs worldwide.
   - **Strategic Goal**: By maintaining their own private backbone, content providers **bypass the public Tier-1 Internet core entirely** for the majority of their traffic, drastically cutting transit costs and gaining fine-grained control over user latency.

### 3.5 Circuit-Switched Network Topologies & Routing Capacity

*(Directly derived from Lecture Slides 130–132: Circuit-Switched Ring Topology Problem)*

```
                       [ Switch A ]
                      /            \
        14 circuits  /              \  16 circuits
                    /                \
        [ Switch D ]                  [ Switch B ]
                    \                /
        17 circuits  \              /  20 circuits
                      \            /
                       [ Switch C ]
```

Consider a 4-node ring network consisting of circuit switches $A$, $B$, $C$, and $D$ interconnected by dedicated links with differing circuit capacities:
- Link $(A, B)$: $14$ circuits
- Link $(B, C)$: $16$ circuits
- Link $(C, D)$: $20$ circuits
- Link $(D, A)$: $17$ circuits

#### Analytical Questions & Official Solutions:

1. **What is the maximum number of simultaneous 1-hop connections that can be ongoing in the network at any one time?**
   - **Solution**: The absolute maximum number of simultaneous 1-hop connections occurs when all circuits on every physical link are fully utilized:
   $$\text{Max 1-Hop Connections} = 14 + 16 + 20 + 17 = \mathbf{67\text{ connections}}$$
2. **Suppose these maximum 67 connections are currently active. What happens when another call connection request arrives at the network? Will it be accepted?**
   - **Solution**: **No, it will be blocked (rejected).** In circuit switching, resources must be reserved in advance. When all circuits are saturated, no new call can be established until an ongoing call terminates and frees its reserved circuit.
3. **Suppose every connection requires 2 consecutive hops, and all calls are routed strictly clockwise ($A \to B \to C$, $B \to C \to D$, $C \to D \to A$, $D \to A \to B$). What is the maximum number of simultaneous 2-hop connections that can be supported?**
   - **Solution**:
     - Let $x_1$ be calls on $A \to B \to C$ (uses link $AB$ and $BC$).
     - Let $x_2$ be calls on $B \to C \to D$ (uses link $BC$ and $CD$).
     - Let $x_3$ be calls on $C \to D \to A$ (uses link $CD$ and $DA$).
     - Let $x_4$ be calls on $D \to A \to B$ (uses link $DA$ and $AB$).
     - The link capacity constraints are:
       - Link $AB$: $x_1 + x_4 \le 14$
       - Link $BC$: $x_1 + x_2 \le 16$
       - Link $CD$: $x_2 + x_3 \le 20$
       - Link $DA$: $x_3 + x_4 \le 17$
     - Summing all four inequalities:
    $$(x_1 + x_4) + (x_1 + x_2) + (x_2 + x_3) + (x_3 + x_4) \le 14 + 16 + 20 + 17 = 67$$
    $$2(x_1 + x_2 + x_3 + x_4) \le 67 \implies \text{Total Connections} \le \lfloor 67 / 2 \rfloor = \mathbf{33\text{ connections}}$$
     - A valid allocation achieving 33 is: $x_1 = 14, x_2 = 2, x_3 = 17, x_4 = 0$, giving $14 + 2 + 17 + 0 = 33$ calls.
4. **Suppose 12 connections are requested from $A \to C$ and 15 connections are requested from $B \to D$. Can the network accommodate all 27 connections simultaneously?**
   - **Solution**: **Yes!**
     - $A \to C$ requires links $AB$ and $BC$ ($x_1 = 12$).
     - $B \to D$ requires links $BC$ and $CD$ ($x_2 = 15$).
     - Check link $BC$: $x_1 + x_2 = 12 + 15 = 27$ circuits would be needed on link $BC$ if routed clockwise! But link $BC$ only has $16$ circuits!
     - However, we can route $A \to C$ clockwise ($A \to B \to C$, using 12 on $AB$ and 12 on $BC$) and route the excess $B \to D$ calls **counter-clockwise** ($B \to A \to D \to C$, or using leftover paths). Total required calls is $27$, which is well below the network-wide 2-hop maximum capacity of $33$.

---

### 3.6 Quantitative Comparison of Packet Switching and Circuit Switching

*(Directly derived from Lecture Slides 121–126)*

#### Scenario 1: 150 Mbps Shared Link with 10 Mbps Users (30% Active)
- Link transmission capacity: $R = 150\text{ Mbps}$.
- Bandwidth requirement per active user: $B = 10\text{ Mbps}$.
- User activity factor: $p = 0.30$ (transmits 30% of the time).

1. **Circuit Switching Maximum Users ($N_{cs}$)**:
   $$N_{cs} = \frac{R}{B} = \frac{150\text{ Mbps}}{10\text{ Mbps}} = \mathbf{15\text{ users}}$$
2. **Packet Switching with 29 Users ($N_{ps} = 29$)**:
   - Under circuit switching, 29 users **cannot** be supported ($29 > 15$).
   - Under packet switching, when 1 user is actively transmitting, they use $\frac{10}{150} = \mathbf{0.067}$ ($6.7\%$) of the link capacity.
   - The probability that more than 15 users transmit simultaneously out of 29 is:
   $$P(\text{Overload}) = \sum_{k=16}^{29} \binom{29}{k} (0.3)^k (0.7)^{29-k} < \mathbf{0.0035 \quad (0.35\%)}$$
   - Packet switching supports nearly **double the users ($29$ vs $15$)** with less than $1\%$ probability of queuing congestion!

#### Scenario 2: 200 Mbps Shared Link with 25 Mbps Users (20% Active)
- Link transmission capacity: $R = 200\text{ Mbps}$.
- Bandwidth requirement per active user: $B = 25\text{ Mbps}$.
- User activity factor: $p = 0.20$ (transmits 20% of the time).

1. **Circuit Switching Maximum Users ($N_{cs}$)**:
   $$N_{cs} = \frac{R}{B} = \frac{200\text{ Mbps}}{25\text{ Mbps}} = \mathbf{8\text{ users}}$$
2. **Packet Switching with 15 Users ($N_{ps} = 15$)**:
   - Under circuit switching, 15 users **cannot** be supported ($15 > 8$).
   - When 1 user transmits, they utilize $\frac{25}{200} = \mathbf{0.125}$ ($12.5\%$) of total capacity.
   - The probability that more than 8 users transmit simultaneously is practically zero ($< 0.001$), proving packet switching's superior efficiency for bursty traffic.

---

## 4. Delay, Loss, and Throughput in Packet-Switched Networks

### 4.1 The Four Sources of Nodal Delay

As a packet travels along its path from source host to destination host, it encounters a delay at **every single node (router)** along the route. The total nodal delay ($d_{nodal}$) is the exact sum of four distinct delay components:

$$d_{nodal} = d_{proc} + d_{queue} + d_{trans} + d_{prop}$$

```
                Incoming Link ──► [ Input Buffer ]
                                         │
    1. Processing Delay (d_proc) ◄───────┤ (Examine header, bit error check)
                                         ▼
    2. Queuing Delay (d_queue)   ◄── [ Output Buffer Queue ] (Waiting for link to be free)
                                         │
    3. Transmission Delay (d_trans) ◄────┤ (Pushing L bits onto the wire at rate R)
                                         ▼
    4. Propagation Delay (d_prop)  ◄── [ Physical Link ] (Bits traveling distance d at speed s)
                                         │
                                         ▼ Next Router
```

1. **Nodal Processing Delay ($d_{proc}$)**:
   - The time required for the router to inspect the packet's header fields, verify checksums to detect bit errors, and consult its internal forwarding table to determine the appropriate output link interface.
   - Modern routers execute processing in dedicated, specialized hardware (Application-Specific Integrated Circuits — ASICs), making $d_{proc}$ typically negligible — often **under a few microseconds**.

2. **Queuing Delay ($d_{queue}$)**:
   - The time a packet sits waiting in the router's output buffer queue until the physical link becomes available for transmission.
   - **Dynamic Nature**: Unlike the other three delay components (which are fixed for a given packet length, link speed, and distance), queuing delay is **highly variable**. If the queue is empty, $d_{queue} = 0$. If the link is heavily congested with a long queue of preceding packets, $d_{queue}$ can stretch into tens or hundreds of milliseconds.

3. **Transmission Delay ($d_{trans}$)**:
   - The time required to physically push (serialize) **all $L$ bits of the packet** onto the physical transmission link.
   - **Formula**:
   $$d_{trans} = \frac{L}{R}$$
     - $L$: Packet length in bits.
     - $R$: Link transmission rate (capacity/bandwidth) in bits per second (bps).
   - **Dependency**: Depends strictly on **packet length** and **link bandwidth**. It has **zero relationship to physical distance**.

4. **Propagation Delay ($d_{prop}$)**:
   - The time required for a single physical bit, once pushed onto the medium, to propagate from the beginning of the physical link to the receiving router at the other end.
   - **Formula**:
   $$d_{prop} = \frac{d}{s}$$
     - $d$: Physical length (distance) of the link in meters.
     - $s$: Propagation speed of the electromagnetic wave in the physical medium ($\approx 2 \times 10^8\text{ m/s}$ for copper wire and optical fiber; $\approx 3 \times 10^8\text{ m/s}$ for open-air wireless and vacuum).
   - **Dependency**: Depends strictly on **physical distance** and **medium speed**. It has **zero relationship to packet length or link bandwidth**.

> [!TIP]
> **The Highway Tollbooth Analogy (Transmission vs. Propagation)**:
> Think of a caravan of 10 cars (the packet of bits) traveling on a highway between two tollbooths spaced 100 km apart:
> - **Transmission Delay**: The time it takes for the tollbooth to service all 10 cars and push them onto the highway. If the toll collector takes 12 seconds per car, the transmission delay for the entire caravan is $10 \times 12 = 120\text{ seconds}$.
> - **Propagation Delay**: The time it takes for a car, once through the tollbooth, to physically drive along the 100 km highway at 100 km/h to reach the second tollbooth ($100\text{ km} / 100\text{ km/h} = 1\text{ hour}$).
> - Doubling the car speed reduces propagation delay, but does not change the toll collector's service time ($d_{trans}$). Adding more toll collectors reduces transmission delay, but does not make the cars drive any faster on the highway ($d_{prop}$).

---

### 4.2 End-to-End Delay Across Multiple Links

Consider a packet of length $L$ bits traversing a path made up of $Q$ distinct links, where each link $i$ ($i = 1, 2, \dots, Q$) has its own transmission rate $R_i$, distance $d_i$, and propagation speed $s_i$. 

Assuming processing delay at each intermediate router is $d_{proc, i}$ and queuing delay is $d_{queue, i}$, the total end-to-end delay is:

$$d_{end-to-end} = \sum_{i=1}^{Q} \left( d_{proc, i} + d_{queue, i} + \frac{L}{R_i} + \frac{d_i}{s_i} \right)$$

If the path consists of $N$ identical links with uniform rate $R$, distance $d$, propagation speed $s$, identical processing delays $d_{proc}$, and negligible queuing delays ($d_{queue} \approx 0$):

$$d_{end-to-end} = N \times \left( d_{proc} + \frac{L}{R} + \frac{d}{s} \right)$$

---

### 4.3 Queuing Delay, Traffic Intensity, and Buffer Loss

Queuing delay represents the most volatile component of network delay. It is fundamentally governed by a dimensionless ratio called **Traffic Intensity ($I$)**:

$$I = \frac{L \cdot a}{R}$$

- $L$: Length of each packet in bits.
- $a$: Average packet arrival rate in packets per second (packets/sec).
- $R$: Link transmission rate in bits per second (bps).
- The numerator $L \cdot a$ represents the average rate at which bits arrive at the queue (in bps).

```
Queuing Delay
     ^
     │                                      | (Asymptote at I = 1)
     │                                      |
     │                                     /
     │                                    /
     │                                  /'
     │                              _.-'
     │                     _...--''
     │         _...---''''
     └───────────────────────────────────────► Traffic Intensity (I = La/R)
     0                                      1.0
```

#### Regimes of Traffic Intensity:
1. **$I \approx 0$ (Very Low Traffic Intensity)**:
   Packets arrive infrequently. Arriving packets rarely find another packet being transmitted; average queuing delay is close to **zero**.
2. **$I \to 1$ (Traffic Intensity Approaches 1)**:
   As $I$ enters the range $0.7 \text{ to } 0.9$, arrival bursts cause queues to form rapidly. Average queuing delay escalates **non-linearly** and steeply.
3. **$I > 1$ (Traffic Intensity Exceeds 1)**:
   Bits arrive at the queue faster than the link can physically serialize and transmit them. In a theoretical system with an infinite buffer, the queue would grow indefinitely and average queuing delay would approach **infinity**.

#### Queuing Delay Modeling Formula:
Under idealized M/M/1 queuing assumptions (Poisson packet arrivals, exponential packet length distribution, single server), when $I < 1$, the theoretical average queuing delay is modeled as:

$$d_{queue} = \frac{I \cdot L}{R(1 - I)} = \frac{I}{1 - I} \times \frac{L}{R}$$

Notice that as $I \to 1$, the denominator $(1 - I) \to 0$, causing delay to explode toward infinity.

#### Packet Loss (Buffer Overflow):
In the physical world, routers do not possess infinite memory buffers. An output link buffer has a finite capacity capable of holding $B$ bytes. 
- When an arriving packet finds the buffer completely filled with previously queued packets, the router has nowhere to store it.
- The router executes a drop policy (typically **drop-tail**, dropping the newly arriving packet).
- The dropped packet is **irrecoverably lost**. The router does not generate any native recovery mechanism. Reliability must instead be provided by higher-layer transport protocols (e.g., TCP timeout and retransmission) or handled by the application.

---

### 4.4 Real-World Network Diagnostic Tools: Ping and Traceroute

The concepts of delay, round-trip time, and packet loss are directly observable using command-line diagnostic utilities:

#### 1. Ping
- **Mechanism**: The sending host sends an **ICMP (Internet Control Message Protocol)** **Echo Request** packet (Type 8) to the target host. Upon receipt, the target's operating system generates an **ICMP Echo Reply** packet (Type 0) back to the sender.
- **Metrics Provided**:
  - Measures the elapsed **Round-Trip Time (RTT)** in milliseconds.
  - Reports **packet loss percentage** if reply packets fail to return within a timeout threshold.

#### 2. Traceroute
- **Problem Solved**: How can a user discover the identity and delay of every intermediate router along an end-to-end path through an opaque network core?
- **Operating Mechanism**: Traceroute exploits the **TTL (Time to Live)** field in the IPv4 header (or Hop Limit in IPv6):
  1. The source sends a probe packet (UDP datagram or ICMP packet) addressed to the final destination, but sets **$\text{TTL} = 1$**.
  2. The first router along the path receives the packet, decrements TTL by 1 ($\text{TTL} = 0$), discards the packet, and sends back an **ICMP Time Exceeded message** (Type 11). The source extracts the router's IP address from this message and calculates the RTT to hop 1.
  3. The source sends another packet with **$\text{TTL} = 2$**. The first router decrements TTL to 1 and forwards it; the second router decrements TTL to 0, discards it, and returns an ICMP Time Exceeded message.
  4. This cycle repeats with progressively incremented TTL values ($\text{TTL} = 3, 4, \dots$) until the packet reaches the ultimate destination host. The destination host recognizes the probe and returns an ICMP Port Unreachable or Echo Reply message, terminating the trace.

---

### 4.5 Throughput and the Bottleneck Link

**Throughput** is formally defined as the rate (measured in bits per second) at which data bits are successfully delivered from a sender process to a receiver process.
- **Instantaneous Throughput**: The rate at which the receiver receives data at a specific instant in time.
- **Average Throughput**: The total volume of data delivered divided by the total time taken to transfer that data.

#### The Bottleneck Link Principle:
Consider a communication path traversing $N$ sequential links with transmission rates $R_1, R_2, \dots, R_N$. Assuming no other interfering traffic, the achievable end-to-end throughput is strictly constrained by the link with the smallest transmission capacity:

$$\text{Throughput} = \min\{R_1, R_2, \dots, R_N\}$$

The link that enforces this minimum rate is called the **bottleneck link**.

```
[ Server ] ──(Rs = 2 Mbps)──► [ Router ] ──(Rc = 1 Mbps)──► [ Client ]
                                   ▲
                                   │
                    Bottleneck Link (Rc < Rs)
                    Maximum Achievable Throughput = 1 Mbps
```

- If a server's access link rate is $R_s = 2\text{ Mbps}$ and a client's access link rate is $R_c = 1\text{ Mbps}$, while the network core links operate at $100\text{ Gbps}$, the end-to-end throughput is strictly $1\text{ Mbps}$. The client's access link is the bottleneck.
- If ten simultaneous server-client connections share a single core backbone link of capacity $R = 300\text{ Mbps}$, and the core link divides bandwidth equally, each connection receives $R / 10 = 30\text{ Mbps}$. If each server has $R_s = 50\text{ Mbps}$ and each client has $R_c = 90\text{ Mbps}$, the achievable throughput is $\min(50, 90, 30) = 30\text{ Mbps}$. In this case, the shared backbone link is the bottleneck.

### 4.6 Official Lecture Interactive Exercises & Problem Walkthroughs

*(Directly derived from Lecture Slides 115–146)*

#### 1. The Car-Caravan Tollbooth Analogy (Slides 133–135)
Consider a caravan of $10\text{ cars}$ traveling along a highway between two tollbooths spaced $d = 500\text{ km}$ apart:
- Tollbooth service rate: $1\text{ car every } 2\text{ seconds}$ (service time $t_{service} = 2\text{ s}$).
- Caravan speed: $s = 10\text{ km/s}$ (or $100\text{ km/h}$ depending on problem scale; slide specifies $10\text{ km/s}$).
- Operation rule: The entire caravan must line up and be fully stored at the tollbooth entrance before the first car pays its toll and proceeds.

```
+------------+       Highway (500 km at 10 km/s)       +------------+
| Tollbooth  | ======================================> | Tollbooth  |
|     1      |    Car 1, Car 2, ..., Car 10            |     2      |
+------------+                                         +------------+
```

| Question Number | Slide Question | Authoritative Answer & Explanation |
| :---: | :--- | :--- |
| **Q1** | What is the service time for a single car at the tollbooth? | **$2\text{ seconds}$**. |
| **Q2** | How long does it take for the tollbooth to service all 10 cars in the caravan? | **$20\text{ seconds}$** ($10\text{ cars} \times 2\text{ seconds/car} = 20\text{ s}$). Analogy: Packet transmission delay $t_{trans} = L/R$. |
| **Q3** | How long does it take for a car to travel from the first tollbooth to the second tollbooth? | **$50\text{ seconds}$** ($\frac{500\text{ km}}{10\text{ km/s}} = 50\text{ s}$). Analogy: Link propagation delay $t_{prop} = d/s$. |
| **Q4** | How long does the last car take to travel between tollbooths? | **$50\text{ seconds}$**. Propagation delay is independent of vehicle order; all cars travel at the exact same physical speed. |
| **Q5** | How long until the first car of the caravan begins receiving service at the second tollbooth? | **$68\text{ seconds}$**. Under the store-and-forward rule, Tollbooth 2 cannot service the first car until the **entire caravan** has arrived. The 10th car finishes service at Tollbooth 1 at $t = 20\text{ s}$, then travels for $50\text{ s}$ ($t = 70\text{ s}$). (Or if measuring from first car release: $9 \text{ cars} \times 2\text{ s} + 50\text{ s} = 68\text{ s}$). |
| **Q6** | Can cars receive service at Tollbooth 2 before all 10 cars arrive? | **No**. The store-and-forward discipline requires the entire packet (all bits) to arrive before processing begins. |
| **Q7** | Are there ever zero cars in service at the same time? | **Yes**. Once the 10th car departs Tollbooth 1 and is in transit on the highway, Tollbooth 1 is idle and Tollbooth 2 is waiting, so zero cars are being serviced. |

---

#### 2. One-Hop Transmission Delay & Maximum Packet Rate (Slides 136–137)
A router transmits packets of length $L = 8000\text{ bits}$ onto an outbound link with transmission rate $R = 1\text{ Mbps} = 1{,}000{,}000\text{ bps}$.

1. **Transmission Delay ($t_{trans}$)**:
   $$t_{trans} = \frac{L}{R} = \frac{8000\text{ bits}}{1{,}000{,}000\text{ bps}} = \mathbf{0.008\text{ seconds}} = 8\text{ ms}$$
2. **Maximum Packet Forwarding Rate**:
   $$\text{Max Packets/Second} = \frac{R}{L} = \frac{1{,}000{,}000\text{ bps}}{8000\text{ bits}} = \mathbf{125\text{ packets/second}}$$

---

#### 3. Three-Link End-to-End Delay with 10-Question Quiz (Slides 141–143)

```
[ Host A ] ─── Link 1 ───► [ Router 1 ] ─── Link 2 ───► [ Router 2 ] ─── Link 3 ───► [ Host B ]
```

Given:
- Packet length: $L = 12{,}000\text{ bits}$
- Propagation speed: $s = 3 \times 10^8\text{ m/s}$ across all three links
- **Link 1**: Rate $R_1 = 100\text{ Mbps} = 10^8\text{ bps}$, Distance $d_1 = 3\text{ km} = 3000\text{ m}$
- **Link 2**: Rate $R_2 = 1\text{ Mbps} = 10^6\text{ bps}$, Distance $d_2 = 5000\text{ km} = 5{,}000{,}000\text{ m}$
- **Link 3**: Rate $R_3 = 10\text{ Mbps} = 10^7\text{ bps}$, Distance $d_3 = 1\text{ km} = 1000\text{ m}$

| Step | Parameter Computed | Exact Formula | Numerical Calculation | Result |
| :---: | :--- | :--- | :--- | :--- |
| **Q1** | Link 1 Transmission Delay | $L / R_1$ | $12{,}000 / 10^8$ | $0.00012\text{ s}$ ($0.12\text{ ms}$) |
| **Q2** | Link 1 Propagation Delay | $d_1 / s$ | $3{,}000 / (3 \times 10^8)$ | $0.00001\text{ s}$ ($0.01\text{ ms}$) |
| **Q3** | Link 1 Total Delay | $d_{trans,1} + d_{prop,1}$ | $0.00012 + 0.00001$ | $\mathbf{0.00013\text{ s}}$ ($0.13\text{ ms}$) |
| **Q4** | Link 2 Transmission Delay | $L / R_2$ | $12{,}000 / 10^6$ | $0.012\text{ s}$ ($12\text{ ms}$) |
| **Q5** | Link 2 Propagation Delay | $d_2 / s$ | $5{,}000{,}000 / (3 \times 10^8)$ | $0.01667\text{ s}$ ($16.67\text{ ms} \approx 0.017\text{ s}$) |
| **Q6** | Link 2 Total Delay | $d_{trans,2} + d_{prop,2}$ | $0.012 + 0.01667$ | $\mathbf{0.02867\text{ s}} \approx \mathbf{0.029\text{ s}}$ ($29\text{ ms}$) |
| **Q7** | Link 3 Transmission Delay | $L / R_3$ | $12{,}000 / 10^7$ | $0.0012\text{ s}$ ($1.2\text{ ms}$) |
| **Q8** | Link 3 Propagation Delay | $d_3 / s$ | $1{,}000 / (3 \times 10^8)$ | $3.33 \times 10^{-6}\text{ s}$ ($0.0033\text{ ms}$) |
| **Q9** | Link 3 Total Delay | $d_{trans,3} + d_{prop,3}$ | $0.0012 + 0.00000333$ | $\mathbf{0.001203\text{ s}}$ ($1.2\text{ ms}$) |
| **Q10**| **Total End-to-End Delay** | $D_1 + D_2 + D_3$ | $0.00013 + 0.02867 + 0.0012$ | $\mathbf{0.030\text{ seconds}} = \mathbf{30\text{ ms}}$ |

---

#### 4. Queuing Delay & Buffer Drop Calculations (Slides 138–140)
Given link rate $R = 1{,}900{,}000\text{ bps}$, packet length $L = 4700\text{ bits}$, average arrival rate $a\text{ packets/sec}$.
The queuing delay is calculated as:
$$d_{queue} = \frac{I \cdot L}{R(1 - I)} \quad \text{where } I = \frac{L \cdot a}{R}$$

1. **Does queuing delay vary a lot in practice?**
   - **Yes**. Real-world traffic arrives in bursts, causing queuing delays to fluctuate wildly from zero to buffer drop limits.
2. **Case $a = 35\text{ packets/sec}$**:
   $$I = \frac{4700 \times 35}{1{,}900{,}000} \approx \mathbf{0.0865}$$
   $$d_{queue} = \frac{0.0865 \times 4700}{1{,}900{,}000 \times (1 - 0.0865)} \times 1000\text{ ms} \approx \mathbf{0.23\text{ ms}}$$
3. **Case $a = 77\text{ packets/sec}$**:
   $$I = \frac{4700 \times 77}{1{,}900{,}000} \approx \mathbf{0.190}$$
   $$d_{queue} = \frac{0.190 \times 4700}{1{,}900{,}000 \times (1 - 0.190)} \times 1000\text{ ms} \approx \mathbf{0.58\text{ ms}}$$
4. **Buffer Occupancy Calculation**:
   - Suppose the router buffer is infinite, queuing delay is $0.3815\text{ ms}$, and $613\text{ packets}$ arrive in one second ($a = 613$).
   - The link can transmit $\lfloor 1000\text{ ms} / 0.3815\text{ ms} \rfloor = 2621\text{ packets/sec}$.
   - Packets remaining in buffer 1 second later:
   $$\text{Packets in Buffer} = \max\left(0,\; a - \left\lfloor \frac{1000}{\text{delay}} \right\rfloor \right) = \max(0, 613 - 2621) = \mathbf{0\text{ packets}}$$
5. **Packet Drops with Finite Buffer**:
   - If buffer capacity is capped at $627\text{ packets}$:
   $$\text{Packets Dropped} = \max(0,\; \text{Arrivals} - \text{Capacity}) = \max(0, 613 - 627) = \mathbf{0\text{ packets dropped}}$$
   - *(Note: In Slide 120, when arrivals $a = 1762$ and buffer capacity $= 956$, packets dropped $= 1762 - 956 = \mathbf{806\text{ packets dropped}}$)*.

---

#### 5. End-to-End Throughput & Link Utilizations with Shared Core (Slides 144–146)

```
Server 1 (50M) ──┐                                        ┌──► Client 1 (60M)
Server 2 (50M) ──┼──► [ Router 1 ] ── 300 Mbps ──► [ R2 ] ┼──► Client 2 (60M)
Server 3 (50M) ──┤                  (Shared Hop)          ├──► Client 3 (60M)
Server 4 (50M) ──┘                                        └──► Client 4 (60M)
```

Four independent server-to-client connections traverse a shared middle backbone link:
- Shared link capacity: $R = 300\text{ Mbps}$.
- Server access links: $R_S = 50\text{ Mbps}$ each.
- Client access links: $R_C = 60\text{ Mbps}$ each (or $90\text{ Mbps}$ in Slide 129).

1. **Maximum Achievable End-to-End Throughput per Connection**:
   - The shared middle link divides capacity equally among the 4 pairs:
   $$R_{shared, share} = \frac{R}{4} = \frac{300\text{ Mbps}}{4} = 75\text{ Mbps}$$
   - End-to-end throughput per pair:
   $$\text{Throughput} = \min\left(R_S,\; R_C,\; \frac{R}{4}\right) = \min(50\text{ Mbps},\; 60\text{ Mbps},\; 75\text{ Mbps}) = \mathbf{50\text{ Mbps}}$$
2. **Bottleneck Link Identification**:
   - The link with the absolute minimum capacity along the path is **$R_S$ (the server link)**.
3. **Link Utilizations**:
   - **Server Link Utilization ($U_S$)**:
   $$U_S = \frac{\text{Throughput}}{R_S} = \frac{50}{50} = \mathbf{1.0 \quad (100\%)}$$
   - **Client Link Utilization ($U_C$)**:
   $$U_C = \frac{\text{Throughput}}{R_C} = \frac{50}{60} = \mathbf{0.83 \quad (83.3\%)}$$
     *(If $R_C = 90\text{ Mbps}$: $U_C = 50 / 90 = \mathbf{0.56}$ or $56\%$)*.
   - **Shared Backbone Link Utilization ($U_{shared}$)**:
   $$U_{shared} = \frac{4 \times \text{Throughput}}{R} = \frac{4 \times 50}{300} = \frac{200}{300} = \mathbf{0.67 \quad (66.7\%)}$$

---

## 5. Protocol Layers and Network Devices

### 5.1 Why Layering Exists (The Layered Architecture)

Modern computer networks are vast, intricate engineering systems involving billions of diverse computing devices, heterogeneous transmission media, operating systems, and distributed applications. To manage this immense complexity, network designers organize hardware and software protocols into **hierarchical layers**.

#### Fundamental Principles of Layering:
1. **Modularity and Abstraction**: Each layer is responsible for performing a specific, well-defined subset of communication functions. Each layer provides a clean service interface to the layer immediately above it, while relying on the services provided by the layer directly below it.
2. **Implementation Independence**: The internal implementation of a layer can be modified, optimized, or completely rewritten without requiring any changes to other layers, provided the external service interface remains constant. For example, switching a computer's physical connection from wired Ethernet to wireless WiFi changes only Layers 1 and 2; web browsers and transport protocols at higher layers continue executing completely unaware of the change.
3. **Acknowledged Trade-offs of Layering**:
   - Layering introduces processing and memory overhead due to repeated header encapsulation and decapsulation at each layer.
   - It can lead to **duplicated functionality** across layers (e.g., error detection implemented simultaneously at the link layer, network layer, and transport layer).
   - Occasionally, higher layers require lower-layer physical parameters (e.g., wireless signal strength) to optimize performance, which challenges strict modular separation.

---

### 5.2 The OSI 7-Layer Reference Model

Formulated by the **ISO (International Organization for Standardization)**, the **OSI (Open Systems Interconnection)** reference model defines an idealized 7-layer architecture:

| # | Layer Name | Primary Operational Responsibilities | PDU Name | Typical Protocols |
|---|---|---|---|---|
| **7** | **Application** | Provides human-facing network services directly to end-user software programs. | **Message** | HTTP, SMTP, FTP, DNS, SSH, Telnet |
| **6** | **Presentation** | Handles data representation, syntax translation, character set conversion (e.g., ASCII to EBCDIC), data compression, and cryptographic encryption/decryption. | Message | TLS/SSL, JPEG, MPEG, ASCII |
| **5** | **Session** | Establishes, manages, synchronizes, and terminates dialog sessions between cooperating application processes; manages checkpoints for session recovery. | Message | RPC (Remote Procedure Call), NetBIOS, PPTP |
| **4** | **Transport** | Manages end-to-end, process-to-process communication; provides service-point addressing (port numbers), message segmentation and reassembly, connection control, flow control, and error recovery. | **Segment** | TCP, UDP, SCTP |
| **3** | **Network** | Delivers packets across multiple networks from the source host to the destination host; performs logical addressing (IP addressing) and path determination (routing). | **Datagram (Packet)** | IPv4, IPv6, ICMP, OSPF, BGP |
| **2** | **Data Link** | Provides error-free data transfer between two physically adjacent nodes connected across a single communication link; handles framing, physical addressing (MAC addresses), channel access control, and link-level flow/error control. | **Frame** | Ethernet (IEEE 802.3), WiFi (IEEE 802.11), PPP |
| **1** | **Physical** | Transmits raw, unstructured bit streams over physical transmission media; specifies mechanical, electrical, functional, and procedural interfaces (voltages, pinouts, bit timing). | **Bit** | 1000Base-T, RS-232, Optical pulses, Radio RF |

---

### 5.3 The TCP/IP 5-Layer Protocol Suite

The real-world Internet does not strictly implement the 7-layer OSI model. Instead, it runs on the pragmatic **TCP/IP Protocol Suite** (often referred to as the **Internet Reference Model**):

```
       OSI Model (7 Layers)                   TCP/IP Suite (5 Layers)
   ┌───────────────────────────┐           ┌───────────────────────────┐
 7 │     Application Layer     │ ──┐       │                           │
   ├───────────────────────────┤   │       │     Application Layer     │
 6 │    Presentation Layer     │ ──┼─────► │   (HTTP, DNS, SMTP, ...)  │
   ├───────────────────────────┤   │       │                           │
 5 │       Session Layer       │ ──┘       ├───────────────────────────┤
   ├───────────────────────────┤           │      Transport Layer      │
 4 │      Transport Layer      │ ────────► │        (TCP, UDP)         │
   ├───────────────────────────┤           ├───────────────────────────┤
 3 │       Network Layer       │ ────────► │       Network Layer       │
   │                           │           │         (IP, ICMP)        │
   ├───────────────────────────┤           ├───────────────────────────┤
 2 │      Data Link Layer      │ ────────► │      Data Link Layer      │
   │                           │           │      (Ethernet, WiFi)     │
   ├───────────────────────────┤           ├───────────────────────────┤
 1 │      Physical Layer       │ ────────► │      Physical Layer       │
   │                           │           │     (Wires, Fiber, RF)    │
   └───────────────────────────┘           └───────────────────────────┘
```

#### What Happened to Presentation and Session Layers in TCP/IP?
In the TCP/IP suite, Presentation and Session layers do not exist as independent, separate operating system protocol layers:
- If an application requires data compression, data formatting, or cryptographic encryption (Presentation layer functions), the **application developer builds that logic directly into the application program itself** (e.g., HTTPS embedding TLS within the application layer).
- If an application requires session synchronization, token management, or checkpointing (Session layer functions), this logic is similarly handled directly within application-layer code.

---

### 5.4 OSI vs. TCP/IP — Head-to-Head Comparison

| Attribute | OSI Reference Model | TCP/IP Protocol Suite |
|---|---|---|
| **Origin & Purpose** | Developed theoretically by ISO committee as a comprehensive, universal standard before protocols were built. | Developed pragmatically under DARPA / IETF sponsorship; protocols were implemented first, and the model described existing reality. |
| **Number of Layers** | **7 layers** | **4 or 5 layers** (4 layers if Link and Physical are combined into Network Access; 5 layers in modern pedagogy). |
| **Separation of Concepts** | Strictly distinguishes between services, interfaces, and protocols. | Does not clearly distinguish between services and protocols; protocols are tightly coupled to layers. |
| **Presentation & Session** | Dedicated, independent Layers 5 and 6. | Features no separate Presentation or Session layers; functions are folded directly into the Application layer. |
| **Transport Layer Service** | Strictly connection-oriented. | Offers both connection-oriented (**TCP**) and connectionless (**UDP**) services. |
| **Network Layer Service** | Supports both connectionless and connection-oriented network services (X.25 virtual circuits). | Strictly **connectionless** (IP datagram service); all intelligence and connection state are pushed to the edge hosts. |
| **Industry Adoption** | Remains a theoretical, educational, and reference model. | The universal operational standard powering the real-world global Internet. |

---

### 5.5 Encapsulation and Decapsulation (Protocol Data Units)

Data moves down the protocol stack at the sending host, traverses intermediate packet switches, and moves up the stack at the receiving host. At each layer, data is packaged into a specific **Protocol Data Unit (PDU)**:

```
Sending Host Stack:
[ Application Layer ]   Payload Data (M)
        │
        ▼ Appends Transport Header (Ht)
[ Transport Layer ]     [ Ht | M ]                         ──► Segment
        │
        ▼ Appends Network Header (Hn)
[ Network Layer ]       [ Hn | Ht | M ]                    ──► Datagram (Packet)
        │
        ▼ Appends Link Header (Hl) and Trailer (Tl)
[ Data Link Layer ]     [ Hl | Hn | Ht | M | Tl ]          ──► Frame
        │
        ▼ Serializes into raw physical signals
[ Physical Layer ]      1 0 1 1 0 0 1 0 1 0 0 0 1 ...      ──► Bits
```

1. **Encapsulation (at Sending Host)**:
   - The application process passes an application-layer **Message ($M$)** down to the transport layer.
   - The transport layer encapsulates the message by prepending a **Transport Header ($H_t$)** containing source/destination port numbers, sequence numbers, and checksums. This unit is a **Segment**.
   - The network layer encapsulates the segment by prepending a **Network Header ($H_n$)** containing source/destination IP addresses. This unit is a **Datagram** (or **Packet**).
   - The data link layer encapsulates the datagram by prepending a **Link Header ($H_l$)** (containing source/destination MAC addresses) and appending a **Link Trailer ($T_l$)** (containing a Cyclic Redundancy Check — CRC for link-level error detection). This unit is a **Frame**.
   - The physical layer serializes the frame bytes into raw **bits** transmitted as electrical, optical, or radio signals across the physical medium.

2. **Decapsulation (at Receiving Host)**:
   - The receiving host's physical layer reconstructs bits into a frame.
   - The data link layer verifies the frame trailer CRC for bit errors, strips the link header and trailer ($H_l, T_l$), and delivers the datagram up to the network layer.
   - The network layer inspects the destination IP address, strips the network header ($H_n$), and delivers the segment up to the transport layer.
   - The transport layer uses port numbers to locate the target socket, verifies segment integrity, strips the transport header ($H_t$), and places the pristine application message ($M$) into the application's receive buffer.

---

### 5.6 Full Taxonomy of Network Devices and Layer Mapping

Different network devices operate at different layers of the protocol stack. A device can only inspect and process header fields up to the highest layer it natively implements:

```
Host (End System):   [ Layer 1 ] [ Layer 2 ] [ Layer 3 ] [ Layer 4 ] [ Layer 5 ] (All Layers)
Gateway / Firewall:  [ Layer 1 ] [ Layer 2 ] [ Layer 3 ] [ Layer 4 ] [ Layer 5 ] (Up to Application)
Router:              [ Layer 1 ] [ Layer 2 ] [ Layer 3 ]                           (Layers 1 to 3)
Switch / Bridge:     [ Layer 1 ] [ Layer 2 ]                                       (Layers 1 to 2)
Repeater / Hub:      [ Layer 1 ]                                                   (Layer 1 Only)
```

| Device Name | OSI Layer | Primary Operational Role | Addressing Used | Key Features |
|---|---|---|---|---|
| **Repeater** | Layer 1 (Physical) | Regenerates, cleans, and amplifies weakening signals over extended distances. | None | Operates at bit level; no packet inspection. |
| **Hub** | Layer 1 (Physical) | Multi-port repeater connecting multiple hosts in a physical star / logical bus topology. | None | Blindly broadcasts every arriving bit to all other ports. Single collision domain; half-duplex. |
| **Modem** | Layer 1 (Physical) | Modulates digital computer bits into analog carrier waves; demodulates incoming analog signals into digital bits. | None | Bridges digital equipment over analog telephone or cable infrastructure. |
| **NIC (Network Interface Card)** | Layer 2 (Data Link) | Physical expansion card or integrated motherboard chip providing hardware interface to network medium. | Hardware **MAC Address** (48 bits / 6 bytes) | Formats frames, detects bit errors, enforces link-layer access control. |
| **Bridge** | Layer 2 (Data Link) | Connects two separate LAN segments; inspects MAC addresses to filter or forward frames. | MAC Address | Divides a network into two smaller collision domains, reducing congestion. |
| **Switch (L2 Switch)** | Layer 2 (Data Link) | High-speed, multi-port bridge connecting multiple devices in a LAN. | MAC Address | Maintains a **MAC Address Table (CAM Table)**. Forwards frames selectively to the specific destination port; provides dedicated bandwidth per port and full-duplex communication. |
| **Wireless Access Point (AP)** | Layer 2 (Data Link) | Bridges untethered wireless devices (IEEE 802.11) onto an adjacent wired Ethernet LAN (IEEE 802.3). | MAC Address | Acts as a wireless Layer 2 bridge; translates between 802.11 and 802.3 frame formats. |
| **Router** | Layer 3 (Network) | Interconnects disparate, heterogeneous networks; forwards packets toward destination based on IP addresses. | Logical **IP Address** (IPv4 32 bits / IPv6 128 bits) | Executes dynamic routing protocols, maintains forwarding tables, performs **NAT (Network Address Translation)**, fragments packets, decrements TTL. |
| **Layer 3 Switch (L3 Switch)** | Layers 2 + 3 | Combines multi-port Layer 2 switching hardware with high-speed Layer 3 IP routing hardware. | MAC Address and IP Address | Forwards intra-VLAN traffic at Layer 2 wire speed; routes inter-VLAN traffic at Layer 3 using dedicated hardware ASICs. The modern standard in enterprise networks. |
| **Wireless LAN Controller (WLC)** | Management / Layer 2 | Centrally monitors, configures, and controls large numbers of lightweight wireless access points across an enterprise. | IP and MAC Address | Coordinates seamless client roaming between APs, dynamic radio frequency channel allocation, and security enforcement. |
| **Gateway** | All Layers (1 through 7) | Connects networks operating on fundamentally incompatible communication architectures and protocols. | All addressing types | Acts as an active **protocol converter**; translates data formats and protocols across all layers (e.g., connecting a legacy SNA network to an IP network). |
| **Firewall** | Layers 3, 4, and 7 | Monitors and filters inbound and outbound network traffic based on predefined security rulesets. | IP addresses, Port numbers, Application signatures | **Packet-Filtering Firewall**: Inspects IP addresses and TCP/UDP port numbers (Layers 3–4).<br>**Application-Level Gateway (Proxy / WAF)**: Inspects actual application payload contents (Layer 7). |

---

### 5.7 Collision Domains vs. Broadcast Domains

Understanding the exact physical and logical boundaries created by network devices is essential for network design:

> [!IMPORTANT]
> **Definitions**:
> - **Collision Domain**: A physical network segment where data packets can physically collide with one another if two devices transmit simultaneously. Collisions occur in shared, half-duplex media.
> - **Broadcast Domain**: A logical network segment in which any device can transmit a broadcast frame (e.g., destination MAC `FF:FF:FF:FF:FF:FF` or IP `255.255.255.255`), and that frame will be received and processed by every other device in that segment.

| Network Device | Impact on Collision Domains | Impact on Broadcast Domains |
|---|---|---|
| **Hub / Repeater** | **Does NOT split** collision domains. All connected ports belong to one single, shared collision domain. | **Does NOT split** broadcast domains. All ports belong to a single broadcast domain. |
| **Bridge / Switch** | **Splits collision domains**. Every individual switch port constitutes its own independent, isolated collision domain. | **Does NOT split** broadcast domains. By default, all switch ports belong to the same single broadcast domain (unless partitioned into separate VLANs). |
| **Router** | **Splits collision domains**. Every router interface forms an isolated collision domain. | **Splits broadcast domains**. Routers do not forward Layer 2 or Layer 3 broadcast frames by default. Every router interface defines an independent broadcast domain. |

---

### 5.5 The Airline Travel Layering Analogy

*(Directly derived from Lecture Slides 80–81 & Textbook Chapter 1)*

To illustrate why complex distributed systems are structured into hierarchical protocol layers, computer networks courses introduce the **Airline Travel Analogy**:

```
AIRLINE FUNCTIONALITY LAYERS                            CORRESPONDING INTERNET LAYERS
+------------------------------------+                  +------------------------------------+
|  Ticket (Purchase / Baggage Claim) |                  |     Application Layer (HTTP, DNS)  |
+------------------------------------+                  +------------------------------------+
|  Baggage (Check-in / Unload)       |                  |     Transport Layer (TCP, UDP)     |
+------------------------------------+                  +------------------------------------+
|  Gates (Boarding pass check)       |                  |     Network Layer (IP Routing)     |
+------------------------------------+                  +------------------------------------+
|  Runway (Takeoff / Landing)        |                  |     Data Link Layer (Ethernet/WiFi)|
+------------------------------------+                  +------------------------------------+
|  Airplane (Airborne Flight Path)   |                  |     Physical Layer (Bits over Wire)|
+------------------------------------+                  +------------------------------------+
```

- Each layer provides a specific service to the layer directly above it.
- **Layer Independence**: If the airline upgrades its aircraft from Boeing 737 to Airbus A350 (physical layer change), ticket purchasing and baggage handling procedures remain completely unchanged. Similarly, changing from fiber to WiFi does not alter HTTP.

---

### 5.6 Extended Network Devices Taxonomy: NIC and Firewalls

*(Directly derived from Lecture Slides 101–111)*

Beyond Hubs, Switches, and Routers, production enterprise networks rely on two additional critical physical/logical devices:

#### 1. NIC (Network Interface Card)
- **Operational Layer**: **Data Link Layer (Layer 2) and Physical Layer (Layer 1)**.
- **Hardware Role**: A printed circuit board or integrated silicon chip that provides physical electronics connectivity between a computer and a transmission medium (Ethernet port or WiFi radio).
- **Addressing**: Every NIC comes with a globally unique **48-bit MAC (Media Access Control) Address** permanently burned into its ROM (Read-Only Memory).
- **Core Functions**:
  1. Serializes parallel computer bus data into serial bit streams for transmission.
  2. Inspects destination MAC addresses on incoming physical frames; passes matching frames up to the operating system kernel and discards non-matching frames.

#### 2. Network Firewall
- **Operational Layer**: Operates across **Network Layer (Layer 3), Transport Layer (Layer 4), and Application Layer (Layer 7)**.
- **Security Role**: Monitors and controls incoming and outgoing network traffic based on predetermined security rules.
- **Architectural Variations**:
  - **Packet-Filtering Firewall (Layer 3/4)**: Inspects source/destination IP addresses and port numbers (e.g., blocking inbound traffic to port 23 Telnet while permitting outbound traffic to port 443 HTTPS).
  - **Stateful Inspection Firewall**: Tracks TCP connection states (`SYN`, `ESTABLISHED`, `FIN`), ensuring incoming packets belong to an established outbound session.
  - **Application-Layer Firewall / WAF (Web Application Firewall, Layer 7)**: Inspects HTTP payload data to detect SQL injection, cross-site scripting (XSS), and malicious payload attacks.

---

### 5.7 Official Slide Summary Comparison Tables

#### Table 1: Hub vs. Switch Comparison (Slide 110)

| Feature Dimension | Network Hub (Physical Layer - L1) | Network Switch (Data Link Layer - L2) |
| :--- | :--- | :--- |
| **Primary Function** | Broadcasts incoming electrical signals to **all connected ports** blindly | Forwards frames selectively to the **specific destination port** using MAC table |
| **Operating Layer** | **Physical Layer (Layer 1)** | **Data Link Layer (Layer 2)** |
| **Data Handling Unit** | Raw electrical bits | Structured **Frames** |
| **Bandwidth Sharing** | All connected devices share a **single common bandwidth pool** | Every port enjoys **dedicated full bandwidth** |
| **Collision Domain** | **1 Single Collision Domain** across all ports (high collision rate) | **Separate Collision Domain** per port (zero collisions in full duplex) |
| **Transmission Mode** | Half-Duplex only | **Full-Duplex** supported simultaneously |

#### Table 2: Switch vs. Router Comparison (Slide 111)

| Feature Dimension | Network Switch (Layer 2) | Network Router (Layer 3) |
| :--- | :--- | :--- |
| **Network Scope** | Connects multiple devices within the **same local network (LAN)** | Interconnects **multiple different networks** across the global Internet |
| **Addressing Used** | **MAC Addresses** (48-bit physical addresses) | **IP Addresses** (32-bit IPv4 / 128-bit IPv6 logical addresses) |
| **Forwarding Table** | **MAC Address Table / CAM Table** (learned via source MAC snooping) | **Routing Table** (computed via routing algorithms OSPF, BGP) |
| **Broadcast Domain** | **Single Broadcast Domain** (broadcast frames flood all ports) | **Separates Broadcast Domains** (blocks broadcast packets by default) |
| **Key Added Services**| VLAN segmentation | NAT (Network Address Translation), DHCP server, Firewall filtering |

---

## 6. Network Application Principles

### 6.1 Network Application Architectures (Client-Server, P2P, Hybrid)

Application software executes strictly on end systems at the network edge. Core routers do not execute application-layer code. Application architectures conform to one of three fundamental models:

#### 1. The Client-Server Architecture
- **Server**: An always-on host with a permanent, static, globally accessible IP address. To handle immense request volumes, servers are clustered inside commercial data centers.
- **Clients**: Intermittently connected devices with dynamic, transient IP addresses that initiate contact with the server.
- **Key Characteristic**: **Clients do not communicate directly with one another**. All communication flows strictly client-to-server and server-to-client.
- **Examples**: The World Wide Web (HTTP), Email (IMAP/SMTP), File Transfer (FTP).

#### 2. The Peer-to-Peer (P2P) Architecture
- **Zero Dedicated Server Infrastructure**: Eliminates reliance on central, always-on servers.
- **Direct Communication**: Arbitrary pairs of intermittently connected end systems, called **peers**, communicate directly with one another.
- **Self-Scalability**: When a new peer joins a P2P system to download a file, it simultaneously uploads chunks of the file to other requesting peers. As system demand grows, system service capacity grows proportionally.
- **Trade-off**: Highly complex to manage and secure due to peers' transient connectivity, dynamic IP addresses, and churn.
- **Examples**: BitTorrent file sharing, original Skype VoIP architecture, Blockchain networks (Bitcoin, Ethereum).

#### 3. Hybrid Architectures
- Combines elements of both client-server and P2P paradigms.
- **Example (Instant Messaging / Early Skype)**: Centralized servers are used to track user presence, authenticate logins, and resolve user search queries (client-server); once two users identify each other, audio, video, and direct text messages stream directly peer-to-peer.

---

### 6.2 Processes and Inter-Process Communication (IPC)

In operating systems terminology, a program actively running within an end system is a **process**.
- **Inter-Process Communication on the Same Host**: Two processes running on the exact same physical machine communicate using operating-system-defined mechanisms (shared memory, pipes, message queues, UNIX domain sockets).
- **Inter-Process Communication Across a Network**: Two processes running on separate, geographically distant hosts communicate by exchanging discrete **messages** across the network protocol stack.

#### The Client Process vs. Server Process Distinction:
Within any given communication session:
- The process that **initiates communication** (sends the first message) is designated the **client process**.
- The process that **waits passively to be contacted** is designated the **server process**.
- In P2P applications, an end system acts as a client process when downloading data from a peer, and acts as a server process when uploading data to another peer.

---

### 6.3 Sockets — The Application Programming Interface (API)

A process sends and receives messages through the network via a software interface called a **socket**.
- **The Door Analogy**: A process is analogous to a room inside a house, and its socket is the door leading to the outside hallway. When a process wants to send a message to another process across the Internet, it pushes the message out its socket-door.
- **The API Boundary**: The socket represents the formal Application Programming Interface (API) bridging the application layer (written and controlled by the application programmer) and the transport layer (implemented and managed entirely within the operating system kernel).

```
[ Application Process (Application Layer) ]  ◄── Controlled by Application Developer
════════════════════════════════════════════
                 [ Socket ]                  ◄── The Interface (API)
════════════════════════════════════════════
[ Transport Layer: TCP / UDP ]               ◄── Controlled by Operating System Kernel
[ Network Layer: IP ]
```

#### Division of Control Across the Socket Interface:
- **Controlled by the Application Developer**: Complete control over all application-layer logic, message formatting, method selection, and the choice of transport protocol (TCP or UDP).
- **Controlled by the Operating System / Network**: Buffer allocations, packetization, congestion control, window sizing, routing, and physical link transmission. The application developer can tune only a few parameters (e.g., maximum buffer size, socket timeout).

---

### 6.4 Addressing a Process: IP Addresses and Port Numbers

To deliver a message across the Internet to a specific receiving process, an **IP address alone is strictly insufficient**:
- A single physical host (identified by its 32-bit IPv4 address or 128-bit IPv6 address) can simultaneously run dozens of independent, network-connected processes (a web server, an SSH daemon, a mail server, a web browser, a streaming music player).
- To pinpoint the specific destination process on the target host, a complete network identifier requires **two components**:
  1. The **IP Address** of the host (identifies the physical machine).
  2. The **Port Number** assigned to the specific socket on that host (identifies the destination process).

#### Port Number Standards:
Port numbers are 16-bit unsigned integers ranging from **$0 \text{ to } 65{,}535$**:
- **Well-Known Ports ($0 \text{ to } 1023$)**: Formally reserved and assigned by **IANA (Internet Assigned Numbers Authority)** to standard network services. Binding to these ports requires administrative (root/superuser) privileges.
  - Port 20: FTP (Data)
  - Port 21: FTP (Control)
  - Port 22: SSH (Secure Shell)
  - Port 23: Telnet (Unencrypted Remote Terminal)
  - Port 25: SMTP (Simple Mail Transfer Protocol)
  - Port 53: DNS (Domain Name System)
  - Port 80: HTTP (HyperText Transfer Protocol)
  - Port 443: HTTPS (HTTP Secure / TLS)
- **Registered Ports ($1024 \text{ to } 49{,}151$)**: Assigned by IANA to commercial software vendors upon request.
- **Dynamic / Private / Ephemeral Ports ($49{,}152 \text{ to } 65{,}535$)**: Allocated dynamically by the operating system kernel to client applications when establishing outbound connections.

---

### 6.5 Transport Service Requirements of Applications

Different network applications have fundamentally distinct requirements for data transport. These requirements are classified along four dimensions:

| Service Dimension | Technical Definition | Application Types & Real-World Examples |
|---|---|---|
| **Data Integrity (Loss Tolerance)** | Does the application require 100% reliable data delivery with zero packet loss, or can it tolerate dropped packets? | **Loss-Tolerant Applications**: Audio/video streaming, VoIP, online video games (a dropped audio packet causes an imperceptible 20 ms click).<br>**Loss-Intolerant Applications**: File transfer (FTP), web browsing (HTTP), remote terminal (SSH), financial transactions, email (SMTP) — a single corrupted or lost bit invalidates a software download or executable. |
| **Throughput (Bandwidth Sensitivity)** | Does the application require a guaranteed minimum transmission rate to function, or can it dynamically adapt to whatever rate is available? | **Bandwidth-Sensitive Applications**: High-definition video conferencing (requires at least 2–5 Mbps continuous throughput; dropping below this renders the video unusable).<br>**Elastic Applications**: Email, file downloads, web browsing — can utilize any available bandwidth; higher bandwidth finishes the transfer faster, but low bandwidth does not crash the application. |
| **Timing (Latency Sensitivity)** | Does the application require tight, strict upper bounds on end-to-end packet delivery delay? | **Time-Sensitive Applications**: Interactive VoIP telephony, competitive multiplayer gaming, financial algorithmic trading — delays exceeding 100–150 ms cause unnatural conversational pauses or gaming lag.<br>**Time-Insensitive Applications**: Email, file transfer, background data backups. |
| **Security** | Does the application require cryptographic confidentiality (encryption), data integrity, and mutual endpoint authentication? | Applications handling sensitive data (banking, passwords, confidential emails, medical records) require transport-layer encryption (TLS). |

---

### 6.6 Transport Services Provided by the Internet: TCP vs. UDP

The Internet architecture provides exactly two foundational transport-layer protocols to satisfy application needs:

#### 1. Transmission Control Protocol (TCP)
- **Connection-Oriented Service**: Requires a formal **three-way handshake** control exchange between client and server to initialize buffers and sequence numbers *before* any application data can flow.
- **Reliable Data Transfer**: Guarantees that every byte sent into the socket arrives at the receiving process in exact sequential order, without gaps, duplicates, or bit corruption. If packets are lost in transit, TCP automatically detects the loss and retransmits them.
- **Flow Control**: Throttles the sender's transmission rate so it never overflows the receiving host's socket buffer.
- **Congestion Control**: Throttles the sender's transmission rate during broader network congestion to prevent catastrophic core link collapse.
- **What TCP Does NOT Provide**: Does not provide timing guarantees (no latency bounds) or throughput guarantees (cannot guarantee a minimum bandwidth).

#### 2. User Datagram Protocol (UDP)
- **Connectionless Service**: Zero handshake overhead. A process pushes a datagram into a UDP socket, and it is immediately serialized onto the network.
- **Unreliable (Best-Effort) Data Transfer**: Does not guarantee delivery. Datagrams may arrive out of order, or be dropped and lost entirely. UDP implements **no automatic retransmission**.
- **No Flow Control and No Congestion Control**: A UDP sender can blast data into the network as fast as its application process generates it.
- **Lightweight**: Minimal 8-byte header overhead (compared to TCP's 20-byte header).
- **Why Choose UDP?** Real-time applications (VoIP, live gaming, DNS lookups) prefer UDP because TCP's retransmission mechanism introduces unpredictable queuing delays (Head-of-Line blocking) that ruin live interactivity.

#### Real-World Protocol Transport Mapping:

| Application | Application-Layer Protocol | Underlying Transport Protocol | Architectural Justification |
|---|---|---|---|
| **World Wide Web** | HTTP / HTTPS | TCP | Cannot tolerate lost HTML, CSS, or JS code; requires reliable delivery. |
| **Electronic Mail** | SMTP / IMAP / POP3 | TCP | Cannot tolerate corrupted or lost email text/attachments. |
| **Remote Terminal** | SSH / Telnet | TCP | Commands and responses must execute sequentially and reliably. |
| **File Transfer** | FTP | TCP | Entire binary file must arrive bit-for-bit intact. |
| **Domain Name System** | DNS | UDP (primarily) | Lookup queries are small and fit inside a single packet; avoids connection handshake delay; client re-queries if timed out. |
| **Internet Telephony** | SIP / RTP / proprietary | UDP (fallback to TCP) | Can tolerate minor audio loss; cannot tolerate TCP retransmission delays. |
| **Streaming Multimedia** | DASH / HTTP Video | TCP | Large playback buffers absorb retransmission delays; firewall-friendly (Port 80/443). |
| **Network Management** | SNMP | UDP | Must be able to report alarms when the network is congested and TCP fails. |

## 7. The Web, HTTP, and HTTPS

### 7.1 Overview of HTTP (HyperText Transfer Protocol)

The **HyperText Transfer Protocol (HTTP)** is the Web's primary application-layer protocol, specified in standards such as RFC 1945 (HTTP/1.0), RFC 2616 / RFC 7230 (HTTP/1.1), RFC 7540 (HTTP/2), and RFC 9114 (HTTP/3).

- **Client/Server Model**:
  - **HTTP Client**: Web browser (Chrome, Firefox, Safari) that requests, receives, and renders web objects.
  - **HTTP Server**: Web server (Apache, Nginx, Microsoft IIS) that stores and serves web objects upon request.
- **Web Pages and Objects**:
  - A **Web page** (document) consists of multiple **objects**.
  - An **object** is simply a file — a base HTML file, a JPEG graphic, a CSS stylesheet, a JavaScript source file, an MP4 video clip — that is addressable by a single **URL (Uniform Resource Locator)**.
  - A URL consists of two primary components:

$$\underbrace{\text{http://www.example.com}}_{\text{Server Hostname}} \quad \underbrace{\text{/departments/cs/faculty.html}}_{\text{Object Path Name}}$$

- **Underlying Transport**: HTTP runs over **TCP**. The client first initiates a TCP connection to the server on **port 80** (or port 443 for HTTPS). Once established, HTTP request and response messages are exchanged across the TCP socket interface. Because TCP provides guaranteed, in-order byte stream delivery, HTTP does not need to implement error detection or retransmission logic.

---

### 7.2 Statelessness — Architectural Rationale and Impact

> [!IMPORTANT]
> **HTTP is a Stateless Protocol**:
> An HTTP server maintains **zero historical state** about its clients. If a client requests the identical object twice within 5 seconds, the server treats the second request completely independently, resending the object without remembering that it just served it to that same client.

#### Why Was HTTP Designed as Stateless?
1. **Dramatic Simplification of Server Design**: Stateful protocols require complex memory structures to track active sessions across thousands of simultaneous clients. If a stateful server crashes, all client state is corrupted, requiring complex state reconciliation protocols.
2. **High Scalability**: A stateless server can process an incoming request immediately without consulting session history, allowing web servers to scale horizontally across server farms behind load balancers with minimal synchronization overhead.
3. **The Workaround**: Because modern web applications (e-commerce shopping carts, user authentication logins, personal preferences) require session tracking, state maintenance is achieved using **Cookies** (Section 8.1).

---

### 7.3 Non-Persistent vs. Persistent HTTP Connections

When a web page contains multiple referenced objects (e.g., a base HTML document referencing 10 separate images), the browser must retrieve 11 total objects. How are the underlying TCP connections managed?

#### 1. Non-Persistent HTTP (HTTP/1.0 Default)
- **Mechanism**: Each TCP connection is opened to transfer **at most one single web object**, and is immediately closed by the server upon completion.
- To retrieve a base HTML file and 10 referenced images, the browser must sequentially initiate **11 separate, distinct TCP connections**.
- **Delay Cost per Object**:
  - **1 RTT** to complete the TCP three-way handshake (SYN, SYNACK).
  - **1 RTT** for the HTTP request message to travel to the server and the first bytes of the HTTP response to return.
  - Plus the actual physical file transmission delay ($L/R$).
  - **Total Baseline Delay per Object = $2\text{ RTT} + \frac{L}{R}$**.
- **Severe Disadvantages**:
  - Operating system overhead: Both client and server must allocate TCP buffer memory and control blocks for every new connection.
  - High cumulative latency: Retrieving 11 objects sequentially takes $11 \times 2\text{ RTT} = 22\text{ RTT}$.
  - TCP Slow Start: Every new connection starts with a small congestion window, underutilizing link capacity.

#### 2. Persistent HTTP (HTTP/1.1 Default)
- **Mechanism**: The server leaves the TCP connection **open** after sending its response. Subsequent HTTP requests and responses between the same client-server pair can be sent over the **same pre-existing open connection**.
- **Without Pipelining**: The client sends a new request only after receiving the complete response for the previous object (1 RTT per referenced object).
- **With Pipelining (HTTP/1.1 Specification)**: As soon as the client parses the base HTML file and discovers referenced objects, it transmits requests for all referenced objects **back-to-back into the socket without waiting for individual replies**. All referenced objects can be requested within a single round-trip time unit.
- **Latency Comparison for 1 Base HTML + 10 Referenced Images (11 Objects Total)**:
  - Non-Persistent (sequential): $11 \times 2\text{ RTT} = \mathbf{22\text{ RTT}}$
  - Persistent, without Pipelining: $2\text{ RTT (first object)} + 10 \times 1\text{ RTT} = \mathbf{12\text{ RTT}}$
  - Persistent, with Pipelining: $2\text{ RTT (first object)} + 1\text{ RTT (all 10 remaining)} = \mathbf{3\text{ RTT}}$

---

### 7.4 HTTP Message Format (Request and Response Messages)

HTTP messages are formatted in human-readable **ASCII (American Standard Code for Information Interchange)** text.

#### 1. HTTP Request Message:
```http
GET /somedir/page.html HTTP/1.1\r\n
Host: www.someschool.edu\r\n
Connection: keep-alive\r\n
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n
Accept: text/html,application/xhtml+xml\r\n
Accept-Language: en-US,en;q=0.9\r\n
\r\n
```

```
[ Request Line: Method | SP | URL | SP | Version | CRLF ]
[ Header Line:  Field Name | : | SP | Field Value | CRLF ]
[ Header Line:  Field Name | : | SP | Field Value | CRLF ]
[ CRLF (Empty Line separating headers from entity body)  ]
[ Entity Body (Optional: present in POST, PUT)           ]
```

- **Request Line**: Contains the **Method** (`GET`), the requested **URL path** (`/somedir/page.html`), and the **HTTP Version** (`HTTP/1.1`), terminated by Carriage Return and Line Feed (`\r\n`).
- **Header Lines**:
  - `Host: www.someschool.edu`: Explicitly declares the destination hostname. **Crucial**: Allows a single physical web server with one IP address to host hundreds of distinct virtual domains (Virtual Hosting) and informs web proxies of the target origin server.
  - `Connection: keep-alive`: Informs the server that the client wants a persistent connection.
  - `User-Agent:`: Identifies the requesting browser type and operating system.
  - `Accept-Language:`: Content negotiation header specifying preferred languages.
- **Empty Line (`\r\n`)**: Strictly required to separate headers from the entity body.
- **Entity Body**: Contains data payloads (e.g., form submissions in POST requests).

#### 2. HTTP Response Message:
```http
HTTP/1.1 200 OK\r\n
Date: Tue, 18 Aug 2026 12:00:00 GMT\r\n
Server: Apache/2.4.52 (Ubuntu)\r\n
Last-Modified: Mon, 17 Aug 2026 15:30:00 GMT\r\n
Content-Length: 6821\r\n
Content-Type: text/html; charset=UTF-8\r\n
Connection: keep-alive\r\n
\r\n
<!DOCTYPE html><html><head>... (6821 bytes of HTML data) ...
```

- **Status Line**: Contains the **HTTP Version** (`HTTP/1.1`), numeric **Status Code** (`200`), and readable **Reason Phrase** (`OK`).
- **Header Lines**:
  - `Date:`: Timestamp when the server generated and transmitted the response.
  - `Server:`: Identifies server software vendor and version.
  - `Last-Modified:`: Date and time when the underlying file on disk was last altered. Essential for **Web Caching** (Section 8.3).
  - `Content-Length:`: Exact size of the enclosed entity body payload in bytes.
  - `Content-Type:`: Declares the official MIME type of the payload (e.g., `text/html`, `image/jpeg`).
- **Entity Body**: Contains the raw bytes of the requested web resource.

---

### 7.5 HTTP Request Methods and Status Codes

#### HTTP Request Methods:
| Method | Operational Purpose | Idempotent? | Safe? |
|---|---|---|---|
| **GET** | Retrieves the resource identified by the target URL from the server. Form data can be transmitted by appending parameters to the URL string (query string, e.g., `?q=networks`). | Yes | Yes |
| **POST** | Submits entity data to the server for processing (e.g., HTML form input, file upload). Data is encapsulated inside the **entity body**, not in the URL. | No | No |
| **HEAD** | Identical to a GET request, but instructs the server to return **only the headers** — the entity body is omitted. Used by caches to inspect `Last-Modified` timestamps or check resource existence. | Yes | Yes |
| **PUT** | Uploads a new resource to the server, completely replacing any existing file located at the specified URL with the payload in the entity body. | Yes | No |
| **DELETE** | Requests that the origin server permanently delete the resource identified by the target URL. | Yes | No |
| **OPTIONS** | Queries the server to describe the communication options and permitted HTTP methods available for the specified URL. | Yes | Yes |

*(A method is **safe** if it does not alter server resource state; it is **idempotent** if executing it multiple identical times produces the same server state as executing it once).*

#### HTTP Status Codes:
Status codes are 3-digit integers categorized into five classes:

| Code Range | Category | Common Codes, Phrases, and Meanings |
|---|---|---|
| **1xx** | Informational | **100 Continue**: The initial request headers have been received; client should proceed to send the entity body. |
| **2xx** | Success | **200 OK**: The request succeeded; requested resource is delivered in entity body.<br>**206 Partial Content**: Server is delivering only a specified byte range of the resource (used in resumable downloads). |
| **3xx** | Redirection | **301 Moved Permanently**: The requested resource has been assigned a new permanent URL, specified in the `Location:` header.<br>**302 Found (Temporary Redirect)**: The resource resides temporarily under a different URL.<br>**304 Not Modified**: The cached copy of the resource is still fresh and unmodified (returned in response to a Conditional GET; contains zero entity body). |
| **4xx** | Client Error | **400 Bad Request**: The server could not understand the request due to malformed syntax.<br>**401 Unauthorized**: Authentication credentials (username/password) are missing or invalid.<br>**403 Forbidden**: Server understood request but refuses to authorize access.<br>**404 Not Found**: The requested resource does not exist on this server.<br>**408 Request Timeout**: Server timed out waiting for the client's request. |
| **5xx** | Server Error | **500 Internal Server Error**: Generic server crash or unexpected internal failure.<br>**502 Bad Gateway**: An intermediate proxy received an invalid response from an upstream server.<br>**503 Service Unavailable**: Server is temporarily overloaded or down for maintenance.<br>**505 HTTP Version Not Supported**: Server does not support the HTTP protocol version used in the request. |

---

### 7.6 Protocol Evolution: HTTP/1.1, HTTP/2, and HTTP/3

| Feature | HTTP/1.1 | HTTP/2 (RFC 7540) | HTTP/3 (RFC 9114) |
|---|---|---|---|
| **Underlying Transport Protocol** | **TCP** | **TCP** | **QUIC** (runs over **UDP**) |
| **Message Encoding** | Textual (plain ASCII text) | **Binary Framing Layer** (parses binary frames) | **Binary Framing** (built natively into QUIC) |
| **Multiplexing Mechanics** | None; pipelining exists but suffers from application HOL blocking | True multiplexing: Multiple requests/responses interleave simultaneously over **1 single TCP connection** | True multiplexing: Independent streams over UDP; **zero transport-layer HOL blocking** |
| **Head-of-Line (HOL) Blocking** | **Severe application-layer HOL blocking**: Server must respond to pipelined requests in strict FCFS order. A large image stalls all subsequent requests. | Eliminates application-layer HOL blocking via frames, but **still suffers from transport-layer TCP HOL blocking**: A single dropped TCP packet stalls all multiplexed streams until retransmitted. | **Completely eliminated**: QUIC streams are strictly independent. Packet loss on Stream A never stalls Stream B. |
| **Header Compression** | None (redundant ASCII headers sent on every request) | **HPACK Compression** (maintains shared client/server header table) | **QPACK Compression** (adapted for out-of-order UDP delivery) |
| **Server Push** | Not supported | Supported: Server can proactively push resources to client before client asks. | Supported |
| **Connection Setup Latency** | High: 1 RTT (TCP) + 1–2 RTT (TLS) | High: 1 RTT (TCP) + 1 RTT (TLS 1.3) | **Ultra-Low**: Combined transport + cryptographic handshake in **1 RTT** (or **0-RTT** resumption). |

---

### 7.7 HTTPS and Transport Layer Security (TLS 1.3)

**HTTPS (HyperText Transfer Protocol Secure)** is not a separate protocol; it is standard application-layer HTTP running over a secure cryptographic tunnel provided by **TLS (Transport Layer Security)**. HTTPS communicates universally over **port 443**.

#### The Dual-Phase Hybrid Cryptographic Architecture:
Cryptographic algorithms are divided into:
1. **Asymmetric Encryption (Public-Key Cryptography)**: Computationally intensive. Uses a mathematically paired public key (openly published) and private key (kept secret). Data encrypted with the public key can only be decrypted by the matching private key.
2. **Symmetric Encryption (Shared-Secret Cryptography)**: Computationally fast and lightweight. Uses the exact same secret session key to both encrypt and decrypt data.

**TLS Hybrid Operation**:
- **Phase 1 (Handshake)**: Uses **Asymmetric Encryption** to authenticate the server's identity and securely negotiate a shared symmetric session key without an eavesdropper being able to calculate it.
- **Phase 2 (Data Transfer)**: Once the session key is established, both parties immediately switch to **Symmetric Encryption** to encrypt all bulk HTTP payload data at wire speed.

#### TLS 1.3 Structural Advancements (RFC 8446):
TLS 1.3 represents a major redesign of transport security:
1. **Handshake Latency Eradication (1-RTT & 0-RTT)**:
   - Legacy TLS required a 2-RTT handshake before encrypted application data could flow.
   - **1-RTT Handshake**: In TLS 1.3, the client sends its supported cipher suites *and* its ephemeral Diffie-Hellman key share parameters inside the very first `ClientHello` packet. The server responds with its chosen parameters and certificate in `ServerHello`. By round-trip 1, symmetric keys are derived and application data can flow immediately.
   - **0-RTT Resumption (Pre-Shared Key — PSK)**: Returning clients that have previously connected can transmit encrypted application data in the very first message. (Note: 0-RTT carries vulnerability to replay attacks, so it is restricted to idempotent requests).
2. **Mandatory Ephemeral Forward Secrecy**:
   - TLS 1.3 completely bans static RSA key transport.
   - All key agreements must use **ephemeral Diffie-Hellman (ECDHE)**.
   - **Security Value**: Even if an adversary steals the server's master private key 10 years in the future, they **cannot retroactively decrypt historically recorded network traffic**, because each past session's temporary encryption key was wiped from memory when the session closed.
3. **Cryptographic Hardening**:
   - Outdated, insecure cryptographic primitives have been purged: RC4 stream ciphers, MD5 hash algorithms, SHA-1, DES/3DES, and static RSA exchanges are completely removed.
   - Enforces strict **AEAD (Authenticated Encryption with Associated Data)** ciphers, such as **AES-GCM (Advanced Encryption Standard in Galois/Counter Mode)** and **ChaCha20-Poly1305**, guaranteeing confidentiality and message integrity simultaneously.
4. **Encrypted Handshake Metadata**:
   - The server's digital certificate is encrypted before transmission, preventing passive eavesdroppers from observing which domain or identity is being accessed.

#### Authentication Flow via Digital Certificates:
1. To prevent **Man-in-the-Middle (MitM) attacks**, the server presents an **X.509 Digital Certificate** signed cryptographically by a trusted **Certificate Authority (CA)** (e.g., Let's Encrypt, DigiCert).
2. The browser contains pre-installed public keys of trusted root CAs. It cryptographically verifies the CA's digital signature on the server's certificate.
3. The server proves ownership of the certificate by generating a cryptographic signature on dynamic handshake data using its secret **Private Key**. The browser verifies this signature using the certificate's **Public Key**, definitively authenticating the server's identity before any user credentials or data are transmitted.

### 7.6 Real-World Wireshark Packet Capture Analysis for HTTP

*(Directly derived from Lecture Slides 181 & 184)*

Understanding real-world packet traces is a core exam competency. Below is the detailed field-by-field breakdown of actual Wireshark packet captures from class lectures:

#### 1. HTTP Request Packet Capture (Slide 181)
When a browser visits `http://gaia.cs.umass.edu/cs453/index.html`:

```http
GET /cs453/index.html HTTP/1.1\r\n
Host: gaia.cs.umass.edu\r\n
User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.7.2) Gecko/20040804 Netscape/7.2 (ax)\r\n
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5\r\n
Accept-Language: en-us,en;q=0.5\r\n
Accept-Encoding: gzip,deflate\r\n
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7\r\n
Keep-Alive: 300\r\n
Connection: keep-alive\r\n
\r\n
```

- **Request Line**: `GET /cs453/index.html HTTP/1.1` specifies the HTTP method, requested resource URI, and HTTP version.
- **`Host: gaia.cs.umass.edu`**: Mandatory in HTTP/1.1. Enables virtual web hosting (multiple websites hosted on a single IP address).
- **`User-Agent`**: Identifies client browser version and operating system to the web server.
- **`Accept-Encoding: gzip,deflate`**: Informs the server that the browser can decompress compressed files, saving network bandwidth.
- **`Connection: keep-alive`**: Requests persistent TCP connection reuse.
- **Empty Line (`\r\n`)**: Terminates the HTTP request header section.

#### 2. HTTP Response Packet Capture (Slide 184)
The origin web server replies with:

```http
HTTP/1.1 200 OK\r\n
Date: Tue, 07 Mar 2006 12:39:45 GMT\r\n
Server: Apache/2.0.52 (Fedora)\r\n
Last-Modified: Sat, 10 Dec 2005 18:27:46 GMT\r\n
ETag: "27b3-76-e137c440"\r\n
Accept-Ranges: bytes\r\n
Content-Length: 118\r\n
Connection: close\r\n
Content-Type: text/html; charset=ISO-8859-1\r\n
\r\n
<data payload: 118 bytes of HTML source code>
```

- **Status Line**: `HTTP/1.1 200 OK` confirms successful request processing.
- **`Last-Modified` & `ETag`**: Provides caching timestamps and hash tokens used later in Conditional GET requests (`If-Modified-Since`).
- **`Content-Length: 118`**: Exact byte length of the enclosed HTML payload.
- **`Content-Type: text/html`**: MIME type instructing the browser to parse the payload as an HTML document.

---

## 8. User-Server Interaction: Cookies and Web Caching

### 8.1 Cookies — Maintaining State on a Stateless Web

Because HTTP is fundamentally stateless (Section 7.2), websites use **Cookies** (RFC 6265) to create stateful user sessions atop stateless HTTP transactions.

```
Client Browser                                             Web Server
      │                                                         │
      │ ─── 1. Initial HTTP Request (No Cookie Header) ───────► │ (Generates ID: 1678 in DB)
      │                                                         │
      │ ◄── 2. HTTP Response (Set-Cookie: 1678) ─────────────── │
      │                                                         │
(Stores 1678 in local                                           │
 cookie file for domain)                                        │
      │                                                         │
      │ ─── 3. Subsequent Request (Cookie: 1678) ─────────────► │ (Queries DB for ID: 1678,
      │                                                         │  restores shopping cart/state)
      │ ◄── 4. HTTP Response (Personalized Data) ────────────── │
```

#### The Four Concrete Components of the Cookie Architecture:
1. A `Set-Cookie:` header line included in the HTTP **response message** sent by the web server.
2. A `Cookie:` header line included in every subsequent HTTP **request message** automatically generated by the client browser to that same domain.
3. A **cookie file** stored, managed, and secured locally on the client's host by the web browser.
4. A **back-end database** at the web server that cross-references each unique cookie ID against the user's stored account details, shopping cart items, and navigation history.

#### Cookie Categories and Privacy Implications:
- **Session Cookies**: Stored in temporary RAM memory; automatically erased as soon as the user closes the web browser.
- **Persistent Cookies**: Written to disk with an explicit expiration date (`Expires=` or `Max-Age=`); persist across system reboots until expiration.
- **First-Party Cookies**: Set directly by the domain visible in the browser's address bar. Used for user login state, language preferences, and shopping carts.
- **Third-Party Cookies**: Injected by external third-party domains (such as ad tracking networks or analytics scripts) embedded within the host site. When multiple unrelated sites embed scripts from the same ad provider, the ad provider reads the user's third-party tracking cookie across all visited sites, compiling an unauthorized cross-site behavioral profile of the user.

---

### 8.2 Web Caching (Proxy Servers)

A **Web Cache (Proxy Server)** is a network entity that stores copies of recently requested web objects in local storage, satisfying subsequent HTTP requests on behalf of origin web servers.

```
[ Client Browser ] ──► (Local Request) ──► [ Web Cache / Proxy ] ──(Cache Hit: Instant Response)──► Client
                                                   │
                                            (Cache Miss)
                                                   ▼ (Opens TCP, requests object)
                                           [ Origin Web Server ]
```

#### Dual Identity of a Proxy Server:
A proxy server operates simultaneously as **both a server and a client**:
- When it receives HTTP requests from an end-user browser, it acts as a **server**.
- When it queries an upstream origin server on a cache miss, it acts as a **client**.

#### Operational Benefits:
1. **Drastically Reduces Client Response Time**: A request satisfied from a local LAN cache completes in a few milliseconds, completely avoiding round-trip delays across the public Internet.
2. **Mitigates Bottlenecks on Institutional Access Links**: Up to 30–60% of web traffic consists of repeated object requests. Caching satisfies these requests locally, slashing traffic demand on expensive enterprise access links and preventing access-link queuing delay collapse.

---

### 8.3 The Conditional GET Protocol

Caching introduces a critical technical problem: *how does the proxy ensure that its stored copy of an object is still valid and has not been modified on the origin server?*

HTTP solves this using the **Conditional GET** mechanism (RFC 7232):

```
Client Browser                    Web Cache (Proxy)                         Origin Server
      │                                   │                                       │
      │ ─── 1. HTTP GET /pic.jpg ───────► │                                       │
      │                                   │ ─── 2. Conditional GET /pic.jpg ────► │
      │                                   │        If-Modified-Since: [Date]      │
      │                                   │                                       │
      │                                   │ ◄── 3. HTTP/1.1 304 Not Modified ──── │
      │                                   │        (Zero Entity Body!)            │
      │ ◄── 4. Delivers Cached Copy ───── │                                       │
```

#### Exact Step-by-Step Sequence:
1. **Initial Cache Miss**:
   - The cache fetches the object from the origin server.
   - The origin server's response includes a `Last-Modified:` header specifying the file's disk alteration timestamp (e.g., `Last-Modified: Mon, 17 Aug 2026 15:30:00 GMT`).
   - The cache stores the object along with this timestamp.
2. **Subsequent Request & Verification (Conditional GET)**:
   - When another client requests the same object, the cache generates a **Conditional GET** message to the origin server, injecting the `If-Modified-Since:` header set to the stored `Last-Modified` timestamp:
     ```http
     GET /pic.jpg HTTP/1.1
     Host: www.example.com
     If-Modified-Since: Mon, 17 Aug 2026 15:30:00 GMT
     ```
3. **Origin Server Evaluation**:
   - **Scenario A (Object Unmodified)**: The object has not changed since that date. The server returns:
     ```http
     HTTP/1.1 304 Not Modified
     Date: Tue, 18 Aug 2026 12:00:00 GMT
     ```
     **Critical Efficiency**: The response body is completely **empty**. Zero image or HTML data is retransmitted across the access link, saving precious bandwidth. The cache safely serves its stored copy to the client.
   - **Scenario B (Object Modified)**: The object was altered. The server returns a standard `200 OK` response with the new object payload in the entity body and an updated `Last-Modified` header. The cache updates its local storage and forwards the fresh object to the client.

### 8.4 Web Caching Institutional Network Performance Analysis

*(Directly derived from Lecture Slides 201–202 & Kurose & Ross)*

```
                       Origin Servers
                             ▲
                             │ Public Internet
                             ▼
                    +-----------------+
                    | 1.54 Mbps Link  |  (Access Link)
                    +-----------------+
                             ▲
                             │
            +----------------┴----------------+
            |      Institutional Network      |
            |           1 Gbps LAN            |
            |                                 |
            |   [ Browsers ]   [ Web Cache ]  |
            +---------------------------------+
```

#### Problem Scenario:
- **LAN Bandwidth**: $1\text{ Gbps}$ Ethernet.
- **Access Link Bandwidth**: $R_{access} = 1.54\text{ Mbps}$ (T1 connection).
- **Average Web Object Size**: $L = 100{,}000\text{ bits} = 100\text{ Kbits}$.
- **Average Request Rate from Browsers**: $a = 15\text{ requests/second}$.
- **Round-Trip Time from Institutional Router to Origin Server**: $RTT = 2.0\text{ seconds}$.

---

#### 1. Baseline Performance Without Local Web Cache:
1. **Average Data Rate to Browsers**:
   $$\text{Traffic Rate} = a \times L = 15\text{ req/s} \times 100{,}000\text{ bits} = 1{,}500{,}000\text{ bps} = \mathbf{1.50\text{ Mbps}}$$
2. **Access Link Traffic Intensity / Utilization**:
   $$I_{access} = \frac{\text{Traffic Rate}}{R_{access}} = \frac{1.50\text{ Mbps}}{1.54\text{ Mbps}} = \mathbf{0.974 \quad (97.4\%)}$$
3. **LAN Utilization**:
   $$I_{LAN} = \frac{1.50\text{ Mbps}}{1000\text{ Mbps}} = \mathbf{0.0015 \quad (0.15\%)}$$
4. **End-to-End Delay Impact**:
   - Because access link utilization is $97.4\%$, queuing delay explodes towards infinity ($d_{queue} \to \infty$ as $I \to 1$).
   - Total response time ranges from **tens of seconds to minutes**, rendering web browsing unusable.
   - Upgrading the access link to $155\text{ Mbps}$ (ATM/OC-3) solves delay but costs thousands of dollars per month.

---

#### 2. Performance After Installing a Local Web Cache (Cache Hit Rate = 40%):
Suppose an inexpensive local proxy server is installed on the institutional LAN, achieving a cache hit rate of $h = 0.40$ ($40\%$):
- $40\%$ of requests are satisfied immediately by the local cache over the 1 Gbps LAN (delay $\approx 10\text{ ms} = 0.01\text{ s}$).
- Only $60\%$ of requests ($1 - h = 0.60$) traverse the access link to origin servers.

1. **New Data Rate Over Access Link**:
   $$\text{New Traffic Rate} = 0.60 \times 1.50\text{ Mbps} = \mathbf{0.90\text{ Mbps}}$$
2. **New Access Link Utilization**:
   $$I_{access, cache} = \frac{0.90\text{ Mbps}}{1.54\text{ Mbps}} = \mathbf{0.584 \quad (58.4\%)}$$
   Because utilization dropped from $97.4\%$ to $58.4\%$, queuing delay on the access link becomes virtually negligible ($< 5\text{ ms}$).
3. **Average End-to-End Response Time**:
   $$\text{Avg Delay} = (1 - h) \times (\text{Origin Server Delay}) + h \times (\text{Cache Delay})$$
   $$\text{Avg Delay} = 0.60 \times (2.01\text{ s}) + 0.40 \times (0.01\text{ s}) \approx 1.206\text{ s} + 0.004\text{ s} \approx \mathbf{1.21\text{ seconds}}$$

**Conclusion**: Web caching drops average response time from minutes down to **$1.21\text{ seconds}$** without purchasing expensive access link bandwidth upgrades!

---

## 9. Solved Numericals (Comprehensive Exam-Style Problems)

### 9.1 End-to-End Nodal Delay Across Multiple Links

**Problem Statement (From Course Notes)**:
A packet of length $L = 8000\text{ bits}$ is transmitted from a source host to a destination host across three sequential links. The propagation speed on all physical links is $s = 3 \times 10^8\text{ m/s}$. The link parameters are:
- **Link 1**: Transmission rate $R_1 = 100\text{ Mbps}$, length $d_1 = 3\text{ km}$
- **Link 2**: Transmission rate $R_2 = 1000\text{ Mbps}$ ($1\text{ Gbps}$), length $d_2 = 500\text{ km}$
- **Link 3**: Transmission rate $R_3 = 10\text{ Mbps}$, length $d_3 = 3\text{ km}$

Calculate the individual transmission delay, propagation delay, total link delay for each hop, and the total end-to-end system delay (assuming processing and queuing delays are negligible).

#### Step-by-Step Solution:

**1. Link 1 Analysis**:
- Transmission Delay:
  $$d_{trans, 1} = \frac{L}{R_1} = \frac{8000\text{ bits}}{100 \times 10^6\text{ bps}} = 80 \times 10^{-6}\text{ s} = \mathbf{80\ \mu s}$$
- Propagation Delay:
  $$d_{prop, 1} = \frac{d_1}{s} = \frac{3000\text{ m}}{3 \times 10^8\text{ m/s}} = 10 \times 10^{-6}\text{ s} = \mathbf{10\ \mu s}$$
- Total Link 1 Delay:
  $$d_{link, 1} = 80\ \mu s + 10\ \mu s = \mathbf{90\ \mu s}$$

**2. Link 2 Analysis**:
- Transmission Delay:
  $$d_{trans, 2} = \frac{L}{R_2} = \frac{8000\text{ bits}}{1000 \times 10^6\text{ bps}} = 8 \times 10^{-6}\text{ s} = \mathbf{8\ \mu s}$$
- Propagation Delay:
  $$d_{prop, 2} = \frac{d_2}{s} = \frac{500{,}000\text{ m}}{3 \times 10^8\text{ m/s}} = 1666.67 \times 10^{-6}\text{ s} \approx \mathbf{1670\ \mu s}\ (1.67\text{ ms})$$
- Total Link 2 Delay:
  $$d_{link, 2} = 8\ \mu s + 1670\ \mu s = \mathbf{1678\ \mu s}$$

**3. Link 3 Analysis**:
- Transmission Delay:
  $$d_{trans, 3} = \frac{L}{R_3} = \frac{8000\text{ bits}}{10 \times 10^6\text{ bps}} = 800 \times 10^{-6}\text{ s} = \mathbf{800\ \mu s}$$
- Propagation Delay:
  $$d_{prop, 3} = \frac{d_3}{s} = \frac{3000\text{ m}}{3 \times 10^8\text{ m/s}} = 10 \times 10^{-6}\text{ s} = \mathbf{10\ \mu s}$$
- Total Link 3 Delay:
  $$d_{link, 3} = 800\ \mu s + 10\ \mu s = \mathbf{810\ \mu s}$$

**4. Total System End-to-End Delay**:
$$d_{end-to-end} = d_{link, 1} + d_{link, 2} + d_{link, 3} = 90\ \mu s + 1678\ \mu s + 810\ \mu s = \mathbf{2578\ \mu s} = \mathbf{2.578\text{ ms}}$$

> [!NOTE]
> **Key Exam Insight**: Link 2 has the fastest transmission rate (1000 Mbps), requiring only $8\ \mu s$ to push the bits onto the wire. However, Link 2 completely dominates the total delay ($1678\ \mu s$) because of its physical distance (500 km). This illustrates that high bandwidth cannot overcome propagation delay enforced by the physical speed of light over large geographical distances.

---

### 9.2 Queuing Delay Modeling via Traffic Intensity (Non-Linear Growth)

**Problem Statement (From Course Notes)**:
Consider a communication link with constant transmission rate $R = 1{,}900{,}000\text{ bps}$ ($1.9\text{ Mbps}$) and constant packet length $L = 4700\text{ bits}$. For traffic intensity $I < 1$, the average queuing delay is modeled as:
$$d_{queue} = \frac{I \cdot L}{R(1 - I)}$$
where Traffic Intensity $I = \frac{L \cdot a}{R}$.

Compute the traffic intensity and average queuing delay (in milliseconds) for:
- **Scenario A**: Packet arrival rate $a = 35\text{ packets/second}$
- **Scenario B**: Packet arrival rate $a = 77\text{ packets/second}$
- **Scenario C**: A separate link with $R = 1{,}800{,}000\text{ bps}$, $L = 7300\text{ bits}$, evaluated at arrival rates $a_1 = 30\text{ packets/s}$ and $a_2 = 76\text{ packets/s}$.

#### Step-by-Step Solution:

**Scenario A ($a = 35\text{ packets/s}$)**:
1. Traffic Intensity:
   $$I_A = \frac{4700 \times 35}{1{,}900{,}000} = \frac{164{,}500}{1{,}900{,}000} \approx \mathbf{0.08658}$$
2. Average Queuing Delay:
   $$d_{queue, A} = \frac{0.08658 \times 4700}{1{,}900{,}000 \times (1 - 0.08658)} = \frac{406.926}{1{,}900{,}000 \times 0.91342} = \frac{406.926}{1{,}735{,}498} \approx 0.000234\text{ s} \approx \mathbf{0.23\text{ ms}}$$

**Scenario B ($a = 77\text{ packets/s}$)**:
1. Traffic Intensity:
   $$I_B = \frac{4700 \times 77}{1{,}900{,}000} = \frac{361{,}900}{1{,}900{,}000} \approx \mathbf{0.19047}$$
2. Average Queuing Delay:
   $$d_{queue, B} = \frac{0.19047 \times 4700}{1{,}900{,}000 \times (1 - 0.19047)} = \frac{895.209}{1{,}900{,}000 \times 0.80953} = \frac{895.209}{1{,}538{,}107} \approx 0.000582\text{ s} \approx \mathbf{0.58\text{ ms}}$$

**Scenario C ($R = 1{,}800{,}000\text{ bps}$, $L = 7300\text{ bits}$)**:
- At $a_1 = 30\text{ packets/s}$:
  $$I_{C1} = \frac{7300 \times 30}{1{,}800{,}000} = \frac{219{,}000}{1{,}800{,}000} \approx \mathbf{0.12167}$$
  $$d_{queue, C1} = \frac{0.12167 \times 7300}{1{,}800{,}000 \times (1 - 0.12167)} \approx \frac{888.191}{1{,}581{,}000} \approx 0.000561\text{ s} = \mathbf{0.561\text{ ms}}$$
- At $a_2 = 76\text{ packets/s}$:
  $$I_{C2} = \frac{7300 \times 76}{1{,}800{,}000} = \frac{554{,}800}{1{,}800{,}000} \approx \mathbf{0.30822}$$
  $$d_{queue, C2} = \frac{0.30822 \times 7300}{1{,}800{,}000 \times (1 - 0.30822)} \approx \frac{2250.006}{1{,}245{,}204} \approx 0.001806\text{ s} = \mathbf{1.806\text{ ms}}$$

> [!NOTE]
> **Key Exam Insight**: In Scenario C, when arrival rate increased by a factor of $76 / 30 \approx 2.53\times$, queuing delay increased by a factor of $1.806 / 0.561 \approx \mathbf{3.22\times}$. This clearly confirms that queuing delay grows **non-linearly**, with delay compounding at an accelerating rate as traffic intensity climbs.

---

### 9.3 Throughput and Bottleneck Links (Shared Backbone & File Download)

#### Problem 1: Shared Backbone Capacity
Ten active client-server connections simultaneously share a common backbone link with transmission capacity $R = 300\text{ Mbps}$. The backbone link divides capacity equally among all 10 connections. Each server connects to the network via an access link $R_s = 50\text{ Mbps}$, and each client connects via an access link $R_c = 90\text{ Mbps}$. 

Calculate the maximum achievable end-to-end throughput per connection, and identify which link acts as the bottleneck.

**Solution**:
- Fair share of backbone link per connection:
  $$R_{backbone, share} = \frac{R}{10} = \frac{300\text{ Mbps}}{10} = 30\text{ Mbps}$$
- Throughput per connection is the minimum capacity along the transmission path:
  $$\text{Throughput} = \min(R_s, R_c, R_{backbone, share}) = \min(50\text{ Mbps}, 90\text{ Mbps}, 30\text{ Mbps}) = \mathbf{30\text{ Mbps}}$$
- **Conclusion**: Even though the server can send at 50 Mbps and the client can receive at 90 Mbps, the **shared core backbone link** acts as the bottleneck.

#### Problem 2: MP3 File Download Time
A client downloads a $32{,}000{,}000\text{-bit}$ ($4\text{ Megabytes}$) MP3 audio file from a web server. The server's outbound access link rate is $R_s = 2\text{ Mbps}$, and the client's access link rate is $R_c = 1\text{ Mbps}$. Assume all network core links have massive Terabit bandwidth and ignore processing, queuing, and propagation delays.

Calculate the average download throughput and total time required to download the complete file.

**Solution**:
1. Bottleneck Throughput:
   $$\text{Throughput} = \min(R_s, R_c) = \min(2\text{ Mbps}, 1\text{ Mbps}) = \mathbf{1\text{ Mbps}} = 1{,}000{,}000\text{ bps}$$
2. Download Time:
   $$\text{Download Time} = \frac{\text{File Size}}{\text{Throughput}} = \frac{32{,}000{,}000\text{ bits}}{1{,}000{,}000\text{ bps}} = \mathbf{32\text{ seconds}}$$

---

### 9.4 Circuit Switching (TDM) — Transmission Time on Homogeneous & Heterogeneous Links

#### Case 1: Homogeneous Links
A user transmits a $640{,}000\text{-bit}$ file across a circuit-switched network using Time Division Multiplexing (TDM). Each link has a total bandwidth of $1.536\text{ Mbps}$ and employs a 24-slot TDM frame. Circuit setup signaling requires $500\text{ ms}$ ($0.5\text{ s}$) before transmission begins.
1. Calculate the dedicated transmission rate of the circuit.
2. Calculate the total time required to transfer the file.
3. How does this transmission time change if the path traverses 1 link versus 100 links?

**Solution**:
1. Circuit Transmission Rate:
   $$R_{circuit} = \frac{\text{Total Link Bandwidth}}{\text{Number of TDM Slots}} = \frac{1{,}536{,}000\text{ bps}}{24} = \mathbf{64{,}000\text{ bps}} = 64\text{ kbps}$$
2. Transmission Time:
   $$t_{trans} = \frac{640{,}000\text{ bits}}{64{,}000\text{ bps}} = \mathbf{10\text{ seconds}}$$
3. Total Time (including circuit setup):
   $$t_{total} = t_{trans} + t_{setup} = 10\text{ s} + 0.5\text{ s} = \mathbf{10.5\text{ seconds}}$$
4. **Hops Invariance**: In circuit switching with identical rates across all links, bits are placed directly into recurring slots and propagate continuously without intermediate store-and-forward buffering. Therefore, the 10-second transmission time is **independent of the number of links**.

#### Case 2: Heterogeneous Links (Variable Slot Rates)
Suppose the path traverses three links with differing link bandwidths ($R_i$) and differing TDM slot allocations ($n_i$):
- **Link 1**: $R_1 = 1.536\text{ Mbps}$, $n_1 = 24\text{ slots} \implies \text{Rate}_1 = 1536 / 24 = 64\text{ kbps}$
- **Link 2**: $R_2 = 2.048\text{ Mbps}$, $n_2 = 32\text{ slots} \implies \text{Rate}_2 = 2048 / 32 = 64\text{ kbps}$
- **Link 3**: $R_3 = 512\text{ kbps}$, $n_3 = 16\text{ slots} \implies \text{Rate}_3 = 512 / 16 = \mathbf{32\text{ kbps}}$

**Solution**:
- Effective End-to-End Rate = $\min(\text{Rate}_1, \text{Rate}_2, \text{Rate}_3) = \min(64, 64, 32) = \mathbf{32\text{ kbps}} = 32{,}000\text{ bps}$.
- New Transmission Time:
  $$t_{trans} = \frac{640{,}000\text{ bits}}{32{,}000\text{ bps}} = \mathbf{20\text{ seconds}}$$
- Total Time (with $0.5\text{ s}$ setup) = $20\text{ s} + 0.5\text{ s} = \mathbf{20.5\text{ seconds}}$.

---

### 9.5 Non-Persistent vs. Persistent HTTP — Round-Trip Time (RTT) Accounting

**Problem Statement**:
A web browser requests a web document from a web server. The document consists of a base HTML file referencing 10 distinct JPEG image files (11 objects total). All objects reside on the same server. Let $\text{RTT}$ denote the round-trip time between client and server. Neglect raw file transmission delays and server processing times.

Calculate the total delay in units of RTT under:
1. Non-Persistent HTTP without parallel connections.
2. Non-Persistent HTTP with up to 5 parallel TCP connections.
3. Persistent HTTP without pipelining.
4. Persistent HTTP with pipelining.

#### Step-by-Step Solution:

1. **Non-Persistent HTTP (Sequential / No Parallelism)**:
   - Base HTML: 1 RTT (TCP handshake) + 1 RTT (HTTP request/response) = 2 RTT.
   - Each of the 10 images requires its own separate TCP connection: $10 \times 2\text{ RTT} = 20\text{ RTT}$.
   - **Total Delay = $2\text{ RTT} + 20\text{ RTT} = \mathbf{22\text{ RTT}}$**.

2. **Non-Persistent HTTP (5 Parallel Connections)**:
   - Base HTML: 2 RTT.
   - 10 referenced images requested in parallel batches of 5:
     - Batch 1 (images 1–5 in parallel): 2 RTT.
     - Batch 2 (images 6–10 in parallel): 2 RTT.
   - **Total Delay = $2\text{ RTT} + 2\text{ RTT} + 2\text{ RTT} = \mathbf{6\text{ RTT}}$**.

3. **Persistent HTTP (Without Pipelining)**:
   - Base HTML: 1 RTT (TCP handshake) + 1 RTT (request/response) = 2 RTT.
   - The TCP connection remains open.
   - Each referenced image requires 1 RTT for its request/response: $10 \times 1\text{ RTT} = 10\text{ RTT}$.
   - **Total Delay = $2\text{ RTT} + 10\text{ RTT} = \mathbf{12\text{ RTT}}$**.

4. **Persistent HTTP (With Pipelining)**:
   - Base HTML: 1 RTT (TCP handshake) + 1 RTT (request/response) = 2 RTT.
   - As soon as HTML is parsed, client pipelines requests for all 10 images back-to-back into the open connection.
   - All 10 requests and responses return in parallel within 1 single round trip.
   - **Total Delay = $2\text{ RTT} + 1\text{ RTT} = \mathbf{3\text{ RTT}}$**.

---

### 9.6 Web Caching — Access Link Utilization and End-to-End Delay Reduction

**Problem Statement (Directly from Course Notes & Kurose & Ross)**:
An institutional network has an internal $1\text{ Gbps}$ Local Area Network (LAN) connected to the public Internet through a constrained $1.54\text{ Mbps}$ access link. The institutional users generate an average request rate of $15\text{ requests/second}$ for web objects averaging $100{,}000\text{ bits}$ ($100\text{ Kbits}$) in size. The average two-way Internet delay (from the institutional edge router to any origin server and back) is $2.0\text{ seconds}$. The LAN internal delay is negligible ($\approx 10\text{ ms} = 0.01\text{ s}$).

1. Calculate the required data rate on the access link and the link utilization **without a cache**.
2. Explain the impact on queuing delay and total end-to-end response time.
3. If the institution installs a local web proxy cache that achieves a **$40\%$ hit rate**, calculate the new data rate on the access link, the new link utilization, and the new weighted average end-to-end response time.

#### Step-by-Step Solution:

**Part 1: Without Web Caching**:
- Arrival data rate on the access link:
  $$\text{Data Rate} = 15\text{ requests/s} \times 100{,}000\text{ bits} = 1{,}500{,}000\text{ bps} = \mathbf{1.50\text{ Mbps}}$$
- Access Link Capacity = $1.54\text{ Mbps}$.
- Access Link Utilization:
  $$\text{Utilization} = \frac{1.50\text{ Mbps}}{1.54\text{ Mbps}} \approx \mathbf{0.974\ (97.4\%)}$$

**Part 2: Delay Impact Without Caching**:
- Because traffic intensity approaches unity ($I = 0.974 \approx 1$), queuing delay on the access link escalates asymptotically toward infinity. Packets experience severe buffer queuing and repeated drops. Total response time stretches into **minutes**, rendering web browsing effectively unusable.

**Part 3: With Web Caching ($40\%$ Cache Hit Rate)**:
- $40\%$ of requests are satisfied immediately from the local cache on the 1 Gbps LAN.
- Only the remaining $60\%$ ($1 - 0.40 = 0.60$) traverse the $1.54\text{ Mbps}$ access link.
- New Data Rate on Access Link:
  $$\text{New Data Rate} = 0.60 \times 1.50\text{ Mbps} = \mathbf{0.90\text{ Mbps}}$$
- New Access Link Utilization:
  $$\text{New Utilization} = \frac{0.90\text{ Mbps}}{1.54\text{ Mbps}} \approx \mathbf{0.584\ (58.4\%)}$$
- Because utilization drops from $97.4\%$ to a stable $58.4\%$, queuing delay on the access link becomes negligible ($\approx 0.01\text{ s}$).
- Delay for a Cache Hit = LAN delay $\approx 0.01\text{ s}$ ($10\text{ ms}$).
- Delay for a Cache Miss = Internet delay + LAN delay $\approx 2.0\text{ s} + 0.01\text{ s} = 2.01\text{ s}$.
- Weighted Average End-to-End Response Time:
  $$\text{Average Delay} = (0.40 \times 0.01\text{ s}) + (0.60 \times 2.01\text{ s}) = 0.004 + 1.206 = \mathbf{1.21\text{ seconds}}$$

> [!NOTE]
> **Key Exam Insight**: Installing a relatively inexpensive local cache reduces average delay from minutes down to **1.21 seconds**. Even if the institution had paid an ISP thousands of dollars to upgrade the physical access link to 100 Mbps (reducing access utilization to near zero), the average delay would still be **2.0 seconds** (the unavoidable Internet propagation delay). Thus, **caching produces a faster average user experience than purchasing raw bandwidth upgrades alone**.

---

### 9.7 Message Segmentation and Pipelining Benefit in Packet Switching

**Problem Statement**:
A host wishes to send a large file message of $M = 7{,}500{,}000\text{ bits}$ ($7.5\text{ Mbits}$) across a path consisting of $3$ links ($N = 3$) separated by $2$ intermediate routers. Each link has a transmission rate of $R = 1.5\text{ Mbps} = 1{,}500{,}000\text{ bps}$. Ignore propagation, processing, and queuing delays.

1. Calculate the total time to transmit the file if it is transmitted as **one single massive packet** without segmentation.
2. Calculate the total time to transmit the file if it is **segmented into 5000 small packets**, each of length $L = 1500\text{ bits}$.
3. Explain the mechanical reason why segmentation produces a dramatic speedup.

#### Step-by-Step Solution:

**1. Without Segmentation (One Massive Packet)**:
- Store-and-forward transmission requires the entire packet to be received at each router before forwarding.
- Time to transmit $M$ on Link 1:
  $$t_1 = \frac{M}{R} = \frac{7{,}500{,}000\text{ bits}}{1{,}500{,}000\text{ bps}} = 5\text{ seconds}$$
- Router 1 receives the packet at $t = 5\text{ s}$, and transmits it onto Link 2, finishing at $t = 10\text{ s}$.
- Router 2 receives the packet at $t = 10\text{ s}$, and transmits it onto Link 3, finishing at $t = 15\text{ s}$.
- Total Time:
  $$T_{unsegmented} = 3 \times \frac{M}{R} = 3 \times 5\text{ s} = \mathbf{15\text{ seconds}}$$

**2. With Segmentation ($5000\text{ Packets}$, $L = 1500\text{ bits}$)**:
- Time to transmit one small packet onto one link:
  $$t_{pkt} = \frac{L}{R} = \frac{1500\text{ bits}}{1{,}500{,}000\text{ bps}} = 0.001\text{ s} = 1\text{ ms}$$
- Time for the **first packet** to traverse all 3 links and arrive at the destination:
  $$t_{first} = 3 \times t_{pkt} = 3 \times 1\text{ ms} = 3\text{ ms}$$
- Because of **pipelining**, while Packet 2 is being transmitted on Link 2, Packet 3 is being transmitted on Link 1 simultaneously. Once the pipeline is primed, a new packet arrives at the destination **every $1\text{ ms}$**.
- The remaining $4999$ packets arrive at intervals of $1\text{ ms}$:
  $$T_{segmented} = t_{first} + (4999 \times t_{pkt}) = 3\text{ ms} + 4999\text{ ms} = 5002\text{ ms} = \mathbf{5.002\text{ seconds}}$$

**3. Architectural Explanation**:
Segmentation achieves a speedup from $15\text{ seconds}$ to **$5.002\text{ seconds}$** (roughly $3\times$ faster) because it enables **parallel pipelined transmission across multiple links**. In unsegmented transmission, intermediate links sit completely idle while the first link transmits. Segmentation maximizes link concurrency.

---

*End of Unit 1 Comprehensive Study Notes.*
