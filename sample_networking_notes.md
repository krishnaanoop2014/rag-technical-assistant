# Networking Basics: TCP vs UDP

## TCP (Transmission Control Protocol)
TCP is a connection-oriented protocol. Before data is sent, TCP performs a
three-way handshake (SYN, SYN-ACK, ACK) to establish a reliable connection
between client and server. TCP guarantees ordered, error-checked delivery of
data and will retransmit lost packets. This reliability comes at the cost of
extra overhead and higher latency compared to UDP.

Common use cases: web browsing (HTTP/HTTPS), email (SMTP), file transfer (FTP).

## UDP (User Datagram Protocol)
UDP is a connectionless protocol. It sends packets ("datagrams") without
establishing a connection first, and it does not guarantee delivery, order,
or error correction. This makes UDP faster and lower-overhead than TCP, at
the cost of reliability.

Common use cases: video streaming, online gaming, DNS lookups, VoIP.

## Choosing between them
Use TCP when correctness and completeness of data matter more than speed
(e.g., transferring a file where every byte must arrive intact). Use UDP
when speed and low latency matter more than occasional data loss (e.g., a
live video call where a dropped frame is preferable to a delayed one).
