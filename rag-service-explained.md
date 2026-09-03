# RAG-as-a-Service — Explained Simply

*What this project is, and what each part of it does.*

---

## The short version

Imagine you have a big pile of documents — company files, manuals, papers, anything. You want to ask a question in normal language and get a real answer, based on what is actually written in those documents.

That is what this system does. You send one question to one web address, and you get back an answer that comes from your own documents.

The interesting part is **not** the AI models. We do not build or train any model. We download ready-made ones and use them as they are.

The real work is everything around them: running these programs on real servers, connecting them, making them fast, watching them, and keeping them alive when traffic gets heavy. That is the project.

---

## The big picture

The system is made of **four services** that talk to each other:

| Service | Job | Type of work |
|---|---|---|
| Embedding service | Turns text into numbers | GPU, no memory of the past |
| Vector database | Stores those numbers and searches them | Disk + RAM, keeps data |
| LLM service | Writes the final answer | GPU, no memory of the past |
| Gateway | The boss that connects all three | Light, just coordination |

Plus a fifth piece that watches everything: the **monitoring stack**.

The three main services do very different kinds of work. Two of them need a GPU and can be thrown away and restarted any time. One of them holds important data and must never lose it. Almost every hard problem in this project comes from that difference.

---

## 1. The Embedding Service

### What it does
It takes a piece of text and turns it into a long list of numbers — usually a few hundred numbers. This list is called a **vector**.

The useful trick: texts with similar meaning get similar numbers. "How do I reset my password?" and "I forgot my login details" end up close to each other, even though they share almost no words. This is how the system finds the right documents later.

We use a small, free, ready-made model for this (like `bge-base` or `all-MiniLM`).

### The hard part: batching
A GPU is like a big truck. Sending one text at a time is like driving the truck with one small box inside. The truck is barely used.

So the service waits a very short moment — maybe 5 to 20 milliseconds — collects all the texts that arrive in that window, and sends them through the GPU together in one go. This is called **dynamic batching**. Done right, it is the difference between handling tens of texts per second and hundreds, on the exact same hardware.

But there is a tension here. When we load documents in bulk, we want big batches, because we care about total speed. When a user is waiting for an answer, we want a short wait, because we care about their delay. The same setting cannot be perfect for both, so we have to tune it.

### The other hard part: slow start
The container for this service is big — several gigabytes, because it carries GPU libraries plus the model file. When a new copy starts, the web server is ready in a second, but the model is still loading. If we tell Kubernetes "ready" too early, it sends real traffic to a service that cannot answer yet.

So the health check must wait for the **model** to be loaded, not just the web server.

### What we measure
Requests per second, batch sizes, how many requests are waiting in line, GPU usage, and response time.

---

## 2. The Vector Database

### What it does
It stores all those number-lists from our documents, and answers one question very fast: *"Here is a new vector — give me the 5 stored ones closest to it."*

We plan to use **Qdrant** (other options: Weaviate, Milvus, pgvector).

### Why it is the scariest part
The two GPU services are disposable. Kill one, a new one starts, nothing is lost.

The database is different. **It holds the data.** If we set it up wrong on Kubernetes, the next time we update the system, the whole index quietly disappears. No error message — the data is just gone.

So it must run as a **StatefulSet** with a **PersistentVolume**, which is the Kubernetes way of saying "this service owns a disk that survives restarts." We also need backups, and later a plan for splitting the data across machines as it grows.

### Searching is a trade-off
Finding the *exactly* closest vectors in a huge collection is too slow. So we use an approximate method called **HNSW**, which finds *almost* the best answers, much faster.

It has settings that trade one good thing for another:
- Build-time settings decide how good and how heavy the index is.
- A search-time setting decides: search harder and slower, or faster and slightly less accurate?

Defaults are rarely right. These have to be tested against our own data.

### The slow leak
This is the component that gets slower **gradually** as the document collection grows. It is fine, fine, fine — and then one day it is the bottleneck. That is why we must watch the trend over weeks, not just check if it is healthy right now.

### What we measure
Search time, how many documents are stored, memory pressure, index size.

---

## 3. The LLM Service

### What it does
This is the part people think of as "the AI." It receives the user's question plus the document pieces we found, and writes the final answer in normal language.

We use **vLLM**, a specialized program for running language models efficiently, with a mid-size model (7–8 billion parameters) that fits on one graphics card.

### Hard part one: memory math
vLLM reserves a chunk of GPU memory in advance for its working memory (the "KV cache"). Getting this number right is fiddly:

- Too much → when several users arrive at once, the service runs out of memory and crashes.
- Too little → it can only handle a few users at a time, and speed collapses.

There is no formula. We find it by testing under real load.

### Hard part two: streaming
Answers should appear word by word, like a person typing, instead of the user staring at a blank screen for three seconds. This is nicer, but it makes the gateway more complicated, and it changes how we measure speed.

With streaming, "how long did it take?" splits into two different questions:
- **Time to first token** — how long until the first word appears? This is what the user *feels*.
- **Time per output token** — how fast do the following words come?

A slow answer that starts instantly feels better than a fast answer that starts after two seconds of nothing.

### Hard part three: knowing when to add more
The usual way to decide "we need another copy of this service" is to watch CPU usage. For this service, that is the wrong signal — the CPU can look calm while the GPU is completely full.

The right signals are: how many requests are being worked on, and how long they wait in line. vLLM already reports these, and we just have to connect them to the autoscaler.

### What we measure
Time to first token, speed per word, running and waiting requests, memory cache usage, GPU usage.

---

## 4. The Gateway

### What it does
This is the front door and the coordinator. It is the only part the user talks to. For every question it:

1. Asks the embedding service to turn the question into numbers.
2. Asks the database for the most relevant document pieces.
3. Builds one text prompt out of those pieces plus the question.
4. Asks the LLM to write the answer, and streams it back to the user.

### Why this is actually the hardest piece
This is where three separate programs become **one system** — and one system can fail in ways that no single program can.

**Things break one at a time.** The database might be healthy while the LLM is restarting. The embedder might be slow because of a sudden burst. The gateway has to handle each case: set a time limit for every step, retry when it makes sense, and return a clear error instead of leaving the user hanging forever.

**Building the prompt is real work.** The retrieved pieces may repeat each other. Their order matters. And there is a hard limit on how much text the model can accept, so we must fit inside that budget and decide what to cut.

**Following one request across three services.** When something is slow, we need to know *which* step was slow. So the gateway attaches an ID to each request and passes it along to every service. Later we can look at one request and see exactly where the time went.

### What we measure
Total time, time per step, error rate per step, retry counts.

---

## 5. The Monitoring Stack

This does not serve users. It tells us what is happening inside.

- **Prometheus** collects numbers from every service.
- **DCGM exporter** reports what the GPUs are really doing — usage, memory, temperature, power.
- **Grafana** turns all of it into dashboards we can look at.
- **OpenTelemetry** tracks single requests across all three services, so a 4-second answer can be split into "0.1s embedding, 0.05s search, 3.8s generating."
- **Alertmanager** sends a warning when something breaks our targets.

### Why this matters more than it sounds
Here is the core idea of the whole project:

> **The bottleneck moves.**

- When loading documents in bulk, the embedding service is the wall.
- When answering questions with long context, the LLM is the wall.
- As the collection grows, the database becomes the wall.

If you guess wrong, you spend money on an expensive GPU and the system does not get any faster. The monitoring exists so we can **prove** which part is the problem at any moment, instead of guessing.

---

## How one question travels through the system

1. You send a question to the gateway.
2. Gateway → embedding service → your question becomes numbers.
3. Gateway → database → the 5 most relevant document pieces come back.
4. Gateway builds a prompt from those pieces and your question.
5. Gateway → LLM service → the answer streams back to you word by word.

Your total wait is all of these added together, plus any queueing at each step. One slow part ruins the whole request. That is why we track each step separately.

There is also a second, very different path: **loading documents**. Files get split into pieces, all pieces get embedded in big batches, and the results are saved into the database. During this, the embedder and the database work hard while the LLM does nothing — the exact opposite of normal use.

---

## How each part grows under load

Each service gets more copies based on **its own** pressure signal, not one shared number:

| Service | Grows when… | How |
|---|---|---|
| Embedding | GPU is full or the queue is long | Add more copies |
| LLM | Too many requests in flight or waiting | Add more copies |
| Database | Memory pressure or slow searches | Bigger machine, then read copies / splitting |
| Gateway | High request rate | Add more copies (it is cheap) |

The database is the odd one out. The others are cheap to copy because they hold nothing. The database holds data, so adding copies means deciding how to split or duplicate that data — a much bigger decision.

---

## What "done" looks like

- One working API that answers questions from documents, with streaming.
- All parts running on Kubernetes, each growing on its own signal.
- We kill the database pod on purpose, and the data is still there afterwards.
- A written report: "we found the bottleneck here, we fixed it, here are the before and after numbers."
- A real cost figure: what does 1,000 questions cost, and why.
- A trace of one slow request showing exactly where the time went.

---

## The one-line summary

The models are ready-made parts we buy off the shelf. **The engineering — deploying them, connecting them, scaling them, watching them, and keeping them alive under real traffic — is the actual project.**
