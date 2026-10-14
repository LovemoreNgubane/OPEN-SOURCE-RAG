# Open-Source RAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system built entirely from open-source components. Local embeddings, self-hosted vector store, and support for Meta's Llama 3.2. No OpenAI, no paid APIs, no cloud lock-in.

Built as a submission for the Meta AI Academy South Africa Pitchathon and as the foundation of a data + AI platform for South African banks, telcos, and insurers.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Why This Project](#why-this-project)
- [Architecture](#architecture)
- [Stack](#stack)
- [Prerequisites](#prerequisites)
- [Setup - Step by Step](#setup---step-by-step)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Running the Pipeline](#running-the-pipeline)
- [Results](#results)
- [Next Steps](#next-steps)
- [License](#license)

---

## What is RAG?

Retrieval-Augmented Generation is a technique that lets a Large Language Model answer questions using your own data, not just what it was trained on.

Instead of fine-tuning an expensive model, you:

1. Split your documents into small chunks.
2. Convert each chunk into a vector (a list of numbers that captures meaning).
3. Store the vectors in a database that can search them by similarity.
4. When a user asks a question, convert the question to a vector, find the closest chunks, and feed them to the LLM as context.
5. The LLM answers using that context.

This reduces hallucination, keeps data private, and costs almost nothing.

---

## Why This Project

Enterprise RAG usually requires paid services: OpenAI for embeddings, Pinecone or Timescale for vectors, Anthropic or OpenAI for generation. This project replaces every paid component with an open-source equivalent that runs on a laptop and can scale to a Kubernetes cluster.

| Paid Service | Open-Source Replacement | Runs Locally |
|---|---|---|
| OpenAI text-embedding-3-small | sentence-transformers/all-MiniLM-L6-v2 (PyTorch) | Yes |
| Pinecone / Timescale Cloud | PostgreSQL 16 + pgvector | Yes (Docker) |
| OpenAI GPT-4 / Claude | Llama 3.2 via Ollama | Yes |
| Databricks / Snowflake | Apache Spark + Apache Iceberg | Yes (Docker) |

The entire stack runs offline, on CPU, with no API keys.

---

## Architecture
