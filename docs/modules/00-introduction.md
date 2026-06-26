
# Introduction and ML Lifecycle

This course is built for learners starting from scratch and progressing to production
MLOps thinking. The goal is not only to define terms, but to build intuition for how
real ML systems are designed, shipped, and operated.

## Who this is for

- Beginners who know little or nothing about ML.
- Engineers who can code but need end-to-end ML platform understanding.
- Teams preparing to deploy Azure ML workloads in production.

## Learning outcomes

By the end of this module, you should be able to:

1. Explain the difference between AI, ML, and data science.
2. Identify major AI and ML categories.
3. Describe the Azure ML lifecycle from problem framing to monitoring.
4. Explain how a deployed model is exposed as an API/web service.

## AI vs ML vs Data Science

| Topic | What it is | Goal | Typical output |
|---|---|---|---|
| AI (Artificial Intelligence) | Broad field of building systems that perform tasks requiring human-like intelligence | Reason, plan, perceive, generate, decide | Intelligent behavior |
| ML (Machine Learning) | Subset of AI where systems learn patterns from data | Predict/estimate outcomes from examples | Trained model |
| Data Science | Interdisciplinary practice of extracting insight from data | Understand data and support decisions | Analysis, dashboards, models |

Key relationship for beginners:

- **AI** is the umbrella.
- **ML** is one major way to build AI systems.
- **Data science** uses statistics, ML, and domain knowledge to solve business problems
  and communicate insights.

In short: AI is the mission, ML is one method, and data science is the broader practice.

## Major AI categories

| Category | Description | Real-world examples |
|---|---|---|
| Symbolic / Rule-based AI | Explicit rules and logic created by humans | Expert systems, business rule engines |
| Machine Learning AI | Learns from data instead of hard-coded rules | Fraud scoring, demand forecasting |
| Generative AI | Learns to generate new content | Text generation, image generation, code assistants |
| Classical Search/Planning | Finds actions to optimize a goal | Route planning, scheduling |

Practical note: many enterprise solutions combine categories. Example: a fraud system can
use supervised ML scoring plus rule-based guardrails.

## Types of ML at a glance

| Type | Data requirement | Typical task |
|---|---|---|
| Supervised learning | Labeled data $(X, y)$ | Classification, regression |
| Unsupervised learning | Unlabeled data $X$ | Clustering, anomaly detection |
| Reinforcement learning | Environment + reward signal | Sequential decision/control |
| Semi-supervised learning | Small labeled + large unlabeled | Classification with sparse labels |
| Self-supervised learning | Labels generated from data itself | Representation learning (NLP/CV) |

## Common confusion points

- A model can be "accurate" but still unusable if latency is too high.
- A model can be statistically strong but fail fairness/compliance checks.
- A model is not a product by itself; the surrounding data and ops system matters.
- Deep learning is a subset of ML, not a separate thing; it uses neural networks with many layers.
- "Training" a model means finding parameter values that minimise a loss function on data â€” not teaching in the human sense.

## Real-world example: e-commerce recommendation

To make this concrete, here is how the full technology stack maps to a product:

| Concern | Technology choice | ML lifecycle stage |
|---|---|---|
| Collect user events | Event streaming (Kafka, Event Hub) | Data ingestion |
| Store features | Feature store or Azure Data Lake | Data preparation |
| Train model | Azure ML training job | Training |
| Serve recommendations | Online endpoint (AKS) | Deployment |
| Detect stale model | Azure ML data drift monitor | Monitoring |

The model is one component. The pipeline around it is what makes it reliable.

## Why Azure ML matters

Azure Machine Learning gives you the managed platform to run the full lifecycle with
reproducibility and governance: versioned data/model assets, tracked runs, deployment
endpoints, and monitoring.

Azure Machine Learning organizes the end-to-end lifecycle:

1. Problem framing
2. Data preparation
3. Training and validation
4. Model registration
5. Deployment
6. Monitoring and retraining

This is not a linear path. Production systems continuously loop from monitoring back to
data and training when model quality or data distributions change.

```mermaid
flowchart LR
  A[Problem Framing] --> B[Data Preparation]
  B --> C[Training and Validation]
  C --> D[Model Registration]
  D --> E[Deployment]
  E --> F[Monitoring]
  F --> C
```

### What each stage does

| Stage | Main question | Key output |
|---|---|---|
| Problem framing | What decision are we trying to improve? | Business KPI definition, success criteria |
| Data preparation | Do we trust the data and labels? | Validated, versioned dataset |
| Training | Which model learns the signal best? | Candidate models with tracked metrics |
| Registration | Is the artifact versioned and reproducible? | Registered model with lineage |
| Deployment | Can consumers call this model safely? | Live endpoint with auth and monitoring |
| Monitoring | Is quality stable over time in production? | Drift and quality alerts, retraining signals |

Note: Stage 1 (problem framing) is often underinvested. The single most common reason ML projects fail is a poorly defined business objective, not a weak model.

![ML infrastructure tools for production](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml-infrastructure-tools-for-production.png)

> Image explanation: This visual shows ml infrastructure tools for production. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![ML workflow stages](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml_workflow_stages.png)

> Image explanation: This visual shows ml workflow stages. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Overview ML flow](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Overview_ML_flow.png)

> Image explanation: This visual shows overview ml flow. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Web Service vs API

- A deployed Azure ML model is typically exposed as a REST API endpoint.
- In practice, teams often say "web service" for the deployed scoring interface.

Practical distinction:

- **API** describes the contract (request/response schema, authentication, versioning).
- **Web service** is the hosted implementation of that API.
- In Azure ML online endpoints, you design the API contract through the scoring payload
  and endpoint auth, and Azure hosts the service.

### Inference request flow (simple)

1. Client sends JSON payload to endpoint URI.
2. Endpoint auth validates identity/key.
3. Scoring script parses input and runs model.
4. API returns prediction response + metadata.

### Inference request flow (detailed)

```mermaid
sequenceDiagram
  participant C as Client App
  participant G as API Gateway / Auth
  participant E as Scoring Endpoint
  participant M as Model
  participant L as Logging
  C->>G: POST /score {features: [...]}
  G->>G: Validate auth token / key
  G->>E: Forward request
  E->>E: Deserialize and validate schema
  E->>M: Run model.predict(features)
  M-->>E: Return raw prediction
  E->>E: Post-process / format response
  E->>L: Log input hash + prediction
  E-->>C: Return {prediction, confidence}
```

Key production considerations for this flow:

- **Schema validation in the scoring script** protects against unexpected input shapes.
- **Logging input hashes** (not raw PII) enables later drift analysis and auditing.
- **Timeouts and retries** must be defined at both the gateway and client layers to avoid silent failures.

![Web service vs API](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/webservice_vs_api.png)

> Image explanation: This visual shows web service vs api. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Web service vs API table](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/table_webservice_vs_api.png)

> Image explanation: This visual shows web service vs api table. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Quick self-check

1. Is every AI system an ML system?
2. In production, which stage catches drift issues?
3. What is the difference between API and web service?

