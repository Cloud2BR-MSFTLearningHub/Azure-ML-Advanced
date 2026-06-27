
# Azure Machine Learning Training Hub

This training hub covers Azure Machine Learning from fundamentals to production
operations, with formulas, architecture, deployment guidance, and visual references.

Progression model:

- Beginner: understand AI/ML fundamentals and platform basics.
- Intermediate: build data, training, and evaluation workflows.
- Advanced: deploy, monitor, debug, and govern production ML systems.

## Evolution of Machine Learning

Machine learning as a scientific field started in the 1950s, then moved through
multiple eras based on compute power, data availability, and algorithmic advances.

![Machine Learning evolution timeline](assets/img/ml-evolution-timeline.svg)

> **Note - How to read this timeline:** Each colored band is the *dominant paradigm* of an era, and the dots on the axis are the
> breakthroughs that triggered the next shift. Read it to see why modern Azure ML must support
> *both* classic ML (still best for tabular business data) and deep/foundation models (best for
> unstructured data) : the platform spans the whole history, not just the latest era.

### Milestone eras

1. Foundations (1950s-1970s): Turing test ideas, perceptrons, nearest-neighbor methods.
2. Expert systems era (1980s): rule-based AI in enterprise workflows.
3. Statistical ML era (1990s-2000s): SVMs, random forests, probabilistic modeling.
4. Deep learning era (2010s): GPUs and large datasets enabled deep neural networks.
5. Foundation model era (2020s+): transformers, large language models, multimodal AI.

### Why this matters for Azure ML learners

- It explains why modern MLOps includes both classic ML and deep learning workflows.
- It clarifies when simpler models can outperform larger neural models in tabular data.
- It frames current production needs: monitoring, governance, safety, and cost control.

## Learning Path

<div class="home-grid">
	<a class="home-card" href="modules/01-math-prerequisites/">
		<span class="card-badge">Module 01</span>
		<h3>Math Prerequisites</h3>
		<p>Probability, linear algebra, calculus, and statistics foundations needed for all ML math.</p>
	</a>
	<a class="home-card" href="modules/02-e2e-overview/">
		<span class="card-badge">Module 02</span>
		<h3>End-to-End ML Overview</h3>
		<p>A map of the whole ML workflow and how every stage connects, from problem framing to production monitoring.</p>
	</a>
	<a class="home-card" href="modules/03-introduction/">
		<span class="card-badge">Module 03</span>
		<h3>Introduction and Lifecycle</h3>
		<p>AI vs ML vs data science, AI categories, and end-to-end Azure ML lifecycle.</p>
	</a>
	<a class="home-card" href="modules/04-ml-foundations/">
		<span class="card-badge">Module 04</span>
		<h3>ML Foundations</h3>
		<p>Full ML taxonomy: supervised, unsupervised, RL, semi/self-supervised, with formulas and selection guidance.</p>
	</a>
	<a class="home-card" href="modules/05-neural-networks/">
		<span class="card-badge">Module 05</span>
		<h3>Neural Networks and Deep Learning</h3>
		<p>Perceptron to Transformers: backpropagation, CNNs, RNNs, attention, and training at scale.</p>
	</a>
	<a class="home-card" href="modules/06-azure-ml-environment/">
		<span class="card-badge">Module 06</span>
		<h3>Azure ML Environment</h3>
		<p>Workspace taxonomy, compute types, model registry, and endpoints.</p>
	</a>
	<a class="home-card" href="modules/07-environment-setup/">
		<span class="card-badge">Module 07</span>
		<h3>Environment Setup</h3>
		<p>Conda/pip setup, package validation, and runtime consistency.</p>
	</a>
	<a class="home-card" href="modules/08-data-preparation/">
		<span class="card-badge">Module 08</span>
		<h3>Data Preparation</h3>
		<p>Data collection, cleaning, schema handling, and split strategy.</p>
	</a>
	<a class="home-card" href="modules/09-model-types/">
		<span class="card-badge">Module 09</span>
		<h3>Model Types</h3>
		<p>Algorithm families with representative mathematical formulations.</p>
	</a>
	<a class="home-card" href="modules/10-training-automl/">
		<span class="card-badge">Module 10</span>
		<h3>Training and AutoML</h3>
		<p>AutoML search flow, compute choices, and practical training pipeline.</p>
	</a>
	<a class="home-card" href="modules/11-performance-metrics/">
		<span class="card-badge">Module 11</span>
		<h3>Performance Metrics</h3>
		<p>Classification and regression metrics, formulas, and interpretation.</p>
	</a>
	<a class="home-card" href="modules/12-results-explainability/">
		<span class="card-badge">Module 12</span>
		<h3>Results and Explainability</h3>
		<p>Result analysis, drift detection, and explainability methods.</p>
	</a>
	<a class="home-card" href="modules/13-deployment/">
		<span class="card-badge">Module 13</span>
		<h3>Deployment</h3>
		<p>Registration, scoring, endpoint deployment, and serving patterns.</p>
	</a>
	<a class="home-card" href="modules/14-deployment-debug-k8s/">
		<span class="card-badge">Module 14</span>
		<h3>Deployment Debugging</h3>
		<p>Kubernetes-focused troubleshooting for production endpoint issues.</p>
	</a>
</div>

## Reference

<div class="home-grid">
	<a class="home-card" href="reference/">
		<span class="card-badge">Ref 01</span>
		<h3>Reference Home</h3>
		<p>Supporting material for implementation and operations.</p>
	</a>
	<a class="home-card" href="reference/model-math-deep-dive/">
		<span class="card-badge">Ref 02</span>
		<h3>Mathematical Model Deep Dive</h3>
		<p>Formula-level explanations, objectives, assumptions, and trade-offs for core ML models.</p>
	</a>
	<a class="home-card" href="reference/cli-commands/">
		<span class="card-badge">Ref 03</span>
		<h3>CLI Commands</h3>
		<p>Command-line references for setup, run, and deployment tasks.</p>
	</a>
	<a class="home-card" href="reference/glossary/">
		<span class="card-badge">Ref 04</span>
		<h3>Glossary</h3>
		<p>Core Azure ML and MLOps terms used throughout this training.</p>
	</a>
</div>

