# Grounding comparison

UTC 20260905-124600 | repeats per arm: 2


# ===== TUTORING: questions the check FIRES on =====

4 of 10 questions.


## [TUTORING-FIRES] Q1: What's the latest thinking on whether chain-of-thought prompting actually helps?

### grounding ON — run 1

- ok: `True` | attempts: 1 | 21812 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `latest`
- sources: ['https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/', 'https://techsy.io/en/blog/chain-of-thought-prompting', 'https://www.ibm.com/think/topics/chain-of-thoughts']

> The latest research shows that chain-of-thought (CoT) prompting still generally boosts performance for non-reasoning AI models, but its effectiveness varies widely depending on the specific model you use. For instance, a tech report from the Wharton School found that CoT prompting led to significant accuracy gains for models like Gemini Flash 2.0 (a 13.5% improvement) and Claude 3.5 Sonnet (an 11.7% improvement), but had a negligible, statistically insignificant impact on others like GPT-4o-mini (https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/). Interestingly, they also discovered that the exact phrasing of your CoT prompt doesn't matter much; simply removing restrictive formatting constraints often allows models to naturally generate their own step-by-step reasoning. At the same time, you have to be careful, as CoT can quietly degrade performance in some models while helping others (https://techsy.io/en/blog/chain-of-thought-prompting). 

Ultimately, the current best practice is to test CoT directly on your specific model and task to ensure the boost in accuracy is worth the extra latency and token cost.

### grounding ON — run 2

- ok: `True` | attempts: 1 | 26922 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `latest`
- sources: ['https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/', 'https://techsy.io/en/blog/chain-of-thought-prompting', 'https://www.ibm.com/think/topics/chain-of-thoughts']

> The latest consensus is that chain-of-thought (CoT) prompting still provides a significant boost to many non-reasoning models, but its effectiveness depends heavily on the specific model you are using and the task at hand. According to a research report from the Wharton School published in April 2026, CoT prompting led to notable performance improvements in models like Gemini Flash 2.0 (a 13.5% increase) and Anthropic's Sonnet 3.5 (an 11.7% increase), though it had a negligible, statistically insignificant effect on others like GPT-4o-mini (https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/). 

Interestingly, the same Wharton study revealed that the specific way you write your CoT prompt doesn't matter as much as we once thought; different prompt variations yielded negligible differences in performance. In fact, simply removing strict formatting constraints and letting a model naturally output a "default" thinking process before answering is often highly effective. However, you do need to exercise caution. Other industry analysis from 2026 notes that while CoT is highly beneficial for breaking down complex reasoning tasks, it can actually quietly break or degrade performance in certain other models (https://techsy.io/en/blog/chain-of-thought-prompting). 

The current best practice is to avoid treating CoT as a universal fix. Instead, you should test it specifically on your target model and task, weighing the accuracy gains against the extra time and token costs that come with generating step-by-step reasoning.

### grounding OFF — run 1

- ok: `True` | attempts: 1 | 39750 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> The latest research shows that chain-of-thought prompting is incredibly powerful but highly context-dependent. It significantly boosts performance on complex reasoning, mathematics, and symbolic logic tasks where a model needs to break down a problem. However, for simpler tasks like factual retrieval, basic classification, or creative writing, it often adds unnecessary latency and cost, and can sometimes even introduce errors or "hallucinations" in the reasoning chain that lead to a wrong final answer.

An important shift in the current thinking revolves around "faithfulness." Researchers have found that a model's written chain of thought does not always reflect its actual internal calculation. Sometimes, a model decides on an answer first and then writes a plausible-looking explanation to justify it, or it might make a mistake in its reasoning chain but still happen to arrive at the correct answer. This means we cannot always take the model's self-explanation at face value as a true representation of its cognitive process.

Furthermore, with the rise of newer, reasoning-focused models like OpenAI's o1 series, the need for users to manually prompt for chain-of-thought is decreasing. These models are trained to perform this structured reasoning natively and implicitly behind the scenes, making manual "let's think step-by-step" prompts redundant for state-of-the-art systems. Essentially, if you are working with standard models on complex logic, it is still one of the best tools available, but it is no longer seen as a universal default for every task.

### grounding OFF — run 2

- ok: `True` | attempts: 1 | 18453 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> Chain-of-thought prompting definitely helps, but the latest research shows its benefits are highly dependent on the model's size and the complexity of the task, and there is growing skepticism about whether the model's written "thoughts" represent its actual reasoning process. 

For complex, multi-step challenges like math word problems, symbolic logic, or dense coding tasks, giving a model the space to write out its reasoning step-by-step consistently boosts accuracy. This works because it allows the model to dedicate computational power to one sub-step at a time. However, for simple tasks that require straightforward retrieval or intuition, chain-of-thought is actually counterproductive, as it wastes time and tokens without improving accuracy, and can even introduce new errors.

The most significant shift in recent thinking focuses on "faithfulness." Several prominent studies have revealed that the reasoning steps a model writes down are often post-hoc justifications rather than the actual path the model took to reach the answer. In other words, a model might decide on a biased or incorrect answer first, and then write a highly convincing, logical-looking explanation to justify that predetermined conclusion. This means we cannot always trust the generated chain of thought to be a true reflection of the model's internal processing.

Because of these limitations, the frontier of AI research is shifting away from simply prompting a model to "think step-by-step." Instead, developers are training models using reinforcement learning to internalize reasoning pathways, as seen in newer reasoning-focused models. This approach encourages genuine error correction and systematic thinking before the model outputs its final response.


## [TUTORING-FIRES] Q2: What's the best way to serve an ML model in production these days?

### grounding ON — run 1

- ok: `True` | attempts: 1 | 12984 ms | 3 calls `{'AssessAndBranch': 1, 'EmbedAndSearchFacts': 1, 'DisambiguationOptions': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> Which of these did you mean?

### grounding ON — run 2

- ok: `True` | attempts: 1 | 76094 ms | 3 calls `{'AssessAndBranch': 1, 'EmbedAndSearchFacts': 1, 'DisambiguationOptions': 1}`
- fired: `False` | marker: `None`
- sources: (none)
- warnings: ["disambiguation_typed_past: student typed past the prior turn's options -- superseded, threaded as context into this turn's AssessAndBranch instead of being matched against"]

> Which of these did you mean?

### grounding OFF — run 1

- ok: `True` | attempts: 1 | 48406 ms | 3 calls `{'AssessAndBranch': 1, 'EmbedAndSearchFacts': 1, 'DisambiguationOptions': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> Which of these did you mean?

### grounding OFF — run 2

- ok: `True` | attempts: 1 | 41219 ms | 3 calls `{'AssessAndBranch': 1, 'EmbedAndSearchFacts': 1, 'DisambiguationOptions': 1}`
- fired: `False` | marker: `None`
- sources: (none)
- warnings: ["disambiguation_typed_past: student typed past the prior turn's options -- superseded, threaded as context into this turn's AssessAndBranch instead of being matched against"]

> Which of these did you mean?


## [TUTORING-FIRES] Q3: How do teams usually do experiment tracking in ML projects nowadays?

### grounding ON — run 1

- ok: `True` | attempts: 1 | 29453 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `nowadays`
- sources: ['https://www.trail-ml.com/blog/first-steps-experiment-tracking', 'https://launchdarkly.com/blog/ml-experiment-tracking/', 'https://viso.ai/deep-learning/experiment-tracking/']
- warnings: ["disambiguation_typed_past: student typed past the prior turn's options -- superseded, threaded as context into this turn's AssessAndBranch instead of being matched against"]

> Modern machine learning teams typically handle experiment tracking by using dedicated MLOps software platforms, moving away from manual methods to automate the capture of run-level metadata. When a developer runs a training job, these tracking tools automatically log hyperparameters, performance metrics, the exact code state, environment configurations, and the specific datasets used, answering the fundamental question of what was done and what results were achieved (https://launchdarkly.com/blog/ml-experiment-tracking/). 

The primary goal of this process is to ensure reproducibility. By keeping an exact record of every experiment, data scientists can easily look back at past runs, reconstruct the precise conditions under which a model was trained, and iterate from there with confidence (https://www.trail-ml.com/blog/first-steps-experiment-tracking/). While very small teams or early-stage projects sometimes start with basic manual tracking in spreadsheets or simple version control, scaling up to complex models—including modern large language models that require prompt and configuration versioning—almost always requires adopting automated, specialized tracking platforms to prevent operational gaps (https://viso.ai/deep-learning/experiment-tracking/, https://launchdarkly.com/blog/ml-experiment-tracking/).

### grounding ON — run 2

- ok: `True` | attempts: 1 | 41766 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `nowadays`
- sources: ['https://www.trail-ml.com/blog/first-steps-experiment-tracking', 'https://launchdarkly.com/blog/ml-experiment-tracking/', 'https://viso.ai/deep-learning/experiment-tracking/']

> Nowadays, machine learning teams typically rely on dedicated, automated experiment tracking tools to manage their workflows rather than manual spreadsheets, which quickly become disorganized. These platforms integrate directly into the training code to capture "run-level" details automatically every time a model is run. This includes logging the exact hyperparameters, performance metrics, training data versions, and code states used (https://launchdarkly.com/blog/ml-experiment-tracking/). Having this centralized, automated record is essential for reproducibility, allowing team members to easily look back at past trials, compare results, and reconstruct the exact environment to build upon previous work (https://www.trail-ml.com/blog/first-steps-experiment-tracking/). While very small teams or early-stage research projects might still use simpler methods like software version control repositories or basic manual spreadsheets, scaling up to production almost always requires these specialized MLOps tools to manage the sheer volume of experiment metadata (https://viso.ai/deep-learning/experiment-tracking/). Let me know if you would like to explore how to set up one of these trackers in a basic Python script.

### grounding OFF — run 1

- ok: `True` | attempts: 1 | 22188 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)
- warnings: ["disambiguation_typed_past: student typed past the prior turn's options -- superseded, threaded as context into this turn's AssessAndBranch instead of being matched against"]

> Most machine learning teams today have moved away from manual spreadsheets and instead rely on dedicated MLOps platforms to automatically track their experiments. The industry standards right now are tools like Weights & Biases, MLflow, and Comet. These platforms act as a centralized repository where every run of a training script automatically logs hyperparameters, training metrics like loss and accuracy, system metrics like GPU usage, and even the final model weights and dataset versions.

To set this up, a developer usually just imports the tool's library into their Python code, initializes a run, and uses API calls to log key variables. This is incredibly helpful because it allows team members to compare different runs side-by-side on interactive dashboards, share results instantly, and reproduce any specific model version later on.

For larger enterprises or teams working within specific cloud ecosystems, experiment tracking is often integrated directly into broader platforms like Amazon SageMaker or Azure Machine Learning, or managed self-hosted on Kubernetes. Regardless of the specific software, the core practice revolves around automated logging, centralized visualization, and strict version control for both code and data. Let me know if you would like to see a simple code example of how one of these tools integrates into a training script.

### grounding OFF — run 2

- ok: `True` | attempts: 1 | 27687 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> Most machine learning teams nowadays rely on dedicated, automated experiment tracking platforms rather than manual spreadsheets or local text files. The industry standard is to use specialized libraries like MLflow, Weights & Biases, Comet, or Neptune, which are integrated directly into the training code with just a few lines of Python. As the training script runs, these libraries automatically log hyperparameters, loss metrics, validation scores, model architectures, and even hardware utilization to a centralized, web-based dashboard. This setup allows anyone on the team to visually compare different training runs side-by-side, collaborate on results, and instantly identify which configuration yielded the best model. To ensure complete reproducibility, teams usually pair these tracking tools with data versioning systems like DVC and Git to tie every logged experiment back to the exact dataset version and code commit that produced it. We can look at how to set up a simple tracking integration in code, or dive into the differences between some of these specific platforms if you want to explore further.


## [TUTORING-FIRES] Q4: What Python version should I target for a new project in 2026?

### grounding ON — run 1

- ok: `True` | attempts: 1 | 44813 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `2026`
- sources: ['https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/', 'https://devguide.python.org/versions/', 'https://apxml.com/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/choosing-python-version']

> For a new project in 2026, you should target Python 3.14.x. This is the latest stable release, making it the ideal choice for most new developments because it offers the newest language features, active support, and excellent compatibility with modern libraries ([1](https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/)).

Choosing the latest stable version ensures your project begins on a foundation that will receive active bug and security fixes for years to come. In Python's official release cycle, once a version is fully released, it enters a maintenance phase where binaries and fixes are regularly updated to keep the environment secure and reliable ([2](https://devguide.python.org/versions/)). If your project requires niche, legacy third-party packages that have not yet updated their dependencies, you could consider Python 3.13 as a fallback, but targeting Python 3.14 is the best way to future-proof your code. Let me know what kind of application you are planning to build, and we can look at the best tools to manage your dependencies.

### grounding ON — run 2

- ok: `True` | attempts: 1 | 33891 ms | 6 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1, 'GroundTimeSensitive': 1}`
- fired: `True` | marker: `2026`
- sources: ['https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/', 'https://devguide.python.org/versions/', 'https://apxml.com/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/choosing-python-version']

> For a new project in 2026, you should target **Python 3.14**. This is the latest stable release, offering the best mix of modern language features, performance improvements, strong library compatibility, and straightforward installation on Windows and macOS (https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/). Targeting Python 3.14 ensures your project benefits from the newest capabilities while enjoying active maintenance and support.

According to the official Python developer guide, once a version is fully released as a stable version, it enters a "bugfix" or maintenance phase where it receives regular bug and security updates roughly every two months (https://devguide.python.org/versions/). Eventually, older versions transition to a security-only phase where no new binaries are released, and finally reach end-of-life five years after their initial release (https://devguide.python.org/versions/). Choosing Python 3.14 gives your new project the longest possible support window and keeps you aligned with the active ecosystem.

### grounding OFF — run 1

- ok: `True` | attempts: 1 | 35828 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> You should target Python 3.14 for a new project in 2026, using Python 3.13 as a safe fallback if your specific dependencies have not yet updated. Python 3.14, which released in late 2025, represents the ideal balance of modern language features, performance improvements, and ecosystem readiness for the year 2026. If you are starting your project late in 2026, Python 3.15 will be newly released, but it is usually best to wait a few months for major libraries to fully support it before adopting it in production.

Sticking to Python 3.13 or 3.14 ensures you benefit from major recent performance boosts, such as the specializing adaptive interpreter, the newer Just-In-Time compiler, and experimental free-threaded execution options. It also guarantees that your project remains securely supported for several years, as the Python core team maintains each version with security updates for five years after its initial release.

### grounding OFF — run 2

- ok: `True` | attempts: 1 | 61953 ms | 5 calls `{'FinalAnswer': 1, 'AssessAndBranch': 1, 'WriteLearnerFact': 2, 'EmbedAndSearchFacts': 1}`
- fired: `False` | marker: `None`
- sources: (none)

> You should target Python 3.13 as your primary baseline for a new project in 2026, though Python 3.14 is an excellent alternative if you want the absolute latest stable features. 

By 2026, Python 3.13 will be highly mature and virtually all major third-party libraries and frameworks will have robust, stable support for it. This version introduces major milestones, such as significant performance enhancements, much improved error traceback messages, and the experimental free-threaded build that allows you to run Python without the Global Interpreter Lock. This makes 3.13 the sweet spot for balancing cutting-edge features, speed, and ecosystem compatibility.

If you want to be on the leading edge and your specific library dependencies support it, Python 3.14—which releases in late 2025—is also a great choice. However, you should avoid targeting anything older than Python 3.12. Older versions like Python 3.10 will reach their official end-of-life in late 2026, meaning they will no longer receive security patches. Starting your project with Python 3.13 or 3.14 ensures your codebase remains modern, secure, and maintainable for years to come.


# COST SUMMARY — TUTORING-FIRES

| Q | fired | on ms (r1/r2) | off ms (r1/r2) | on calls | off calls |
|---|---|---|---|---|---|
| 1 | True | 21812/26922 | 39750/18453 | 6/6 | 5/5 |
| 2 | False | 12984/76094 | 48406/41219 | 3/3 | 3/3 |
| 3 | True | 29453/41766 | 22188/27687 | 6/6 | 5/5 |
| 4 | True | 44813/33891 | 35828/61953 | 6/6 | 5/5 |

mean latency delta on firing questions: -1200 ms (NOTE: end-to-end, includes Gemini variance/retries — see the isolated search-latency measurement for the clean number)


# EXCERPT STABILITY — TUTORING-FIRES

Did the two identical grounded runs retrieve the same sources? If not, a single-run verdict is noise.

| Q | run1 sources | run2 sources | identical |
|---|---|---|---|
| 1 | ['https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/', 'https://techsy.io/en/blog/chain-of-thought-prompting', 'https://www.ibm.com/think/topics/chain-of-thoughts'] | ['https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/', 'https://techsy.io/en/blog/chain-of-thought-prompting', 'https://www.ibm.com/think/topics/chain-of-thoughts'] | True |
| 2 | (none) | (none) | True |
| 3 | ['https://www.trail-ml.com/blog/first-steps-experiment-tracking', 'https://launchdarkly.com/blog/ml-experiment-tracking/', 'https://viso.ai/deep-learning/experiment-tracking/'] | ['https://www.trail-ml.com/blog/first-steps-experiment-tracking', 'https://launchdarkly.com/blog/ml-experiment-tracking/', 'https://viso.ai/deep-learning/experiment-tracking/'] | True |
| 4 | ['https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/', 'https://devguide.python.org/versions/', 'https://apxml.com/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/choosing-python-version'] | ['https://www.krishnamohanproductions.com/blog/tech-blog/which-python-version-should-i-use-2026/', 'https://devguide.python.org/versions/', 'https://apxml.com/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/choosing-python-version'] | True |


# ===== TUTORING: questions the check MISSES =====

6 of 10 questions. Both arms are identical here by construction, so this is ONE run each: what the student actually gets today, ungrounded. Read these for staleness — that is the real cost of the miss rate.


## [TUTORING-MISSED] Q1: What's the current standard approach to fine-tuning a large language model on a small dataset?

- ok: `True` | 26609 ms | 5 calls | fired: `False`

> The current standard approach is Parameter-Efficient Fine-Tuning (PEFT), specifically using a technique called Low-Rank Adaptation, or LoRA. When you are working with a small dataset, training all of a model's billions of parameters is highly likely to cause overfitting, where the model memorizes your specific data but loses its general reasoning capabilities. LoRA avoids this by freezing the original model weights entirely and only training a tiny, auxiliary set of adapter weights inserted into the model's layers. This drastically reduces the memory footprint and prevents the model from forgetting its pre-trained knowledge.

To make this process even more accessible, developers frequently use QLoRA, which quantizes the base model to 4-bit precision to save memory while still using LoRA adapters for the actual training. Alongside this technical setup, the standard practice for small datasets heavily emphasizes data quality over quantity. Because the model has fewer examples to learn from, each example must be clean, accurately labeled, and formatted consistently using a clear prompt template. When running the training, you should use a very low learning rate, moderate dropout, and a small number of epochs to gently steer the model toward your desired output style without degrading its core capabilities. 

Would you like to explore how LoRA mathematically adapts those weights, or should we look at how to structure your specific dataset for this process?


## [TUTORING-MISSED] Q2: Is dropout still recommended in transformer architectures?

- ok: `True` | 24875 ms | 5 calls | fired: `False`

> Yes, dropout is still recommended and used in transformer architectures, but its application has shifted significantly depending on the scale of your model and the size of your dataset. 

In the original Transformer paper, dropout was applied liberally—usually at a rate of 10%—to the outputs of every sub-layer, including the attention weights and the feed-forward networks, to prevent overfitting. This remains the standard practice if you are training smaller transformer models from scratch on modest datasets.

However, in modern, massive Large Language Models (LLMs) like GPT-3 or LLaMA, dropout is often set to zero during the initial pre-training phase. Since these models are trained on trillions of tokens of data, they are highly unlikely to overfit on a single pass through the dataset. Disabling dropout also speeds up the training process and reduces memory consumption, which is a massive benefit when training at scale. 

Despite this shift in pre-training, dropout is still highly recommended when you are fine-tuning these large models on smaller, specialized datasets. In those scenarios, the risk of overfitting returns, and applying a small amount of dropout helps the model generalize better to your specific task. If you are working on a project, deciding whether to use dropout will largely depend on whether you are pre-training a large model or fine-tuning an existing one on a limited dataset.


## [TUTORING-MISSED] Q3: How do people handle really long context windows in LLMs now?

- ok: `True` | 11515 ms | 3 calls | fired: `False`

> Which of these did you mean?


## [TUTORING-MISSED] Q4: Is RAG still the standard way to give a model access to private documents?

- ok: `True` | 25266 ms | 5 calls | fired: `False`

> Yes, Retrieval-Augmented Generation (RAG) remains the industry standard for giving models access to private documents. While there have been major advancements in other areas—most notably the rise of models with massive context windows that can hold hundreds of pages of text at once—RAG is still the most practical, secure, and cost-effective approach for the vast majority of applications.

The main reason RAG holds its crown is efficiency. Loading millions of words directly into a model's active memory for every single query is incredibly expensive and slow. RAG solves this by acting like a smart search engine: it quickly finds only the few relevant paragraphs from your private database and feeds just those to the model. This keeps API costs down and response times fast. Furthermore, RAG makes it easy to update your data instantly by simply adding or deleting files from your database. In contrast, fine-tuning a model on private data is expensive, doesn't guarantee the model will actually remember the facts, and risks leaking sensitive information.

Recently, we have seen RAG evolve rather than get replaced. Developers are now using advanced techniques like "GraphRAG," which maps out relationships between concepts in your documents, and "Agentic RAG," where the model decides for itself how and where to search for information. 

If you are designing a system to query private data, starting with a RAG pipeline is still the best approach. Are you currently planning a specific project, or are you trying to decide between RAG and another method like fine-tuning?


## [TUTORING-MISSED] Q5: Should I still be using pandas for dataframes, or has something replaced it?

- ok: `True` | 37453 ms | 5 calls | fired: `False`

> Yes, you should definitely still use pandas as your primary library for dataframes, though it is no longer the only major player in the space. It remains the absolute industry standard, meaning almost every data science tool, machine learning library, and tutorial is built to work with it. If you are collaborating with others or looking for a job in data science, pandas is still a mandatory skill.

However, there are two main reasons people look beyond pandas today: speed and data size. If you are working with larger datasets and finding pandas to be too slow, Polars has emerged as a highly popular and incredibly fast alternative. Polars is written in Rust, utilizes all of your computer's processor cores, and is much more memory-efficient. On the other extreme, if your data is so massive that it cannot fit on a single computer at all, then distributed frameworks like PySpark or Dask are the standard tools. 

For most day-to-day data analysis and learning, pandas is still the best place to start because of its massive community support, but Polars is well worth learning once you run into performance bottlenecks.


## [TUTORING-MISSED] Q6: Has anything replaced attention for long sequence modelling?

- ok: `True` | 36750 ms | 5 calls | fired: `False`

> State Space Models, most notably an architecture called Mamba, along with hybrid models and advanced recurrent networks, have emerged as highly successful replacements for traditional attention in long-sequence modeling. While the attention mechanism in Transformers is powerful, its computational and memory costs grow quadratically with the length of the sequence, making it incredibly expensive for very long documents, audio, or video. 

Mamba and other modern State Space Models solve this by processing data with linear complexity, meaning the resources required scale proportionally with the sequence length rather than squaring. They achieve this by mathematically modeling the sequence as a continuous system, similar to classical control theory. In practice, this allows them to act like Recurrent Neural Networks during inference—maintaining a fixed-size memory state that does not grow with sequence length—while still being able to train efficiently on modern hardware like Convolutional Neural Networks.

Another prominent alternative is RWKV, which stands for Receptive Weighted Key Value. It redesigns the traditional recurrent neural network to allow for parallelized training like a Transformer, but retains the linear scaling and tiny memory footprint of an RNN during deployment. 

Currently, the frontier of long-sequence modeling is moving toward hybrid architectures. Instead of completely abandoning attention, researchers are combining a few attention layers, which are excellent at precise, needle-in-a-haystack recall, with State Space Model layers, which handle the massive volume of the surrounding context efficiently. This combination gives the best of both worlds: the high accuracy of attention and the infinite-scaling potential of state space models.


# ===== TUTORING: stable controls =====

Must not fire. One run each, to confirm no search happened.

- Q1: fired=`False` | 5 calls | Can you explain how backpropagation actually works?

- Q2: fired=`False` | 5 calls | Why is quicksort O(n log n) on average?

- Q3: fired=`False` | 5 calls | What is the derivative of sin(x) times cos(x)?

- Q4: fired=`False` | 5 calls | How does a pendulum's period depend on its length?
