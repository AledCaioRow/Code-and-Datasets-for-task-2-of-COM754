# Code-and-Datasets-for-task-2-of-COM754

Repository Overview

This repository contains all scripts and datasets used to examine divergence between human and AI-generated empathetic listener responses across emotional contexts using the EmpatheticDialogues framework.

The pipeline follows a transparent sequence:
raw data → filtering → AI generation → cleaning → benchmarking → descriptive statistics → hypothesis testing.

Dataset Files
ED dataset.csv

Raw EmpatheticDialogues training dataset (~25,000 conversations).
Each conversation contains a speaker-elicited emotion, a situation description, and human listener responses.

Purpose:
Baseline human data where emotional context is explicitly provided by speakers, not inferred post hoc.

PreFilteredDataset.csv

Subset of the ED dataset after:

Removing ambiguous emotion labels

Restricting data to emotions clearly classifiable by valence and complexity

Purpose:
Improves construct validity before AI generation.

The Final Dataset.csv

Main analysis dataset containing:

Human listener responses

AI-generated listener responses

Emotional category labels

Benchmark scores and composite divergence measures

Purpose:
Used for all descriptive statistics and hypothesis testing.

THE TEST.csv

Small test subset used to validate:

API calls

Prompt construction

Data integrity before full generation

Purpose:
Prevents costly or invalid large-scale API execution.

Code Files
Filtering.py

Assigns emotional valence (positive/negative) and complexity (simple/complex).
Balances the dataset across the 2×2 emotional framework at the conversation level.

Purpose:
Ensures equal representation across emotional categories and prevents confounding by imbalance.

GPT API Prompter.py

Generates AI empathetic listener responses using the OpenAI API.

Key constraints:

Stateless calls (no memory across conversations)

Only prior speaker turns used as context

Human listener responses never shown to the model

Purpose:
Creates AI responses under matched conversational conditions.

dataset cealing.py

Cleans NLP artefacts including:

Placeholder tokens (e.g. _comma_)

Encoding noise

Excess whitespace

Purpose:
Prevents benchmark distortion due to non-semantic text artefacts.

Descriptive Stats.py

Produces:

Conversation counts per emotional cell

Utterance distributions

Summary statistics of divergence scores

Purpose:
Provides transparency and sanity checks before inferential testing.

Analysis.py

Conducts hypothesis testing using frequentist ANOVA:

Main effect of emotional complexity

Main effect of emotional valence

Complexity × valence interaction

Reports effect sizes (η²).

Purpose:
Formal statistical evaluation of AI–human divergence.
