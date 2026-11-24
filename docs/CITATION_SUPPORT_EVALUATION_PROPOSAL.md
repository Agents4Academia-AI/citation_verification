# Proposal: evaluate citation support and evidence traceability separately

Status: proposal for maintainer review. No evaluation has been run, no new
dependency is required, and no implementation owner or recurring monitoring is
assigned. This can be accepted, reduced, or deferred independently of the
[full-text evidence-attribution fix](https://github.com/Agents4Academia-AI/citation_verification/pull/47).

## Existing foundation

The repository already has offline tests, schema validation, and an evaluation
harness covering correctness, relevance, priority and abstention. The
[evaluation README](../evals/README.md) documents 18 hand-written smoke examples
from fictional papers and an external CitationHallucinationBench dataset.
Those tests are useful engineering validation; they do not by themselves
establish real-world claim-support accuracy or faithful evidence localization.

The smoke directory currently contains gold examples but no predictions. A
successful invocation that reports `pred=0` is not an accuracy result. Any
benchmark report should expose gold, predicted, matched, missing and excluded
counts, and refuse to describe zero matched cases as a successful evaluation.

## Two resources worth considering

**SciCiteVal: human-annotated claim/evidence pairs.** The
[LREC 2026 paper](https://aclanthology.org/2026.lrec-1.125/) describes 1,034
examples: 302 Correct, 302 Incorrect and 430 Unrelated. Each pairs a citation
context with evidence from the cited work. The
[dataset card](https://huggingface.co/datasets/birdie0111/SciCiteVal) lists
CC-BY-4.0 and fields `Citation_context`, `Cited_content`, `Label` and
`Twist_category`. This could test support reasoning with supplied evidence,
separately from source retrieval. Controlled alterations mean its class
frequencies should not be interpreted as natural citation-error prevalence.

**SemanticCite: a second checker, not gold labels.** Its
[public repository](https://github.com/sebhaan/SemanticCite) provides a
claim-plus-reference-text interface, four support classes, evidence snippets,
and hybrid retrieval/reranking. Its README states MIT licensing; model licenses
and pinned versions should be checked before reuse. It could provide a second
set of judgments on frozen inputs. Agreement is not proof of correctness,
especially if models or training data overlap. We have not independently
validated its accuracy or its integration with this project.

## Small, bounded first study

1. **Agree on the rubric before running models.** Judge the subclaim attributed
   to each citation, not whether that source proves every clause of a compound
   sentence. Distinguish narrower evidence, contradiction, absence after search,
   and unavailable sources. Do not equate average underconfidence with a
   pointwise guarantee, or an observed effect with a universal claim.
2. **Start with 30 supplied-evidence examples** sampled across the three
   SciCiteVal labels. Record sample IDs, dataset revision, sampling seed, model
   identifiers and prompts. Keep examples sharing an underlying source or
   original context together when separating development from held-out cases;
   distorted variants must not leak across the split. Use the small sample for
   feasibility/error analysis, not a headline performance claim.
3. **Keep label mapping explicit.** SciCiteVal has no direct equivalent of all
   four RefWarden labels. Initially retain raw predictions and report a
   three-by-four cross-tabulation, the fraction of Incorrect/Unrelated examples
   judged `supports`, and abstention coverage. Do not silently force `partial`
   or `inconclusive` into Incorrect. Establish an adjudicated mapping before
   reporting mapped macro-F1 or accuracy.
4. **Audit evidence separately.** Measure whether the quotation occurs in the
   exact supplied context, whether the locator is correct, and whether a human
   reviewer considers it sufficient for the attributed subclaim. A literal
   match is a provenance check, not entailment. Report missing quotations and
   valid reasons for abstention; absence judgments need not invent a quotation.
5. **Optionally compare SemanticCite on the same cases.** Use the same frozen
   citation contexts and source text where its interface permits; document
   any differences in claim extraction or retrieval. Inspect disagreements
   against human labels and source passages, rather than using one checker
   to grade the other. Check training overlap before treating this as held-out.

If maintainers find the feasibility study useful, they can choose an owner and
budget for a larger held-out run, followed separately by end-to-end retrieval
evaluation. Human review should cover a random sample as well as disagreements;
checking only disagreements can miss shared errors. Record reviewer disagreement
and avoid manufacturing consensus when the source is ambiguous.

## Suggested deliverable

A single reproducible report with pinned inputs, raw per-pair outputs, label
mapping, denominator counts, uncertainty intervals where meaningful, runtime
and usage, and a short error analysis. Retrieval failures, support decisions,
evidence-location failures and bibliography errors should remain separate.
For revision experiments, include unchanged manuscripts as controls: changed
support counts alone do not distinguish repaired claims from judge variability.

This proposal does not change runtime behavior, freeze a new public schema,
install either resource, or commit anyone to maintaining a benchmark service.
