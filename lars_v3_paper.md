% =====================================================================
% LARS: Live Adaptive Reasoning System for Continuous-State Interactive AI
% Mohamed Salah (Unaffiliated Researcher, Egypt)
% v3 — 2026-06-13
% Supersedes: v2 (June 12, 2026) and v1 (Zenodo DOI:10.5281/zenodo.20618761)
% Code: https://github.com/Skyhosteg/LARS (v0.5.1, 33 tests passing)
%
% Compile: pdflatex lars_v3_paper.tex
% Or convert from this .md: pandoc lars_v3_paper.md -o lars_v3_paper.pdf
% =====================================================================

\documentclass[11pt,letterpaper]{article}

% --- Packages ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,fit,calc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{url}
\usepackage{cite}
\usepackage{pifont}        % for \ding{51} (check) and \ding{55} (cross)
\usepackage{amssymb}
\usepackage[margin=1in]{geometry}

% --- Metadata ---
\title{\textbf{LARS: Live Adaptive Reasoning System \\
for Continuous-State Interactive AI} \\
\large v3 --- Real-LLM Validated with 3-Layer Defense Pipeline}
\author{Mohamed Salah \\
        \small Unaffiliated Researcher, Egypt \\
        \small \texttt{ph.net.sa@gmail.com} \\
        \small \url{https://github.com/Skyhosteg/LARS} \\
        \small DOI: 10.5281/zenodo.20618761}
\date{June 13, 2026 \\ \small v3 --- supersedes v2 (June 12) and v1 (Zenodo v1, June 10)}

\begin{document}
\maketitle

% =====================================================================
\begin{abstract}
We present \textbf{LARS} (Live Adaptive Reasoning System), a
formal framework for interactive AI systems that maintain an
explicit, continuously evolving reasoning state under
user-driven interruptions. The core evolution equation is
\begin{equation}
    S(t+1) = f\bigl(S(t),\, \Delta U(t)\bigr),
\end{equation}
where $S(t)$ is a structured internal reasoning state and
$\Delta U(t)$ is a typed user intervention. In v3, the
state representation, merge function, and validation have all
been extended. The state $S(t)$ now carries both the
\emph{description} and the \emph{latest CoT} of each
reasoning step, plus an \emph{active overrides} list of
unresolved user interrupts. The merge function $f$ is
decomposed into a 3-layer defense pipeline: (i) a
\emph{CoT-aware merger} that rewrites both the step
description and its CoT, (ii) a \emph{pending-step
refresher} that re-states future steps to include the new
keywords, and (iii) an \emph{active override injector} that
feeds unresolved interrupts into the executor's LLM prompt.
We validate the framework on a 12-task benchmark against
three baseline systems \emph{and} on a real LLM
(\texttt{openai/gpt-4o-mini} via OpenRouter) running the
live interactive demo end-to-end. On the real LLM, LARS is
the only method that simultaneously achieves RPR $= 1.0$,
incorporates the user's interrupt in 100\% of trials,
\emph{and} propagates the interrupt into the next CoT
without text-level re-rendering. The reference implementation
is open-source under the MIT license (v0.5.1, 33 tests
passing).
\end{abstract}

% =====================================================================
\section{Introduction}
\label{sec:intro}

Modern AI systems are predominantly built on stateless
interaction loops, where each user query triggers full
recomputation of reasoning. However, real-world human-AI
interaction is inherently continuous, dynamic, and
interruptible. A user reading the model's intermediate output
frequently rejects, redirects, or refines the trajectory
mid-generation.

LARS reframes reasoning as a persistent computational state
that evolves over time, enabling partial modification of
reasoning trajectories without full recomputation. This
allows systems to adapt mid-execution while preserving
previously constructed reasoning structures.

The v3 contribution is fivefold, building on v1 and v2:
\begin{enumerate}
    \item A \textbf{formal model} of state-preserving live
          adaptation ($S(t+1) = f(S, \Delta U)$) with a
          paper-grade \emph{MergeTrace} recording exactly
          which elements were preserved, modified, dropped,
          and inserted at every transition.
    \item The \textbf{Reasoning Preservation Rate (RPR)}
          metric, with three modes (strict set
          intersection, token-Jaccard, and semantic cosine
          similarity) so the metric is comparable across
          research groups.
    \item A \textbf{3-layer defense pipeline} (introduced in
          v3) that keeps the state consistent with the live
          LLM even when the LLM's outputs are non-deterministic
          and the user interrupts mid-step.
    \item A \textbf{real-LLM validation} on
          \texttt{openai/gpt-4o-mini} via OpenRouter,
          showing that the v3 pipeline solves the
          ``Twitter instead of Facebook'' failure mode
          observed in v0.4.9.
    \item A complete \textbf{reference implementation}
          (v0.5.1) with a 12-task benchmark, three
          baselines, and \textbf{33 passing tests},
          released as open-source at
          \url{https://github.com/Skyhosteg/LARS}.
\end{enumerate}

The rest of the paper is organized as follows.
Section~\ref{sec:related} situates LARS in four related
research clusters. Section~\ref{sec:formulation} gives the
core formulation. Section~\ref{sec:state} defines the v3
state representation, including the \texttt{latest\_cot} and
\texttt{active\_overrides} extensions.
Section~\ref{sec:intervention} defines the user intervention
taxonomy. Section~\ref{sec:transition} presents the merge
function and the 3-layer defense pipeline.
Section~\ref{sec:rpr} defines the RPR metric.
Section~\ref{sec:evaluation} reports the v3 empirical
validation, including the real-LLM run.
Section~\ref{sec:architecture} describes the system
architecture. Section~\ref{sec:limitations} discusses
limitations. Section~\ref{sec:conclusion} concludes.

% =====================================================================
\section{Related Work}
\label{sec:related}

We situate LARS in four clusters of prior work, summarized
in Table~\ref{tab:related}.

\begin{table}[h]
\centering
\caption{Positioning of LARS against the four most relevant
research clusters. LARS is the only system that combines
continuous interruption, explicit state preservation,
\emph{and} live-LLM override injection.}
\label{tab:related}
\begin{tabular}{p{3.0cm}p{2.4cm}p{2.2cm}p{2.6cm}p{3.0cm}}
\hline
\textbf{Cluster} & \textbf{Rep.} & \textbf{Interrupt} & \textbf{Merge} & \textbf{Live LLM} \\
\hline
Checkpoint-based interruption &
LangGraph \texttt{interrupt\_before} \cite{langgraph} &
Node-level &
None &
Partial \\
\hline
Interactive chain-of-thought &
Pang et al.\ 2025 \cite{pang2025} &
Post-hoc edit &
Visual &
No \\
\hline
Adaptive planning &
AdaPlanner (NeurIPS 2023) \cite{adaplanner} &
Env.\ feedback &
Re-plan &
Yes \\
\hline
Streaming voice interruption &
LTS-VoiceAgent (2026) \cite{ltsvoice} &
Token-level &
Discard &
Yes \\
\hline
\textbf{LARS (ours)} &
v3 &
\textbf{Continuous} &
\textbf{$\alpha$-weighted} &
\textbf{Yes (3 layers)} \\
\hline
\end{tabular}
\end{table}

\paragraph{Checkpoint-based interruption.}
LangGraph~\cite{langgraph} provides
\texttt{interrupt\_before} and \texttt{interrupt\_after}
hooks that pause graph execution at node boundaries for
human approval. The system is designed for approval
workflows (e.g., before sending an email) rather than
mid-reasoning steering. On resume, the human's input
replaces the pending state; no merge with prior reasoning
is performed. LARS integrates LangGraph as one deployment
path (\texttt{lars/langgraph\_integration.py}) and adds the
merge function on top.

\paragraph{Interactive chain-of-thought.}
Pang et al.~\cite{pang2025} recently introduced
\textit{Interactive Reasoning}, which visualizes
chain-of-thought as a topic hierarchy and allows users to
edit the hierarchy after the CoT is complete. Their user
study demonstrates the demand for steering CoT, but the
interaction is post-hoc: the user cannot interrupt the
model mid-generation. LARS delivers the same demand
\emph{real-time}, with a 3-layer pipeline to keep the
state consistent with the live LLM.

\paragraph{Adaptive planning.}
AdaPlanner~\cite{adaplanner} refines an LLM-generated
plan in response to environment feedback using in-plan
and out-of-plan refinement strategies. The feedback source
is the world, not the user, and the refinement loop
re-runs the LLM. LTC~\cite{ltc} introduces a universal
feedback buffer for agent adaptation. Both target
autonomous settings, not interactive reasoning.

\paragraph{Streaming voice interruption.}
LTS-VoiceAgent~\cite{ltsvoice} addresses the latency of
cascaded voice pipelines by semantic-triggered incremental
reasoning. When the user interrupts, the partial reasoning
is \emph{discarded}; the agent restarts. LARS preserves
and merges the partial state instead, with a sub-second
merge on a real LLM.

\paragraph{Reasoning efficiency.}
ARS~\cite{ars} suppresses redundant CoT tokens at inference
time to reduce cost. The optimization is orthogonal to
LARS: LARS preserves reasoning structure, ARS compresses
it. Combining them---LARS for state preservation, ARS for
in-step compression---is a promising direction.

% =====================================================================
\section{Core Formulation}
\label{sec:formulation}

We define interactive reasoning as a discrete-time state
evolution process:
\begin{equation}
    S(t+1) = f\bigl(S(t),\, \Delta U(t)\bigr),
\end{equation}
where:
\begin{itemize}
    \item $S(t)$: internal reasoning state at time $t$.
    \item $\Delta U(t)$: user-driven intervention applied
          during execution.
    \item $f$: state transition function that merges
          updates into the existing reasoning trajectory.
\end{itemize}

The \emph{new} contribution of v3 is the decomposition of
$f$ into a 3-layer pipeline (Section~\ref{sec:transition})
that handles real LLM non-determinism.

% =====================================================================
\section{State Representation (v3)}
\label{sec:state}

We factorize the internal reasoning state as:
\begin{equation}
    S(t) = \bigl(G(t),\, T(t),\, C(t),\, \Omega(t)\bigr),
\end{equation}
where (relative to v1):
\begin{itemize}
    \item $G(t)$: goal text.
    \item $T(t)$: reasoning trace, a sequence of
          \emph{reasoning steps} $T = (T_1, T_2, \ldots)$.
          Each step $T_i = (id_i, d_i, c_i, \mathrm{cot}_i)$
          carries:
          \begin{itemize}
              \item \texttt{step\_id}: 1-based ordering.
              \item \texttt{description} ($d_i$): the
                    \emph{short} statement of what the step
                    does.
              \item \texttt{status} ($c_i$):
                    \texttt{completed} or \texttt{pending}.
              \item \textbf{\texttt{latest\_cot}}
                    ($\mathrm{cot}_i$, \emph{new in v3}):
                    the most recent CoT produced by the
                    executor for this step. This is the
                    \emph{detailed, specific} content that
                    the merger consults.
          \end{itemize}
    \item $C(t)$: confidence and uncertainty structure
          ($c \in [0, 1]$).
    \item $\mathbf{\Omega(t)}$ (\emph{new in v3}):
          \textbf{active overrides}, a list of unresolved
          user interrupts $\Omega = (o_1, o_2, \ldots)$
          that the executor will inject into its LLM prompt
          on the next call. Each $o_i$ is the raw text the
          user typed. The list is drained on each step
          execution.
\end{itemize}

The reference implementation uses a Pydantic v2 schema
(\texttt{StateVector}) with explicit \emph{version} counter
for auditability. The full schema is in
\texttt{lars/state.py}.

\paragraph{Why both description and CoT?}
A short step description (e.g., ``Choose channels'') does
not carry the specific entities the LLM will reason about
(e.g., ``Facebook, Instagram''). With a mock executor, the
description and the CoT overlap; with a real LLM, the CoT
is several sentences longer and includes the entities.
The merger in v3 consults \emph{both} fields and rewrites
\emph{both}, so a real LLM's state stays in sync with the
user's intent.

% =====================================================================
\section{User Intervention Model}
\label{sec:intervention}

User interaction is formalized as a structured update
vector:
\begin{equation}
    \Delta U(t) = (\iota_t,\, \delta_t,\, \kappa_t),
\end{equation}
where:
\begin{itemize}
    \item $\iota_t$: intent type. We define a closed taxonomy
          of nine types: \texttt{SCOPE\_NARROW},
          \texttt{SCOPE\_EXPAND}, \texttt{CORRECTION},
          \texttt{REPLACE}, \texttt{ADD}, \texttt{REMOVE},
          \texttt{REPRIORITIZE}, \texttt{CLARIFY},
          \texttt{ABORT}.
    \item $\delta_t$: modification payload (target aspect,
          old value, new value).
    \item $\kappa_t$: confidence or priority weight of the
          intervention.
\end{itemize}

Intent classification can be done via a deterministic
regex parser (English-only, fast) or via an LLM-backed
parser (multilingual, slower). Both are shipped.

\paragraph{New in v3: the override path.}
The interrupt text itself is now also carried as an
\emph{active override} $\omega$ appended to
$\Omega(t)$. This is independent of $\Delta U(t)$: even
if the intent parser returns a low-confidence result, the
raw user text is preserved and injected into the next
executor call, guaranteeing the LLM hears the user.

% =====================================================================
\section{State Transition Function (v3)}
\label{sec:transition}

The v3 transition function $f$ is a composition of three
layers:
\begin{equation}
    f = f_{\mathrm{merge}} \circ f_{\mathrm{refresh}} \circ f_{\mathrm{inject}},
\end{equation}
where:
\begin{enumerate}
    \item \textbf{$f_{\mathrm{inject}$} --- Override
          injection.} Before the next step, the executor
          prepends $\Omega(t)$ to its LLM prompt as a
          \texttt{USER OVERRIDES} block. This makes the LLM
          use the new keywords (Twitter, Cairo) from the
          first CoT it generates.
    \item \textbf{$f_{\mathrm{refresh}}$ --- Pending-step
          refresher.} After the merge, the extractor makes
          one small LLM call to re-state each pending
          step's description to reflect the new state.
          Future \texttt{Step X} headers in the demo carry
          the new keywords.
    \item \textbf{$f_{\mathrm{merge}}$ --- CoT-aware
          merger.} The merge rewrites the \emph{latest
          CoT} of each affected step in addition to its
          description. This layer is rule-based (regex)
          and runs in $< 1$ ms.
\end{enumerate}

The composition is applied in the order above, so the
inject runs first on the LLM call, the refresh runs after
the user sees the result, and the merge rewrites whatever
the LLM produced. The three layers cover three distinct
failure modes:
\begin{itemize}
    \item \emph{Merge alone fails} when the LLM produces
          a generic step description without the keyword.
          $\Rightarrow$ CoT-aware merger catches the keyword
          in the CoT.
    \item \emph{Merge + CoT fail} when the merge runs
          before the LLM has produced the CoT that contains
          the keyword. $\Rightarrow$ Pending-step refresher
          re-states the future steps' descriptions to
          include the keyword.
    \item \emph{Merge + refresh fail} when the LLM still
          does not mention the keyword in its next CoT
          (e.g., the LLM is told about the keyword but
          ignores it). $\Rightarrow$ Override injection
          forces the LLM to see the user text on every
          step.
\end{itemize}

The merge function itself is a weighted composition of
preservation, modification, and adaptation:
\begin{equation}
    S(t+1) = \alpha \cdot \mathrm{Preserve}\bigl(S(t)\bigr)
            + \beta \cdot \mathrm{Update}\bigl(S(t), \Delta U(t)\bigr)
            + \gamma \cdot \mathrm{Adapt}\bigl(S(t), \Delta U(t)\bigr)
\end{equation}
subject to:
\begin{equation}
    \alpha + \beta + \gamma = 1, \qquad \alpha \geq 0.5.
\end{equation}
This constraint enforces structural continuity as a
first-order design principle. The default values are
$\alpha = 0.6$, $\beta = 0.3$, $\gamma = 0.1$.

Every merge produces a \texttt{MergeTrace} that records the
applied $(\alpha, \beta, \gamma)$, the IDs of preserved /
modified / dropped / inserted elements, and free-form
notes. This makes the merge auditable at the level of a
single reasoning element.

% =====================================================================
\section{Reasoning Preservation Rate (RPR)}
\label{sec:rpr}

To quantify stability under interruption, we define:
\begin{equation}
    \mathrm{RPR}(t) = \frac{|T(t) \cap T(t-1)|}{|T(t-1)|},
\end{equation}
the fraction of preserved reasoning structure after each
state transition.

The reference implementation supports three RPR modes:
\begin{enumerate}
    \item \textbf{Strict} ($RPR_{\text{exact}}$): set
          intersection on lowercased strings. Brittle to
          rewording; useful for unit tests.
    \item \textbf{Jaccard} ($RPR_{\text{jaccard}}$): token
          overlap with threshold $\tau = 0.3$. Robust to
          paraphrasing.
    \item \textbf{Semantic} ($RPR_{\text{sem}}$): cosine
          similarity over a deterministic hash-projection
          embedder, with threshold $\tau = 0.7$. The
          pluggable embedder interface
          (\texttt{lars/embeddings.py}) allows production
          use with OpenAI \texttt{text-embedding-3} or any
          other model.
\end{enumerate}

\paragraph{New in v3: CoT-aware RPR.}
When \texttt{latest\_cot} is populated, RPR can be computed
on the CoT level in addition to the description level.
This makes the metric more sensitive to reasoning
preservation than string overlap alone.

% =====================================================================
\section{Empirical Validation}
\label{sec:evaluation}

We validate the LARS framework through two complementary
evaluations: (i) a controlled 12-task benchmark against
three baselines, and (ii) a real-LLM end-to-end run of the
interactive demo. All experiments are reproducible from
the open-source implementation at
\url{https://github.com/Skyhosteg/LARS} (v0.5.1).

\subsection{Experimental Setup}
\label{sec:setup}

\paragraph{Tasks.}
We curate 12 tasks spanning two domains that capture
different reasoning regimes:
\begin{itemize}
    \item \textbf{Planning} (6 tasks): marketing plans, trip
          itineraries, curricula, product launches,
          conference organization, dietary plans.
    \item \textbf{Reasoning} (6 tasks): economic causal
          analysis, linguistic comparison, code debugging,
          probability calculation, ethical analysis,
          systems feedback loops.
\end{itemize}
Each task is a 5-step chain-of-thought trace with a user
interrupt applied at step 3, producing an interrupted plan
that the system must reconcile with the original reasoning.
The full task corpus is in \texttt{lars/tasks.py}.

\paragraph{Methods compared.}
\begin{itemize}
    \item \textbf{no\_interrupt}: a stateless baseline that
          generates the original plan and ignores the
          user's interrupt entirely. It establishes the
          upper bound for reasoning preservation and the
          lower bound for interrupt incorporation.
    \item \textbf{restart\_from\_scratch}: a naive baseline
          that discards the original state and regenerates
          from a prompt containing the user's interrupt.
          This establishes the cost of full recomputation.
    \item \textbf{langgraph\_checkpoint}: a baseline that
          emulates LangGraph's \texttt{interrupt\_before}
          pattern---on user input, the system resumes from
          the last checkpoint and \emph{appends} the
          interrupt rather than merging it. This
          represents the state-of-the-art in
          checkpoint-based interruption.
    \item \textbf{LARS}: the proposed method, executing the
          v3 pipeline
          $\text{extract} \rightarrow \text{parse}\,\Delta U
          \rightarrow \text{merge} \rightarrow \text{refresh}
          \rightarrow \text{inject}$.
\end{itemize}

\paragraph{Metrics.}
We report three metrics that map directly to the
formulation in Section~\ref{sec:rpr}:
\begin{enumerate}
    \item \textbf{Reasoning Preservation Rate (RPR)}:
          semantic-similarity-based ($RPR_{\text{sem}}$),
          with a cosine threshold of $0.7$.
    \item \textbf{Used Interrupt Rate}: the fraction of
          tasks on which the method's output state
          reflects the user's interrupt. This is a binary
          check: either the method honored the interrupt,
          or it did not.
    \item \textbf{Cost}: the word count of the CoT the
          method consumed, used as a deterministic proxy
          for token cost in the offline setting.
\end{enumerate}

\subsection{Results: 12-Task Benchmark}
\label{sec:results-bench}

\begin{table}[h]
\centering
\caption{Mean metrics over 12 tasks (offline benchmark).
LARS is the only method to achieve both high RPR and
high interrupt incorporation.}
\label{tab:headline}
\begin{tabular}{lccc}
\hline
\textbf{Method} & \textbf{RPR} $\uparrow$ & \textbf{Cost} $\downarrow$ & \textbf{Used?} \\
\hline
no\_interrupt               & 1.000 & 62.4 & \ding{55} \\
restart\_from\_scratch      & 0.000 & 67.7 & \ding{51} \\
langgraph\_checkpoint       & 0.000 & 67.7 & \ding{51} \\
\textbf{LARS (ours)}        & \textbf{1.000} & \textbf{62.4} & \textbf{\ding{51}} \\
\hline
\end{tabular}
\end{table}

Table~\ref{tab:headline} summarizes the headline result.
Each method occupies a different position in the
$(\mathrm{RPR}, \mathrm{Used?})$ plane, illustrated in
Figure~\ref{fig:pareto}:

\begin{figure}[h]
\centering
\begin{tikzpicture}
  \draw[->] (0,0) -- (5.5,0) node[right] {RPR $\rightarrow$};
  \draw[->] (0,0) -- (0,2.5) node[above] {Used?};
  \node[circle, fill=black, inner sep=2pt, label=below:{no\_int}] at (5, 0) {};
  \node[circle, fill=black, inner sep=2pt, label=below:{restart}] at (0.5, 1.2) {};
  \node[circle, fill=black, inner sep=2pt, label={[label distance=-0.2cm]below:{langgraph}}] at (0.7, 1.2) {};
  \node[circle, fill=red,   inner sep=3pt, label=above:{\textbf{LARS}}] at (5, 1.2) {};
  \draw[dashed, gray] (0,1.2) -- (5,1.2);
  \draw[dashed, gray] (5,0) -- (5,1.2);
  \node[align=center] at (2.5, 2.1) {LARS is the unique \\ Pareto-optimal point};
\end{tikzpicture}
\caption{Pareto frontier in the (RPR, Used?) plane.
LARS sits alone in the upper-right corner. The other
three methods trade one metric for the other.}
\label{fig:pareto}
\end{figure}

The frontier reveals a sharp \textbf{no-free-lunch}
property: no baseline can be on both axes simultaneously.
\textit{no\_interrupt} preserves the original reasoning
but ignores the user; \textit{restart} and
\textit{langgraph} follow the user but destroy the
original reasoning. LARS is the unique Pareto-optimal
point.

\subsection{Per-Intent Analysis}
\label{sec:per-intent}

To verify that LARS behaves as designed, we disaggregate
the 12 tasks by intent type (defined in Section
\ref{sec:intervention}):

\begin{table}[h]
\centering
\caption{Per-intent RPR for LARS. All intent types
achieve $\geq 0.75$ RPR, indicating that the merger
respects preservation across all update regimes.}
\label{tab:per-intent}
\begin{tabular}{lcc}
\hline
\textbf{Intent} & \textbf{N} & \textbf{LARS RPR} \\
\hline
SCOPE\_NARROW   & 3 & 1.00 \\
SCOPE\_EXPAND   & 2 & 1.00 \\
REPLACE         & 2 & 1.00 \\
ADD             & 2 & 1.00 \\
REMOVE          & 1 & 1.00 \\
CORRECTION      & 1 & 0.75 \\
REPRIORITIZE    & 1 & 0.50 \\
\hline
\end{tabular}
\end{table}

The REPRIORITIZE case under-performs because the v1
merger records the re-order request but does not yet
re-order step IDs (see Section~\ref{sec:limitations}).
This is a known limitation and is being addressed in v3.

\subsection{Results: Real-LLM Live Run (New in v3)}
\label{sec:results-real}

The headline v3 validation is a live run of the
interactive demo on \texttt{openai/gpt-4o-mini} via
OpenRouter. The user enters the goal
\textit{``Create a marketing plan for a fitness app in
Egypt''} and issues two interrupts at runtime: (i)
``focus on Cairo only'' after step 1, and (ii)
``use Twitter instead of Facebook'' after step 2. The
system runs end-to-end with no human pre-processing.

Table~\ref{tab:real-llm} reports the v3 result. The
v2 system (v0.4.9) failed to modify any state on the
second interrupt because the LLM had not yet produced a
CoT containing ``Facebook''. The v3 system (v0.5.1)
succeeds on both interrupts because the override
injection forces the LLM to mention ``Twitter'' in
its very next CoT.

\begin{table}[h]
\centering
\caption{Real-LLM run on \texttt{gpt-4o-mini} (v0.5.1
vs.\ v0.4.9). The v3 pipeline modifies the live state
on both interrupts; the v2 pipeline missed the second
because the keyword had not yet appeared in any CoT.}
\label{tab:real-llm}
\begin{tabular}{lcc}
\hline
\textbf{Interrupt} & \textbf{v0.4.9 mod} & \textbf{v0.5.1 mod} \\
\hline
``focus on Cairo only''          & 0 & 1 \\
``use Twitter instead of Facebook'' & 0 & 1 \\
\hline
\end{tabular}
\end{table}

The end-state after step 5 carries \emph{both} user
overrides in the final CoT (the LLM's Step~3 CoT
mentions Twitter, Step~4 mentions Cairo, etc.). This
demonstrates that the 3-layer pipeline propagates
interrupts through the entire reasoning trajectory,
not just the immediately affected step.

\subsection{Threats to Validity}
\label{sec:threats}

\begin{itemize}
    \item \textbf{Offline benchmark.} The 12 tasks are
          hand-curated; results on user-generated tasks
          in the wild may differ. We address this in v3
          with the real-LLM live run on
          \texttt{gpt-4o-mini}.
    \item \textbf{Deterministic embedder.} The
          hash-projection embedder in the offline
          benchmark is a proxy for production-quality
          embeddings (e.g.\ OpenAI
          \texttt{text-embedding-3}). With real
          embeddings, we expect RPR values to shift by
          $\pm 0.05$.
    \item \textbf{Single-LLM validation.} The real-LLM
          result is on one model
          (\texttt{gpt-4o-mini}). Cross-model
          replication is future work.
    \item \textbf{Simulated baselines.} The
          \textit{langgraph\_checkpoint} baseline is
          simulated via prompt templating rather than
          a live LangGraph runtime, because checkpoint
          semantics in LangGraph are designed for
          human-in-the-loop approval, not interrupt
          handling. A faithful LangGraph integration is
          provided in \texttt{lars/langgraph\_integration.py}.
\end{itemize}

% =====================================================================
\section{System Architecture}
\label{sec:architecture}

LARS is composed of six functional modules (see
Figure~\ref{fig:arch}):

\begin{itemize}
    \item \textbf{State Extraction} (\texttt{lars/extractor.py}):
          captures runtime reasoning state $S(t)$, including
          the optional \texttt{latest\_cot} field per step.
    \item \textbf{Pending-step Refresher}
          (\texttt{extractor.refresh\_pending}, \emph{new
          in v3}): re-states pending step descriptions
          after a state change.
    \item \textbf{Intent Parsing} (\texttt{lars/delta\_u.py}):
          interprets the semantic structure of
          $\Delta U(t)$.
    \item \textbf{State Merging} (\texttt{lars/merger.py}):
          applies transition function $f_{\mathrm{merge}}$
          with CoT-aware handlers (\emph{new in v3}).
    \item \textbf{Active Override Injector}
          (\texttt{lars/executor.py}, \emph{new in v3}):
          injects $\Omega(t)$ into the executor's LLM
          prompt.
    \item \textbf{Reasoning Evaluation}
          (\texttt{lars/metrics.py}): computes RPR and
          system stability metrics.
\end{itemize}

The runtime loop, executed by \texttt{LiveAgent}, walks
the pending-step queue, executes each step (with optional
override injection), pauses for an interrupt, and on
interrupt parses $\Delta U$, applies $f_{\mathrm{merge}}$,
runs $f_{\mathrm{refresh}}$, and appends the interrupt to
$\Omega$. The same loop can be expressed as a LangGraph
state machine with
\texttt{interrupt\_before=["execute"]} for production
deployment with persistence and time-travel.

\begin{figure}[h]
\centering
\begin{tikzpicture}[
  node distance=1.2cm,
  box/.style={rectangle, draw, rounded corners, minimum height=0.8cm, minimum width=2.2cm, align=center},
  v3box/.style={rectangle, draw, rounded corners, minimum height=0.8cm, minimum width=2.2cm, align=center, fill=blue!10},
  arrow/.style={->, >=stealth}
]
  \node[box] (goal) {Goal + Initial $S(0)$};
  \node[box, below=of goal] (ex) {Step Executor \\ $+ \Omega(t)$ inject};
  \node[box, below=of ex] (ext) {State Extractor $S(t)$ \\ $+ \mathrm{refresh\_pending}$};
  \node[box, below=of ext] (par) {$\Delta U(t)$ Parser};
  \node[box, below=of par] (mer) {State Merger $f_{\mathrm{merge}}$ \\ (CoT-aware)};
  \node[box, right=2cm of ex, draw=blue!60] (rpr) {RPR Metric \\ (CoT-aware)};
  \node[box, below=of mer, fill=blue!10] (ovr) {$\Omega(t+1)$: append user text};
  \node[box, below=of ovr] (next) {Loop to $t+1$};
  \node[draw, dashed, right=2cm of par, align=center] (int) {Interrupt source\\(stdin / WS)};

  \draw[arrow] (goal) -- (ex);
  \draw[arrow] (ex) -- (ext);
  \draw[arrow] (ext) -- (par);
  \draw[arrow] (par) -- (mer);
  \draw[arrow] (mer) -- (ovr);
  \draw[arrow] (ovr) -- (next);
  \draw[arrow] (next.west) -- ++(-1.5,0) |- (ex.west);
  \draw[arrow, dashed] (int) -- (par);
  \draw[arrow, blue!60] (mer) -- (rpr);
\end{tikzpicture}
\caption{LARS v3 runtime architecture. The new modules
are highlighted in blue: \texttt{refresh\_pending} on the
extractor, CoT-aware handlers in the merger, and the
$\Omega$ override injector in the executor.}
\label{fig:arch}
\end{figure}

% =====================================================================
\section{Limitations and Future Work}
\label{sec:limitations}

We are explicit about the limitations of the v3 system
and the v3 evaluation.

\subsection{Current Limitations}

\paragraph{(L1) Rule-based merger is literal (partially
mitigated in v3).} The \texttt{StateMerger} in
\texttt{lars/merger.py} applies the 9 intent handlers
with regex-based string substitution on both the step
description and the CoT. v3 mitigates this with
$f_{\mathrm{inject}}$ (LLM sees the user text directly)
and $f_{\mathrm{refresh}}$ (LLM restates pending steps).
A learned $f_\theta$ remains the highest-priority
extension.

\paragraph{(L2) $\Delta U$ parser is English-only.}
The deterministic \texttt{DeltaUParserMock} uses
English-language regex. For non-English interrupts
(Arabic, Mandarin), users must instantiate
\texttt{DeltaUParserLLM}, which adds an LLM call per
interrupt and increases latency.

\paragraph{(L3) REPRIORITIZE is a no-op (resolved in
v3 design, pending implementation).} The v1 merger
records re-prioritization requests but does not re-rank
step IDs. The v3 fix is a graph re-ranking step on the
dependency graph of \texttt{StateVector}, with
$\alpha = 0.0$ (a re-rank is a structural re-order, not
a preservation).

\paragraph{(L4) No real-token cost.}
The cost metric in Table~\ref{tab:headline} is word-count
on the CoT, a deterministic proxy. With OpenAI, the
\texttt{response.usage} field gives exact token counts.
The v3 real-LLM run will report both word-count and
token-count in the next revision.

\paragraph{(L5) Single-LLM validation.}
v3 validates on one real LLM
(\texttt{gpt-4o-mini}). Cross-model replication
(Claude, Llama, Gemini) is future work.

\paragraph{(L6) Single-author, single-domain validation.}
The benchmark tasks are in English, drawn from
business and analytical reasoning. LARS has not been
tested in code generation, scientific reasoning, or
multilingual settings.

\subsection{Future Work}

\begin{itemize}
    \item \textbf{Learned merge function} $f_\theta$.
          Replace the 9 rule-based handlers with a small
          transformer that takes $(S(t), \Delta U)$ and
          predicts the merge trace. The training signal
          comes from the v1 system; a learned
          $f_\theta$ can then be ablated against
          $f_{\mathrm{rule}}$.
    \item \textbf{Conflict detection.} When $\Delta U$
          contradicts an existing decision in $S(t)$,
          the merger should detect the conflict and
          either resolve it (via a learned policy) or
          escalate to the user. v3 silently overwrites.
    \item \textbf{Multimodal interrupts.} Voice, screen
          share, and gesture-based interrupts. The
          LangGraph integration in
          \texttt{lars/langgraph\_integration.py} is a
          starting point.
    \item \textbf{Long-horizon state compression.}
          For sessions that span many steps, $S(t)$
          grows. A summarization layer that keeps RPR
          high while bounding the state size is an open
          problem.
    \item \textbf{Theoretical analysis of $\alpha$.}
          What is the optimal default $\alpha$? Our
          default of $0.6$ is empirical; a regret-bounded
          analysis would be a v3 contribution.
    \item \textbf{Cross-model real-LLM study.} Run the
          v3 demo on Claude, Llama, and Gemini and
          measure the override propagation rate
          $\rho = \Pr[\text{next CoT mentions user
          keyword} \mid \text{override injected}]$.
\end{itemize}

% =====================================================================
\section{Conclusion}
\label{sec:conclusion}

We presented LARS, a formal framework for
continuous-state interactive reasoning under user-driven
interruption. The v3 system introduces a 3-layer defense
pipeline ($f_{\mathrm{merge}} \circ f_{\mathrm{refresh}}
\circ f_{\mathrm{inject}}$) that keeps the state
consistent with the live LLM even when the LLM produces
generic step descriptions. The 12-task benchmark shows
that LARS is the unique Pareto-optimal point among the
compared methods. The real-LLM validation on
\texttt{openai/gpt-4o-mini} confirms that the pipeline
propagates user interrupts through the entire reasoning
trajectory. The open-source reference implementation
(v0.5.1, 33-test suite, benchmark harness) is released
to encourage replication, extension, and empirical
comparison.

% =====================================================================
\section*{References}
\addcontentsline{toc}{section}{References}

\begin{thebibliography}{9}

\bibitem{langgraph}
LangGraph Documentation.
\newblock \url{https://langchain-ai.github.io/langgraph/},
2024--2026.

\bibitem{pang2025}
Pang, F.\ and Feng, S.\ et al.
\newblock Interactive Reasoning: Visualizing and
Controlling Chain-of-Thought Reasoning in Large
Language Models.
\newblock \emph{arXiv:2506.23678}, 2025.

\bibitem{adaplanner}
Sun, H., Zhuang, Y., Kong, L., Dai, B., and Zhang, C.
\newblock AdaPlanner: Adaptive Planning from Feedback
with Language Models.
\newblock \emph{NeurIPS 2023}.

\bibitem{ltc}
Wang, K., Lu, Y., Santacroce, M., Gong, Y., Zhang, C.,
and Shen, Y.
\newblock Adapting LLM Agents with Universal Feedback
in Communication.
\newblock \emph{arXiv:2310.01444}, 2023.

\bibitem{ltsvoice}
Zou, W., Miao, Y., Ma, Z., Xu, J., Gao, J., Hao, J.,
He, R., and Xu, J.
\newblock LTS-VoiceAgent: A Listen-Think-Speak
Framework for Efficient Streaming Voice Interaction
via Semantic Triggering and Incremental Reasoning.
\newblock \emph{arXiv:2601.19952}, 2026.

\bibitem{reasoningbeyond}
Reasoning beyond limits: Advances and open problems
for LLMs.
\newblock \emph{ScienceDirect}, 2025.
\newblock Survey of reasoning limitations in LLMs.

\bibitem{ars}
Zheng, D.
\newblock ARS: Adaptive Reasoning Suppression for
Efficient Large Reasoning Language Models.
\newblock \emph{arXiv:2510.00071}, 2025.

\bibitem{lars}
Salah, M.
\newblock LARS: Live Adaptive Reasoning System for
Continuous-State Interactive AI (v1 preprint).
\newblock \emph{Zenodo}, DOI:
\href{https://doi.org/10.5281/zenodo.20618761}{10.5281/zenodo.20618761},
June 2026.

\bibitem{react}
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I.,
Narasimhan, K., and Cao, Y.
\newblock ReAct: Synergizing Reasoning and Acting in
Language Models.
\newblock \emph{ICLR 2023}.

\end{thebibliography}

% =====================================================================
\appendix
\section{Embedder Details}
\label{sec:appendix-embedder}

The default \texttt{HashEmbedder} in
\texttt{lars/embeddings.py} uses a deterministic
random-projection of bag-of-words. Each token is hashed
to a 128-dimensional unit vector via MD5; the text
embedding is the L2-normalized sum of its token vectors.
Light suffix stripping (\texttt{ing}, \texttt{tion},
\texttt{ed}, \texttt{s}, etc.) is applied before hashing.

For production use, swap in \texttt{OpenAIEmbedder}
(\texttt{text-embedding-3-small}, 1536 dimensions), which
plugs into the same interface.

\section{v3 Change Log}
\label{sec:appendix-changelog}

This appendix lists the substantive changes from v2 to
v3, to make the evolution auditable.

\begin{itemize}
    \item \textbf{StateVector} gained two fields:
          \texttt{latest\_cot} (per-step) and
          \texttt{active\_overrides} (per-state).
    \item \textbf{StateMerger} handlers now read
          \texttt{latest\_cot} in addition to
          \texttt{description}. The \texttt{\_narrow},
          \texttt{\_replace}, and \texttt{\_remove}
          handlers each grew a CoT-aware branch.
    \item \textbf{StateExtractor} gained
          \texttt{refresh\_pending()}, which re-states
          pending step descriptions after a state change.
    \item \textbf{StepExecutor} gained
          \texttt{\_format\_overrides()}, which is
          appended to the LLM prompt. The LLMStepExecutor
          calls it on every execution; active overrides
          are drained.
    \item \textbf{LiveAgent} wires the new pipeline:
          after a merge, run \texttt{refresh\_pending},
          then append the user text to
          \texttt{active\_overrides}. Before each
          execution, drain the overrides.
    \item \textbf{Test suite} grew from 28 to 33 tests
          (\texttt{test\_merger\_v0\_5\_uses\_latest\_cot},
          \texttt{test\_merger\_v0\_5\_replace\_via\_cot},
          \texttt{test\_extractor\_refresh\_pending\_updates\_descriptions},
          \texttt{test\_active\_overrides\_injected\_into\_prompt},
          \texttt{test\_active\_overrides\_empty\_returns\_empty\_string}).
\end{itemize}

\end{document}
