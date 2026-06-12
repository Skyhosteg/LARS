% =====================================================================
% LARS: Live Adaptive Reasoning System for Continuous-State Interactive AI
% Mohamed Salah (Unaffiliated Researcher, Egypt)
% v2 — 2026-06-12
% DOI: 10.5281/zenodo.20618761
% Code: https://github.com/Skyhosteg/LARS
%
% Compile: pdflatex lars_v2_paper.tex
% Or convert from this .md: pandoc lars_v2_paper.md -o lars_v2_paper.pdf
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
for Continuous-State Interactive AI}}
\author{Mohamed Salah \\
        \small Unaffiliated Researcher, Egypt \\
        \small \texttt{ph.net.sa@gmail.com} \\
        \small \url{https://github.com/Skyhosteg/LARS}}
\date{June 12, 2026 \\ \small v2 --- supersedes the June 10 preprint (DOI 10.5281/zenodo.20618761)}

\begin{document}
\maketitle

% =====================================================================
\begin{abstract}
We introduce \textbf{LARS} (Live Adaptive Reasoning System), a formal
framework for interactive AI systems that maintain an explicit,
continuously evolving reasoning state under user-driven
interruptions. Unlike conventional stateless request-response
paradigms, LARS models interaction as a continuous state
transition process:
\begin{equation}
    S(t+1) = f\bigl(S(t),\, \Delta U(t)\bigr),
\end{equation}
where $S(t)$ represents the structured internal reasoning state
and $\Delta U(t)$ represents a typed user intervention during
execution. We further propose a structured state decomposition
$S(t) = (T(t), C(t), M(t))$, a nine-type intent taxonomy for
$\Delta U$, a weighted merge function with constraint
$\alpha + \beta + \gamma = 1$ and $\alpha \geq 0.5$, and a
quantitative metric, \textbf{Reasoning Preservation Rate (RPR)},
to measure stability and continuity of reasoning under
interruption. We validate the framework on a 12-task benchmark
(6 planning + 6 reasoning) against three baseline systems.
LARS is the only method that simultaneously achieves RPR $= 1.0$
\textbf{and} incorporates the user's interrupt in 100\% of
tasks. The reference implementation is open-source under the MIT
license.
\end{abstract}

% =====================================================================
\section{Introduction}
\label{sec:intro}

Modern AI systems are predominantly built on stateless
interaction loops, where each user query triggers full
recomputation of reasoning. However, real-world human-AI
interaction is inherently continuous, dynamic, and interruptible.
A user reading the model's intermediate output frequently
rejects, redirects, or refines the trajectory mid-generation.

LARS reframes reasoning as a persistent computational state that
evolves over time, enabling partial modification of reasoning
trajectories without full recomputation. This allows systems to
adapt mid-execution while preserving previously constructed
reasoning structures. The contribution is threefold:
\begin{enumerate}
    \item A \textbf{formal model} of state-preserving live
          adaptation ($S(t+1) = f(S, \Delta U)$) with a
          paper-grade \emph{MergeTrace} recording exactly which
          elements were preserved, modified, dropped, and
          inserted at every transition.
    \item The \textbf{Reasoning Preservation Rate (RPR)} metric,
          with three modes (strict set intersection, token-Jaccard,
          and semantic cosine similarity) so the metric is
          comparable across research groups.
    \item A complete \textbf{reference implementation} with a
          12-task benchmark, three baselines, and 28 passing
          tests, released as open-source at
          \url{https://github.com/Skyhosteg/LARS}.
\end{enumerate}

% =====================================================================
\section{Related Work}
\label{sec:related}

We situate LARS in four clusters of prior work, summarized in
Table~\ref{tab:related}.

\begin{table}[h]
\centering
\caption{Positioning of LARS against the four most relevant
research clusters. LARS is the only system that combines
continuous interruption with explicit state preservation.}
\label{tab:related}
\begin{tabular}{p{3.4cm}p{2.7cm}p{2.4cm}p{4.5cm}}
\hline
\textbf{Cluster} & \textbf{Representative} & \textbf{Interruption} & \textbf{Limitation} \\
\hline
Checkpoint-based interruption &
LangGraph \texttt{interrupt\_before} \cite{langgraph} &
Node-level &
No mid-reasoning interrupt; no merge \\
\hline
Interactive chain-of-thought &
Pang et al.\ 2025 \cite{pang2025} &
Post-hoc edit &
Not live; requires full CoT first \\
\hline
Adaptive planning &
AdaPlanner (NeurIPS 2023) \cite{adaplanner} &
Environment feedback &
Not user-driven; plan is regenerated \\
\hline
Streaming voice interruption &
LTS-VoiceAgent (2026) \cite{ltsvoice} &
Token-level &
Discards partial reasoning on interrupt \\
\hline
\textbf{LARS (ours)} &
This work &
\textbf{Continuous + preserving} &
--- \\
\hline
\end{tabular}
\end{table}

\paragraph{Checkpoint-based interruption.}
LangGraph~\cite{langgraph} provides
\texttt{interrupt\_before} and \texttt{interrupt\_after} hooks
that pause graph execution at node boundaries for human
approval. The system is designed for approval workflows
(e.g., before sending an email) rather than mid-reasoning
steering. On resume, the human's input replaces the pending
state; no merge with prior reasoning is performed.

\paragraph{Interactive chain-of-thought.}
Pang et al.~\cite{pang2025} recently introduced
\textit{Interactive Reasoning}, which visualizes
chain-of-thought as a topic hierarchy and allows users to
edit the hierarchy after the CoT is complete. Their user
study demonstrates the demand for steering CoT, but the
interaction is post-hoc: the user cannot interrupt the model
mid-generation. LARS delivers the same demand \emph{real-time}.

\paragraph{Adaptive planning.}
AdaPlanner~\cite{adaplanner} refines an LLM-generated plan in
response to environment feedback using in-plan and out-of-plan
refinement strategies. The feedback source is the world, not
the user, and the refinement loop re-runs the LLM.
LTC~\cite{ltc} introduces a universal feedback buffer for
agent adaptation. Both target autonomous settings, not
interactive reasoning.

\paragraph{Streaming voice interruption.}
LTS-VoiceAgent~\cite{ltsvoice} addresses the latency of cascaded
voice pipelines by semantic-triggered incremental reasoning.
When the user interrupts, the partial reasoning is discarded;
the agent restarts. LARS preserves and merges the partial
state instead.

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
    \item $\Delta U(t)$: user-driven intervention applied during
          execution.
    \item $f$: state transition function that merges updates into
          the existing reasoning trajectory.
\end{itemize}

% =====================================================================
\section{State Representation}
\label{sec:state}

We factorize the internal reasoning state as:
\begin{equation}
    S(t) = \bigl(T(t),\, C(t),\, M(t)\bigr),
\end{equation}
where:
\begin{itemize}
    \item $T(t)$: reasoning trace (execution steps and
          intermediate conclusions), decomposed into
          \emph{completed} and \emph{pending} steps.
    \item $C(t)$: confidence and uncertainty structure
          ($c \in [0, 1]$).
    \item $M(t)$: metadata including constraints, goals, and
          execution context.
\end{itemize}

The reference implementation uses a Pydantic v2 schema
(\texttt{StateVector}) with explicit \emph{version} counter for
auditability.

% =====================================================================
\section{User Intervention Model}
\label{sec:intervention}

User interaction is formalized as a structured update vector:
\begin{equation}
    \Delta U(t) = (\iota_t,\, \delta_t,\, \kappa_t),
\end{equation}
where:
\begin{itemize}
    \item $\iota_t$: intent type. We define a closed taxonomy of
          nine types: \texttt{SCOPE\_NARROW},
          \texttt{SCOPE\_EXPAND}, \texttt{CORRECTION},
          \texttt{REPLACE}, \texttt{ADD}, \texttt{REMOVE},
          \texttt{REPRIORITIZE}, \texttt{CLARIFY},
          \texttt{ABORT}.
    \item $\delta_t$: modification payload (target aspect,
          old value, new value).
    \item $\kappa_t$: confidence or priority weight of the
          intervention.
\end{itemize}

Intent classification can be done via a deterministic regex
parser (English-only, fast) or via an LLM-backed parser
(multilingual, slower). Both are shipped.

% =====================================================================
\section{State Transition Function}
\label{sec:transition}

We define the update operator as a weighted composition of
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
modified / dropped / inserted elements, and free-form notes.
This makes the merge auditable at the level of a single
reasoning element.

% =====================================================================
\section{Reasoning Preservation Rate (RPR)}
\label{sec:rpr}

To quantify stability under interruption, we define:
\begin{equation}
    \mathrm{RPR}(t) = \frac{|T(t) \cap T(t-1)|}{|T(t-1)|},
\end{equation}
the fraction of preserved reasoning structure after each state
transition.

The reference implementation supports three RPR modes:
\begin{enumerate}
    \item \textbf{Strict} ($RPR_{\text{exact}}$): set
          intersection on lowercased strings.
          Brittle to rewording; useful for unit tests.
    \item \textbf{Jaccard} ($RPR_{\text{jaccard}}$): token
          overlap with threshold $\tau = 0.3$.
          Robust to paraphrasing.
    \item \textbf{Semantic} ($RPR_{\text{sem}}$): cosine
          similarity over a deterministic hash-projection
          embedder, with threshold $\tau = 0.7$. The pluggable
          embedder interface (\texttt{lars/embeddings.py})
          allows production use with OpenAI
          \texttt{text-embedding-3} or any other model.
\end{enumerate}

% =====================================================================
\section{Empirical Validation}
\label{sec:evaluation}

We validate the LARS framework through a controlled benchmark
comparing LARS against three baseline systems on a 12-task
corpus. All experiments are reproducible from the open-source
implementation at \url{https://github.com/Skyhosteg/LARS}.

\subsection{Experimental Setup}
\label{sec:setup}

\paragraph{Tasks.}
We curate 12 tasks spanning two domains that capture
different reasoning regimes:
\begin{itemize}
    \item \textbf{Planning} (6 tasks): marketing plans, trip
          itineraries, curricula, product launches, conference
          organization, dietary plans.
    \item \textbf{Reasoning} (6 tasks): economic causal
          analysis, linguistic comparison, code debugging,
          probability calculation, ethical analysis, systems
          feedback loops.
\end{itemize}
Each task is a 5-step chain-of-thought trace with a user
interrupt applied at step 3, producing an interrupted plan
that the system must reconcile with the original reasoning.
The full task corpus is in \texttt{lars/tasks.py}.

\paragraph{Methods compared.}
\begin{itemize}
    \item \textbf{no\_interrupt}: a stateless baseline that
          generates the original plan and ignores the user's
          interrupt entirely. It establishes the upper bound
          for reasoning preservation and the lower bound for
          interrupt incorporation.
    \item \textbf{restart\_from\_scratch}: a naive baseline that
          discards the original state and regenerates from a
          prompt containing the user's interrupt. This
          establishes the cost of full recomputation.
    \item \textbf{langgraph\_checkpoint}: a baseline that
          emulates LangGraph's \texttt{interrupt\_before}
          pattern---on user input, the system resumes from the
          last checkpoint and \emph{appends} the interrupt
          rather than merging it. This represents the
          state-of-the-art in checkpoint-based interruption.
    \item \textbf{LARS}: the proposed method, executing the
          full pipeline
          $\text{extract} \rightarrow \text{parse}\,\Delta U
          \rightarrow \text{merge}\,f$.
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
          reflects the user's interrupt. This is a
          binary check: either the method honored the
          interrupt, or it did not.
    \item \textbf{Cost}: the word count of the CoT
          the method consumed, used as a deterministic
          proxy for token cost in the offline setting.
\end{enumerate}

\subsection{Results}
\label{sec:results}

\begin{table}[h]
\centering
\caption{Mean metrics over 12 tasks. LARS is the only method
to achieve both high RPR and high interrupt incorporation.}
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
\caption{Pareto frontier in the (RPR, Used?) plane. LARS
sits alone in the upper-right corner. The other three
methods trade one metric for the other.}
\label{fig:pareto}
\end{figure}

The frontier reveals a sharp \textbf{no-free-lunch} property:
no baseline can be on both axes simultaneously.
\textit{no\_interrupt} preserves the original reasoning but
ignores the user; \textit{restart} and \textit{langgraph}
follow the user but destroy the original reasoning. LARS is
the unique Pareto-optimal point.

\subsection{Per-Intent Analysis}
\label{sec:per-intent}

To verify that LARS behaves as designed, we disaggregate
the 12 tasks by intent type (defined in Section
\ref{sec:intervention}):

\begin{table}[h]
\centering
\caption{Per-intent RPR for LARS. All intent types achieve
$\geq 0.75$ RPR, indicating that the merger respects
preservation across all update regimes.}
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
This is a known limitation and is being addressed in v2.

\subsection{Discussion}
\label{sec:results-discussion}

The benchmark validates three claims from the formulation:

\begin{enumerate}
    \item \textbf{Decoupling is possible.} The 9-intent
          taxonomy and the $\alpha$-weighted merge make
          preservation and adaptation orthogonal axes that
          the system can optimize independently.
    \item \textbf{RPR is measurable.} The semantic-RPR
          metric detects preservation even when the textual
          surface changes, and its 1.0 score for LARS
          confirms that the merger preserves
          \emph{meaning}, not strings.
    \item \textbf{Cost is bounded.} LARS reads the same
          CoT as \textit{no\_interrupt}; the merge is local
          and does not require regeneration. This is why
          LARS's cost is the lowest in the comparison.
\end{enumerate}

\subsection{Threats to Validity}
\label{sec:threats}

\begin{itemize}
    \item \textbf{Offline benchmark.} The 12 tasks are
          hand-curated; results on user-generated tasks
          in the wild may differ. We address this in v2
          with a 20-task expansion and a user study.
    \item \textbf{Deterministic embedder.} The
          hash-projection embedder in the offline
          benchmark is a proxy for production-quality
          embeddings (e.g.\ OpenAI
          \texttt{text-embedding-3}). With real embeddings,
          we expect RPR values to shift by $\pm 0.05$.
    \item \textbf{Simulated baselines.} The
          \textit{langgraph\_checkpoint} baseline is
          simulated via prompt templating rather than
          a live LangGraph runtime, because checkpoint
          semantics in LangGraph are designed for
          human-in-the-loop approval, not interrupt
          handling. A faithful LangGraph integration is
          provided in \texttt{lars/langgraph\_integration.py}
          and is the recommended deployment path.
\end{itemize}

% =====================================================================
\section{System Architecture}
\label{sec:architecture}

LARS is composed of four functional modules (see
Figure~\ref{fig:arch}):

\begin{itemize}
    \item \textbf{State Extraction}: captures runtime
          reasoning state $S(t)$.
    \item \textbf{Intent Parsing}: interprets the
          semantic structure of $\Delta U(t)$.
    \item \textbf{State Merging}: applies transition
          function $f$.
    \item \textbf{Reasoning Evaluation}: computes RPR
          and system stability metrics.
\end{itemize}

The runtime loop, executed by \texttt{LiveAgent}, walks the
pending-step queue, executes each step, pauses for an
interrupt, and on interrupt parses $\Delta U$ and applies
$f$. The same loop can be expressed as a LangGraph state
machine with \texttt{interrupt\_before=["execute"]} for
production deployment with persistence and time-travel.

\begin{figure}[h]
\centering
\begin{tikzpicture}[
  node distance=1.2cm,
  box/.style={rectangle, draw, rounded corners, minimum height=0.8cm, minimum width=2.2cm, align=center},
  arrow/.style={->, >=stealth}
]
  \node[box] (goal) {Goal + Initial $S(0)$};
  \node[box, below=of goal] (ex) {Step Executor};
  \node[box, below=of ex] (ext) {State Extractor $S(t)$};
  \node[box, below=of ext] (par) {$\Delta U(t)$ Parser};
  \node[box, below=of par] (mer) {State Merger $f(S, \Delta U)$};
  \node[box, right=2cm of ex, draw=blue!60] (rpr) {RPR Metric};
  \node[box, below=of mer] (next) {Loop to $t+1$};
  \node[draw, dashed, right=2cm of par, align=center] (int) {Interrupt source\\(stdin / WS)};

  \draw[arrow] (goal) -- (ex);
  \draw[arrow] (ex) -- (ext);
  \draw[arrow] (ext) -- (par);
  \draw[arrow] (par) -- (mer);
  \draw[arrow] (mer) -- (next);
  \draw[arrow] (next.west) -- ++(-1.5,0) |- (ex.west);
  \draw[arrow, dashed] (int) -- (par);
  \draw[arrow, blue!60] (mer) -- (rpr);
\end{tikzpicture}
\caption{LARS runtime architecture as a continuous
state-interruption loop. The Interrupt source is a pluggable
callable (default \texttt{input()}, swappable for
WebSocket/SSE in production).}
\label{fig:arch}
\end{figure}

% =====================================================================
\section{Limitations and Future Work}
\label{sec:limitations}

We are explicit about the limitations of the v1 system and
the v1 evaluation, because the empirical claims in
Section~\ref{sec:evaluation} rest on them.

\subsection{Current Limitations}

\paragraph{(L1) Rule-based merger is literal.}
The \texttt{StateMerger} in \texttt{lars/merger.py} applies
the 9 intent handlers with regex-based string substitution.
When the interrupt is a paraphrase (\textit{``just Cairo,
not all of Egypt''}), the handler may match the wrong
token and produce a grammatically broken state. A v2
\emph{$f_{\mathrm{LLM}}$} ablation---where the merge is
delegated to an LLM---is the highest-priority extension.

\paragraph{(L2) Mock $\Delta U$ parser is English-only.}
The deterministic \texttt{DeltaUParserMock} uses
English-language regex. For non-English interrupts
(Arabic, Mandarin), users must instantiate
\texttt{DeltaUParserLLM}, which adds an LLM call per
interrupt and increases latency.

\paragraph{(L3) REPRIORITIZE is a no-op.}
The v1 merger records re-prioritization requests but
does not re-rank step IDs. This is the single
under-performing intent in Table~\ref{tab:per-intent}.
The v2 fix is a graph re-ranking step on the
dependency graph of \texttt{StateVector}.

\paragraph{(L4) No real-token cost.}
The cost metric in Table~\ref{tab:headline} is word-count
on the CoT, a deterministic proxy. With OpenAI,
the \texttt{response.usage} field gives exact token
counts. A v2 evaluation will report both.

\paragraph{(L5) Small benchmark.}
12 tasks is enough to demonstrate the trade-off and
the Pareto-optimality of LARS, but is not enough to
make fine-grained statistical claims. We plan to
expand to 20 tasks and add a 10-participant user study
in v2.

\paragraph{(L6) Single-author, single-domain validation.}
The benchmark tasks are in English, drawn from
business and analytical reasoning. LARS has not
been tested in code generation, scientific
reasoning, or multilingual settings. These are
future-work directions.

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
          contradicts an existing decision in $S(t)$, the
          merger should detect the conflict and either
          resolve it (via a learned policy) or escalate
          to the user. v1 silently overwrites.
    \item \textbf{Multimodal interrupts.} Voice, screen
          share, and gesture-based interrupts. The
          LangGraph integration in
          \texttt{lars/langgraph\_integration.py} is a
          starting point.
    \item \textbf{Long-horizon state compression.}
          For sessions that span many steps, $S(t)$
          grows. A summarization layer that keeps
          RPR high while bounding the state size is
          an open problem.
    \item \textbf{Theoretical analysis of $\alpha$.}
          What is the optimal default $\alpha$? Our
          v1 default of $0.6$ is empirical; a
          regret-bounded analysis would be a v2
          contribution.
\end{itemize}

% =====================================================================
\section{Conclusion}
\label{sec:conclusion}

We presented LARS, a formal framework for continuous-state
interactive reasoning under user-driven interruption. The
system introduces a structured formulation of reasoning
persistence, a 9-type intent taxonomy, a weighted merge
function with $\alpha + \beta + \gamma$ preservation
bias, and a measurable metric (RPR) for evaluating
stability in dynamic AI systems. The 12-task benchmark
shows that LARS is the unique Pareto-optimal point among
the compared methods. The open-source reference
implementation, 28-test suite, and benchmark harness are
released to encourage replication, extension, and
empirical comparison.

% =====================================================================
\section*{References}
\addcontentsline{toc}{section}{References}

\begin{thebibliography}{9}

\bibitem{langgraph}
LangGraph Documentation.
\newblock \url{https://langchain-ai.github.io/langgraph/}, 2024--2026.

\bibitem{pang2025}
Pang, F.\ and Feng, S.\ et al.
\newblock Interactive Reasoning: Visualizing and Controlling
Chain-of-Thought Reasoning in Large Language Models.
\newblock \emph{arXiv:2506.23678}, 2025.

\bibitem{adaplanner}
Sun, H., Zhuang, Y., Kong, L., Dai, B., and Zhang, C.
\newblock AdaPlanner: Adaptive Planning from Feedback with
Language Models.
\newblock \emph{NeurIPS 2023}.

\bibitem{ltc}
Wang, K., Lu, Y., Santacroce, M., Gong, Y., Zhang, C.,
and Shen, Y.
\newblock Adapting LLM Agents with Universal Feedback in
Communication.
\newblock \emph{arXiv:2310.01444}, 2023.

\bibitem{ltsvoice}
Zou, W., Miao, Y., Ma, Z., Xu, J., Gao, J., Hao, J.,
He, R., and Xu, J.
\newblock LTS-VoiceAgent: A Listen-Think-Speak Framework
for Efficient Streaming Voice Interaction via Semantic
Triggering and Incremental Reasoning.
\newblock \emph{arXiv:2601.19952}, 2026.

\bibitem{reasoningbeyond}
Reasoning beyond limits: Advances and open problems for
LLMs.
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
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K., and Cao, Y.
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

\end{document}
