% LARS Paper v2 — Section 2: Related Work
% Mohamed Salah, 2026
% Inserts as a new Section 2; renumber subsequent sections.

\section{Related Work}
\label{sec:related}

We situate LARS in four clusters of prior work, summarized
in Table~\ref{tab:related}.

\begin{table}[h]
\centering
\caption{Positioning of LARS against the four most relevant
research clusters. LARS is the only system that combines
continuous interruption with explicit state preservation.}
\label{tab:related}
\begin{tabular}{p{3.5cm}p{2.5cm}p{2.5cm}p{4cm}}
\hline
\textbf{Cluster} & \textbf{Representative} & \textbf{Interruption} & \textbf{Limitation} \\
\hline
Checkpoint-based interruption &
LangGraph \texttt{interrupt\_before} &
Node-level &
No mid-reasoning interrupt; no merge \\
\hline
Interactive chain-of-thought &
Pang et al.\ 2025 (arXiv 2506.23678) &
Post-hoc edit &
Not live; requires full CoT first \\
\hline
Adaptive planning &
AdaPlanner (Sun et al., NeurIPS 2023) &
Environment feedback &
Not user-driven; plan is regenerated \\
\hline
Streaming voice interruption &
LTS-VoiceAgent (arXiv 2601.19952, 2026) &
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
LangGraph~\cite{langgraph} provides \texttt{interrupt\_before}
and \texttt{interrupt\_after} hooks that pause graph execution
at node boundaries for human approval. The system is designed
for approval workflows (e.g., before sending an email) rather
than mid-reasoning steering. On resume, the human's input
replaces the pending state; no merge with prior reasoning is
performed.

\paragraph{Interactive chain-of-thought.}
Pang et al.~\cite{pang2025} recently introduced
\textit{Interactive Reasoning}, which visualizes chain-of-thought
as a topic hierarchy and allows users to edit the hierarchy
after the CoT is complete. Their user study demonstrates the
demand for steering CoT, but the interaction is post-hoc: the
user cannot interrupt the model mid-generation. LARS delivers
the same demand \emph{real-time}.

\paragraph{Adaptive planning.}
AdaPlanner~\cite{adaplanner} refines an LLM-generated plan in
response to environment feedback using in-plan and out-of-plan
refinement strategies. The feedback source is the world, not
the user, and the refinement loop re-runs the LLM. LTC~\cite{ltc}
introduces a universal feedback buffer for agent adaptation.
Both target autonomous settings, not interactive reasoning.

\paragraph{Streaming voice interruption.}
LTS-VoiceAgent~\cite{ltsvoice} (Jan 2026) addresses the latency
of cascaded voice pipelines by semantic-triggered incremental
reasoning. When the user interrupts, the partial reasoning is
discarded; the agent restarts. LARS preserves and merges the
partial state instead.

\paragraph{Conceptual neighbors.}
Outside interactive agents, our model has a natural reading in
\emph{event-driven systems} and \emph{stream processing}. The
$\alpha + \beta + \gamma$ decomposition is reminiscent of
CRDT merge operators, but our setting differs in that the
update vector $\Delta U$ is human-authored and unconstrained.
We discuss this connection in Section~\ref{sec:future}.

% References for the citations above (add to bibliography):
% \bibitem{langgraph} LangGraph Documentation.
%   \url{https://langchain-ai.github.io/langgraph/}, 2024-2026.
% \bibitem{pang2025} Pang, Feng et al.
%   Interactive Reasoning: Visualizing and Controlling
%   Chain-of-Thought Reasoning in Large Language Models.
%   arXiv:2506.23678, 2025.
% \bibitem{adaplanner} Sun, H. et al.
%   AdaPlanner: Adaptive Planning from Feedback with
%   Language Models. NeurIPS 2023.
% \bibitem{ltc} Wang, K. et al.
%   Adapting LLM Agents with Universal Feedback in
%   Communication. arXiv:2310.01444, 2023.
% \bibitem{ltsvoice} Zou, W. et al.
%   LTS-VoiceAgent: A Listen-Think-Speak Framework for
%   Efficient Streaming Voice Interaction. arXiv:2601.19952, 2026.
