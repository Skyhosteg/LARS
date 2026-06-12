% LARS Paper v2 — Section: Limitations
% Mohamed Salah, 2026
% Replaces / extends the original Section 8 (Discussion and Future Work).

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
not all of Egypt''}), the handler may match the
wrong token and produce a grammatically broken state. A v2
\emph{$f_\text{LLM}$} ablation---where the merge is delegated
to an LLM---is the highest-priority extension.

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
    \item \textbf{Learned merge function} $f_\theta$. Replace
          the 9 rule-based handlers with a small
          transformer that takes $(S(t), \Delta U)$ and
          predicts the merge trace. The training signal
          comes from the v1 system; a learned
          $f_\theta$ can then be ablated against $f_\text{rule}$.
    \item \textbf{Conflict detection.} When $\Delta U$
          contradicts an existing decision in $S(t)$, the
          merger should detect the conflict and either
          resolve it (via a learned policy) or escalate
          to the user. v1 silently overwrites.
    \item \textbf{Multimodal interrupts.} Voice, screen
          share, and gesture-based interrupts. The
          LARS-VoiceAgent integration in
          \texttt{lars/} is a starting point.
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
