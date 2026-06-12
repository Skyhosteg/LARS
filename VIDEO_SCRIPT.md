# LARS Demo Video — Production Script
**Target length:** 60 seconds (Twitter/LinkedIn) or 2 minutes (YouTube)
**Format:** 1080p, 30fps, dark terminal, large font
**Style:** No face, just terminal + on-screen text

---

## Pre-Recording Checklist

- [ ] OBS Studio installed (or Win+G ready)
- [ ] Windows Terminal open, dark theme, font 16+
- [ ] Repository cloned and extracted
- [ ] Dependencies installed (`pip install pydantic openai`)
- [ ] Audio test (if doing voice-over)
- [ ] Phone silent, notifications off

---

## Take A: "60-Second Twitter Cut"

### Scene 1 — Cold Open (0:00 - 0:05)

**Visual:** Zoomed terminal, dark theme. Static frame.

**On-screen text (add later in CapCut):**
```
Every LLM ignores you OR restarts from scratch.
What if it just... continued?
```

**Action:** Static, no typing. 5 seconds.

---

### Scene 2 — Launch (0:05 - 0:10)

**Type this in PowerShell:**
```powershell
cd C:\Users\Dell\Downloads\LARS-v0.4.0\lars
python examples/demo_live.py
```

**Wait for:** LARS extracts initial state. ~2 sec.

---

### Scene 3 — The Goal (0:10 - 0:18)

**When prompted "Enter your goal":**
```
Create a marketing plan for a fitness app in Egypt
```

**Wait for:** LARS to plan and execute Step 1.

**Highlight:** The status line: `steps=1✓/4⏳`

---

### Scene 4 — First Interrupt (0:18 - 0:32) ⭐ THE MONEY SHOT

**After Step 1 completes, when you see the prompt "press Enter to continue":**

**Type:**
```
focus on Cairo only
```

**Wait. Don't touch anything.** Let the camera catch:
- `[USER] focus on Cairo only` (red)
- `[ΔU] type=scope_narrow target=geo new='Cairo' conf=0.90` (magenta)
- `[F] MergeTrace: ... α=0.60 β=0.30 γ=0.10` (magenta)
- `[STATE] S(v=2) ... conf=0.48` (green)

**Duration:** ~14 seconds. This is the climax — let it breathe.

---

### Scene 5 — Continue (0:32 - 0:50)

**Press Enter, Enter, Enter** to let LARS execute Steps 2, 3, 4, 5.

**After Step 3 completes, type:**
```
use Twitter instead of Facebook
```

**After Step 4 completes, press Enter.**

**After Step 5 completes, session ends.**

---

### Scene 6 — Outro (0:50 - 1:00)

**Visual:** Switch to browser showing the GitHub repo.

**On-screen text:**
```
github.com/Skyhosteg/LARS
zenodo.org/records/20618761

Live Adaptive Reasoning under Continuous Interruption
```

**No narration needed. Just hold the frame.**

---

## Take B: "2-Minute YouTube Cut"

Use the same flow but add:

### Insert 1 (after Scene 2): Show the architecture
- Open `docs/ARCHITECTURE.md` in browser for 10 sec
- Or show the figure in the PDF for 10 sec

### Insert 2 (after Scene 4): Show the benchmark
- Run `python examples/run_benchmark.py` (no input needed)
- Let it print the trade-off table
- Hold on the final table for 5 sec

### Insert 3 (after Scene 6): Voice-over
If you have a mic, narrate:
> "LARS is the first LLM runtime that preserves chain-of-thought
> when the user interrupts. The benchmark shows it's the only
> method with both high reasoning preservation AND interrupt
> incorporation. Code is MIT-licensed, paper is on Zenodo."

---

## Take C: "5-Minute Deep Dive" (for YouTube long form)

Add:
- The paper PDF walkthrough
- Live code review of `lars/merger.py`
- Live benchmark with real OpenAI (if you have a key)
- Q&A from comments

---

## Editing Checklist (CapCut, free)

After recording, do these edits in **CapCut Desktop**:

1. **Trim dead time** at the start and end
2. **Add zoom on the merge trace** (the climax at 0:18-0:32)
3. **Captions on key moments**:
   - "LARS plans the steps..."
   - "I interrupt mid-reasoning..."
   - "It preserves 60%, applies 30%, adapts 10%"
4. **Background music** (royalty-free, low volume)
5. **Title card** at 0:00 with "LARS — Live Adaptive Reasoning"
6. **End card** with links to GitHub + Zenodo

---

## Filming Tips

- **DO** use a dark terminal theme. White-on-black is standard for coding videos.
- **DO** keep the font large. People watch on phones.
- **DO** pause after typing — let the viewer read.
- **DON'T** talk over the merge trace. The visual IS the message.
- **DON'T** add music during the demo. Save it for intro/outro.
- **DON'T** apologize for mistakes. Just re-take the scene.

---

## Upload Plan

| Platform | Use this cut | Title |
|---|---|---|
| Twitter | Take A (60s) | "I built an LLM that survives being interrupted" |
| LinkedIn | Take A (60s) | "LARS: the first LLM runtime that preserves CoT under user interruption" |
| YouTube Shorts | Take A (60s, vertical) | "Your LLM should not restart when you interrupt it" |
| YouTube long | Take C (5min) | "LARS: Live Adaptive Reasoning under Continuous Interruption" |
| GitHub README | Take A → GIF (10MB) | Alt text: "Demo of LARS mid-reasoning interruption" |

---

**Pro tip:** The 60-second Take A is the most important.
If you only have time for one video, do that one.
The merge trace at 0:18-0:32 is the ONLY part that needs to be perfect.

Press record when ready. 🎬
