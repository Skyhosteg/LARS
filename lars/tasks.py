"""
tasks.py — The 12 benchmark tasks

Two domains, six tasks each:

  PLANNING: tasks where the user asks for a multi-step plan
  REASONING: tasks where the user asks for a logical analysis

Each task has:
  - id
  - domain
  - goal (the user request)
  - initial_cot (a pre-generated chain-of-thought for the original request)
  - interrupt (a free-form user correction mid-reasoning)
  - intent (what the interrupt should classify to)
  - expected_preservation (rough expectation: how much should be preserved)

The interrupt is at step 3 of 5 — i.e., after 60% of the plan is
generated, the user changes direction. This is the most realistic
real-world scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkTask:
    id: str
    domain: str  # "planning" | "reasoning"
    goal: str
    initial_cot: str
    interrupt: str
    intent: str
    expected_rpr: float = 0.5  # rough expectation
    notes: str = ""


# --------------------------------------------------------------------------- #
# PLANNING (6 tasks)
# --------------------------------------------------------------------------- #

PLANNING_TASKS = [
    BenchmarkTask(
        id="plan-marketing-fitness-egypt",
        domain="planning",
        goal="Create a marketing plan for a fitness app targeting young adults in Egypt.",
        initial_cot=(
            "Step 1: Analyze the market. Egypt has 110M people, 60% under 30, growing fitness app adoption. "
            "Target all major cities: Cairo, Alexandria, Giza. "
            "Step 2: Define the audience. Primary: 18-30 urban middle income, health-conscious. "
            "Step 3: Choose channels. Instagram for engagement, TikTok for reach, local fitness influencers for trust. "
            "Step 4: Allocate budget. 40% paid social, 30% influencers, 20% content, 10% events. "
            "Step 5: Define KPIs. Downloads, 30-day retention, CAC, LTV."
        ),
        interrupt="focus on Cairo only",
        intent="scope_narrow",
        expected_rpr=0.75,
        notes="Classic LARS demo. Should preserve channels, modify geo.",
    ),
    BenchmarkTask(
        id="plan-trip-japan",
        domain="planning",
        goal="Plan a 7-day trip to Japan in October for a couple interested in culture and food.",
        initial_cot=(
            "Step 1: Pick cities. Tokyo (4 days) for culture, Kyoto (3 days) for traditional. "
            "Step 2: Book flights. Round-trip from Cairo to Tokyo, ~$1200 per person. "
            "Step 3: Reserve hotels. Mid-range ryokan in Kyoto, business hotel in Tokyo. "
            "Step 4: Plan activities. Senso-ji, Meiji shrine, Tsukiji market, Arashiyama bamboo. "
            "Step 5: Budget. Total ~$5000 including flights, hotels, food, JR pass."
        ),
        interrupt="actually we want to include Osaka too",
        intent="scope_expand",
        expected_rpr=0.85,
        notes="Add a new city — channels/structure preserved.",
    ),
    BenchmarkTask(
        id="plan-curriculum-python",
        domain="planning",
        goal="Design a 12-week curriculum to teach Python to high school students with no programming experience.",
        initial_cot=(
            "Step 1: Weeks 1-2: Python basics (variables, types, I/O). "
            "Step 2: Weeks 3-4: Control flow (if/else, loops). "
            "Step 3: Weeks 5-6: Functions and modules. "
            "Step 4: Weeks 7-8: Lists, dicts, and data structures. "
            "Step 5: Weeks 9-10: A small project (CLI tool). "
            "Step 6: Weeks 11-12: Intro to OOP and final project."
        ),
        interrupt="actually make it 8 weeks not 12",
        intent="correction",
        expected_rpr=0.50,
        notes="Compression of the curriculum, content should be condensed.",
    ),
    BenchmarkTask(
        id="plan-product-launch",
        domain="planning",
        goal="Plan the launch of a new B2B SaaS product in Q2 2026.",
        initial_cot=(
            "Step 1: Define ICP. Mid-market SaaS companies, 50-500 employees, $5M-$50M ARR. "
            "Step 2: Build waitlist. Landing page, content marketing, partner referrals. "
            "Step 3: Beta program. 20 design partners, weekly feedback, monthly releases. "
            "Step 4: Pricing. $99-$499/mo tiered, 14-day trial. "
            "Step 5: Launch channels. LinkedIn ads, conference sponsorships, partner co-marketing."
        ),
        interrupt="use Twitter instead of LinkedIn",
        intent="replace",
        expected_rpr=0.80,
        notes="Single channel swap, structure preserved.",
    ),
    BenchmarkTask(
        id="plan-event-conference",
        domain="planning",
        goal="Plan a 2-day AI conference in Cairo for 200 attendees with a $30K budget.",
        initial_cot=(
            "Step 1: Pick venue. The Greek Campus or Zewail City, both fit 200 and have A/V. "
            "Step 2: Set agenda. Keynotes (4), talks (12), workshops (4), panels (2). "
            "Step 3: Invite speakers. Mix of local AI researchers and 2 international. "
            "Step 4: Sponsors. Reach out to 10 local tech companies. "
            "Step 5: Marketing. LinkedIn, Twitter, university mailing lists. "
            "Step 6: Logistics. Catering, badges, AV tech, recording."
        ),
        interrupt="drop the workshops",
        intent="remove",
        expected_rpr=0.80,
        notes="Remove a category, other structure preserved.",
    ),
    BenchmarkTask(
        id="plan-diet-vegetarian",
        domain="planning",
        goal="Plan a 7-day vegetarian meal plan for a family of 4 with a $150 weekly budget.",
        initial_cot=(
            "Step 1: Protein sources. Eggs, lentils, chickpeas, tofu, beans. "
            "Step 2: Vegetables. Seasonal Egyptian produce: molokhia, bamia, fuul, salads. "
            "Step 3: Grains. Rice, bulgur, whole-wheat bread, oats. "
            "Step 4: Plan 3 meals/day across 7 days, repeating ingredients to control cost. "
            "Step 5: Shopping list and budget per category."
        ),
        interrupt="add a vegan day on Wednesday",
        intent="add",
        expected_rpr=0.90,
        notes="Additive — should preserve most of the plan.",
    ),
]


# --------------------------------------------------------------------------- #
# REASONING (6 tasks)
# --------------------------------------------------------------------------- #

REASONING_TASKS = [
    BenchmarkTask(
        id="reason-causal-economic",
        domain="reasoning",
        goal="Explain why Egypt's inflation rate rose from 5% to 35% between 2021 and 2024.",
        initial_cot=(
            "Step 1: Note the devaluation of the Egyptian pound in early 2022 and again in 2024. "
            "Step 2: Trace supply chain disruptions from the Russia-Ukraine war (wheat, fertilizer). "
            "Step 3: Discuss the impact of foreign currency shortages on import-dependent goods. "
            "Step 4: Analyze fiscal expansion and reduced subsidies on food and fuel. "
            "Step 5: Summarize: currency devaluation + supply shocks + import dependency + fiscal expansion = 35% inflation."
        ),
        interrupt="focus more on the currency devaluation",
        intent="reprioritize",
        expected_rpr=0.60,
        notes="Reweighting, not replacement. v1 limitation: noted as no-op.",
    ),
    BenchmarkTask(
        id="reason-comparative-languages",
        domain="reasoning",
        goal="Compare the grammatical structure of Egyptian Arabic and Modern Standard Arabic.",
        initial_cot=(
            "Step 1: Word order. EA: VSO or SVO flexible. MSA: predominantly VSO. "
            "Step 2: Verb conjugation. EA: simpler, mostly aspect-based. MSA: rich, person/number/gender agreement. "
            "Step 3: Definite article. Both use الـ / el-, but EA prefixes it to any noun freely. "
            "Step 4: Pronouns. EA has clitic pronoun suffixes. MSA separates them. "
            "Step 5: Use case. MSA: partial case marking. EA: none."
        ),
        interrupt="use Twitter instead of Facebook",
        intent="replace",
        expected_rpr=0.50,
        notes="Misaligned interrupt — not relevant to reasoning. Tests robustness.",
    ),
    BenchmarkTask(
        id="reason-debug-code",
        domain="reasoning",
        goal="Find the bug in this Python code: def add(a,b): return a-b print(add(2,3))",
        initial_cot=(
            "Step 1: Read the function definition. It defines `add(a, b)` but the body returns `a - b`. "
            "Step 2: Identify the bug. The function name implies addition but the operation is subtraction. "
            "Step 3: Confirm with the print statement. add(2, 3) returns 2-3 = -1, which is wrong. "
            "Step 4: Propose the fix. Change the body to `return a + b`."
        ),
        interrupt="actually the function name is right, the print is wrong",
        intent="correction",
        expected_rpr=0.50,
        notes="User pushes back — the bug analysis should adapt.",
    ),
    BenchmarkTask(
        id="reason-probability-cards",
        domain="reasoning",
        goal="What is the probability of drawing two aces in a row from a standard deck without replacement?",
        initial_cot=(
            "Step 1: Probability of first ace. 4/52 = 1/13. "
            "Step 2: After drawing one ace, 51 cards remain with 3 aces. "
            "Step 3: Probability of second ace. 3/51. "
            "Step 4: Multiply. (4/52) * (3/51) = 12/2652 = 1/221 ≈ 0.00452. "
            "Step 5: Express as percentage. ~0.452%."
        ),
        interrupt="with replacement",
        intent="correction",
        expected_rpr=0.60,
        notes="Switch scenario — keeps formula structure, changes values.",
    ),
    BenchmarkTask(
        id="reason-ethics-trolley",
        domain="reasoning",
        goal="Analyze the trolley problem from a utilitarian and a deontological perspective.",
        initial_cot=(
            "Step 1: Frame the scenario. A runaway trolley will kill 5 people on the main track. You can pull a lever to divert it to a side track where 1 person is tied. "
            "Step 2: Utilitarian analysis. Divert — saves 4 net lives. The morally right action maximizes aggregate welfare. "
            "Step 3: Deontological analysis. Diverting uses the 1 person as a means to an end, violating their dignity. The moral rule is: do not use a person as a means. "
            "Step 4: Virtue ethics. A virtuous agent would act with courage and practical wisdom. "
            "Step 5: Conclusion. Utilitarianism and deontology diverge here, revealing the limits of monistic moral theories."
        ),
        interrupt="also include care ethics",
        intent="add",
        expected_rpr=0.85,
        notes="Additive dimension, most reasoning preserved.",
    ),
    BenchmarkTask(
        id="reason-systems-feedback",
        domain="reasoning",
        goal="Explain why a system with negative feedback tends to be stable while positive feedback tends to amplify.",
        initial_cot=(
            "Step 1: Define negative feedback. Output opposes input. Example: thermostat. "
            "Step 2: Define positive feedback. Output reinforces input. Example: microphone squeal. "
            "Step 3: Stability analysis. Negative feedback damps deviations from equilibrium. "
            "Step 4: Amplification analysis. Positive feedback increases deviations, leading to runaway. "
            "Step 5: Real-world examples. Climate (ice-albedo positive feedback), economic cycles (negative feedback through prices)."
        ),
        interrupt="focus on biological examples",
        intent="reprioritize",
        expected_rpr=0.50,
        notes="Reweighting — should preserve framework, swap examples.",
    ),
]


# All 12 tasks
ALL_TASKS = PLANNING_TASKS + REASONING_TASKS
