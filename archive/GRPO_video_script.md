# Video Script & Scene Guide (< 3 Minutes)

**Pro Tip**: Record the screen first, then record voiceover.

---

## 0:00 - 0:30 | Intro & Problem
**Visual**: 
Face camera (or Title Slide: "Tunix Zero-Cost: Reasoning").
Cut to: Screen recording of a standard Gemma 2B model failing a complex math question (just giving a wrong number instantly).

**Audio**:
"Hi, I'm [Your Name], and this is my solution for the Google Tunix Hackathon.
We all know small models like Gemma 2B have a problem: they answer too fast. They hallucinate because they don't 'think' before they speak.
Training them to reason usually costs a fortune in API credits for synthetic data. But today, I'll show you how we built a reasoning pipeline for zero dollars."

## 0:30 - 1:15 | The Strategy (Architecture)
**Visual**:
Show the "Format-Align-Reinforce" Diagram (from your Writeup). Zoom in on each box as you mention it.

**Audio**:
"Our strategy is called 'Format, Align, Reinforce.'
First, we leverage **Gemma-2B-IT's** existing instruction following. We use structure rewards to enforce the XML format: Reasoning first, Answer second.
Next, we use **GRPO** with Tunix. Instead of a heavy critic model, GRPO lets us optimize for truth in Math and Coding efficiently on a single TPU.
We generated our own variants of GSM8K and MBPP using public data scripts."

## 1:15 - 2:00 | The Code (Tunix Deep Dive)
**Visual**:
Screen record scrolling through your Notebook (`tunix_zero_cost_train.ipynb`).
*   Pause on the `structure_reward` function.
*   Pause on the `GRPOLearner` config.

**Audio**:
"Here's the implementation in Tunix.
Inside the notebook, we define custom reward functions. We used a 'Structure Reward' to enforce the XML tags, and a 'Symbolic Math Reward' using SymPy to verify answers more robustly than simple string matching.
The entire pipeline fits into the 9-hour Kaggle TPU limit. We focus strictly on RL for maximal reasoning improvements."

## 2:00 - 2:40 | The Demo (Results)
**Visual**:
Show the model answering a NEW hard question (e.g., "Solve 2x + 5 = 15"). Highlight the `<reasoning>` output block appearing *before* the answer.

**Audio**:
"Let's see it in action.
Here I ask it a math problem. Notice how it doesn't rush. It opens a reasoning tag, breaks down the algebra step-by-step, and only then provides the final answer.
This 'Chain of Thought' drastically improves accuracy on verifiable tasks."

## 2:40 - 3:00 | Outro
**Visual**:
Face camera or "Thank You / Link in Bio" slide.

**Audio**:
"We also pushed boundaries with an Unrestricted Mode, chaining sessions to double our training time.
Check out the full code and write-up in the link below. Thanks for watching!"

---

## What to Record (Checklist)
- [ ] Your Face (Intro/Outro) - optional but good.
- [ ] Slide: "Tunix Zero-Cost" Title.
- [ ] Diagram: Base Model -> GRPO flow.
- [ ] Code: Reward Functions in Notebook.
- [ ] Demo: Model generating output with `<reasoning>` tags.
