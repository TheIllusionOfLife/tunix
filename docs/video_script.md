# Video Script (GlaiveAI-Only Strategy) - < 3 Minutes

**Pro Tip**: Record screen first, add voiceover second.

---

## 0:00 - 0:30 | Intro & Problem

**Visual**: Face camera or title slide: "Teaching Reasoning Through Demonstration"

**Audio**:
"Hi, I'm [Name], and this is my solution for the Google Tunix Hackathon.

Small models like Gemma 2B have a problem: they answer too fast without thinking through the problem. The usual approach is reinforcement learning on math puzzles, but we took a different path.

**What if we could teach the model HOW to think by showing it examples?**"

---

## 0:30 - 1:15 | The Strategy

**Visual**: Show diagram: `Gemma 2B` → `SFT on 180K GlaiveAI` → `Thinking Model`

**Audio**:
"Our strategy: Supervised Fine-Tuning on a single, high-quality dataset called GlaiveAI.

Why just one dataset? Quality over quantity. We found that 2023 datasets can't compare to 2025 reasoning quality.

GlaiveAI was created this year using DeepSeek-R1-Distill-70B - state of the art. And it focuses on exactly what the competition values: creative writing, social science, analytical reasoning - NOT math and code.

Every example shows the full thinking process with explicit `<reasoning>` tags."

---

## 1:15 - 2:00 | The Code

**Visual**: Scroll through notebook, pause on SFT trainer setup

**Audio**:
"Here's our Tunix implementation.

We use the PeftTrainer for supervised learning. Each sample follows the format: question, reasoning trace, then answer.

180,000 samples. 4 epochs. 7 hours on a single TPU.

The model learns to generate the thinking process first, then the conclusion."

---

## 2:00 - 2:40 | Demo

**Visual**: Show model answering a creative/analytical question with `<reasoning>` output

**Audio**:
"Let's see it work.

I'll ask a philosophical question: 'What are the ethical implications of AI art?'

Watch - it opens a reasoning tag, considers multiple perspectives, weighs the tradeoffs, and only then provides a thoughtful response.

This is the power of learning from high-quality examples."

---

## 2:40 - 3:00 | Outro

**Visual**: Face camera or "Thank You" slide

**Audio**:
"For unrestricted mode, we continued training on another 100K fresh samples from GlaiveAI.

The lesson? One great dataset beats four mediocre ones. Check out the full code in the writeup. Thanks for watching!"

---

## Recording Checklist

- [ ] Title slide: "Teaching Reasoning Through Demonstration"
- [ ] Diagram: SFT pipeline with GlaiveAI focus
- [ ] Code: PeftTrainer setup
- [ ] Demo: Model with `<reasoning>` output
- [ ] Face (optional): Intro and outro
