# Video Script (SFT Strategy) - < 3 Minutes

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

**Visual**: Show diagram: `Gemma 2B` → `SFT on 100K reasoning traces` → `Thinking Model`

**Audio**:
"Our strategy: Supervised Fine-Tuning on diverse reasoning traces.

We found that the competition values creative and analytical thinking more than math. So instead of training on equations, we trained on 100,000 examples of step-by-step reasoning across philosophy, ethics, commonsense, and creative tasks.

We used datasets like Raiden-DeepSeek-R1 and glaiveai's reasoning corpus - all public, all properly licensed.

The key: every example shows the full thinking process with explicit `<reasoning>` tags."

---

## 1:15 - 2:00 | The Code

**Visual**: Scroll through notebook, pause on SFT trainer setup

**Audio**:
"Here's our Tunix implementation.

We use the PeftTrainer for supervised learning. Each sample follows the format: question, reasoning trace, then answer.

The model learns to generate the thinking process first, then the conclusion.

In 8 hours on a single TPU, we processed over 100,000 reasoning examples - that's 10 times more than reinforcement learning would allow."

---

## 2:00 - 2:40 | Demo

**Visual**: Show model answering a creative/analytical question with `<reasoning>` output

**Audio**:
"Let's see it work.

I'll ask a philosophical question: 'What are the ethical implications of AI art?'

Watch - it doesn't just give an answer. It opens a reasoning tag, considers multiple perspectives, weighs the tradeoffs, and only then provides a thoughtful response.

This is the power of learning by example."

---

## 2:40 - 3:00 | Outro

**Visual**: Face camera or "Thank You" slide

**Audio**:
"For unrestricted mode, we scaled up to millions of samples using the glaiveai dataset.

The lesson? Sometimes teaching is more effective than training. Check out the full code in the writeup. Thanks for watching!"

---

## Recording Checklist

- [ ] Title slide: "Teaching Reasoning Through Demonstration"
- [ ] Diagram: SFT pipeline
- [ ] Code: PeftTrainer setup
- [ ] Demo: Model with `<reasoning>` output
- [ ] Face (optional): Intro and outro
