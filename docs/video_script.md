# Video Script (GlaiveAI-Only Strategy) - Under 3 Minutes

## Recording Tips
- **Record screen WHILE reading script out loud (muted)** - naturally paces your actions
- Add voiceover separately using same script
- Timing cues show [SCREEN ACTION] paired with audio duration

---

## 0:00 - 0:30 | Intro & Problem (30 seconds)

**[SCREEN ACTIONS]**
- 0:00-0:10: Title slide "Teaching Reasoning Through Demonstration" (hold 10s)
- 0:10-0:20: Animate/reveal Gemma 2B logo or model icon (10s)
- 0:20-0:30: Show competition FAQ quote about "much lower weights" for math/code (10s)

**[VOICEOVER - speak at normal pace, ~30 seconds]**
> "Hi, I'm [Name], and this is my solution for the Google Tunix Hackathon.
> 
> Small models like Gemma 2B answer too fast without thinking through problems. The usual fix is reinforcement learning on math puzzles - but the competition says math and code have 'much lower weights.'
> 
> So we took a different path: What if we teach the model HOW to think by showing examples?"

---

## 0:30 - 1:15 | The Strategy (45 seconds)

**[SCREEN ACTIONS]**
- 0:30-0:45: Show workflow diagram: Gemma 2B → SFT → GlaiveAI 180K → Trained Model (15s)
- 0:45-1:00: Zoom into "GlaiveAI 180K" node, show HuggingFace page or dataset stats (15s)
- 1:00-1:15: Show sample from dataset with `<think>` tags highlighted (15s)

**[VOICEOVER - ~45 seconds]**
> "Our strategy: Supervised Fine-Tuning on one high-quality dataset - GlaiveAI.
> 
> Why just one? Quality over quantity. This is a 2025 dataset from DeepSeek-R1-Distill-70B - state of the art.
> 
> It focuses on creative writing, social science, and analytical reasoning - exactly what the competition values. No math, no code.
> 
> One hundred eighty thousand examples, each showing the full thinking process with explicit reasoning tags."

---

## 1:15 - 2:00 | The Code (45 seconds)

**[SCREEN ACTIONS]**
- 1:15-1:30: Open notebook, scroll to config section showing SFT_STEPS=22500 (15s)
- 1:30-1:45: Scroll to data loading, pause on format standardization function (15s)
- 1:45-2:00: Scroll to training loop, show progress bar or WandB loss chart (15s)

**[VOICEOVER - ~45 seconds]**
> "Here's the Tunix implementation.
> 
> We use the PeftTrainer with LoRA for efficient fine-tuning.
> 
> Each sample follows: question, reasoning trace, then answer.
> 
> 180K samples over 4 epochs - that's 22,500 training steps in about 7 hours on a single TPU.
> 
> The model learns to generate thinking first, then the conclusion."

---

## 2:00 - 2:40 | Demo (40 seconds)

**[SCREEN ACTIONS]**
- 2:00-2:10: Type philosophical prompt: "What are the ethical implications of AI art?" (10s)
- 2:10-2:25: Show model generating `<reasoning>` tags, wait for output (15s)
- 2:25-2:40: Highlight the structured output - reasoning → answer (15s)

**[VOICEOVER - ~40 seconds]**
> "Let's see it work.
> 
> I'll ask a philosophical question: 'What are the ethical implications of AI art?'
> 
> Watch - it opens a reasoning tag, considers multiple perspectives, weighs the tradeoffs...
> 
> Then provides a thoughtful answer. Not just a quick response - actual thinking.
> 
> This is the power of learning from high-quality examples."

---

## 2:40 - 3:00 | Outro (20 seconds)

**[SCREEN ACTIONS]**
- 2:40-2:50: Show "Thank You" slide with key stats: 180K samples, 7 hours, GlaiveAI (10s)
- 2:50-3:00: Show GitHub/repo link or competition submission page (10s)

**[VOICEOVER - ~20 seconds]**
> "For unrestricted mode, we continued with another 100K fresh samples.
> 
> The lesson? One great dataset beats four mediocre ones.
> 
> Check out the full code in the writeup. Thanks for watching!"

---

## Recording Checklist

**Before Recording:**
- [ ] Open notebook in browser
- [ ] Prepare demo model (or record output separately)
- [ ] Print/open this script on second monitor
- [ ] Test screen recording software

**Screen Recording Order:**
1. [ ] Title slide (10s)
2. [ ] Model icon animation (10s)
3. [ ] FAQ quote highlight (10s)
4. [ ] Workflow diagram (15s)
5. [ ] Dataset page/stats (15s)
6. [ ] Sample with tags (15s)
7. [ ] Notebook config (15s)
8. [ ] Data loading code (15s)
9. [ ] Training progress/WandB (15s)
10. [ ] Type prompt (10s)
11. [ ] Model output (15s)
12. [ ] Highlight output (15s)
13. [ ] Thank you slide (10s)
14. [ ] GitHub link (10s)

**Total: ~180 seconds (3 minutes)**
