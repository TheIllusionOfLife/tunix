# Video Script - Under 3 Minutes

## Recording Tips
- **Record screen WHILE reading script out loud (muted)** - naturally paces your actions
- Add voiceover separately using same script
- Timing cues show [SCREEN ACTION] paired with audio duration

---

## 0:00 - 0:40 | Intro & Problem (40 seconds)

**[SCREEN ACTIONS]**
- 0:00-0:40: Show a slide which visualizes the problem and the task of the hackathon.

**[VOICEOVER - speak at normal pace, ~40 seconds]**
> "Hi, I'm Yuya, and this is my solution for the Google Tunix Hackathon.
> 
> Small models like Gemma2 2B answer too fast without thinking through problems. The usual fix is reinforcement learning on math puzzles, but verifiable problems like math and code are often not the main use cases for small models. The main use cases are often non-verifiable problems like summarization, creative ideation, or writings. In this hackathon, we're tasked to finetune a small model by using Google's lightweight LLM post-training library, Tunix, to answer better with reasoning traces, especially for non-verifiable problems."
---

## 0:40 - 1:10 | The Strategy (30 seconds)

**[SCREEN ACTIONS]**
- 0:40-1:10: Show a slide which visualizes our strategy and the dataset we used for the hackathon.

**[VOICEOVER - ~30 seconds]**
> "Our strategy is Supervised Fine-Tuning on a dataset which contains reasoning traces for diverse non-verifiable topics.
> Why not reinforcement learning? Because small models often fail to explore to get rewards, and it is not always possible to define a reward function for non-verifiable problems.
> The dataset, glaiveai/reasoning-v1-20m, was generated using DeepSeek-R1-Distill-Llama-70B, and it is available at HuggingFace."
---

## 1:10 - 1:50 | The Code (40 seconds)

**[SCREEN ACTIONS]**
- 1:10-1:50: Show multiple slides which contain key code of Tunix implementation in our notebook. First slide for the first example, second slide for the second example.

**[VOICEOVER - ~40 seconds]**
> "Here are snippets of our code.
> 
> We utilize Tunix's `nnx` module for efficient model loading on TPUs. By using `eval_shape`, we initialize the model abstractly without consuming memory until sharding is defined."
> 
> ```python
> def get_gemma_model(ckpt_path):
>     # Abstract shape evaluation for TPU memory efficiency
>     abs_gemma = nnx.eval_shape(
>         lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
>     )
>     # Sharding across TPU mesh
>     abs_state = jax.tree.map(
>         lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
>         nnx.state(abs_gemma), nnx.get_named_sharding(abs_state, mesh)
>     )
>     return nnx.merge(graph_def, restored_params)
> ```
> 
> "Then, we use the `PeftTrainer` with Low-Rank Adaptation. This captures the reasoning patterns without retraining the entire 2-billion parameter model."
> 
> ```python
> # Tunix PeftTrainer for TPU alignment
> trainer = peft_trainer.PeftTrainer(
>     model=lora_model,
>     optimizer=optimizer,
>     training_config=training_config
> )
> 
> # Efficient training on the TPU mesh
> with mesh:
>     trainer.train(train_ds=train_iter, skip_jit=False)
> ```
>
> "We trained it for less than 7 hours on Kaggle's TPU."
---

## 1:50 - 2:40 | Demo (50 seconds)

**[SCREEN ACTIONS]**
- 1:50-2:40: Show a slide which compares the outputs of raw gemma2-2b-it model and our Tunix-trained model.

**[VOICEOVER - ~50 seconds]**
> "Let's see how the model improved. We asked both to 'Propose innovative uses for AI in education.'
> 
> The raw model's answer is mixing up the points. It places `<reasoning>` tags inside the final answer.
> 
> The Tunix model behaves like thinking in the `<reasoning>` block. It brainstorms, then critiques itself.
> *'Wait, I should make sure these ideas are innovative. I also need to consider ethical implications.'*
> 
> It self-corrects *before* generating the final answer. The answer is a clean, well-structured proposal that includes an Ethics section, something the raw model completely missed.
> 
> The model successfully generates a better proposal with reasoning traces.
---

## 2:40 - 3:00 | Outro (20 seconds)

**[SCREEN ACTIONS]**
- 2:40-3:00: Show a slide which contains thanks note for Tunix, glaiveai, and Kaggle. The slide also contains a list of Tunix functionality, and a link to my writeup and code. No logos in the slide because they are very sensitive assets.

**[VOICEOVER - ~20 seconds]**
> We greatly appreciate the hackathon to let us explore the opportunity to apply Tunix to fine tune a small model to reason, and the glaiveai to provide the dataset to the public.
> We haven't explored the full functionality of Tunix which provides Reinforcement Learning, Preference Fine-tuning, Knowledge Distillation, and more which we couldn't have time to cover in this hackathon.
> Check out my writeup and code for more details. Thanks for watching!"

---

## Recording Checklist

**Before Recording:**
- [ ] Open notebook in browser
- [ ] Prepare demo model (or record output separately)
- [ ] Print/open this script on second monitor
- [ ] Test screen recording software


**Total: ~180 seconds (3 minutes)**
