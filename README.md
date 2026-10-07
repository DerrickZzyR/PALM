# Training

**TCCM → SSE → description generation → LongCLIP encoding → CRP**

Run the following PowerShell commands from `minimal_palm`. This example uses **Batch1, tail200, 15 channels and two classes**. Use a Python environment with PyTorch, NumPy, scikit-learn, tqdm, Matplotlib, the OpenAI SDK and the official LongCLIP dependencies. Replace `cuda` with `cpu` if needed.


## 1. Train TCCM

TCCM learns normal sequence dynamics using healthy training samples (`label=0`).

```powershell
python main.py --device cuda --train_tccm 1 --train_sse 0 --train_crp 0 --data_path "$Dataset" --time_steps 200 --win_size 6 --num_channels 15 --num_classes 2 --tccm_epochs 500 --use_topology_mask 1 --topology_mask_path ../topology_mask_ali.json --tccm_checkpoint "$TccmCheckpoint"
```

## 2. Train SSE

SSE uses residuals and normalized targets from the frozen TCCM. The CLI defaults are embedding 128, patch size 4, stride 2 and SSE RevIN disabled. Epochs must exceed warmup epochs; checkpoints are selected by training loss after warmup.

```powershell
python main.py --device cuda --train_tccm 0 --train_sse 1 --train_crp 0 --data_path "$Dataset" --time_steps 200 --win_size 6 --num_channels 15 --num_classes 2 --sse_epochs 100 --warmup_epochs 40 --tccm_checkpoint "$TccmCheckpoint" --sse_checkpoint "$SseCheckpoint"
```

## 3. Generate descriptions

Use the trained TCCM/SSE checkpoints before training CRP. Batch1 encoding requires complete `all` descriptions; CRP still trains only on `train` samples. Use `generate_descriptions.py` directly, since `main.py --gene_des` has no execution branch.

Set `PALM_LLM_API_KEY` or `OPENAI_API_KEY` in your environment and replace the API URL below. The vision endpoint must support `temperature` and `max_tokens`. Copy only IDs and labels into the text output directory.

```powershell
python generate_descriptions.py --device cuda --data_root "$Dataset" --splits all --num_shards 1 --shard_id 0 --time_steps 200 --win_size 6 --num_channels 15 --num_classes 2 --tccm_checkpoint "$TccmCheckpoint" --sse_checkpoint "$SseCheckpoint" --health_prior_path ../healthy_causal_relations.json --llm_model gpt-5 --description_model_tag gpt_5 --llm_base_url "$LlmBaseUrl" --output_file "$TextPath/desc_all_len200_gpt_5.json" --save_images "$RunRoot/images"
```

One shard covers all samples. The explicit output file matches the encoder's `gpt5` directory. Specify `--sse_checkpoint` and keep SSE architecture settings consistent between training and generation.

Every ID must have a nonempty description. Rerun the same command to recover failures listed in `desc_all_len200_gpt_5_failures.json` before encoding. `--dry_run 1` only renders images.

## 4. Encode descriptions with LongCLIP

**Prepare the LongCLIP checkpoint before running this step.** Download the official pretrained [`longclip-B.pt` weights](https://huggingface.co/BeichenZhang/LongCLIP-B/blob/main/longclip-B.pt). Save the file as `../checkpoints/longclip/longclip-B.pt` relative to the directory where you run the command below, or pass your local file path with `--checkpoint`. The encoding script requires an existing local checkpoint; it does not download the weights automatically.

Prepare the [official Long-CLIP source](https://github.com/beichenzbc/Long-CLIP) at `../Long-CLIP`, or point `--longclip_root` to your clone, and follow its [installation instructions](https://github.com/beichenzbc/Long-CLIP#installation) for dependencies.

This loads `src/LongClipTextEncoder.py` with the official repository and weights in the parent directory. It writes `[N,248,512]` float16 token features, boolean masks, IDs and metadata to `$TextPath`, ordered by `id_all.npy`. This entry point currently processes tail200 descriptions.

```powershell
python data_provider/Encode_descriptions.py --device cuda --data_root "$TextDataRoot" --encoder_root . --batch 1 --model gpt5 --batch_size 32 --checkpoint ../checkpoints/longclip/longclip-B.pt --longclip_root ../Long-CLIP
```

## 5. Train CRP

After encoding completes, train the model defined in `src/Co_Refinement_Predictor.py` through `main.py`. TCCM and SSE remain frozen; CRP trains patch, text and multi-port fusion with features aligned by sample ID.

```powershell
python main.py --device cuda --train_tccm 0 --train_sse 0 --train_crp 1 --data_path "$Dataset" --multi_port_path "$Dataset" --text_path "$TextPath" --text_feature_tag longclip_b_ctx248_trunc_model_gpt_5 --time_steps 200 --win_size 6 --num_channels 15 --num_classes 2 --crp_epochs 50 --tccm_checkpoint "$TccmCheckpoint" --sse_checkpoint "$SseCheckpoint" --crp_checkpoint "$CrpCheckpoint"
```

Keep `--train_sse 0`: SSE training is enabled by default, and `main.py` executes one action per run. The final model is saved to `$CrpCheckpoint`.
