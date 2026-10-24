# Audio model candidates (A6)

Source facts below were read from the cited pages on 2026-10-24. "Not stated" means the page
did not say; it does not mean the property is absent. Nothing here is gated (no HF login used).

| Role | Candidate | Facts (source) | Notes / risks |
|---|---|---|---|
| Multi-speaker separation (benchmarked) | ConvTasNet Libri2Mix sep_clean 16k, 5.07 M params | 16 kHz, trained Libri2Mix train-360, reported 15.24 dB SI-SDR; license tag cc-by-sa-4.0 ([HF card](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)) | Non-causal (global norm); run as sliding window. Needs `asteroid`. Clean-speech only: no noise/reverb robustness. |
| Multi-speaker separation (benchmarked) | SepFormer WHAMR! 16k | 16 kHz, 13.5 dB SI-SNRi on WHAMR! (noise+reverb), Apache-2.0 ([HF card](https://huggingface.co/speechbrain/sepformer-whamr16k)) | Non-causal, heavy; see benchmark. |
| Multi-speaker separation (not benchmarked) | SepFormer Libri2Mix | 8 kHz model ([HF](https://huggingface.co/speechbrain/sepformer-libri2mix)) | Wrapper registered in models.py; needs 16k->8k resample. |
| Causal target-sound extraction (not run) | Waveformer | MIT; ~10 ms chunk, past-only context, e2e <20 ms claimed, RTF 0.66-0.94 on single i5 thread; checkpoints at targetsound.cs.washington.edu/files/experiments.zip ([GitHub](https://github.com/vb000/Waveformer)) | Only truly causal pretrained model found; extracts by sound CLASS, not speaker separation. Sample rate/class list not stated on page; check config.json. Checkpoint is on a university server (not run here). |
| Text-queried separation (not run) | AudioSep | MIT; 32 kHz; text query; weights in HF Space `Audio-AGI/AudioSep` checkpoint folder; chunked inference only for memory, no streaming claim ([GitHub](https://github.com/Audio-AGI/AudioSep)) | Offline model; use for Accurate mode only. Gating of the weights not stated; verify before download. |
| Sound labeler | PANNs Cnn14_16k (mAP 0.438) / Cnn14_DecisionLevelMax (frame-wise, mAP 0.385, 32 kHz) | MIT repo; Zenodo 3987831 ([GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn)) | Cheap; 16k variant matches live rate. AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, 86.6M params, bsd-3-clause) is the heavier alternative ([HF](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)). Not run. |
| Speaker embedding | WeSpeaker (ResNet34 / ECAPA / CAM++; Apache-2.0 repo) or SpeechBrain `spkrec-ecapa-voxceleb` | ([WeSpeaker](https://github.com/wenet-e2e/wespeaker)) ECAPA_TDNN reaches 0.728% EER with LM+AS-Norm on vox1-O-clean (recipe result, not a specific checkpoint); weight licenses not stated | Not run. Use for tracking identity across windows (replace the energy-correlation permutation matcher). |
| Speech enhancement | DeepFilterNet (MIT/Apache-2.0) | 48 kHz only for CLI; real-time version/LADSPA exists; DFN3 weights availability not confirmed on page ([GitHub](https://github.com/Rikorose/DeepFilterNet)) | Needs 16k<->48k resampling (~adds latency). Alternative: ClearerVoice-Studio (Apache-2.0; FRCRN enhancement, MossFormer separation; exact 16k model names unconfirmed) ([GitHub](https://github.com/modelscope/ClearerVoice-Studio)). Not run. |

Search result (not verified in a paper): no pretrained *causal* Libri2Mix speaker separator was
found; causal research models exist (e.g. arXiv 2505.13094 TFACM) but no checkpoint was confirmed.
