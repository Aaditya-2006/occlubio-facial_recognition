A modular face-recognition pipeline — detect → track → align → quality-gate → embed → 1:N search — designed to be extended for recognition under occlusion (masks, sunglasses, caps, side profiles, low light, motion blur).
 
It ships as a working web platform and CLI built on pretrained models. The occlusion-aware recognizer that gives the project its name is **designed and its training code is written, but it has not been trained yet**, so this repo does not currently claim any accuracy under occlusion. See [Status](#status) and [Benchmarks](#benchmarks).
 
Design rationale, papers, datasets and deployment plan: [OCCLUSION_ROBUST_FR_ARCHITECTURE.md](OCCLUSION_ROBUST_FR_ARCHITECTURE.md) · Platform design: [PLATFORM.md](PLATFORM.md) · Full write-up: [REPORT.md](REPORT.md)
 
---
 
## Status
 
| Stage | Today | Planned |
|---|---|---|
| Detection | SCRFD (InsightFace `buffalo_l`) | Fine-tune on masked / helmeted faces |
| Tracking | IoU tracker | ByteTrack |
| Alignment | 5-point similarity transform → 112×112 | Partial-affine fallback for profiles |
| Quality gate | Proxy FIQA: detector score + Laplacian sharpness + exposure bounds | CR-FIQA |
| Occlusion estimate | Heuristic, region-based | Trained classifier + face parsing |
| Embedding | **ArcFace `w600k_r50`, 512-d, pretrained** | EdgeFace + AdaFace + feature masking (training code written, **not trained**) |
| Track fusion | Mean of the best *K* frames per track | — |
| 1:N search | FAISS `IndexFlatIP` (exact cosine) | IVF-PQ / HNSW / Milvus beyond ~10⁵ identities |
| Platform | FastAPI, SQLite, background jobs, analytics, web UI | PostgreSQL, Celery + Redis, Docker |
| Anti-spoofing, template protection, gait / body re-ID, DeepStream | **Hooks only** — marked `# HOOK:` in code, not implemented | — |
 
**Verified end to end (CPU):** registration and login with hashed passwords; quality-gated enrollment; duplicate-face rejection (HTTP 409); video identification job with JSON report; unknown-person clustering; annotated output video; web UI.
 
**Not done:** the occlusion-aware model, anti-spoofing, protected templates, Jetson / DeepStream deployment, and any accuracy evaluation on occluded or masked benchmarks.
 
## Benchmarks
 
### 1:N gallery search
 
Search was tested at **50,000 enrolled templates** (512-d, L2-normalised).
 
Gallery contents: real face embeddings of 7864 distinct identities / synthetic vectors.
 
Reproduce: `python scripts/bench_search.py --n 50000 --index flat` (add `--embeddings path/to/embeddings.npy` to use real templates).
 
This measures **search only**. It does not include detection, alignment or embedding.
 
### End to end, per face
 
On CPU only, detection + embedding costs roughly 30–100 ms per face, and un-strided video runs at about 0.4 FPS (use `--stride`). Measured on: TODO (CPU model).
 
Recognition runs once per track on the best quality-gated frames, not once per frame, which is what keeps video processing tractable.
 
The Jetson and server-GPU figures in the architecture document are **latency budgets, not measurements**.
 
### Accuracy
 
Not yet evaluated: IJB-C TAR@FAR, masked / occluded sets (MFR2, O-LFW), per-demographic breakdown. These are Phase 3 deliverables.
 
---
 
## Quickstart
 
Python 3.10–3.12 (3.14 has no wheels for the vision stack).
 
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[infer,api]"       # core + inference + web API
# GPU box:  pip install onnxruntime-gpu faiss-gpu   (replace the CPU wheels)
```
 
The first run downloads the `buffalo_l` model pack (~300 MB).
 
### Web platform
 
```bash
uvicorn occlubio.api.app:app --host 0.0.0.0 --port 8000
# UI:        http://localhost:8000
# API docs:  http://localhost:8000/docs
```
 
### Command line
 
```bash
# enroll: data/enroll/<person_name>/*.jpg, one folder per identity
python scripts/enroll.py --images data/enroll --gallery gallery_store
 
# identify people in a video (report + annotated MP4)
python scripts/identify_video.py --source data/my_video.mp4 \
       --gallery gallery_store --out data/out.mp4 \
       --report data/out.json --stride 2 --no-display
 
# build a synthetically occluded test set (masks / sunglasses / caps)
python scripts/make_occluded_dataset.py --src data/enroll --dst data/enroll_occluded
 
# benchmarks
python scripts/benchmark.py --source clip.mp4 --frames 300
python scripts/bench_search.py --n 50000 --index flat
```
 
Annotated videos use the `mp4v` codec; open them in VLC rather than a browser.
 
### Tests
 
```bash
pytest -q        # smoke tests; no network or model download required
```
 
---
 
## How recognition works
 
```
video → decode → SCRFD detect → track → 5-pt align (112×112) → quality gate
      → occlusion estimate → ArcFace embed → per-track fusion → FAISS 1:N
      → identity + confidence → analytics → JSON report + annotated video
```
 
Three design rules:
 
1. **The database is the source of truth; the FAISS index is derived** and rebuilt from it. FAISS can later be swapped for Milvus or Qdrant, and SQLite for PostgreSQL, without touching business logic.
2. **Cheap stages run every frame; expensive ones run once per track.** Embeddings from a track's best frames are averaged before matching, which is also more stable under occlusion than a single frame.
3. **Alignment is pinned.** One alignment routine is used for training and inference. Mismatched alignment degrades embeddings silently, so it is the first thing to check when accuracy is worse than expected.
Key thresholds live in `configs/default.yaml`; the two that matter most are `gallery.match_threshold` (0.45, deliberately conservative — a false match is worse than no match) and `OCCLUBIO_DUP_THRESHOLD` (0.5, duplicate-face rejection at enrollment).
 
## Training the occlusion-aware model (Phase 3)
 
The augmentation utilities and training script are implemented. **No model has been trained**: it needs a licensed face dataset and GPU time.
 
```bash
pip install -e ".[train]"      # adds torch, timm
python -m occlubio.training.train --data /path/to/aligned_faces --epochs 20 --out runs/edge_adaface
# then set recognition.custom_onnx in configs/default.yaml to runs/edge_adaface/model.onnx
```
 
Plan: EdgeFace (or MobileFaceNet) backbone, AdaFace loss, feature masking, heavy synthetic occlusion with a curriculum that starts from clean faces, distilled from an IResNet-100 teacher. The pretrained ArcFace baseline stays in place as the control every change is benchmarked against.
 
## Layout
 
```
occlubio/
  pipeline/     detector, aligner, quality, occlusion, antispoof (hook), engine
  tracking/     lightweight IoU tracker (ByteTrack hook)
  gallery/      FAISS enroll / search / persistence
  analytics/    track merging, unknown-person clustering, reports
  api/ service/ db/    FastAPI app, background jobs, SQLite
  data/         synthetic occlusion + photometric augmentation
  training/     AdaFace head, dataset, train + ONNX export
web/            browser UI
scripts/        enroll / identify_video / benchmark / make_occluded_dataset
deepstream/     sample DeepStream / TensorRT config (template, untested on hardware)
tests/          smoke tests
configs/default.yaml   single source of truth for pipeline settings
```
 
## Roadmap
 
| Phase | Work |
|---|---|
| P1–P2 | Detection, tracking, alignment, pretrained baseline, FAISS gallery — **done on CPU** |
| P3 | Train the occlusion-aware recognizer; CR-FIQA; occlusion classifier; periocular sub-model; evaluate against the baseline |
| P4 | Anti-spoofing, template protection, one extra modality (gait / body re-ID) with score-level fusion |
| P5 | Quantisation (FP16 embedder, INT8 auxiliaries), on-device benchmarking, DeepStream / Jetson, field pilot |
 
Platform: M1 done. Next are JWT auth and per-user job history, Docker, PostgreSQL, then Celery + Redis workers.
 
---
 
## Responsible use and licensing
 
This is a biometric identification system. Before any test with real people, get written authorisation, define the scope, and complete a data protection impact assessment. Real-time remote biometric identification in public spaces is heavily restricted under the EU AI Act, and India's DPDP Act 2023 treats biometric data as sensitive personal data.
 
- **Templates are not yet protected.** Embeddings are stored as plain float32 blobs. Encrypt them or use a protected template scheme before onboarding real users.
- **Bias is unevaluated.** Occlusion and low light raise false-match rates unevenly across groups. Measure per-demographic TAR at a fixed FAR before relying on this system.
- **A human stays in the loop** for any consequential decision. Generative face restoration (GFPGAN, CodeFormer) must never feed the recognition path or be presented as evidence; restored pixels are invented, not recovered.
- **Model and dataset licences.** InsightFace's pretrained packs are distributed for non-commercial research use, and several face datasets (WebFace260M, MS-Celeb derivatives) carry research-only or withdrawn licences. Confirm terms before any commercial use.

Occlubio is a team project by **Aaditya Raj** and **Shashwat Kumar Pandey**, carried out under the supervision of **Dr. Harsh Kasyap**. It is currently being continued under his PhD student, **Ritesh Mishra**.
 
The project report and architecture documents were written by Shashwat Kumar Pandey. The original repository is at [github.com/SHASHWATPANDEYOFFCOG/occlubio](https://github.com/SHASHWATPANDEYOFFCOG/occlubio).

