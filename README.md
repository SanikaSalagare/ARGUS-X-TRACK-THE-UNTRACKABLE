






ARGUS-X — Track the Untrackable
A Python-based object detection, multi-object tracking, persistent identity management, and visual re-identification system.

ARGUS-X is designed to detect objects in video, follow them across frames, maintain an identity for each confirmed track, and attempt to recover a previously seen identity when the object disappears and later returns.

Project status: Active prototype / research project
Current documented version: v0.3.1
Primary interface: Streamlit
Detection model: YOLO26n
Short-term tracker: ByteTrack
Appearance matching: Colour-based appearance descriptors (experimental)

Contents
Overview
What ARGUS-X does
Application screenshots
System architecture
Identity lifecycle
Technology stack
Project structure
Current capabilities and limitations
Benchmark snapshot
Installation and setup
Run the application
Run automated tests
How to use the dashboard
Configuration notes
Roadmap
Acknowledgements
Overview
Most object trackers assign a temporary tracking ID while an object remains visible. When the object is occluded, leaves the frame, or the tracker loses it, that temporary ID may be lost.

ARGUS-X adds an identity-management layer on top of object detection and tracking. It maintains application-level ARGUS-XXXX identities, stores appearance information for confirmed tracks, records identity events, and attempts to associate a newly confirmed track with a recently missing identity.

The goal is to move beyond frame-by-frame detection toward persistent visual identity across a video session.

ARGUS-X is a prototype. Its appearance matching is heuristic and experimental; it is not a guarantee that the same real-world person will always be recognized correctly.

What ARGUS-X does
Detects objects: Uses the YOLO26n model through the Ultralytics inference stack.
Tracks detections: Uses ByteTrack to associate detections across nearby frames.
Confirms new tracks: A candidate must be observed for a configurable number of frames before an ARGUS identity is assigned. The documented default is four frames.
Maintains identity memory: Stores identity metadata, recent observations, appearance samples, and lifecycle state.
Attempts re-identification: Compares a confirmed new track with recently missing identities using appearance similarity and plausibility checks.
Records lifecycle events: Tracks states such as active, missing, and archived.
Provides a dashboard: Offers video/camera processing, live tracking, object management, identity history, analytics, and settings pages.
Version 0.3.1 highlights
The existing project notes describe the following changes in v0.3.1:

Re-identification enabled by default with the colour appearance model and a similarity threshold of 0.80.
Recovered identities can be marked on processed video with a RECOVERED label, an orange box, and a line toward the last known location; missing identities can be represented with a grey ghost box.
New tracks require a configurable confirmation period (default: four frames) before receiving an ARGUS ID.
Colour fingerprints use the torso region and Bhattacharyya similarity.
A location plausibility gate and bounding-box size checks help reject implausible matches.
Live preview is throttled and kept small; final video re-encoding uses FFmpeg's veryfast preset.
Analytics is organized into Overview, Identities, Re-ID Recoveries, Timeline, Movement, and Sessions views.
The exact availability of individual overlays and export features depends on the current application path and should be verified with a real processing session before being presented as production-ready.

Application screenshots
The screenshots below are from the running ARGUS-X Streamlit interface. They show selected pages rather than every dashboard view.

Dashboard
The dashboard summarizes identity counts, processing sessions, recent activity, and system information.



Video & Camera
The Video & Camera page provides video upload and processing controls, including confidence threshold, image size, frame limit, object classes, and appearance re-identification options.



Object Manager
The Object Manager presents identities for a selected session, including ARGUS ID, class, tracker ID, status, first and last frame, frames seen, re-identification count, and confidence.



Analytics
The Analytics page summarizes session-level identity and detection information and provides tabbed analysis views.



System architecture
The following diagram describes the logical processing flow and the major responsibilities documented for the project. It is a high-level architecture view, not a claim that every box is implemented as an independent service.

Input sourcesUploaded video / local camera
Streamlit frontendVideo & Camera / Live Tracking
Processing configurationConfidence · image size ·classes · tracker
Video frame reader
YOLO26n detectorObject detections + confidence
ByteTrackShort-term track association
ARGUS Engine
Track confirmationRequire N observed frames
Identity matching
Plausible existing identity?
Recover existing ARGUS ID
Create a new ARGUS ID
Identity memory & lifecycle
Database / session records
Track history & identity events
Video overlays / preview
Object Manager
Identity History
Analytics
Processed output / dashboard
Yes
No / uncertain
Architecture layers
Layer	Responsibility	Main role
User interface	Collects input and processing settings; displays results	Streamlit pages
Input and frame handling	Opens video/camera input and supplies frames to processing	OpenCV / video pipeline
Object detection	Finds supported object classes in each frame	YOLO26n via Ultralytics
Short-term tracking	Associates detections between nearby frames	ByteTrack
Identity management	Confirms tracks, assigns ARGUS IDs, maintains active/missing state	ArgusEngine
Appearance matching	Compares appearance samples for possible identity recovery	Colour descriptor; optional neural feature path is experimental
Persistence	Stores sessions, identities, and event/history information	Project database layer
Analytics and presentation	Builds identity summaries, timelines, charts, and management views	Analytics data layer + Streamlit
Important distinction: tracker ID vs ARGUS ID
Tracker ID: A short-term identifier supplied by the tracking algorithm. It can change when tracking is interrupted.
ARGUS ID: The application-level identity, such as ARGUS-0181, managed by ARGUS-X. The engine attempts to retain or recover this identity when a track is interrupted.
An ARGUS ID is a system-generated visual track identity. It does not establish a person's real-world identity.

Identity lifecycle
A simplified identity lifecycle is:

Provisional
Active
;
new
ARGUS
ID
assigned
prior
identity
recovered
Missing
Archived
Detection / new tracker
Confirmation framesaccumulating
Confirmed
Confirmed
Track observed
Track no longer observed
Plausible appearance match
Archive timeout reached
Session / lifecycle policy
Re-identification decision (high level)
A new tracker is observed and collected as a provisional candidate.
The candidate must pass the configured confirmation-frame requirement.
If appearance re-identification is enabled, the engine compares its appearance representation against eligible missing identities.
The engine applies class, time-window, and location/size plausibility checks.
A sufficiently strong, unambiguous candidate may recover an existing ARGUS ID.
If there is no acceptable match, ARGUS-X creates a new identity rather than forcing an uncertain association.
The current colour-based method is experimental. Similar clothing, lighting changes, camera viewpoint, occlusion, and crowded scenes can cause false matches or missed recoveries.

Technology stack
Area	Technology
Language	Python
User interface	Streamlit
Object detection	Ultralytics YOLO26n
Multi-object tracking	ByteTrack
Image and video processing	OpenCV
Numerical operations	NumPy
Appearance features	Colour descriptors; optional MobileNetV3-small ImageNet features
Video encoding	FFmpeg
Evaluation	motmetrics and custom MOT-format evaluation script
Testing	pytest
Hardware used during local development: NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB VRAM), with CUDA-enabled PyTorch available in the development environment.

Project structure
The following is a practical overview of the important project areas. Some supporting files are omitted for readability.

ARGUS-X2/
│
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── argus_engine.py       # Identity management and ReID matching
│   │   │   └── video_job.py          # Video-processing workflow
│   │   ├── database/
│   │   │   └── db.py                 # Database access and persistence
│   │   └── reidentification/
│   │       └── embedder.py           # Optional MobileNet feature embedder
│   └── test_identity_memory.py       # Identity-memory tests
│
├── benchmarks/
│   └── run_mot17.py                  # MOT17 evaluation runner
│
├── datasets/                         # Local datasets (not committed)
├── docs/
│   └── images/                       # README screenshots
├── frontend/
│   ├── app.py                        # Streamlit application entry point
│   └── analytics_data.py             # Analytics data preparation
│
├── models/
│   └── yolo26n.pt                    # YOLO26n model weights (download separately)
│
├── outputs/                          # Generated results (not committed)
├── requirements.txt
└── README.md
This tree highlights the known core files; it is not an exhaustive listing of every module in the repository.

Current capabilities and limitations
The project notes distinguish implemented, tested, experimental, and not-built functionality. That distinction matters: a visible control or code path does not automatically mean a feature has been validated under real-world conditions.

Tested in automated tests
The project has automated tests for:

Identity recovery logic and lifecycle events.
Confirmation behaviour and rejection of short-lived flickers.
Location plausibility gating.
ID numbering across sessions.
Database round trips and identity renaming.
Analytics data preparation and dashboard rendering with seeded/stand-in data.
Run the test suite using the command in Run automated tests.

Exercised on local video / hardware
The developer has exercised the main dashboard workflow, video upload and processing, live preview, stop controls, history, analytics, settings, and webcam-based live tracking in the local development environment.

This is development testing, not a claim of production readiness or broad camera compatibility.

Experimental or requiring further validation
Appearance re-identification on varied real-world footage.
The optional MobileNetV3-small ImageNet feature path. It is not a person-ReID-trained model.
Recovery overlays and ghost-box behaviour across difficult occlusions.
Thumbnail generation and trajectory visualization.
Tracking CSV export and processed-video playback across different media.
RTSP/HTTP stream URL and alternative tracker options.
Performance across different resolutions, cameras, and GPU/CPU hardware.
Not built / not supported as a project capability
Cross-camera identity sharing.
Open-vocabulary detection of arbitrary unseen object categories.
Vehicle-specific re-identification.
A dedicated person-ReID-trained embedding model.
YOLO26n detects classes in its trained label set; ARGUS-X does not literally detect every possible moving object.

Benchmark snapshot
A local 600-frame run was recorded on the MOT17-02-FRCNN training sequence using the project benchmark script.

Metric	Recorded result
Sequence	MOT17-02-FRCNN
Frames processed	600
Valid pedestrian ground-truth rows	18,581
Matches	5,675
False positives	1,563
Misses	12,865
Identity switches	41
MOTA	0.221301
MOTP	0.204142
IDF1	0.360944
Processing time	27.28 s
Average processing speed	21.99 FPS
Evaluation note: These are preliminary results from the project's custom evaluation on one sequence. The script filters the MOT17 ground truth using the selected mark/class fields; it does not implement the complete official MOTChallenge ignore-region protocol. Therefore, these numbers must not be described as official MOTChallenge scores or as a comprehensive dataset benchmark. Results can vary with model settings, hardware, and code version.

Installation and setup
Requirements
Windows PowerShell (commands below are written for Windows).
Python version compatible with the dependencies in requirements.txt (the development environment used Python 3.12).
Git, if cloning the repository.
NVIDIA GPU and compatible CUDA-enabled PyTorch are optional; CPU execution may be slower.
YOLO26n model weights.
1. Clone the repository
git clone https://github.com/SanikaSalagare/ARGUS-X-TRACK-THE-UNTRACKABLE.git
cd ARGUS-X-TRACK-THE-UNTRACKABLE
If you already have the project folder, open PowerShell in that folder instead.

2. Create and activate a virtual environment (optional)
If you are setting up a fresh environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If using the existing development environment, use its Python executable instead of creating another environment.

3. Install dependencies
For the existing local environment used during development:

& "C:\ARGUS-X\.venv\Scripts\python.exe" -m pip install -r requirements.txt
4. Add the model weights
Place the YOLO26n weights file here:

models/yolo26n.pt
The model file is not included in the Git repository and must be obtained separately from a trusted model source.

Run the application
From the project root:

& "C:\ARGUS-X\.venv\Scripts\python.exe" -m streamlit run frontend/app.py --server.port 8503
Then open:

http://localhost:8503
To stop the app, press Ctrl+C in the PowerShell terminal running Streamlit.

If you created and activated your own virtual environment, you can instead run:

python -m streamlit run frontend/app.py
Run automated tests
The repository includes a pytest configuration that scopes test discovery to the backend directory and excludes local output/dataset folders.

Run all discovered tests:

& "C:\ARGUS-X\.venv\Scripts\python.exe" -m pytest -q
Run the identity-memory test file directly:

& "C:\ARGUS-X\.venv\Scripts\python.exe" -m pytest -q .\backend\test_identity_memory.py
The current verified local result is 11 passed.

How to use the dashboard
Start the Streamlit application.
Open Video & Camera to upload a supported video file and configure processing.
Choose the confidence threshold, image size, frame limit, and object classes as needed.
Enable appearance re-identification to allow the engine to attempt identity recovery.
Use Live Tracking to run a local camera source (camera index 0 was tested in the development environment).
Use Object Manager to inspect identities and their status for a session.
Use Identity History to inspect identity records and events.
Use Analytics to review session summaries and identity-related charts.
Supported upload formats shown by the interface include MP4, AVI, MOV, MKV, and MPEG4, with a displayed 200 MB per-file limit. Actual playback and codec compatibility depend on the local FFmpeg/OpenCV environment.

Configuration notes
Common processing controls exposed by the interface include:

Setting	Meaning
Confidence threshold	Minimum detector confidence for retaining detections
Image size	Inference image size; larger sizes may improve small-object detection at additional compute cost
Maximum frames	Optional processing limit; 0 means process all frames in the interface
Object classes	Classes selected for detection/tracking
Appearance re-identification	Enables the experimental appearance-based identity recovery path
Appearance model	Selects the configured appearance representation (colour is the documented default)
Similarity threshold	Minimum appearance similarity for a candidate identity match
Re-link window	Maximum missing interval considered for identity recovery
Tracker memory	Controls how long ByteTrack retains short-term track state
The engine also uses track confirmation, class consistency, and plausibility checks. The exact effective defaults should be checked in the current code when changing configuration or reproducing an experiment.

Roadmap
Potential next development steps:

Validate re-identification on more varied video sequences and longer occlusions.
Evaluate using a dedicated person-ReID-trained embedding model.
Improve handling of similar-looking people and appearance changes.
Complete and validate trajectory, thumbnail, and export workflows.
Test supported camera and network-stream inputs systematically.
Add reproducible evaluation across multiple MOT17 sequences using the official evaluation protocol.
Explore cross-camera identity association as a separate future capability.
Acknowledgements
ARGUS-X builds on open-source computer-vision and tracking tools, including Ultralytics, ByteTrack, OpenCV, PyTorch, Streamlit, and motmetrics. Please refer to the respective projects for their licenses and usage terms.

ARGUS-X — Track the Untrackable
Detect an object. Track it. Remember it. Recognize it again.

You’re out of attachments for now. Upgrade your plan for more, or try again after 10:22 PM.




