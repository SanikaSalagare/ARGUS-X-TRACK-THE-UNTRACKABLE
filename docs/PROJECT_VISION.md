# ARGUS-X GÇö TRACK THE UNTRACKABLE
## Complete Project Vision, Technical Specification, UI/UX Design and Development Roadmap

---

# 1. PROJECT INTRODUCTION

### Project Name
**ARGUS-X GÇö Track the Untrackable**

### Project Category
Artificial Intelligence | Computer Vision | Object Detection | Multi-Object Tracking | Visual Re-Identification | Intelligent Video Surveillance

### Project Description

ARGUS-X is an intelligent computer vision-based object detection, multi-object tracking, and persistent identity management platform.

The primary objective is to detect objects in video streams, track their movements, assign unique identities, and maintain those identities throughout their observed lifetime.

Traditional object detection systems identify objects independently in individual frames. Conventional tracking systems associate detected objects across consecutive frames, but tracking identities can be lost when objects disappear, become occluded, or reappear.

ARGUS-X aims to go beyond conventional object tracking by developing a persistent identity management system.

The long-term vision is to allow the system to recognize an object that has previously appeared, even if it disappears temporarily and later returns with a different tracker ID.

The system should eventually support:
- Uploaded video files.
- Live webcam streams.
- CCTV camera streams.
- IP cameras and RTSP streams.
- Multiple objects tracked simultaneously.
- Persistent ARGUS identity assignment.
- Visual re-identification.
- Object history and movement information.
- Multi-camera identity matching.
- Real-time monitoring through an interactive dashboard.

The central idea is:

**Detect an object. Track it. Remember it. Recognize it again.**

---

# 2. THE MAIN PROBLEM ARGUS-X SOLVES

A conventional object detection system can detect a person, vehicle, or other supported object in a video frame.

However, detection alone does not tell us whether an object appearing in a later frame is the same object seen previously.

For example:

1. A person enters the camera view.
2. ARGUS-X detects the person.
3. The system assigns an identity such as ARGUS-0001.
4. The person walks through the scene.
5. The person disappears behind another object.
6. The person eventually reappears.
7. The tracking algorithm may assign a different tracker ID.
8. ARGUS-X should ideally recognize that the person is the same previously observed object and recover ARGUS-0001.

This is the main challenge the project is intended to address.

### Main Objectives

1. Detect supported object categories from video.
2. Track multiple detected objects simultaneously.
3. Maintain consistent object identities.
4. Store object information and tracking history.
5. Recover identities after temporary disappearance.
6. Handle partial and complete occlusion more effectively.
7. Provide a real-time visual monitoring interface.
8. Maintain records of detected objects and processing sessions.
9. Eventually support identity matching across multiple cameras.

---

# 3. COMPLETE SYSTEM VISION

ARGUS-X should be developed as a modular computer vision platform.

The system should contain the following major components:

1. Video Input System.
2. Video Processing and Frame Extraction.
3. Object Detection Engine.
4. Multi-Object Tracking Engine.
5. Persistent Identity Management System.
6. Visual Re-Identification Engine.
7. Object Lifecycle Management.
8. Annotation and Visualization Engine.
9. Database and Storage System.
10. Interactive Web Dashboard.
11. Analytics and Reporting System.
12. Camera Integration and Deployment System.

Each component should have a clearly defined responsibility.

The architecture must allow future features to be integrated without rewriting the entire application.

---

# 4. COMPLETE SYSTEM ARCHITECTURE

```text
                    ARGUS-X PLATFORM
                           |
              +------------+------------+
              |                         |
         INPUT SOURCES             USER INTERFACE
              |                         |
       +------+------+           +------+------+
       |      |      |           |             |
     Video  Webcam  CCTV      Dashboard    Monitoring
     Upload         RTSP      Controls      Pages
       |      |      |           |
       +------+------+-----------+
                     |
              VIDEO INGESTION
                     |
          Video Validation and Decoder
                     |
              Frame Extraction
                     |
            Background Job Worker
                     |
              ARGUS-X ENGINE
                     |
          +----------+----------+
          |                     |
    OBJECT DETECTION       TRACKING ENGINE
          |                     |
        YOLO26n               ByteTrack
          |                     |
          +----------+----------+
                     |
              IDENTITY MANAGER
                     |
          +----------+----------+
          |                     |
      Identity Memory     ARGUS ID Generator
          |                     |
          +----------+----------+
                     |
          VISUAL RE-IDENTIFICATION
                 (PLANNED)
                     |
          Object Appearance Matching
                     |
              IDENTITY RECOVERY
                     |
          +----------+----------+
          |                     |
      ANNOTATION            DATABASE
          |                     |
    Bounding Boxes       SQLite / Future DB
    Identity Labels      Session Records
          |              Identity Records
          |              Event Records
          +----------+----------+
                     |
              OUTPUT SYSTEM
                     |
          +----------+----------+
          |          |          |
      Annotated   Live       Analytics
       Video     Preview     and History
```

### Important architectural requirement

The identity manager and visual re-identification engine must remain separate concepts.

ByteTrack provides short-term tracking associations.

Identity Memory maintains ARGUS identities associated with tracker IDs.

Visual Re-Identification is a future component that will compare appearance information to recover identities when tracker associations are lost.

Do not claim that the current identity memory already performs visual re-identification.

---

# 5. VIDEO INPUT SYSTEM

The system should accept multiple types of video input.

### 5.1 Uploaded Video

Supported formats should include:

- MP4
- AVI
- MOV
- MKV

Expected workflow:

1. User opens the Video & Camera page.
2. User uploads a video.
3. The system validates the file.
4. Video metadata is extracted.
5. The original video is displayed.
6. User configures processing settings.
7. User starts ARGUS-X processing.
8. The video is processed in the background.
9. Annotated frames are displayed as a live preview.
10. The final processed video is saved and made available for playback and download.

### 5.2 Webcam Input

The system should allow users to access a connected webcam.

Expected functionality:

- Start camera.
- Stop camera.
- Display live video.
- Run object detection.
- Track multiple objects.
- Display bounding boxes and ARGUS IDs.
- Update object statistics.
- Handle camera disconnection gracefully.

### 5.3 CCTV and RTSP

This is a future extension.

The system should eventually support:

- RTSP camera URLs.
- IP camera streams.
- Multiple camera connections.
- Camera connection status.
- Camera-specific tracking sessions.

Do not make CCTV support appear implemented until it has been developed and tested.

---

# 6. OBJECT DETECTION ENGINE

### Technology: YOLO26n

The object detection module uses YOLO26n through the Ultralytics framework.

Its responsibilities include:

- Detecting supported object categories.
- Generating bounding boxes.
- Identifying object classes.
- Calculating confidence scores.
- Returning detection information to the tracking module.

Each detection should contain:

```python
{
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.92,
    "bbox": [120, 80, 250, 420]
}
```

The detection confidence threshold should be configurable.

The interface should eventually allow users to adjust:
- Confidence threshold.
- Image resolution.
- Processing device where supported.
- Detection settings.

### Important limitation

YOLO26n detects supported classes from its trained category set. It does not automatically detect every possible moving object.

Future research may explore open-vocabulary detection or custom-trained models to expand the supported categories.

---

# 7. MULTI-OBJECT TRACKING ENGINE

### Technology: ByteTrack

The tracking module associates detections across consecutive video frames.

Its responsibilities include:

1. Receiving detection results.
2. Associating detections with existing tracks.
3. Assigning tracker IDs.
4. Updating object locations.
5. Maintaining short-term tracking continuity.
6. Returning tracked object information to the identity manager.

Example:

```text
Frame 1:
Person detected GåÆ Tracker ID 1

Frame 2:
Same person detected GåÆ Tracker ID 1

Frame 3:
Same person detected GåÆ Tracker ID 1
```

Multiple objects must be tracked simultaneously.

The tracking engine should support configurable tracking settings and should be designed so alternative tracking algorithms can be integrated later.

### Important distinction

A ByteTrack ID is not the same as a permanent ARGUS ID.

Tracker IDs belong to the tracking algorithm. ARGUS IDs belong to the identity management system.

---

# 8. PERSISTENT IDENTITY MANAGEMENT

This is one of the most important parts of ARGUS-X.

### 8.1 ARGUS ID Generation

Every newly registered tracked object receives a unique identity.

Examples:

```text
ARGUS-0001
ARGUS-0002
ARGUS-0003
ARGUS-0004
```

ARGUS IDs should be independent of ByteTrack tracker IDs.

### 8.2 Identity Memory

The Identity Memory module maintains records for previously registered objects.

Each identity should store:

- ARGUS ID.
- Tracker ID.
- Object class.
- First-seen frame.
- Last-seen frame.
- Last-known bounding box.
- Detection confidence.
- Number of observed frames.
- Current status.
- Relevant timestamps.

### 8.3 Object Lifecycle

An object should move through three main states:

**ACTIVE**

The object is currently visible and being tracked.

**MISSING**

The object has temporarily disappeared from the current video frames.

**ARCHIVED**

The object has been missing beyond the configured archive threshold.

The system should retain archived identity records rather than immediately deleting them.

### 8.4 Rename Identity

Users should be able to rename an object.

Example:

```text
ARGUS-0001 GåÆ Person A
ARGUS-0002 GåÆ Delivery Vehicle
```

The original ARGUS ID must remain unchanged even when the display name is modified.

### 8.5 Identity Recovery

The current implementation associates identities with ByteTrack IDs.

The future system must add appearance-based identity matching.

This is the most important planned intelligence feature.

---

# 9. VISUAL RE-IDENTIFICATION ENGINE

Visual Re-Identification, commonly called ReID, is the feature that will distinguish ARGUS-X from a basic tracking dashboard.

The goal is to recognize an object using its visual appearance rather than depending only on its tracker ID.

### Expected workflow

1. Detect an object.
2. Extract its image crop.
3. Generate an appearance representation or embedding.
4. Store that representation with its identity.
5. When a new object appears, extract its appearance representation.
6. Compare it with previously stored representations.
7. Calculate similarity scores.
8. Evaluate the best candidate against configured matching criteria.
9. Recover the previous ARGUS ID if the evidence is sufficiently strong.
10. Otherwise, create a new identity.

### Future ReID capabilities

- Person re-identification.
- Vehicle re-identification.
- Appearance similarity comparison.
- Identity recovery after disappearance.
- Identity recovery after tracker ID changes.
- Partial occlusion handling.
- Appearance changes due to lighting.
- Viewpoint variation.
- Similar-looking object handling.

### Important design requirement

ReID must not blindly assign an old identity based on a weak similarity score.

It should use configurable thresholds and an uncertainty mechanism.

If the system cannot confidently determine an identity, it should avoid making an unsupported match.

For the first implementation, focus on one object category at a time, such as people or vehicles, before attempting general-purpose ReID.

---

# 10. USER INTERFACE AND UI/UX DESIGN

The ARGUS-X interface should look like a professional AI monitoring and computer vision platform.

### Overall visual direction

The design should be:

- Dark-themed.
- Modern.
- Minimal.
- Technical.
- Professional.
- Easy to navigate.
- Visually organized.
- Responsive where possible.

### Suggested visual style

```text
Background:       #0B0F14
Secondary:        #111821
Cards:            #151E29
Borders:          #263241
Primary Accent:   #00D4FF
Secondary Accent: #7867FF
Success:          #23C98B
Warning:          #F4B942
Danger:           #F05B65
Main Text:        #F3F6FA
Muted Text:       #8A98A9
```

Use cyan as the primary accent color, with restrained secondary colors.

Avoid excessive glow effects, unnecessary animations, and cluttered cards.

The interface should resemble a sophisticated monitoring and analytics product rather than a generic Streamlit demo.

---

# 11. MAIN APPLICATION LAYOUT

### Sidebar

The left sidebar should contain:

- ARGUS-X logo and branding.
- Project tagline: Track the Untrackable.
- Navigation menu.
- System status.
- Processing device information.
- Version information.

### Main Navigation

The application should have these pages:

1. Dashboard.
2. Video & Camera.
3. Live Tracking.
4. Object Manager.
5. Identity History.
6. Analytics.
7. Settings.

Each page should have a consistent visual language and layout.

---

# 12. DETAILED UI DESIGN

## 12.1 Dashboard

Purpose: Provide an overview of the ARGUS-X system.

### Top Section

Display:

- ARGUS-X title.
- Welcome or overview message.
- Current system status.
- Active processing session.
- Processing device information.

### Statistics Cards

Display:

1. Total Identities.
2. Active Objects.
3. Missing Objects.
4. Archived Objects.
5. Frames Processed.
6. Total Sessions.

### Main Content

Include:

- Recent processing sessions.
- Object category distribution.
- Recent identity events.
- Processing activity.
- System performance information.

### Design

Use clean cards with clear labels, compact statistics, and meaningful charts.

Avoid displaying fabricated statistics. All dashboard values should come from actual application data.

---

## 12.2 Video & Camera

Purpose: Provide the primary video-processing interface.

### Layout

The page should contain:

**Left/Main Section**
- Video uploader.
- Original video preview.
- Annotated video preview.
- Processing progress.
- Video playback.

**Right/Settings Section**
- Confidence threshold.
- Maximum frames.
- Image size.
- Tracking configuration.
- Processing controls.

### Controls

- Upload Video.
- Start Processing.
- Stop Processing.
- Reset.
- Download Annotated Video.

### Processing Experience

While processing:

- Show the current frame count.
- Show progress percentage.
- Display the latest annotated preview.
- Display detected object statistics.
- Allow the user to stop processing.

When processing finishes:

- Display the final annotated video.
- Provide playback controls.
- Provide a download option.
- Display processing summary.
- Save session information.

### Important technical requirement

Video processing should run in a background worker.

Do not perform long-running video processing directly inside a normal Streamlit page rerun.

Use safe file handling, proper resource cleanup, and reliable video encoding.

---

## 12.3 Live Tracking

Purpose: Provide real-time object monitoring.

### Interface Components

- Live camera feed.
- Annotated camera feed.
- Start Camera button.
- Stop Camera button.
- Camera status.
- Active identity count.
- Detection statistics.
- Current object list.

### Object Display

Show:

- ARGUS ID.
- Object class.
- Confidence.
- Current status.
- Tracker ID where useful.

The camera processing pipeline should use the same core engine as uploaded video processing wherever practical.

---

## 12.4 Object Manager

Purpose: Allow users to inspect and manage all registered identities.

### Object Table

Display:

| Column | Description |
|---|---|
| ARGUS ID | Unique identity |
| Object Name | User-defined name |
| Class | Detected category |
| Tracker ID | Current or last tracker association |
| Status | Active, Missing, Archived |
| First Seen | First observation |
| Last Seen | Latest observation |
| Frames Seen | Number of observed frames |

### Actions

- Search objects.
- Filter by category.
- Filter by status.
- Open identity details.
- Rename identity.
- View tracking information.

### Identity Detail Panel

When a user selects an identity, show:

- ARGUS ID.
- Display name.
- Object class.
- Current status.
- First-seen and last-seen information.
- Detection history.
- Tracking statistics.
- Available visual evidence.

Future versions should include appearance thumbnails and identity match history.

---

## 12.5 Identity History

Purpose: Maintain a historical record of registered objects.

Display:

- Previously detected identities.
- First-seen information.
- Last-seen information.
- Archived identities.
- Associated processing sessions.
- Identity lifecycle events.

Provide search and filtering capabilities.

Future versions should include visual re-identification events, showing when an old identity was recovered.

---

## 12.6 Analytics

Purpose: Present useful information about the processing session and detected objects.

### Suggested Statistics

- Total objects detected.
- Total unique ARGUS identities.
- Active identities.
- Missing identities.
- Archived identities.
- Frames processed.
- Processing duration.
- Object category distribution.
- Object observation frequency.
- Identity lifecycle events.

### Suggested Charts

1. Object category distribution.
2. Active versus archived identities.
3. Detection activity over time.
4. Frames processed over time.
5. Identity appearance frequency.
6. Object class statistics.

Use real data from the engine and database.

Do not display placeholder values as actual system results.

---

## 12.7 Settings

Purpose: Allow users to configure system behavior.

### Detection Settings

- Confidence threshold.
- Image size.
- Model selection when supported.

### Tracking Settings

- Tracker configuration.
- Missing-frame threshold.
- Archive threshold.

### Processing Settings

- Maximum frames.
- Video processing options.
- Output directory.
- Hardware device selection where supported.

### System Information

- Python version.
- PyTorch version.
- CUDA availability.
- GPU information.
- Model information.
- Application version.

Settings should be validated before being applied.

---

# 13. CODING AND TECHNOLOGY STACK

The project should use Python as the primary programming language.

JavaScript may be introduced only when required for a specific frontend enhancement.

Do not introduce C or C++ application code.

### Backend

- Python 3.12.
- Ultralytics.
- YOLO26n.
- PyTorch.
- OpenCV.
- ByteTrack.
- NumPy.
- SciPy.
- SQLite.
- SQLAlchemy or other suitable database abstractions when needed.

### Frontend

- Streamlit.
- Streamlit components.
- Streamlit WebRTC.
- Custom CSS.
- Plotly or another suitable charting library.

### Video Processing

- OpenCV.
- PyAV.
- streamlit-webrtc.

### Hardware Acceleration

- PyTorch CUDA.
- NVIDIA GPU where available.

The system should also support CPU execution when CUDA is unavailable.

### Development Environment

- Windows.
- PowerShell.
- Visual Studio Code.
- Python virtual environment.

---

# 14. PROJECT DIRECTORY STRUCTURE

Maintain a modular and understandable project structure.

```text
ARGUS-X/
Göé
Gö£GöÇGöÇ backend/
Göé   Gö£GöÇGöÇ app/
Göé   Göé   Gö£GöÇGöÇ core/
Göé   Göé   Göé   Gö£GöÇGöÇ argus_engine.py
Göé   Göé   Göé   GööGöÇGöÇ video_job.py
Göé   Göé   Göé
Göé   Göé   Gö£GöÇGöÇ detectors/
Göé   Göé   Göé   GööGöÇGöÇ yolo_detector.py
Göé   Göé   Göé
Göé   Göé   Gö£GöÇGöÇ tracking/
Göé   Göé   Göé   Gö£GöÇGöÇ object_tracker.py
Göé   Göé   Göé   GööGöÇGöÇ identity_memory.py
Göé   Göé   Göé
Göé   Göé   Gö£GöÇGöÇ reidentification/
Göé   Göé   Göé   GööGöÇGöÇ future ReID modules
Göé   Göé   Göé
Göé   Göé   Gö£GöÇGöÇ database/
Göé   Göé   Göé   GööGöÇGöÇ future database modules
Göé   Göé   Göé
Göé   Göé   GööGöÇGöÇ utils/
Göé   Göé       GööGöÇGöÇ future utility modules
Göé   Göé
Göé   Gö£GöÇGöÇ test_detector.py
Göé   Gö£GöÇGöÇ test_tracking.py
Göé   GööGöÇGöÇ test_identity_memory.py
Göé
Gö£GöÇGöÇ frontend/
Göé   Gö£GöÇGöÇ app.py
Göé   Gö£GöÇGöÇ components/
Göé   Gö£GöÇGöÇ pages/
Göé   GööGöÇGöÇ assets/
Göé
Gö£GöÇGöÇ models/
Göé   GööGöÇGöÇ yolo26n.pt
Göé
Gö£GöÇGöÇ outputs/
Göé   Gö£GöÇGöÇ processed videos
Göé   Gö£GöÇGöÇ database
Göé   GööGöÇGöÇ test outputs
Göé
Gö£GöÇGöÇ uploads/
Göé
Gö£GöÇGöÇ requirements.txt
Gö£GöÇGöÇ README.md
GööGöÇGöÇ .gitignore
```

Additional modules should be introduced only when they provide a clear architectural benefit.

Do not move or rename existing working modules unnecessarily.

---

# 15. CURRENT DEVELOPMENT STATE

The following components have already been developed.

### Implemented and Tested

1. YOLO26n detector.
2. ByteTrack object tracker.
3. ARGUS identity memory.
4. Unique ARGUS ID generation.
5. Object lifecycle management.
6. ARGUS-X core processing engine.
7. Bounding box and identity annotation.
8. SQLite session and identity storage.
9. Standalone background video-processing worker test.

### Developed but Requiring Further Validation

1. Streamlit dashboard.
2. Video upload workflow.
3. Live annotated video preview.
4. Processing progress and stop controls.
5. Final video playback and download.
6. Webcam integration.
7. Dashboard-to-backend end-to-end workflow.

### Not Yet Implemented GÇö Planned Features

1. Appearance-based visual re-identification.
2. Identity recovery after tracker ID changes.
3. Advanced long-term tracking.
4. Cross-camera identity matching.
5. CCTV and RTSP integration.
6. Open-vocabulary object detection.
7. Advanced object trajectory analytics.

The implementation status must be updated whenever a feature is successfully tested.

---

# 16. DEVELOPMENT ROADMAP GÇö FIRST STEP TO FINAL PRODUCT

Development should follow a structured sequence.

Do not jump directly into advanced features before the existing system is stable.

## PHASE 1 GÇö Project Foundation

Objectives:

- Set up the Python environment.
- Install dependencies.
- Establish the directory structure.
- Verify CUDA and GPU availability.
- Load the YOLO26n model.
- Verify video input and OpenCV functionality.

Deliverable: A working development environment.

**Current status: Completed.**

## PHASE 2 GÇö Object Detection

Objectives:

- Implement the YOLO detector.
- Process individual frames.
- Detect supported object categories.
- Return bounding boxes and confidence scores.
- Test detection on sample images and videos.

Deliverable: Working object detection module.

**Current status: Implemented and tested.**

## PHASE 3 GÇö Multi-Object Tracking

Objectives:

- Integrate ByteTrack.
- Associate detections across frames.
- Assign tracker IDs.
- Maintain multiple simultaneous tracks.
- Validate tracking continuity.

Deliverable: Working multi-object tracking module.

**Current status: Implemented and tested.**

## PHASE 4 GÇö Identity Memory

Objectives:

- Create the ARGUS ID generator.
- Associate tracker IDs with ARGUS IDs.
- Store identity information.
- Handle missing and archived objects.
- Support identity renaming.
- Maintain object statistics.

Deliverable: Persistent identity records.

**Current status: Implemented and tested.**

## PHASE 5 GÇö Core Processing Engine

Objectives:

- Integrate detection and tracking.
- Integrate identity memory.
- Draw bounding boxes and labels.
- Generate annotated frames.
- Maintain processing statistics.
- Save session information.

Deliverable: Unified ARGUS-X processing engine.

**Current status: Implemented and tested.**

## PHASE 6 GÇö Video Processing Worker

Objectives:

- Implement background video processing.
- Decode uploaded video.
- Process frames.
- Generate annotated previews.
- Write the final annotated video.
- Handle stop requests.
- Handle errors and resource cleanup.
- Ensure output videos are playable.

Deliverable: Reliable asynchronous video processing.

**Current status: Standalone worker test passed. Dashboard integration requires validation.**

## PHASE 7 GÇö Dashboard and UI Integration

Objectives:

- Finalize the dark-themed dashboard.
- Connect all pages to actual engine data.
- Complete the video upload workflow.
- Validate annotated live previews.
- Validate progress updates.
- Validate stop controls.
- Validate final video playback.
- Validate download functionality.
- Validate webcam processing.
- Test database integration.

Deliverable: Fully working ARGUS-X dashboard.

**Current status: Developed; end-to-end validation remains.**

## PHASE 8 GÇö Object Management and Analytics

Objectives:

- Complete object tables.
- Add search and filters.
- Add identity details.
- Add renaming.
- Add historical records.
- Connect analytics to real database information.
- Improve session summaries.

Deliverable: Complete identity management and monitoring interface.

## PHASE 9 GÇö Visual Re-Identification

Objectives:

- Research suitable ReID models.
- Extract object crops.
- Generate appearance embeddings.
- Store embeddings.
- Compare new observations against previous identities.
- Add similarity scoring.
- Add identity recovery logic.
- Handle uncertain matches safely.

Deliverable: Appearance-based identity recovery.

## PHASE 10 GÇö Advanced Tracking

Objectives:

- Improve long-term identity continuity.
- Handle occlusion.
- Handle camera movement.
- Improve similar-object separation.
- Maintain object trajectories.
- Improve matching under lighting and viewpoint changes.

Deliverable: More robust long-term tracking.

## PHASE 11 GÇö Multi-Camera Intelligence

Objectives:

- Integrate multiple camera streams.
- Create camera-specific processing sessions.
- Share identity information across cameras.
- Perform cross-camera appearance matching.
- Maintain centralized identity history.

Deliverable: Multi-camera tracking platform.

## PHASE 12 GÇö Finalization and Deployment

Objectives:

- Optimize GPU utilization.
- Improve processing speed.
- Improve error handling.
- Add logging and diagnostics.
- Improve database reliability.
- Prepare documentation.
- Add installation instructions.
- Add demonstration videos and screenshots.
- Prepare GitHub repository.
- Evaluate deployment options.

Deliverable: Documented, tested, and deployable ARGUS-X platform.

---

# 17. CODING RULES FOR FUTURE DEVELOPMENT

When developing ARGUS-X, follow these requirements.

### 1. Preserve Existing Working Code

Do not rewrite working modules without a clear reason.

Inspect the existing implementation before suggesting modifications.

### 2. Work Incrementally

Implement one feature at a time.

Test each module before integrating it into the complete system.

### 3. Explain Before Changing

Before modifying important architecture or files, explain:

- What needs to change.
- Why it needs to change.
- Which files are affected.
- What the expected result will be.

### 4. Provide Complete Code

When a file needs to be replaced, provide the complete corrected file rather than disconnected fragments.

Clearly identify the file path.

### 5. Provide Exact PowerShell Commands

The development environment is Windows PowerShell.

Always provide commands that can be copied directly.

Use this Python interpreter:

```powershell
& "C:\ARGUS-X\.venv\Scripts\python.exe"
```

To launch the dashboard:

```powershell
& "C:\ARGUS-X\.venv\Scripts\python.exe" -m streamlit run frontend/app.py
```

Do not assume the local project `.venv` contains the installed dependencies.

### 6. Test Every Important Change

After each change:

- Run the relevant test.
- Check the output.
- Inspect errors.
- Confirm expected behavior.
- Move to the next step only after validation.

### 7. Protect Existing Data

Do not delete:

- Existing model weights.
- Existing output videos.
- Existing database records.
- Existing test files.
- Working project modules.

Create backups before risky modifications.

### 8. Avoid Unnecessary Complexity

Prefer simple, reliable, modular Python implementations.

Do not introduce additional frameworks or services without a clear requirement.

### 9. Keep Documentation Accurate

Clearly label features as:

- Implemented.
- Tested.
- In development.
- Planned.

Never describe a future feature as an existing capability.

---

# 18. PERFORMANCE AND RELIABILITY REQUIREMENTS

ARGUS-X should be developed with performance and reliability in mind.

### GPU Utilization

- Use CUDA when available.
- Support CPU fallback.
- Avoid unnecessary model reloads.
- Manage GPU memory carefully.

### Video Processing

- Use background processing.
- Release video capture resources.
- Release video writers.
- Use appropriate output codecs.
- Verify that output files are finalized correctly.
- Avoid incomplete MP4 files.
- Handle unsupported or corrupted video files.

### Identity Management

- Avoid duplicate ARGUS IDs.
- Avoid accidental identity overwrites.
- Keep tracker IDs and ARGUS IDs separate.
- Preserve identity history.
- Use confidence-aware matching for future ReID.

### Dashboard

- Avoid unnecessary full-page reruns.
- Keep processing controls responsive.
- Display meaningful error messages.
- Avoid duplicate video players.
- Ensure progress and status information are updated correctly.

---

# 19. FINAL EXPECTED PRODUCT

The final ARGUS-X platform should become an intelligent object tracking and identity management system.

A user should be able to:

1. Open the ARGUS-X dashboard.
2. Upload a video or connect a supported camera.
3. Start object detection and tracking.
4. See multiple objects detected simultaneously.
5. View bounding boxes and unique ARGUS identities.
6. Monitor objects in real time.
7. Inspect individual object information.
8. Rename and manage object identities.
9. View object history and tracking statistics.
10. Recover previously observed identities using visual re-identification.
11. Handle temporary object disappearance and occlusion more effectively.
12. Eventually follow object identities across multiple cameras.
13. Export annotated videos and processing information.

The final goal is to transform ARGUS-X from a conventional object detection and short-term tracking application into a persistent object identity intelligence platform.

---

# 20. MASTER DEVELOPMENT INSTRUCTION

Act as a senior computer vision engineer, Python developer, software architect, and UI/UX designer responsible for helping develop ARGUS-X.

Your responsibility is to help build the project from its current implementation to the complete vision described in this specification.

Follow these instructions:

1. Understand the existing project architecture before changing anything.
2. Preserve all working modules.
3. Identify incomplete features and technical limitations.
4. Follow the development roadmap in sequence.
5. Build the backend and frontend as modular components.
6. Keep the user interface consistent with the specified professional dark monitoring dashboard.
7. Use Python as the primary programming language.
8. Use Streamlit for the frontend.
9. Use YOLO26n and ByteTrack for the current detection and tracking pipeline.
10. Keep persistent identity management separate from tracker IDs.
11. Design visual re-identification as a separate future module.
12. Provide complete code and exact file paths when modifications are required.
13. Provide copy-ready PowerShell commands.
14. Test every major implementation step.
15. Explain technical concepts in beginner-friendly language.
16. Never claim that a feature works until it has been tested.
17. Avoid unnecessary rebuilding, refactoring, or dependency changes.
18. Maintain accurate documentation.
19. Prioritize reliability, performance, and a polished user experience.
20. Work step by step, using checkpoints and waiting for test results before moving forward.

**The immediate development priority is to validate and complete the existing video-processing dashboard and webcam workflows before beginning advanced visual re-identification.**

The long-term priority is to implement reliable visual identity recovery and evolve ARGUS-X into a persistent multi-object and multi-camera tracking platform.

---

# PROJECT IDENTITY

**ARGUS-X**

**TRACK THE UNTRACKABLE**

*Beyond Detection. Beyond Tracking. Persistent Identity.*
