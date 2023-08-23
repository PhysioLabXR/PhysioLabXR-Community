---
title: 'PhysioLabXR: A software platform in Python for multi-modal brain-computer interface and real-time experiment pipelines'
tags:
  - Python
  - neuroscience
  - human computer interaction
  - brain-computer interface
  - multi-modality
  - virtual augmented reality
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib
---

# Summary

![PhysioLabXR is built for real-time, multi-modal physiological experiments. It offers features including various 
diverse visualization methods, digital 
signal processing modules, replaying past experiments, and a scripting interface to deploy custom pipelines.\label{fig:teaser}](physiolabxr%20teaser.png)


Recent years have seen growing interest in multi-modal experiments and closed-loop interaction systems in fields of 
neuroscience and human-computer interaction (HCI). This creates a need for versatile software platforms that can handle 
diverse data streams in real time, enable flexible processing pipelines, and support prototyping of interactive paradigms. 

We introduce *PhysioLabXR*, an open-source software platform aimed at streamlining experimentation involving real-time 
physiological data and interactive interfaces. *PhysioLabXR* provides native support for various devices and popular data 
transfer protocols such as LabStreamingLayer and ZMQ. Its Key features include multi-stream visualization, digital signal 
processing modules, robust recording and synchronized replay of multi-modal data, and a Python-based scripting interface to enable custom pipelines. The software architecture utilizes concurrency and parallelism to ensure smooth performance.

*PhysioLabXR* is built for neurosciences and HCI experiments. Benchmarks demonstrate its ability to handle high-throughput 
multi-stream data reliably. Use cases showcase its versatility in areas like VR experiments, P300 spellers, and novel 
gesture interfaces. The software reduces overheads in the research cycle and provides researchers full flexibility to build 
customized systems. It represents an extensible framework to tackle the complexity of modern experiments at the intersection 
of neuroscience, HCI and related fields. By simplifying real-time multi-modal data handling and interactive prototyping, 
it has the potential to accelerate insights and innovations. 


# Statement of need

*Multi-modal experiments and supporting software* In recent years, an increasing body of research in brain-computer interface emphasizes multi-modal and close-loop approaches.
Modern experiments in neuroscience and related fields continuously place more emphasis on multiple modalities. Though
the early-day experiment combining EEG and event markers can be considered multi-input. The potential of multi-modal
experiment shines when different physiological measurement marries together. Sometimes streams can act as events to extract
meaningful intervals of data from other sensors, such as fixation-related potential (FRP) studies where eye behavior and EEG
are combined [@nikolaev2016combining]. Multiple physiological signal can also be combined to enhance the predictive
power for application including emotion recognition [@koelstra2011deap] [@he2020advances] and movement actuation
via sensorimotor rhythms [@sollfrank2016effect]. They also help study how different physiological systems interact like
using pupil dilation as a proxy for locus coeruleus activity [@murphy2014pupil].

Many analysis methods and computational modeling have been proposed specifically for multi-modal data including the
analytical approach of FRP deconvolution [@dimigen2021regression], and the data-driven multi-modal deep learning (DL) models
[@wang2022husformer] [@nie20163d]. Efforts have been taken more recently to design real-time interactive systems (i.e., BCIs)
using multi-modal data [@ahn2017multi] and supporting software were developed
[@kothe2014labstreaminglayer] [@razavi2022opensync]. Most commonly the software are networking protocols focusing
on timestamp synchronization across different modalities. To provide researchers an open-source, community-driven software
tool for the increasing needs in multi-modal experiments, PhysioLabXR offers a complete all-in-one GUI application for
visualizing, recording, replaying past experiments, and deploying end-to-end DSP and ML pipelines.

# Benefits

*Software tools for real-time experiments* Despite the prevalence of real-time behavior or interactive experiments, software tools supporting them are surprisingly 
far between. Researchers often resort to creating their own data pipeline for each new project, adding significant overhead in the way of
high-quality research. The existing software tools for handling physiological data in real-time can be broadly categorized 
into two groups. The first group consists of proprietary software closely tied to specific hardware, often developed by 
the manufacturer of the hardware. These device-coupled software tools aim to provide a tailored experience for the types 
of signals captured by their hardware, offering specialized data visualization interfaces. For example, NIRX Aurora [@nirx] 
provides a 3D scalp and cortex model for visualizing hemoglobin oxidation. Meanwhile, these software often offer powerful 
data analysis tools for the respective hardware, such as fixation detection in eyetracking software suites like Tobii [@tobii]. 
While this line of software excels in their compatibility with the associated device, they often lack support for multi-modal 
experiments (e.g., pair fNIRS with eyetracking), as the data they capture remain isolated within their ecosystem and cannot 
be synchronized with other modalities. To address this limitation, researchers often resort to third-party solutions that 
support integration with all the data sources in their experiment. LSL solves the inter-device compatibility problem as 
a standardized network protocol for synchronizing data from different devices [@kothe2014labstreaminglayer] [@wang2023scoping]. 
ZMQ is another networking protocol popular among researchers for streaming high-throughput data with low latency between 
processes [@michalareas2022scalable] [@macinnes2020pyneal] [@baltruvsaitis2016openface]. These libraries offer bindings 
with various programming languages, making them versatile choices for various platforms. Nevertheless, when devices lack 
built-in support for standardized data streaming protocols, researchers must create scripts to relay the data to the network. 
On the receiving end, LSL comes with a serialization protocol and a standalone software - LabRecorder [@kothe2014labstreaminglayer], 
offering a GUI for monitoring available streams and recording them to the local file system.

At the same time, the data transfer middleware does not cover the ability to visually inspect data streams in real time, 
which is crucial for many experiments, particularly those involving devices prone to failure and artifacts during operation, 
such as EEG and fNIRS. Real-time visualization allows experimenters to react promptly to sensor failures and prevent wasting 
valuable participant time. Additionally, visualizing multiple streams as they come in helps researchers gain deeper insight. 
They can, for example, observe the stimulus event triggers on the event marker stream while inspecting the EEG streams 
for corresponding event-related potential \cite{pei2020brainkilter}.

With that, we come to the second group of device-independent graphic software tools; many have been proposed over the 
years to assist researchers and practitioners in conducting experiments involving real-time data collection and feedback. 
These tools often utilize the networking protocols mentioned earlier for data transfer and may include native plug-ins for 
specific devices, such as Brainflow for OpenBCI devices \cite{brainflow}. Some offer features such as real-time data 
processing modules as in NeuroPype \cite{neuropype}, and MNE S \cite{esch2018mne}. At the time of this article, a majority 
of popular experiment platforms such as OpenVibe, MNE Scan, and FieldTrip are primarily built using statically compiled 
languages (e.g., c and C++) or proprietary languages (e.g., MATLAB). While tools such as OpenVibe and NeuroPype offer 
scripting interfaces where users can add custom scripts in Python, it is worth noting that Python, despite its rising 
popularity as a programming language \cite{srinath2017python} for its ease of use and versatility, is less favored by 
developers as a backbone language for experiment platform that requires high precision and high data throughput. 
Although not impossible, implementing an experiment platform in an interpreted language like Python necessitates 
significant optimization efforts to match the performance level of platforms built with a compiled language. 
Nevertheless, one of the objectives that PhysioLabXR seeks to address is to prioritize ease of understanding, enabling 
users to expand and build upon its open-source foundation. As a result, Python dominates PhysioLabXR's implementation 
from frontend GUI to the backend servers without sacrificing performance owing to its concurrent runtime architecture 
(section \ref{sec: software architecture}). Users are welcome to use the software as a scaffold and leverage Python's 
extensive APIs to shape the platform according to their needs.

Moreover, a scripting interface to add user scripts is offered by most experiment software platforms, allowing users to build flexible 
data processing pipelines. OpenVibe, for instance, is known for its graphical pipeline builder while also allowing the 
addition of Python and LUA scripts \cite{renard2010openvibe}. iMotion uses a platform-specific iMotion scripting language 
to achieve the same goal \cite{iMOTION_Script_Language}. With PhysioLabXR, users write Python scripts to interact with any 
data stream and stream out the processed results with built-in I/O modules. This flexibility allows users to design 
closed-loop systems, including deploying machine learning models and sending predictions to and from PhysioLabXR.


# PhysioLabXR: the experiment platform
To facilitate HCI and neuroscience experiments, prototyping interaction paradigms such as brain-computer interface (BCI), 
we present PhysioLabXR (Reality Navigation Laboratory Application), a software platform designed to simplify the 
research \& development (R\&D) process. PhysioLabXR combines the ease of use and versatility of Python as main language 
in its codebase, while leveraging the performance of C++ for computation-intensive subroutines. As an a open-source, 
cross-platform  experiment tool caters to both academic researcher and industry practitioners. It offers native support 
for various sensing hardware such as electroencephalogram (EEG), eyetracker, mmWave radar, as well as video and audio 
devices. Moreover, PhysioLabXR integrates with popular stimulus-presentation software such as Unity \cite{Unity} and 
PsychoPy \cite{peirce2007psychopy}, and other analysis software including MATLAB \cite{matlab}. For experiments already 
utilizing LabStreamLayer (LSL) and ZeroMQ (ZMQ) for data transfer, PhysioLabXR provides convenient network stream 
connectivity with these two widely-used data middlewares.

A key feature of PhysioLabXR is its ability to visualize multiple data streams using the most suitable plots for each 
physiological data type, such as line charts for eye tracking or functional near infrared spectroscopy (fNIRS) data, and spectrograms 
for EEG or audio data. This function enables users to validate measurement setups prior to and during data collection. 
In addition, user can create real-time data processing pipeline with the built-in digital signal processing (DSP) modules 
such as filters and apply them to data streams. PhysioLabXR has a highly optimized serialization interface to record 
experimental data to file system. These recordings can later be replayed with the playback interface, which includes 
features such as setting the playback location using a slider. 


One of the most powerful and flexible aspects of PhysioLabXR is its scripting interface, where users can create custom data 
processing pipelines, training machine learning models and deploy them to make predictions. On the other hand, when a data 
source is not natively supported and the network streaming over LSL or ZMQ is not an option, PhysioLabXR provides a 
specialized scripting module that enables user to define custom data source simply by overriding a few functions. 
Additionally, PhysioLabXR supports multi-modal experiments that incorporate audio/video data in their pipeline, or when 
the experimenter needs to synchronously record the screen where the stimuli are presented, along with other physiological 
sensors. PhysioLabXR offers native connectivity with screen capture, webcams and microphones on the host computer.


PhysioLabXR is designed with the following objectives 1) providing a user-friendly graphic user interface (GUI) for 
working with both physiological and behavioral data 2)ensuring a reliable and robust backend capable of synchronizing and 
processing multi-modal and high-throughput data in a scalable manner. 3) streamlining the hitherto time-consuming and 
challenging steps in experiment cycles, including visualizing, recording and analyze data offline to understand 
physiological phenomenon, or for online uses such as providing neurofeedback in a brain-computers interface. 4) offering 
flexibility and ease of setup as a cross-platform solution, complemented by extensive developer API, which encourages 
users to extend the platform with custom hardware, and real-time processing scripts.


The development of PhysioLabXR adheres to industry-standard software development guidelines and implements continuous 
integration. Its modular software architecture simplifies the learning curve for users who wish add custom functionalities 
to fit their needs. This includes adding support for novel sensors not natively integrated into PhysioLabXR.


## Working with streams

All features in PhysioLabXR builds on *Streams*:
a stream is a sequence of data points that arrive in real-time, with each frame of data carrying a timestamp, 
whether it is from physiological sensors, video cameras, microphones, 
screen capture, or software-generated data. PhysioLabXR provides a unified interface for working with streams with 
functionalities including
visualization, recording, replaying, and DSP. 
Each feature is designed to address different aspects of needs in experiments involving real-time data collection and 
processing. Here is a brief overview of these features:
* *Data stream API* establishes connection with data sources, either through native plugins or network protocols (LSL or ZMQ). 
* *Visualization* helps users visually inspect their data in real-time to better understand their data. 
* *Recording* lets users capture experimental data in real-time and export them for further analysis. 
* *Replay* enables users to play back data streams from past experiments, and if needed, test their data processing script and algorithm in real-time as if the experiment is running live. 
* *DSP* is another powerful feature in PhysioLabXR that allows users to apply various predefined signal processing 
algorithms to their data streams. 

![Example use case of PhysioLabXR in a memory formation and retrieval experiment involving real-time processing of 
pupillometry and fMRI streams. This example demonstrates the diverse visualization options the software packs.
In this experiment, the participant is asked to navigate a virtual shopping mall and give verbal responses during their task. 
(A) The 3D fMRI visualizer shows fMRI data streamed in real-time (B) The experimenter uses RenaLabApp to monitor and 
record scene from the participant's first person view, while they performs the task. (C) The participant's speech is 
captured using a microphone connected to PhysioLabXR that visualizes the audio data as a spectrogram. (D) Eye movement 
and pupillometry data are recorded through an eyetracker outside of the scanner that receives the participant's eye image 
via a mirror. The time series of the eyetracking data are plotted in a line chart. (E) Simultaneously, a machine learning 
model deployed through RenaLabApp's scripting interface predicts from the fMRI data and pupillometry if a target memory 
is being retrieved, with the two-class inference result visualized as a bar plot..\label{fig:teaser}](ALlPlottingFormat.png)


## Scripting Interface

![Example script setup a fixation-related potential (FRP) experiment. The FixationDetection script (upper right) identify 
fixations from eyetracking stream, while the P300Detector script (bottom right) decodes EEG data locked to detected fixations, 
and determine if a FRP is elicited by a target object. This setup is similar to the experiment by conducted by 
[@rama2010eye]. The labels in the figure shows the following: 
(A) The eyetracking data as an input is processed by the fixation-detection script to check if a fixation is made by 
the participant. (B) The fixation results are streamed through the output LSL outlet named "Fixations". 
(C) In the *.init* function of P300Detector, the P300 classifier model is loaded from the file system path indicated by
the script parameter *model_path*. (D) and (E) If a fixation is detected, the model takes the EEG epoch time locked to 
the fixation and makes a prediction. A loop call is completed by writing the prediction results to the output stream named 
*P300Detect*.](scripting%example.png)

The scripting interface, within PhysioLabXR offers extensive possibility to researchers. 
It enables the execution of user-defined Python scripts, empowering users to create and deploy custom data processing 
pipelines. With python's versatility and open-source libraries encouraging exploration of novel applications like close-loop 
neurofeedback, users can train and run ML models in real-time, and communicate 
with external applications using built-in networking APIs including LSL and ZMQ. The script widget as part of the PhysioLabXR's GUI 
offers a straightforward to add, and run scripts, adjust attributes that influence the data processing pipeline's behavior, 
and monitor their performance.
These attributes include defining what streams go into the script as inputs,
setting input buffer duration, controlling the run frequency, creating outputs to visualize pipeline results or communicate with other programs,
and utilizing exposed parameters 
that allow variable adjustments during runtime. A script in PhysioLabXR consists of three abstract methods—$init$, $loop$, and 
$cleanup$—which users can override to specify behavior. The software ships with a collection of built-in scripts for commonly
used algorithms such as fixation detection and band-power computation, and connection to popular devices such as Tobii eyetrackers [@tobii] and
OpenBCI [@OpenBCI].


All these features can be used in combination with each other, enhancing the overall potential of RenaLabApp. 
For example, filter can first be applied to EEG data, and the user can visualize the filtered stream that can 
simultaneously drive a BCI with custom script.

## Software design principles

Smooth runtime experience is a significant challenge for large-scale Python software when dealing with high-throughput 
data, complex graphics, and frequent i/o operation from serialization, given the interpreted nature of the language.
PhysioLabXR is optimized through a combination of concurrency, parallelism, and a modular software architecture.

![Sequence diagram showing the information exchange between RenaLabApp's threads and processes. The main process contains 
concurrent threads of the main GUI controller, serialization, data worker, and visualization. When the user adds a new 
data source, the GUI thread forks a data worker and a visualization thread. More demanding operations run on separate 
server processes, including replay and scripting. User commands such as *StartStreaming*, *StartReplay*, 
and *StopRecording* are passed from the main GUI to the corresponding threads or processes.](RenaLabApp%Sequence%Diagram.png)

The architecture of PhysioLabXR follows rigorous software design patterns, minimizing maintenance efforts while maximizing 
scalability. Unit testing is at the core of our development process, each software feature, backend and GUI frontend,
is tested on both 
functionality and performance. The software is built with continuous integration, each
commit to the main branch triggers a full test routine to ensure compatibility.


# Future Scope

PhysioLabXR aims to be a versatile platform for any real-time experiments, primarily for, but not limited to, HCI and neuroscience.
It is designed to be a community-driven project with our core team of developers maintaining the architectural integrity and functional correctness
, and we welcome contributions from researchers and practitioners in related fields to build on this scaffold and expand its capabilities.

# Acknowledgements

We would like to express our gratitude for the support from our colleagues at the Laboratory for Intelligent Imaging and Neural Computing (LIINC),
and Computer Graphics and User Interfaces (CGUI) Lab at Columbia University. We would also like thank all the community members who have contributed to PhysioLabXR.

# Licensing and Availability
RenaLabApp is an open-source project distributed under the copyleft license - GNU General Public License v3.0, ensuring the free and open-source nature of the software and any downstream derivatives. Researchers are encouraged to modify the software to meet their specific needs and share their modifications with the community. To facilitate easy access to RenaLabApp and its resources, we provide the following links:

- **Website:** The official RenaLabApp website serves as a central hub for information and updates. It can be accessed at [https://www.physiolabxr.org](https://www.renalabapp.org).

- **Documentation:** Detailed documentation is available to guide users through the various features, tutorials, example use cases, and developer guides. It can be accessed at [https://realitynavigationdocs.readthedocs.io/en/latest/index.html](https://realitynavigationdocs.readthedocs.io/en/latest/index.html).

- **GitHub Repository:** Users can access the repository at [https://github.com/ApocalyVec/RenaLabApp](https://github.com/ApocalyVec/RenaLabApp). Additionally, users can submit bug reports and feature requests through the GitHub issue tracker.


# References

