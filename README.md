[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<h3 align="center">PhysioLabXR</h3>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/PhysioLabXR/PhysioLabXR">
    <img src="physiolabxr/_media/readme/PhysioLabXR Overview.png" alt="Logo">
  </a>


  <p align="center">
    A Python Platform for Real-Time, Multi-modal, Brain–Computer Interfaces and Extended Reality Experiments
    <br />
    <a href="https://physiolabxrdocs.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://physiolabxr.org">Official website</a>
    ·
    <a href="https://github.com/apocalyvec/physiolabxr/issues">Report Bug</a>
    ·
    <a href="https://github.com/apocalyvec/physiolabxr/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![PhysioLabXR Screen Shot][product-screenshot]](physiolabxr/_media/readme/AllPlottingFormat.png)

**PhysioLabXR** is a Python-based App for visualizing, recording, and processing (i.e., make prediction) 
data streams. PhysioLabXR can help you build novel interaction interface like BCIs as well as aid you in 
running experiments. It works best with multi-modal (e.g., combining EEG and eyetracking, camera with speech), high-throughput (~500Mbps/sec), real-time data streams.

[//]: # (Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)


Check out the paper on [Journal of Open Source Software](https://doi.org/10.21105/joss.05854): [![DOI](https://joss.theoj.org/papers/10.21105/joss.05854/status.svg)](https://doi.org/10.21105/joss.05854). Cite:

```
@article{Li2024, doi = {10.21105/joss.05854}, url = {https://doi.org/10.21105/joss.05854}, year = {2024}, publisher = {The Open Journal}, volume = {9}, number = {93}, pages = {5854}, author = {Ziheng 'Leo' Li and Haowen 'John' Wei and Ziwen Xie and Yunxiang Peng and June Pyo Suh and Steven Feiner and Paul Sajda}, title = {PhysioLabXR: A Python Platform for Real-Time, Multi-modal, Brain–Computer Interfaces and Extended Reality Experiments}, journal = {Journal of Open Source Software} }
```

### Built With

* [![Python][Python.org]][Python-url]
* [![QT][QT.io]][QT-url]
* [![PyQtGraph][pyqtgraph.org]][pyqtgraph-url]
* [![NumPy][numpy.org]][numpy-url]
* [![ZMQ][zeromq.org]][zeromq-url]
* [![LSL][LSL.org]][LSL-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Download and run the executable

Download the latest release exe from [here](https://github.com/PhysioLabXR/PhysioLabXR/releases), available for Windows, Mac, and Linux.

### Install with pip

PhysioLabXR currently supports Python 3.9, 3.10, and 3.11. Support for Python 3.12 is coming soon.

Install PhysioLabXR's [PYPI distribution](https://pypi.org/project/physiolabxr/) with

  ```sh
    pip install physiolabxr
  ```
Then run with

  ```sh
    physiolabxr
  ```

### Run from Source

Alternatively, you can clone the repo and run from source.

  ```sh
    git clone https://github.com/PhysioLabXR/PhysioLabXR.git
    cd PhysioLabXR
    pip install -r requirements.txt
  ```

The entry point to PhysioLabXR is `physiolabxr.py`, located in the folder named "physiolabxr". From the root folder, you can run it by:

  ```sh
    python physiolabxr/physiolabxr.py
  ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>




## Usage


_For more examples, please refer to the tutorials in the [documentation](https://physiolabxrdocs.readthedocs.io/en/latest/index.html)_

Tutorials have examples for:

* Real-time fixation detection ([link](https://physiolabxrdocs.readthedocs.io/en/latest/FixationDetection.html))
* Real-time multi-modal event-related-potential classification with EEG and pupillometry
* P300 speller in Unity
* Stroop task with PsychoPy (link [https://physiolabxrdocs.readthedocs.io/en/latest/PsychoPy.html])

More are coming soon!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the 3-Clause BSD License. See `LICENSE.txt` for more information.

© 2023 The Trustees of Columbia University in the City of New York.  This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  To obtain a license to this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

[Ziheng 'Leo' Li](https://www.linkedin.com/in/ziheng-leo-li/) - zl2990@columbia.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to express our gratitude for the support from our colleagues at Columbia University and Worcester Polytechnic Institute. 
We would also like to thank all the community members who have contributed to *PhysioLabXR*.


[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/apocalyvec/renalabapp.svg?style=for-the-badge
[contributors-url]: https://github.com/apocalyvec/renalabapp/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/apocalyvec/renalabapp.svg?style=for-the-badge
[forks-url]: https://github.com/apocalyvec/renalabapp/network/members
[stars-shield]: https://img.shields.io/github/stars/apocalyvec/renalabapp.svg?style=for-the-badge
[stars-url]: https://github.com/apocalyvec/renalabapp/stargazers
[issues-shield]: https://img.shields.io/github/issues/apocalyvec/renalabapp.svg?style=for-the-badge
[issues-url]: https://github.com/apocalyvec/renalabapp/issues
[license-shield]: https://img.shields.io/github/license/apocalyvec/renalabapp.svg?style=for-the-badge
[license-url]: https://github.com/apocalyvec/renalabapp/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/ziheng-leo-li/
[product-screenshot]: physiolabxr/_media/readme/AllPlottingFormat.png

[Python.org]: https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&color=3776AB&logoColor=white
[Python-url]: https://python.org/

[QT.io]: https://img.shields.io/badge/QT-000000?style=for-the-badge&logo=qt
[QT-url]: https://qt.io/

[zeromq.org]: https://img.shields.io/badge/zeromq-000000?style=for-the-badge&logo=zeromq&color=DF0000
[zeromq-url]: https://qt.io/

[lsl.org]: https://img.shields.io/badge/lsl-000000?style=for-the-badge&color=lightgrey
[lsl-url]: https://labstreaminglayer.org

[pyqtgraph.org]: https://img.shields.io/badge/pyqtgraph-000000?style=for-the-badge&color=bbbbff
[pyqtgraph-url]: https://www.pyqtgraph.org/

[numpy.org]: https://img.shields.io/badge/numpy-000000?style=for-the-badge&logo=numpy&color=013243
[numpy-url]: https://www.numpy.org/

[Next-url]: https://python.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
