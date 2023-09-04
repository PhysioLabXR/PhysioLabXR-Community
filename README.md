
_This README is still under construction_



[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/PhysioLabXR/PhysioLabXR">
    <img src="physiolabxr/_media/readme/PhysioLabXR Overview.png" alt="Logo">
  </a>

<h3 align="center">PhysioLabXR</h3>

  <p align="center">
    A Python Software Platform for Multi-Modal Brain-Computer Interface and Real-Time Experiment Pipelines
    <br />
    <a href="https://physiolabxrdocs.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/apocalyvec/physiolabxr">View Demo</a>
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

[![PhysioLabXR Screen Shot][product-screenshot]](physiolabxr/_media/readme/screenshot.png)

**PhysioLabXR** is a Python-based App for visualizing, recording, and processing (i.e., make prediction) 
data streams. PhysioLabXR can help you build novel interaction interface like BCIs as well as aid you in 
running experiments. It works best with multi-modal (e.g., combining EEG and eyetracking, camera with speech), high-throughput (~500Mbps/sec), real-time data streams.

[//]: # (Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



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



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the 3-Clause BSD License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

-->


<!-- CONTACT -->
## Contact

[Ziheng 'Leo' Li](https://www.linkedin.com/in/ziheng-leo-li/) - zl2990@columbia.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to express our gratitude for the support from our colleagues at Columbia University and Worcester Polytechnic Institute. We would also like to thank all the community members who have contributed to *PhysioLabXR*.


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
[product-screenshot]: physiolabxr/_media/readme/screenshot.png

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
