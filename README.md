# Cartpole with Reinforcement Learning
![CartpoleDemo](https://github.com/Waverider144/Assets/blob/main/cartpole_training.gif?raw=true)
## Introdution

As an engineer with a background in **Industrial Automation**, I chose the classic **Cartpole** control problem as my entry point into the realms of Reinforcement Learning and Deep Learning.

Driven by an incurable "need for speed," I have developed multiple iterations focused on computational efficiency. Please navigate the menu below to explore the implementation that best fits your performance requirements.

## MENU
*[vDQNpy](#vdqnpy)
*[vDQNCpp](#vdqncpp)
*[vDQNCUDA](#vdqncuda)


## vDQNpy
<table border="0">
  <tr>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Waverider144/Assets/2da875f20ee63d378c1ca710ac94e51d2ca2d36c/Carpole_v1.svg" alt="Benchmark Origin" style="width:100%;">
      <p align="center"><b>Python v01(Baseline)</b></p>
    </td>
    <td width="50%">
      <img src="https://raw.githubusercontent.com/Waverider144/Assets/2da875f20ee63d378c1ca710ac94e51d2ca2d36c/Carpole_vHP.svg" alt="Design Outcome" style="width:100%;">
      <p align="center"><b>Python vHP(Optimized)</b></p>
    </td>
  </tr>
</table>

I provide four distinct Python versions, ranging from standard implementations to highly optimized architectures:

-   **`cartpole_v1.py`**: A high-fidelity physical model without performance overhead—serving as the project baseline.
    
-   **`cartpole_hp.py`**: A high-performance iteration incorporating **C++-style memory management** strategies to minimize computational latency.
    
-   **`cartpole_hpjit.py`**: Features **kernel merging** within the `optimized_train` segment and **vectorization** via the `VectorizedReplayBuffer`. Continuous optimization is currently underway.
    
-   **`cartpole_hpjax.py`**: An exploration of **JAX**, Google’s high-performance numerical computing paradigm. The results are remarkable, nearly achieving the execution efficiency of native C++ code.

## vDQNCpp
![vDQNC++](https://github.com/Waverider144/Assets/blob/main/vDQNCpp.png?raw=true)
While Python dominates the machine learning landscape, industry-standard software such as **Ansys Fluent**, **Zemax**, and **CodeV** rely on the performance of compiled C++. To natively integrate with C++ I/O streams, I developed **vDQNCpp**. This version utilizes **Libtorch** (the C++ frontend for PyTorch) to handle all neural network operations, bridging the gap between RL research and industrial deployment.





## vDQNCUDA
![enter image description here](https://raw.githubusercontent.com/Waverider144/Assets/e179e72e4bbb86b901e2e22655c196cca4527e09/vDQNCUDA.svg)
A high-octane derivative of vDQNCpp, this version offloads the entire neural network workload to the **GPU**. Watching the training logs flash across the screen at these speeds is, quite frankly, breathtaking. To quote the feeling... **Yes, NVIDIA!** 🚀
