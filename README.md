# Motivation
This controller is based on the well-known MIT CMPC for MIT Cheetah 3. However, the authors find the update frequency is too low when implementing it on our Unitree A1 robot. As we stated in the comparative experiments of our paper, the update frequency of MPC significantly influences the control performance. Therefore, we propose three techniques to accelerate the solving procedure of the MPC control law. In our experiments, these techniques can significantly improve the solving frequency, especially when the problem is large (i.e. the predictive horizon is longer or the contacted legs are more). Therefore we call the controller Fast Convex Model Predictive Control, FCMPC, meaning that it is a faster version of CMPC.

# User Guide

# Citation

