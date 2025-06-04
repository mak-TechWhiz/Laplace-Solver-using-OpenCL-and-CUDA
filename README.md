This project was done using Google Colab GPU. You need to import the 'OpenCL_template.ipynb' file in google colab and run each cell. 

A simple OpenCL wrapper (https://github.com/inf-eth/OpenCL.git) is used to run all the codes. Navigate to OpenclSimpleWrapper/src and you will find main.cpp and kernels.cl.
Replace those files with your own main.cpp and kernels.cl and just run the corresponding cell from Google Colab.

For CUDA you do not need any wrappers. Just upload your .cu file into the memory of colab, compile the file by doing: 

'!nvcc -arch=sm_75 -o laplace_solver laplace_solver.cu'

then run it using:

'!./laplace_solver'
