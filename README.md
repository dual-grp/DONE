# DONE: Distributed Approximate Newton-type Method for Federated Edge Learning.

This repository is for the Experiment Section of the paper:
"DONE: Distributed Approximate Newton-type Method for Federated Edge Learning"

Authors: Canh T. Dinh, Nguyen H. Tran, Tuan Dung Nguyen, Wei Bao, Amir Rezaei Balef

Paper Link: https://arxiv.org/pdf/2012.05625.pdf

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 4 datasets: MNIST, Human Activity, FEMNIST, and Synthetic

- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_32users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 32 and NUM_LABELS = 3
  
- To generate Human Activity Data: 
  - Access data/human_activity and run: "python3 human_acitity_generation"

- To generate FEMNISTy Data: 
  - Access Nist/human_activity and run: "python3 generate_niid_nist_32users.py"

- To generate non-iid Synthetic:
  - Access data/Linear_synthetic and run: "python3 generate_niid_linear_32users_updated.py". Synthetic data is configurable with the number of users, the numbers of labels for each user, and the value of $\kappa$.

- The datasets also are available to download at: https://drive.google.com/drive/folders/1LkBjkP0PzfRNiAY9ImN85r9vBIuW4U6-?usp=sharing

# Produce experiments and figures
- There is a main file "main.py" which allows running all experiments, and 2 files: "main_plot_mnist.py", "main_plot_synthetic.py" to plot all results after runing all experiment.  Only run "plot_mnist.py" and "plot_synthetic.py" after getting the results from training process.

## Performance comparison with different distributed algorithms (table 2 in our paper)
                          | Dataset | Algorithm | Alpha(gamma) | Test Accurancy |
                          |---------|-----------|--------------|----------------|
                          |  Mnist  | DONE      | 0.03         |  92.11 ± 0.01  |
                          |         | DANE      | 0.04         |  91.84 ± 0.02  |
                          |         | Newton    | 0.03         |  92.11 ± 0.01  |
                          |         | GD        | 0.2          |  91.35 ± 0.03  |
                          |---------|-----------|--------------|----------------|
                          | FEMNIST | DONE      | 0.01         |  80.60 ± 0.02  |
                          |         | DANE      | 0.02         |  77.57 ± 0.07  |
                          |         | Newton    | 0.01         |  80.60 ± 0.02  |
                          |         | GD        | 0.02         |  60.58 ± 0.03  |
                          |---------|-----------|--------------|----------------|
                          |  Human  | DONE      | 0.02         |  96.78 ± 0.01  |
                          | Activity| DANE      | 0.05         |  95.82 ± 0.02  |
                          |         | Newton    | 0.02         |  96.78 ± 0.01  |
                          |         | Newton    | 0.03         |  96.90 ± 0.01  |
                          |         | GD        | 0.1          |  80.02 ± 0.03  |
                          |---------|-----------|--------------|----------------|
- For MNIST:
      <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm Newton --batch_size 0 --alpha 0.05 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.04 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.2 --num_global_iters 100 --numedges 32
    </code></pre>
- For FEMNIST:
      <pre><code>
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.02 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm GD --batch_size 0 --learning_rate 0.02 --num_global_iters 100 --numedges 32
    </code></pre>

- For Human Activities:
      <pre><code>
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm Newton --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm Newton --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DANE --batch_size 0 --eta 1 --learning_rate 0.05 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm GD --batch_size 0 --learning_rate 0.1 --num_global_iters 100 --numedges 30
    </code></pre>

## Effect of hyper-parameters: $\alpha$, $R $, and $\kappa$
- To produce the Fig.1 : Effects of various values of $\alpha$ and $R$ on synthetic with diffent $\kappa = 10^2$
 <pre><code>
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.06 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.08 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.2 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 5 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 10 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Linear_synthetic --model linear_regression --algorithm DONE --batch_size 0 --alpha 0.1 --num_global_iters 100 --local_epochs 30 --numedges 32
    </code></pre>

### Effects of various values of $\alpha$ and $R$
  - MNIST
    <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.005 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 32
      
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 10 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 30 --numedges 32
    </code></pre>

  - FEMNIST
    <pre><code>
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.004 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.006 --num_global_iters 100 --local_epochs 40 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.008 --num_global_iters 100 --local_epochs 40 --numedges 32
      
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 10 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 20 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 30 --numedges 32

    </code></pre>
  
  - Human activies
    <pre><code>
       python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.005 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.015 --num_global_iters 100 --local_epochs 40 --numedges 30

      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 10 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 20 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 30 --numedges 30
    </code></pre>

### Effects of mini batch
  - MNIST
    <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 256 --alpha 0.01 --num_global_iters 100 --local_epochs 120 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 128 --alpha 0.01 --num_global_iters 100 --local_epochs 120 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 64 --alpha 0.01 --num_global_iters 100 --local_epochs 120 --numedges 32
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 32 --alpha 0.01 --num_global_iters 100 --local_epochs 120 --numedges 32
    </code></pre>
  - FEMNIST
    <pre><code>
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 256 --alpha 0.005 --num_global_iters 100 --local_epochs 80 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 128 --alpha 0.005 --num_global_iters 100 --local_epochs 80 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 64 --alpha 0.005 --num_global_iters 100 --local_epochs 80 --numedges 32
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 32 --alpha 0.005 --num_global_iters 100 --local_epochs 80 --numedges 32
    </code></pre>
  - Human activies
    <pre><code>
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 64 --alpha 0.01 --num_global_iters 100 --local_epochs 80 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 256 --alpha 0.01 --num_global_iters 100 --local_epochs 80 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 128 --alpha 0.01 --num_global_iters 100 --local_epochs 80 --numedges 30
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 32 --alpha 0.01 --num_global_iters 100 --local_epochs 80 --numedges 30
    </code></pre>

### Effects of sub-sampling users
  - MNIST
    <pre><code>
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 13
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 20
      python3 main.py --dataset Mnist --model mclr --algorithm DONE --batch_size 0 --alpha 0.03 --num_global_iters 100 --local_epochs 40 --numedges 26
    </code></pre>

  - FEMNIST
    <pre><code>
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 13
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 20
      python3 main.py --dataset Nist --model mclr --algorithm DONE --batch_size 0 --alpha 0.01 --num_global_iters 100 --local_epochs 40 --numedges 26
    </code></pre>

  - Human activies
    <pre><code>
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 12
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 18
      python3 main.py --dataset human_activity --model mclr --algorithm DONE --batch_size 0 --alpha 0.02 --num_global_iters 100 --local_epochs 40 --numedges 24
    </code></pre>
