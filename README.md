![arxiv-efficiency](img/arxiv-efficiency-banner.png)

# Daily arXiv papers about efficiency

I'm interested in making deep learning more efficient.
I cover anything relevant to Transformers, LLMs, NLP, Optimization, etc.

arXiv categories covered:
[cs.AI](https://arxiv.org/list/cs.AI/recent)
[cs.CL](https://arxiv.org/list/cs.CL/recent)
[cs.ET](https://arxiv.org/list/cs.ET/recent)
[cs.LG](https://arxiv.org/list/cs.LG/recent)
[cs.NE](https://arxiv.org/list/cs.NE/recent)
[stat.ML](https://arxiv.org/list/stat.ML/recent)

🔥 New: I've added ChatGPT generated summaries. Click on them to reveal.
Feel free to submit PR if you can summarize them better.

Disclaimer: We're not affiliated with arXiv.


## June 17, 2023

No new papers on arXiv

## June 16, 2023

[Neural Network Compression using Binarization and Few Full-Precision Weights](https://arxiv.org/pdf/2306.08960.pdf) <details><summary>Summary</summary><p>
* The paper introduces a novel compression technique called Automatic Prune Binarization (APB) that combines quantization with pruning to enhance the representational capability of binary networks.
* The authors propose efficient algorithms for quantized matrix multiplication on CPU, which outperform existing state-of-the-art solutions.
* APB demonstrates better accuracy/memory trade-off compared to other model compression methods and is faster than 2-bits quantized models.
</p></details>

[When and Why Momentum Accelerates SGD:An Empirical Study](https://arxiv.org/pdf/2306.09000.pdf) <details><summary>Summary</summary><p>
* This paper investigates the impact of momentum in stochastic gradient descent (SGD) and provides insights into when and why momentum acceleration occurs.
* The study establishes a comparison framework to analyze the performance of SGD with Momentum (SGDM) under different learning rates and batch sizes. It reveals that SGDM outperforms SGD when the effective learning rate surpasses a certain threshold, particularly with larger batch sizes.
* The paper also explores the relationship between momentum acceleration and abrupt sharpening, highlighting the importance of momentum in accelerating training and preventing the entrance of EoS (End of Sequence) states.
</p></details>

[Stochastic Re-weighted Gradient Descent via Distributionally Robust Optimization](https://arxiv.org/pdf/2306.09222.pdf) <details><summary>Summary</summary><p>
* The paper introduces a technique called Stochastic Re-Weighted Gradient Descent via Distributionally Robust Optimization, which aims to improve the performance of deep neural networks.
* The approach involves importance weighting of data points during optimization, inspired by distributionally robust optimization with f-divergences. This re-weighting scheme is simple, efficient, and compatible with popular optimization algorithms.
* Empirical results demonstrate the effectiveness of the proposed approach across various tasks, including classification, label imbalance, noisy labels, domain adaptation, and representation learning. Significant improvements are observed on benchmarks such as DomainBed, Tabular, GLUE, and ImageNet-1K.
</p></details>

[PaReprop: Fast Parallelized Reversible Backpropagation](https://arxiv.org/pdf/2306.09342.pdf) <details><summary>Summary</summary><p>
* Reversible Transformations: The paper introduces the concept of reversible transformations, which map inputs to outputs in an invertible process. These transformations utilize intermediate functions F and G, allowing for the recovery of inputs from the outputs. This forms the basis for the memory-efficient reversible training algorithm.
* PaReprop Algorithm: The key contribution of the paper is the PaReprop algorithm, which parallelizes activation re-computation in reversible backpropagation. By performing gradient updates and activation re-computation simultaneously, the algorithm significantly reduces the computation time, making it comparable to vanilla backpropagation. This parallelization technique enables faster training throughput.
* Benchmarking and Extensions: The paper extensively benchmarks the PaReprop algorithm across various model families, data modalities, model sizes, and training batch sizes. The results demonstrate the effectiveness of the algorithm in improving training efficiency. Additionally, the paper proposes extensions to existing reversible architectures, such as Rev-Swin and Rev-RoBERTa models, further enhancing the throughput improvement.
</p></details>

[Sampling-Based Techniques for Training Deep Neural Networks with Limited Computational Resources: A Scalability Evaluation](https://arxiv.org/pdf/2306.09293.pdf) <details><summary>Summary</summary><p>
* This paper focuses on the scalability evaluation of sampling-based approaches for training deep neural networks with limited computation and memory resources. It provides extensive experiments and theoretical analysis to assess the performance and limitations of these approaches.
* The paper presents a taxonomy of sampling-based approaches and discusses two specific methods in detail. It highlights the correlation between the number of hidden layers and approximation error in DNNs under hashing-based methods, providing valuable insights into the trade-offs and considerations in training deep neural networks.
* The experimental results confirm the theoretical analysis and reveal the challenges of feedforward approximation in achieving scalability. The paper identifies areas for further research and emphasizes the need for designing scalable sampling-based approaches for stochastic gradient descent. Overall, it offers valuable insights into the performance and potential of these methods in different settings.
</p></details>

[MinMax Networks](https://arxiv.org/pdf/2306.09253.pdf) <details><summary>Summary</summary><p>
* The paper introduces a discrete MinMax learning algorithm for continuous piece-wise linear functions. This algorithm utilizes combinations of convex and concave neurons to approximate the measurement function. It addresses convergence difficulties of gradient descent on a quadratic error cost, such as saddle points, sub-optimal plateaus, and non-Lipschitz edges.
* The proposed algorithm is based on Contraction Theory with Inequality Constraints. It extends the Contraction Analysis of Continuous Constrained Systems to the discrete case. The paper provides theoretical guarantees of global exponential convergence and stability for the algorithm. It also incorporates time-varying measurements and the time discretization of the gradient into the stability proof.
* The paper emphasizes the practical implications of step size selection in gradient descent. It highlights the challenges of selecting an appropriate step size and avoiding instabilities caused by non-Lipschitz edges. The proposed approach mitigates these issues by introducing intermediate Lagrange constraints and utilizing a linear parametrization.
</p></details>

[Implicit Compressibility of Overparametrized Neural Networks Trained with Heavy-Tailed SGD](https://arxiv.org/pdf/2306.08125.pdf) <details><summary>Summary</summary><p>
* Theoretical Framework: The paper presents a theoretical framework that explores the relationship between compressibility and generalization error in neural network compression. It proposes a modification for SGD that ensures provable compressibility without relying on unverifiable assumptions.
* Empirical Validation: The study validates the proposed theory through empirical results. It investigates the effects of injecting heavy-tailed noise in SGD on the compressibility and train/test performance of a single-hidden-layer neural network with ReLU activations. The Electrocardiogram (ECG), MNIST, and CIFAR10 datasets are used for experimentation.
* Practical Implications: The findings of this research have practical implications for reducing computational requirements in neural network compression. The study discusses the implications of compressibility studies on federated learning and highlights the potential benefits of compressing neural networks in terms of computational efficiency.
</p></details>

[INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation](https://arxiv.org/pdf/2306.08162.pdf) <details><summary>Summary</summary><p>
* The paper introduces a novel method for fine-tuning and error correction in quantized Large Language Models, significantly reducing VRAM requirements and rectifying quantization errors.
* The method utilizes a combination of Low-Rank Adaptation (LoRA) and a hybrid loss function to achieve memory-efficient fine-tuning and improve the quality of responses generated by the model.
* The research demonstrates the potential for efficient fine-tuning strategies, explores the trade-off between quantization levels and model performance, and suggests future directions for scalability and applicability across different domains.
</p></details>

[Contrastive Loss is All You Need to Recover Analogies as Parallel Lines](https://arxiv.org/pdf/2306.08221.pdf) <details><summary>Summary</summary><p>
* The paper explores the use of contrastive-style optimization in word embeddings to recover analogies as parallel lines. It demonstrates that optimizing a contrastive-style objective over word co-occurrences is sufficient to encode analogies as parallel lines, shedding light on the inner workings of word embeddings.
* The research builds upon previous literature and generalizes the understanding of the underlying mechanisms governing the geometry of word embeddings. It highlights that parallel geometry is induced largely from word co-occurrence statistics for any push-pull model.
* The paper also discusses the potential of alternative mechanisms for recovering analogies as parallel lines and suggests further investigation into their ability to achieve similar results. Additionally, it showcases the performance of the proposed approach on analogy-based benchmarks and highlights the significant reduction in training time compared to popular word embedding models.
</p></details>

[Accelerated Convergence of Nesterov's Momentum for Deep Neural Networks under Partial Strong Convexity](https://arxiv.org/pdf/2306.08109.pdf) <details><summary>Summary</summary><p>
* This paper focuses on the theoretical understanding of optimization and machine learning methods, specifically exploring the accelerated convergence of Nesterov's momentum for deep neural networks under partial strong convexity.
* The authors consider the complexity of modern neural networks and raise questions about whether more complicated partition schemes, such as selecting a subset of weights in each layer, will still satisfy the assumptions in their framework.
* The paper provides examples that satisfy the given assumptions and demonstrate the accelerated convergence of Nesterov's momentum, even in nonconvex and possibly non-smooth models. The authors also discuss the potential impact of their work on society, which depends on the application of machine learning models.
</p></details>

[When to Use Efficient Self Attention? Profiling Text, Speech and Image Transformer Variants](https://arxiv.org/pdf/2306.08667.pdf) <details><summary>Summary</summary><p>
* This paper presents a comprehensive study on the efficiency of self-attention-based Transformer variants in text, speech, and vision domains.
* The study explores input length thresholds where efficient Transformer models outperform vanilla models, considering various efficiency metrics.
* The paper emphasizes the importance of selecting the appropriate model based on modality, task type, and resource constraints, highlighting the need for tailored models in different scenarios.
</p></details>

[MetaML: Automating Customizable Cross-Stage Design-Flow for Deep Learning Acceleration](https://arxiv.org/pdf/2306.08746.pdf) <details><summary>Summary</summary><p>
* The paper introduces a novel co-optimization framework for FPGA-based DNN accelerators, which automates the selection and configuration of low-level optimization techniques. This framework enables the development of customized design flows, resulting in significant reductions in DSP resource usage (up to 92%) and LUT usage (up to 89%) while maintaining accuracy.
* The authors propose a library of reusable optimization and transformation tasks that can be easily integrated into the co-optimization framework. These tasks are designed to be customizable and flexible, providing versatility and adaptability to the framework. Some tasks are specific to certain applications and target technologies, while others are agnostic.
* The effectiveness of the proposed framework and its optimization modules are evaluated using multiple benchmarks and different optimization strategies. The evaluation provides insights into the framework's performance under different scenarios, highlighting its benefits and potential for further improvements.
</p></details>

[Noise Stability Optimization for Flat Minima with Optimal Convergence Rates](https://arxiv.org/pdf/2306.08553.pdf) <details><summary>Summary</summary><p>
* This paper introduces an algorithm for noise stability optimization that leverages random noise injection and the symmetry of a given distribution to find approximate first-order stationary points of a weight-perturbed function.
* The algorithm's performance is rigorously analyzed, providing matching upper and lower bounds. It is shown to effectively find flat, local minimizers with optimal convergence rates.
* Empirical experiments on image classification tasks validate the algorithm's effectiveness, demonstrating its ability to improve convergence and achieve better results compared to traditional optimization methods.
</p></details>

## June 15, 2023

No new papers on arXiv

## June 14, 2023

[SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/pdf/2306.07629v1.pdf) <details><summary>Summary</summary><p>
* This paper introduces SqueezeLLM, a post-training quantization framework for Large Language Models (LLMs) that addresses the memory bottleneck in generative inference.
* SqueezeLLM achieves lossless compression and improved quantization performance for LLMs by incorporating a sensitivity-based non-uniform quantization technique and a Dense-and-Sparse decomposition method.
* The framework significantly reduces model sizes and inference time costs, making it a promising solution for generative inference with LLMs.
</p></details>

[Exact Mean Square Linear Stability Analysis for SGD](https://arxiv.org/pdf/2306.07850v1.pdf) <details><summary>Summary</summary><p>
* This paper focuses on the mean square stability analysis of stochastic gradient descent (SGD) in neural network training.
* The authors derive a closed-form expression for the stability threshold of SGD and show that it is closely related to the stability threshold of gradient descent (GD).
* The analysis reveals that reducing the batch size in SGD can negatively impact stability, indicating that larger batch sizes are generally more stable.
</p></details>

[One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning](https://arxiv.org/pdf/2306.07967v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces GLoRA, a parameter-efficient fine-tuning framework that surpasses previous state-of-the-art methods in terms of average accuracy and performance on various benchmarks.
* GLoRA enhances the low-rank adaptation approach by incorporating a more generalized prompt module design per layer, offering enhanced capability and flexibility in fine-tuning.
* GLoRA eliminates the need for retraining the subnet and manual hyperparameter tuning, resulting in superior parameter efficiency and no additional inference cost. It can be effectively deployed in computer vision and natural language processing domains.
</p></details>

## June 13, 2023

[Hidden symmetries of ReLU networks](https://arxiv.org/pdf/2306.06179v1.pdf) <details><summary>Summary</summary><p>
* The paper explores the concept of hidden symmetries in ReLU networks and their impact on function representation. It discusses the degree of redundancy in parameter settings and how it varies across different network architectures.
* The paper examines the mechanisms through which hidden symmetries can arise in network architectures. It highlights the role of permutation invariance and explores the linear mode connectivity of neural networks.
* The paper investigates the probability of a network having no hidden symmetries, and how it changes with increasing depth, width, and input dimension. It provides insights into the relationship between network architecture and the presence of hidden symmetries.
</p></details>

[$`FPDM`$: Domain-Specific Fast Pre-training Technique using Document-Level Metadata](https://arxiv.org/pdf/2306.06190v1.pdf) <details><summary>Summary</summary><p>
* This paper introduces FPDM, a fast pre-training technique that leverages document-level metadata and domain-specific taxonomy to pre-train transformer encoders on domain-specific corpora. By incorporating sentence-level embeddings during pre-training and token-level embeddings during fine-tuning, FPDM achieves superior performance compared to transformer-based baselines.
* The authors highlight the potential for further exploration and improvement of the FPDM model. They suggest extending the contrastive loss beyond triplet loss to learn graded similarity between documents. Additionally, they propose the use of hierarchical topic modeling for creating taxonomies in domains that lack them.
* While the paper does not provide specific results using large transformer encoders like BERT LARGE and RoBERTa LARGE due to GPU resource constraints, the authors believe that the fundamental results and message of the paper would remain unchanged. They also emphasize the need for careful consideration of exposure bias and limited interpretability associated with the output of pre-trained language models like FPDM.
</p></details>

[Understanding the Effect of the Long Tail on Neural Network Compression](https://arxiv.org/pdf/2306.06238v1.pdf) <details><summary>Summary</summary><p>
* This paper explores the field of neural network compression and its impact on maintaining semantic equivalence with the original network while achieving good generalization. It highlights the importance of considering factors beyond overall accuracy when compressing neural networks.
* The authors discuss the long tail phenomenon in computer vision datasets and its relationship to network capacity and memorization. They argue that understanding the mismatches that occur during compression is crucial for developing effective compression techniques.
* The paper presents evidence and insights that contribute to a systematic understanding of mismatches in compressed models. It discusses the use of multi-part loss functions and knowledge distillation techniques to improve alignment between the original and compressed models, leading to better fairness across classes and similarity of attribution maps. However, it acknowledges that some mismatches may be unavoidable and inherent to underparameterization.
</p></details>

[Improving Non-autoregressive Translation Quality with Pretrained Language Model, Embedding Distillation and Upsampling Strategy for CTC](https://arxiv.org/pdf/2306.06345v1.pdf) <details><summary>Summary</summary><p>
* This paper introduces innovative techniques to improve the translation quality of Non-Autoregressive Translation (NAT) models while maintaining fast inference speed.
* The proposed methods include fine-tuning Pretrained Multilingual Language Models (PMLMs), using a MASK insertion scheme for up-sampling, and employing embedding distillation.
* The experiments conducted show that these techniques outperform baseline autoregressive models and achieve state-of-the-art performance on multiple datasets.
</p></details>

[Can Forward Gradient Match Backpropagation?](https://arxiv.org/pdf/2306.06968v1.pdf) <details><summary>Summary</summary><p>
* The paper explores the concept of Forward Gradients in neural network training, specifically focusing on computer vision neural networks.
* The study reveals that using gradients obtained from a local loss as a candidate direction improves the accuracy of gradient guesses.
* The paper proposes a novel approach to strongly bias gradient guesses in more promising directions, highlighting the potential of Forward Gradients in improving gradient estimation methods.
</p></details>

[Unveiling the Hessian's Connection to the Decision Boundary](https://arxiv.org/pdf/2306.07104v1.pdf) <details><summary>Summary</summary><p>
* The generalization of neural networks is connected to the complexity of the decision boundary, which is hard to study in high-dimensional input space.
* The Hessian's top eigenvectors can be used to characterize the decision boundary learned by a neural network and identify minima with simple wide-margin boundaries.
* This connection between the Hessian and decision boundary inspires a new generalization measure and margin estimation technique for identifying well-generalizing minima in deep learning models.
</p></details>

[Benchmarking Neural Network Training Algorithms](https://arxiv.org/pdf/2306.07179v1.pdf) <details><summary>Summary</summary><p>
* The paper evaluates and compares the performance of various neural network training algorithms.
* A standardized benchmarking framework is used to conduct the evaluation and comparison.
* The study provides insights into the strengths and weaknesses of different algorithms, which can inform the development of more efficient and effective neural network training methods.
</p></details>

## June 12, 2023

[Boosting with Tempered Exponential Measures](https://arxiv.org/pdf/2306.05487v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces a new algorithm called t-AdaBoost that extends the popular machine learning algorithm AdaBoost.
* t-AdaBoost uses a new type of probability distribution called tempered exponential measures, which are indexed by a temperature parameter and have improved convergence rates compared to AdaBoost.
* The authors show how to derive a new family of tempered losses for the induction of domain-partitioning classifiers like decision trees using t-AdaBoost, which ensures strict properness for all while their boosting rates span the full known spectrum.
</p></details>

[Reevaluating Loss Functions: Enhancing Robustness to Label Noise in Deep Learning Models](https://arxiv.org/pdf/2306.05497v1.pdf) <details><summary>Summary</summary><p>
* The paper focuses on the challenge of label noise in deep learning models and proposes the use of bounded loss functions to enhance performance even on seemingly clean benchmark datasets.
* The authors conduct a comparative analysis of different loss functions and provide insights into their relative strengths and weaknesses, enabling researchers and practitioners to select the most suitable loss function for a given dataset and task.
* The paper introduces a novel technique of including an output bias, which enhances learning with bounded loss functions and improves generalization performance, particularly in the presence of label noise.
</p></details>

[Asymptotically efficient one-step stochastic gradient descent](https://arxiv.org/pdf/2306.05896v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces a novel method called the one-step procedure, which offers a fast and asymptotically efficient alternative to traditional averaging or adaptivity methods for parametric estimation.
* The one-step procedure involves an initial guess estimator followed by a single step of the gradient descent method on the log-likelihood function to correct the initial estimation and achieve asymptotic efficiency.
* The method has been successfully applied and generalized to various statistical experiments, including diffusion processes, ergodic Markov chains, and inhomogeneous Poisson and Hawkes counting processes, demonstrating its versatility and effectiveness in different domains.
</p></details>

[How Sparse Can We Prune A Deep Network: A Geometric Viewpoint](https://arxiv.org/pdf/2306.05857v1.pdf) <details><summary>Summary</summary><p>
* This paper explores the maximum pruning ratio of deep networks from a high-dimensional geometry perspective, using the fundamental pruning objective of minimizing an l1-regularized loss. It characterizes the sharp phase point of network pruning and presents a novel network pruning algorithm that is characterized by a global one-shot pruning approach.
* The paper introduces the concept of Gaussian width and leverages powerful tools and theorems in high-dimensional geometry, such as the Gordon's Escape theorem, to precisely characterize the phase transition point of network pruning. It also highlights the influence of loss function flatness and weight magnitude on the maximum pruning ratio of the network.
* The paper provides experimental evidence to validate the theoretical results and demonstrates the high performance of the proposed pruning algorithm. It also discovers that networks with smoother loss landscapes and smaller weights have stronger pruning capability, which can offer insights for understanding and comparing various network pruning techniques.
</p></details>

[End-to-End Neural Network Compression via $`\frac{\ell_1}{\ell_2}`$ Regularized Latency Surrogates](https://arxiv.org/pdf/2306.05785v1.pdf) <details><summary>Summary</summary><p>
* This paper introduces a novel technique for compressing neural networks using $`\frac{\ell_1}{\ell_2}`$ regularized latency surrogates. The approach optimizes for both FLOPs and on-device latency, providing a versatile solution for various compression methods.
* The proposed algorithm offers significant training speed-up compared to standard methods, making it fast and efficient. It can be applied to popular compression techniques such as pruning, low-rank factorization, and quantization.
* The technique has achieved impressive results in reducing FLOPs and on-device latency without sacrificing performance or accuracy. It has been successfully applied to compress BERT on GLUE fine-tuning tasks and MobileNetV3 on ImageNet-1K, demonstrating its effectiveness in real-world scenarios.
</p></details>

[Error Feedback Can Accurately Compress Preconditioners](https://arxiv.org/pdf/2306.06098v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces an error-feedback technique that effectively compresses preconditioners for deep learning models. This technique leverages sparsification or low-rank compression to reduce the memory requirements of full-matrix preconditioning without compromising convergence or accuracy.
* Existing approaches for full-matrix preconditioning often come with high storage costs. However, the error-feedback technique presented in this paper offers a more efficient and simple-to-implement solution, effectively removing the memory overhead associated with full-matrix preconditioning.
* Extensive experiments have been conducted on deep neural networks for vision to validate the effectiveness of the error-feedback technique. These experiments demonstrate that the proposed approach successfully reduces the memory requirements of preconditioning without sacrificing the convergence or accuracy of the deep learning models.
</p></details>

[Prodigy: An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/pdf/2306.06101v1.pdf) <details><summary>Summary</summary><p>
* Prodigy and Resetting techniques: The authors propose Prodigy, which estimates the distance to the solution in adaptive methods using a novel approach. They compare Prodigy with the D-Adaptation method and show that Prodigy and Resetting techniques consistently improve convergence.
* Advantages over D-Adaptation: Prodigy and Resetting techniques offer several advantages over the D-Adaptation method. They eliminate the need for manual tuning of hyperparameters, provide faster convergence, and achieve better performance in terms of test accuracy.
* Experimental results: The paper presents experimental results comparing Prodigy with hand-tuned Adam. Prodigy outperforms hand-tuned Adam in terms of test accuracy on various datasets, demonstrating the effectiveness of the proposed approach.
</p></details>

[S$`^{3}`$: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/pdf/2306.06000v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces S$`^{3}`$, a framework designed to improve throughput in serving Transformer-based generative models. S$`^{3}`$ leverages a predictor to estimate the output length of generated sequences and schedules them accordingly, maximizing GPU utilization and increasing throughput.
* S$`^{3}`$ addresses the challenge of memory consumption in generating texts with large language models. By allocating varying memory sizes to different inputs, S$`^{3}`$ acknowledges that not all sequences should be treated equally, expanding the conventional trade-off between latency and throughput.
* The evaluation of S$`^{3}`$ shows promising results. In online scenarios, S$`^{3}`$ can generate up to 6.49 times more sequences while adhering to the same latency service level objective (SLO) constraint. In offline scenarios, S$`^{3}`$ achieves a speedup of up to 6.49 times for different models, demonstrating its effectiveness in improving throughput and cost-efficiency.
</p></details>

## June 11, 2023

No new papers on arXiv

## June 10, 2023

No new papers on arXiv

## June 9, 2023

[Catapults in SGD: spikes in the training loss and their impact on generalization through feature learning](https://arxiv.org/pdf/2306.04815v1.pdf) <details><summary>Summary</summary><p>
* Catapults can lead to better generalization: Empirical results show that a single catapult or multiple catapults can improve test performance in gradient descent for wide neural networks. This suggests that catapults have a positive impact on generalization.
* Catapults increase alignment with the true AGOP: The paper demonstrates that the improved test performance associated with catapults is due to the alignment between the trained network's Average Gradient Outer Product (AGOP) and the true AGOP. This alignment is a measure of feature learning.
* Experimental settings: The experiments in the paper involve four datasets, including two synthetic datasets and two real-world datasets. The details of these experiments can be found in Appendix E of the paper.
</p></details>

[Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts](https://arxiv.org/pdf/2306.04845v1.pdf) <details><summary>Summary</summary><p>
* Generalized Weight Sharing: The paper proposes a formulation that generalizes weight sharing methods, including direct weight sharing and flexible weight sharing. This formulation improves the supernet's expressive power, allowing for more efficient NAS.
* Mixture-of-Experts (MoE): The paper adopts the idea of MoE to improve the model's capability. The weights of the model are dynamically generated based on the activated subnetwork architecture. After training, the MoE can be converted into equivalent static models, reducing retraining time and improving training efficiency.
* SOTA NAS Results: The paper presents comprehensive experiments demonstrating that the proposed supernets achieve state-of-the-art results in building efficient task-agnostic BERT and MT models. Additionally, the supernets reduce retraining time and greatly improve training efficiency.
</p></details>

[Layer-level activation mechanism](https://arxiv.org/pdf/2306.04940v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces a novel activation mechanism called LayerAct, which aims to improve the noise-robustness of neural networks by reducing fluctuations in activation outputs at the layer level.
* LayerAct functions exhibit a zero-like mean activation without restricting the activation output space, which leads to more efficient training.
* Experimental results demonstrate the superiority of LayerAct functions over traditional element-level activation functions in both noisy and clean datasets, making them a promising approach for handling noisy image datasets in image classification tasks.
</p></details>

[Robust Learning with Progressive Data Expansion Against Spurious Correlation](https://arxiv.org/pdf/2306.04949v1.pdf) <details><summary>Summary</summary><p>
* The paper introduces the concept of spurious correlations in deep learning models, where the model learns non-generalizable features that can lead to inaccurate predictions.
* The proposed algorithm, PDE (Progressive Data Expansion), aims to enhance the robustness of deep learning models by preventing the learning of spurious features. It achieves this through a rapid warm-up stage and continuous improvement during the expansion stage.
* Experimental results from both synthetic and real datasets demonstrate the effectiveness of the PDE algorithm in improving model robustness. The worst-group accuracy metric is used to evaluate the model's performance against spurious correlations, and the results show significant improvements compared to existing approaches.
</p></details>

[Mixed-TD: Efficient Neural Network Accelerator with Layer-Specific Tensor Decomposition](https://arxiv.org/pdf/2306.05021v1.pdf) <details><summary>Summary</summary><p>
* Efficient Design Flow: The paper presents a design flow that includes a search stage and a deployment stage. In the search stage, the accuracy and throughput of each design point are queried to identify the optimal design. The optimal design is then fine-tuned to improve its accuracy before being synthesized and deployed on the FPGA device.
* Large Design Space: Due to the fine-grained and layer-specific decisions made by the Mixed-TD method, the design space defined by the tensor decomposition is extremely large. For example, ResNet-18 alone consists of a staggering number of possible candidate designs.
* Impressive Performance: The Mixed-TD method achieves high throughput per DSP compared to existing work. In terms of Throughput per DSP, the method achieves gains ranging from 1.73× to 10.29× compared to other approaches.
</p></details>

[Magnitude Attention-based Dynamic Pruning](https://arxiv.org/pdf/2306.05056v1.pdf) <details><summary>Summary</summary><p>
* Novel Pruning Method: The paper introduces a novel magnitude attention-based pruning method that dynamically explores sparse model structures. This method considers the importance of weights throughout the forward and backward paths, resulting in highly effective pruning compared to other static pruning methods.
* Attention Mechanism: The attention mechanism plays a crucial role in this approach. It helps identify the most significant weights and layers that have the greatest impact on performance. By focusing on these important weights, the sparse model can be optimized while maintaining the well-established structure of the discovered sparse networks.
* Performance and Comparison: The pruned models achieved performance comparable to dense models and outperformed previous pruning methods on CIFAR-10/100 and ImageNet datasets. The effectiveness of the magnitude attention-based pruning method was demonstrated through comparisons with competitive state-of-the-art pruning methods, highlighting its superiority in terms of performance and effectiveness.
</p></details>

[Correlated Noise in Epoch-Based Stochastic Gradient Descent: Implications for Weight Variances](https://arxiv.org/pdf/2306.05300v1.pdf) <details><summary>Summary</summary><p>
* The authors challenge the assumption of uncorrelated noise in epoch-based stochastic gradient descent (SGD) and investigate its impact on weight variances.
* The authors calculate the exact autocorrelation of the noise for training in epochs and investigate the influence of correlations introduced by the epoch-based learning scheme on SGD dynamics.
* The paper primarily focuses on theoretical derivations and provides insights into the stationary distribution of discrete-time SGD with momentum, limited to a quadratic loss.
</p></details>
