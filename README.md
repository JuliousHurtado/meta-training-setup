# Optimizing Reusable Knowledge for Continual Learning via Metalearning

This repositorie es the code of the paper [Optimizing Reusable Knowledge for Continual Learning via Metalearning](https://arxiv.org/pdf/2106.05390.pdf). 

## Paper

When learning tasks over time, artificial neural networks suffer from a problem known as Catastrophic Forgetting (CF). This happens when the weights of a network are overwritten during the training of a new task causing forgetting of old information. To address this issue, we propose MetA Reusable Knowledge or MARK, a new method that fosters weight reusability instead of overwriting when learning a new task. Specifically, MARK keeps a set of shared weights among tasks. We envision these shared weights as a common Knowledge Base (KB) that is not only used to learn new tasks, but also enriched with new knowledge as the model learns new tasks. Key components behind MARK are two-fold. On the one hand, a metalearning approach provides the key mechanism to incrementally enrich the KB with new knowledge and to foster weight reusability among tasks. On the other hand, a set of trainable masks provides the key mechanism to selectively choose from the KB relevant weights to solve each task.

### Main Idea

A schematic view of our proposal is described in the following image:

![MARK](/mark_architecture.png)

The flow of information in MARK is as follows. Input X<sub>i</sub> goes into F<sup>t</sup> to extract the representation F<sup>t</sup><sub>i</sub>. This representation is then used by M<sup>t</sup> to produce the set of masks that condition each of the blocks in the KB. The same input X<sub>i</sub> enters the mask-conditioned KB leading to vector F<sup>t</sup><sub>i,KB</sub> used by the classification head. Finally, classifier C<sup>t</sup> generates the model prediction, where *t* is the task ID associated to input X<sub>i</sub>.

The motivation behind this flow of information is that MARK learns to reuse information stored in the KB. By using M<sup>t</sup>, MARK weights information from the KB, delivering greater value to relevant information and ignoring irrelevant information.

To encourage the reuse of knowledge, we must find knowledge that can be relevant across tasks. MARK uses a metalearning strategy that improves the ability to generalize to future tasks

More details in the paper.

## Code

To run code, first you need to install the libraries listed in requirement.txt. Then, run the following command:

    python main.py --config ./configs_file.yml

where "./configs_file.yml" is the corresponding configuration file. For example, to run experiments in CIFAR100:

    python main.py --config ./configs/config_cifar100.yml

In the configuration files (.yml), we change the hyperparameters of the experiments. For example, the number of epochs, if using or not Meta-Learning or Mask Functions, using a pre-trained Resnet as F<sup>t</sup>, etc.

If you have any questions, do not hesitate to write // Si tienes alguna pregunta, no dudes en escribirme.

To cite our works:
```
@article{hurtado2021optimizing,
  title={Optimizing Reusable Knowledge for Continual Learning via Metalearning},
  author={Hurtado, Julio and Raymond-Saez, Alain and Soto, Alvaro},
  journal={arXiv preprint arXiv:2106.05390},
  year={2021}
}
```
