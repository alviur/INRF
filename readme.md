# Evidence for the intrinsically nonlinear nature of receptive fields in vision

This is the repository for the artificial neural networks experiments described in "Evidence for the intrinsically 
nonlinear nature of receptive fields in vision" paper (see citation).


# Dependencies
- Python 3.5.2
- Numpy 1.16.2
- Pytorch 1.0.0
 - [Adversarial-robustness-toolbox 1.2.0](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

# Instructions 
1_INRFnets_classification: Contains a script for the accuracy on each reported dataset . A CNN version of each architecture is included as a commented-out section.
 
2_INRFnets_adversarial: Contains all the experiments on adversarial  attacks. Contains scripts to run a series of adversarial attacks over the CNN and INRF models trained in MNIST and CIFAR-10. 

# Citation
If you use these models in your research, please cite:

    @article{bertalmio2020inrf,
      title={Evidence for the intrinsically nonlinear nature of receptive fields in vision},
      author={Bertalm{\'i}o, Marcelo and Gomez-Villa, Alex and Mart{\'i}n, Adrian and Vazquez-Corral, Javier and Kane, David and Malo, Jes{\'u}s},
      booktitle={Scientific Reports},
      pages={-},
      year={2020}
    }

# License


INRFnet is a copyrighted work of Universitat Pompeu Fabra, (the "Software"). 

The INRF, is the methodology underlying the INRFnet and has been the object of a Patent Application filed before the European Patent Office, with number  P19750EP00, (the "Method").  

All rights over the Method and the Software are reserved to Universitat Pompeu Fabra. Therefore it is not possible to execute, reproduce, produce, transform, communicate, sell, or use them in any form or mean, except for those limitations established by Law. The use for non-commercial academic research is authorized, with the obligation to mention the ownership, and its authors and inventors. Other uses are expressly forbidden.

If you are interested in obtaining a license, please contact Marcelo Bertalmío at marcelo.bertalmio@upf.edu.
