# Multimodal normative models for studying heterogeneous neurodegenerative diseases

This repository is a collection  of our work on multi-modal autoencoders as normative models applied to brain imaging data.

## Using multi-modal autoencoders as normative models

One of the challenges of studying common neurological disorders is disease heterogeneity including differences in causes, neuroimaging characteristics, comorbidities, or genetic variation. Normative modelling has become a popular method for studying such cohorts where the 'normal' behaviour of a physiological system is modelled and can be used at subject level to detect deviations relating to disease pathology. For many heterogeneous diseases, we expect to observe abnormalities across a range of neuroimaging and biological variables. However, thus far, normative models have largely been developed for studying a single imaging modality. We aim to develop a multi-modal normative modelling framework where abnormality is aggregated across variables of multiple modalities and is better able to detect deviations than uni-modal baselines. We propose two multi-modal VAE normative models to detect subject level deviations across T1 and DTI data. Our proposed models were better able to detect diseased individuals, capture disease severity, and correlate with patient cognition than baseline approaches. We also propose a multivariate latent deviation metric, measuring deviations from the joint latent space, which outperformed feature-based metrics.

For more information, please see our pre-print: 

Ana Lawry Aguila, James Chapman, & Andre Altmann. (2023). Multi-modal Variational Autoencoders for normative modelling across multiple imaging modalities. https://arxiv.org/abs/2303.12706

## Multi-modal normative modelling of Epilepsy

One of the challenges of studying Epilepsy is the vast heterogeneity of the disorder including differences in causes, seizure types, neuroimaging characteristics, comorbidities, and other individual factors to be considered in treatment selection. Previous work[1,2,3] typically compares individual syndromes to control cohorts. However, this approach assumes high within-syndrome homogeneity. Normative modelling has previously been applied to the study of other heterogeneous diseases[4,7]. Recently, deep-learning approaches, using autoencoder models, have been used to measure deviations in the data[5] and latent space[6].

Previous studies have found differences between epilepsy and control cohorts across several imaging modalities[1,2]. Multimodal machine learning can be used to capture interactions across modalities and provide a more complete picture of disease. In the normative modelling setting, we can explore how disease cohorts deviate from the multi-modal normative patterns. Here we used a multi-view Variational Autoencoder (VAE) as a multimodal normative model. We measured deviations in DTI and T1 MRI data from the Enhancing NeuroImaging Genetics through Meta-Analysis (ENIGMA) Epilepsy working group, which combines data from sites around the world into a large-scale study. We explored syndrome deviations in the data space and in the latent space. Results from this work will be presented in poster form at OHBM 2023. Relevant figures and code can be found in the `OHBM2023` folder of this repository. 

### References 

[1] C. Whelan, A. Altmann, J. Bot ́ıa, N. Jahanshad, D. Hibar, J. Absil, S. Alhusaini, M. Alvim, P. Auvinen, E. Bartolini, F. Bergo, T. Bernardes, K. Blackmon, B. Braga, M. Caligiuri, A. Calvo, S. Carr, J. Chen, S. Chen, S. Sisodiya, Structural brain abnormalities in the common epilepsies assessed in a worldwide enigma study, Brain 141 (2018).

[2] S. N. Hatton, K. H. Huynh, L. Bonilha, E. Abela, S. Alhusaini, A. Altmann, M. K. M. Alvim, A. R. Balachandra, E. Bartolini, B. Bender, N. Bernasconi, A. Bernasconi, B. Bernhardt, N. Bargallo, B. Caldairou, M. E. Caligiuri, S. J. A. Carr, G. L. Cavalleri, F. Cendes, L. Concha, E. Davoodi-bojd, P. M. Desmond, O. Devinsky, C. P. Doherty, M. Domin, J. S. Duncan, N. K. Focke, S. F. Foley, A. Gambardella, E. Gleichgerrcht, R. Guerrini, K. Hamandi, A. Ishikawa, S. S. Keller, P. V. Kochunov, R. Kotikalapudi, B. A. K. Kreilkamp, P. Kwan, A. Labate, S. Langner, M. Lenge, M. Liu, E. Lui, P. Martin, M. Mascalchi, J. C. V. Moreira, M. E. Morita-Sherman, T. J. O’Brien, H. R. Pardoe, J. C. Pariente, L. F. Ribeiro, M. P. Richardson, C. S. Rocha, R. Rodr ́ıguez-Cruces, F. Rosenow, M. Severino, B. Sinclair, H. Soltanian-Zadeh, P. Striano, P. N. Taylor, R. H. Thomas, D. Tortora, D. Velakoulis, A. Vezzani, L. Vivash, F. von Podewils, S. B. Vos, B. Weber, G. P. Winston, C. L. Yasuda, A. H. Zhu, P. M. Thompson, C. D. Whelan, N. Jahanshad, S. M. Sisodiya, C. R. McDonald, White matter abnormalities across different epilepsy syndromes in adults: an ENIGMA-Epilepsy study, Brain 143 (8) (2020) 2454–2473.

[3] S. Lariviere, J. Royer, R. Rodr ́ıguez-Cruces, C. Paquola, M. Caligiuri, A. Gambardella, L. Concha, S. Keller, F. Cendes, C. Yasuda, L. Bonilha, E. Gleichgerrcht, N. Focke, M. Domin, F. Podewills, S. Langner, C. Rummel, R. Wiest, P. Martin, B. Bernhardt, Structural network alterations in focal and generalized epilepsy assessed in a worldwide enigma study follow axes of epilepsy risk gene expression, Nature Communications 13 (2022).

[4] Rutherford, S., Kia, S.M., Wolfers, T. et al. The normative modeling framework for computational psychiatry. Nat Protoc 17, 1711–1734 (2022).

[5] W. Pinaya, C. Scarpazza, R. Garcia-Dias, S. Vieira, L. Baecker, P. Ferreira da Costa, A. Redolfi, G. Frisoni, M. Pievani, V. Calhoun, J. Sato, A. Mechelli, Using normative modelling to detect disease progression in mild cognitive impairment and alzheimer’s disease in a cross-sectional multi-cohort study, Scientific Reports 11 (08 2021).

[6] A. Lawry Aguila, J. Chapman, M. Janahi, A. Altmann, Conditional vaes for confound removal and normative modelling of neurodegenerative diseases, in: Medical Image Computing and Computer Assisted Intervention – MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part I, Springer-Verlag, 2022, p.430–440.

[7] Zabihi, M., Floris, D.L., Kia, S.M. et al. Fractionating autism based on neuroanatomical normative modeling. Transl Psychiatry 10, 384 (2020). 

