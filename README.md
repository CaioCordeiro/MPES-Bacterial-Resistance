# MPES-Bacterial-Resistance

This repository contains the code and data for the article:

### "Exploring Feature Selections Algorithms and Datasets in Bacterial Resistance Prediction"
by Caio Cordeiro.

This article is the Master's thesis of Caio Cordeiro, a student of the Professional Master's Program in Computer Science at the CESAR School Univercity.

Status: in progress 📝

## Abstract

The resistance of microorganisms to antimicrobials is a global public health problem that has been the cause of the death of more than 33,000 deaths per year in the US (Centers for Disease Control and Prevention, 2020). Many governments around the world are beginning to be concerned about this serious issue that threatens the accomplishments of modern medicine (CHOWDHURY, 2019; MECESIC, 2020; NGUYEN, 2020; MORRISON, 2020). Antimicrobial-resistant (AMR) genes are developed by some bacteria and make them resistant to substances known to kill them, a great example is the bacteria Klebsiella Pneumonae that in recent years has been developing resistance to polymyxin, a strong “last-line” treatment antibiotic(European Centers for Disease Control and Prevention, 2021;  TACCONELLI, 2017). Computational approaches to classify AMR genes are becoming an extremely viable option and greatly rising given the effort necessary of most methods to identify the susceptibility of bacteria to some antibiotics and with the diverse databases of sequenced whole genomes appearing(CHOWDHURY, 2019; JORGENSEN, 2009; Centers for Disease Control and Prevention; 2021). As AMR genes databases have a high feature dimension it's important that we choose not only an algorithm that selects the best feature subset and that evaluates the importance of features together with other features, but also evaluate the performance of this algorithm in many datasets from different sources to validate its potential of being used more broadly(CHOWDHURY, 2019; JAIN, 1997; ROBNIK-ŠIKONJA, 1997; SUN, 2012).

## References

CORDEIRO, C. Evaluating feature selection algorithms to apply in antimicrobial-resistant genes classification in Gram-negative bacteria. CESAR School, 2021.

Centers for Disease Control and Prevention. About Antibiotic Resistance. Available in: https://www.cdc.gov/drugresistance/about.html, 2020. Access in: 24/07/2021.

CHOWDHURY, A. S., DOUGLAS, R. C., SHIRA L. B. Antimicrobial Resistance Prediction for Gram-Negative Bacteria via Game Theory-Based Feature Evaluation. Scientific Reports, 2019.

MECESIC, Nenad Predicting Phenotypic Polymyxin Resistance in Klebsiella. mSystems, 2020.

NGUYEN, Marcus Developing an in silico minimum inhibitory concentration panel test for Klebsiella pneumonia. Scientific Reports, 2020.
MORRISON, L., & ZEMBOWER, T. R. Antimicrobial Resistance. Gastrointestinal Endoscopy Clinics of North America, 2020.
SHORR, A. F. Review of studies of the impact on Gram-negative bacterial resistance on outcomes in the intensive care unit. Critical Care Medicine, 2009.

INFOGRAPHIC: Antibiotic resistance - an increasing threat to human health. European Centers for Disease Control and Prevention. Available in: https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance. Access in: 24/07/2021.

TACCONELLI, E. Global Priority List of Antibiotic-Resistant Bacteria to Guide Research, Discovery, and Development of New Antibiotics. WHO Publications, 2017.

JORGENSEN, J. H., & FERRARO, M. J. Antimicrobial susceptibility testing: A review of general principles and contemporary practices. Clinical Infectious Diseases, 2009.

BIGGEST Threats and Data. Centers for Disease Control and Prevention. Available in: https://www.cdc.gov/drugresistance/biggest-threats.html. Access in:: 24/07/2021.

## Project Requirements

1. Python > 3.9
2. sra-toolkit
3. Jellyfish
4. fastq_to_fasta

### sra-toolkit

Sra toolkit will hep us fetch data directly from the NCBI database.
To instal this requirements we must follow the [sra-toolkit documentation](https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit).
