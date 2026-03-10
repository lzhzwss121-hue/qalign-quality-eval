{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Bold;\f2\froman\fcharset0 Times-Roman;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Systematic Evaluation of Zero-shot VLM-based Quality Scoring in Image Degradation and Restoration Settings\
\
This repository contains a research-oriented evaluation pipeline for comparing:\
\
- **Fidelity metrics**: PSNR, SSIM\
- **Perceptual metric**: LPIPS\
- **Zero-shot VLM-based quality scoring**: Q-Align\
\
The main focus of the repository is **metric analysis under image degradation settings**.  \
An additional **experimental extension** explores restoration-output evaluation with models such as SwinIR and MambaIRv2.\
\
## Main Pipeline\
\
The stable part of this project evaluates metric behavior under multiple degradation settings, including:\
\
- bicubic\
- blur\
- noise\
\
It includes:\
- raw metric computation\
- correlation analysis\
- case-study mining\
- qualitative board visualization\
\
## Main Findings\
\
- Q-Align behaves differently from traditional fidelity metrics such as PSNR.\
- In several degradation settings, Q-Align shows stronger alignment with perceptual similarity metrics than with pixel-level fidelity measures.\
- Case studies reveal representative disagreement patterns between fidelity-based and VLM-based quality scoring.\
\
## Experimental Restoration Extension\
\
The repository also includes an experimental restoration branch under `scripts/experimental/`.\
\
This part was used to explore restoration-output evaluation for:\
- SwinIR\
- MambaIRv2\
\
During this process, a benchmark mismatch case was identified when integrating MambaIRv2 outputs, highlighting the importance of:\
- GT/reference consistency\
- output scale alignment\
- crop-border protocol\
- benchmark evaluation settings\
\
## Repository Structure\
\
```text\
scripts/\
  run_eval.py\
  analyze_metrics.py\
  plot_results.py\
  make_case_study.py\
  make_case_board_compact.py\
  experimental/\
    generate_degradations.py\
    run_eval_restoration.py\
    run_eval_restoration_official.py\
\
results/\
  raw_metrics.csv\
  summary_stats.csv\
  case_study.csv\
  figures/\
  experimental/\
\
\
\
\pard\pardeftab720\sa298\partightenfactor0

\f1\b\fs36 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Notes\
\pard\pardeftab720\sa240\partightenfactor0

\f2\b0\fs24 \cf0 This repository focuses on 
\f1\b evaluation and analysis
\f2\b0 , rather than training new restoration models.\
The restoration branch is currently kept as an 
\f1\b experimental extension
\f2\b0 , since restoration benchmarking is highly sensitive to evaluation consistency.\
}