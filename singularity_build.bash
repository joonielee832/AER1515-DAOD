#! /bin/bash

sudo SINGULARITY_NOHTTPS=1 singularity build -F misc/aer1515-dgx.sif docker-daemon://aer1515-project-dgx:latest