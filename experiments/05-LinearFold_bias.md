# Experiment 05: LinearFold bias model
 #### **Code version:** initial experiments on bias model variations barplot(fa09a55f5aa1a5d89f1a0ac21cfd2f0eb0be292f)

## Results and Next Steps

FILL



## Objective 

We will now implement a version of the biased model that uses LinearFold predictions to compute the bias matrix instead of a simple Watson-Crick pairing. As usual, an initial investigation on one fold will provide insights before CV.


## Status
**IN PROGRESS** 
- **job names**: FILL

## Expected outcomes
- _Deliverables_: 
- _output directory_: FILL
- _decisions to take_: FILL


## Resources required

1 GPU.

## Duration
21.06.2026

## Experiment description

The structure of the biased model does not have to change as it supports any bias matrix given by the batch collator. However, the collator now has more work to do. LinearFold predictions should probably be pre-computed and saved in a file. The collator should read the files for corresponding sequences and create the bias matrix. 


### example scripts

all present in `jobs/cv_biased_model/`.


```bash



```

## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

