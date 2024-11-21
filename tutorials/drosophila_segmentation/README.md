# Training a UNet to infer introgression

In this tutorial we'll replicate some results given in the paper [IntroUNET: Identifying introgressed alleles via semantic segmentation.] (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1010657).  Make sure to have the pre-requisites installed per the README at the top of this repo.

## Simulating the data

We include some parameters found using DADI:

```
head params/dros.txt
-1542.930648526035      3099.86129705207        327155.49387802556      74735.1004313855        3271554.9387802505      5940211.089003239       15490.092609766634      53133.59924396219       0.010375771827919596    0.14121142707782997
-1416.6879807220432     2847.3759614440864      285850.2602648295       72932.45768301723       2858498.432070652       5553967.987726062       17209.329524868634      51135.82704681309       0.013575969298897714    0.061045598426534536
-1271.882532227674      2557.765064455348       293665.5422031877       78218.80308357006       2936105.346408323       5786630.039753016       17457.066991075993      54040.82125636821       0.01382086221817424     0.05001666725919176
-1283.911398085168      2581.822796170336       304196.9460510423       78077.02744219886       3041967.659656098       5462072.145224885       17943.111452758116      53641.53863811744       0.014260405767451962    0.06483704946048502
...
```

And a script to generate the data at:

```
src/simulations/simulate_msmodified.py
```

This script is only meant for simulating this particular dataset.  It simulates 20 individuals in _D. simulans_ and 14 individuals in _D. melanogaster_.  More information on the demographic model can be found in our manuscript.

We can generate 100 replicates with introgression from population A (_D. simulans_) to population B (_D. melanogaster_) for each parameter set returned by DADI by running:

```
python3 src/simulations/simulate_msmodified.py --direction ab --odir data/dros_ab --n_jobs 1 --n_samples 100
```

If you have access to a SLURM cluster you can submit many many such jobs using:

```
python3 src/simulations/simulate_msmodified.py --direction ab --odir data/dros_ab --n_jobs 10 --n_samples 100 --slurm
```

This will submit 10 jobs to your cluster, each of which will simulate 100 replicates for each DADI parameter set.  The first local command took roughly 12 minutes on my computer and we get a directory structure like:

```
ls data/dros_ab/
iter000000  iter000003  iter000006  iter000009  iter000012  iter000015  iter000018  iter000021  iter000024  iter000027  iter000030  iter000033  iter000036  iter000039  iter000042
iter000001  iter000004  iter000007  iter000010  iter000013  iter000016  iter000019  iter000022  iter000025  iter000028  iter000031  iter000034  iter000037  iter000040
iter000002  iter000005  iter000008  iter000011  iter000014  iter000017 ...
```

Within each are two files that contain the x and y variables respectively, ```ab.mig.msOut.gz``` and ```anc.out.gz```.  

Next we will format the training data using multiple cores with MPI.  In this step each replicate is read and the chromosomes in pop A are sorted via seriation and then matched to an up-sampled set of chromosomes from pop B.






