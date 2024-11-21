# Training a UNet to infer introgression

In this tutorial we'll replicate some results given in the paper [IntroUNET: Identifying introgressed alleles via semantic segmentation.] (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1010657).  

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

This script is only meant for simulating this particular dataset.  It simulates 20 individuals in *D. *Simulans and 
