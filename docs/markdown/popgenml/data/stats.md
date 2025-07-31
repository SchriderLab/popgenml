# Population Genetic Statistics Functions

## `theta_pi(x, pos)`

Computes **nucleotide diversity** (π), the average number of pairwise differences per site.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype array (0/1).
- `pos`: `np.ndarray`, shape `(n_sites,)`  
  Genomic positions (as integers).

### Returns
- `float`  
  Nucleotide diversity (π) over the provided positions.

---

## `watterson_theta(x, pos, ploidy=2)`

Computes **Watterson's theta**, an estimator of genetic diversity based on the number of segregating sites.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype array.
- `pos`: `np.ndarray`, shape `(n_sites,)`  
  Genomic positions (as integers).
- `ploidy`: `int`, default=`2`  
  Ploidy level to convert haplotypes into genotypes.

### Returns
- `float`  
  Watterson’s theta estimate.

---

## `tajimas_d(x, pos, ploidy=2)`

Computes **Tajima's D**, a neutrality test statistic comparing π and Watterson's theta.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype array.
- `pos`: `np.ndarray`, shape `(n_sites,)`  
  Genomic positions.
- `ploidy`: `int`, default=`2`  
  Ploidy level for genotype conversion.

### Returns
- `float`  
  Tajima’s D statistic.

---

## `ld_stats(x, ploidy=2)`

Calculates the **mean and standard deviation of pairwise linkage disequilibrium** using Rogers-Huff R.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype array.
- `ploidy`: `int`, default=`2`  
  Ploidy level to convert haplotypes into genotypes.

### Returns
- `np.ndarray`, shape `(2,)`  
  Mean and standard deviation of LD (Rogers-Huff R values).

---

## `het_diversity(x, ploidy=2)`

Computes **observed heterozygosity** across all sites.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype array.
- `ploidy`: `int`, default=`2`  
  Ploidy level to convert haplotypes into genotypes.

### Returns
- `np.ndarray`, shape `(2,)`  
  Mean and standard deviation of observed heterozygosity.

---

## `sfs(x)`

Calculates the **Site Frequency Spectrum (SFS)** from a binary haplotype matrix.

### Parameters
- `x`: `np.ndarray`, shape `(n_haplotypes, n_sites)`  
  Binary haplotype matrix (0/1).

### Returns
- `np.ndarray`, shape `(n_samples - 2,)`  
  Normalized site frequency spectrum (sums to 1).

---