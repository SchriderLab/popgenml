# I/O Functions for Genetic Data

## `append_to_ms(ofile, X, sites, params=None)`

Appends a simulation (genotype matrix and positions) to a text file in a format identical to how `ms` outputs simulations.

### Parameters
- `ofile`: `file-like object`  
  The output file where the simulation will be appended.
- `X`: `np.ndarray`, shape `(n_samples, n_sites)`  
  The genotype matrix.
- `sites`: `np.ndarray`, shape `(n_sites,)`  
  The positions of the sites (normalized between 0 and 1).
- `params`: `list` or `None`, optional  
  A list of parameters to be written as a header.

### Returns
- None  

---

## `write_to_ms(ofile, X, sites, params=None)`

Writes a genotype matrix (`X`) and positions to a text file in a format identical to how `ms` outputs simulations.

### Parameters
- `ofile`: `str`  
  The path to the output file where the simulation will be written.
- `X`: `np.ndarray`, shape `(n_samples, n_sites)`  
  The genotype matrix.
- `sites`: `np.ndarray`, shape `(n_sites,)`  
  The positions of the sites (normalized between 0 and 1).
- `params`: `list` or `None`, optional  
  A list of parameters to be written as a header.

### Returns
- None  

---

## `split(word)`

Splits a string into a list of characters.

### Parameters
- `word`: `str`  
  The string to be split.

### Returns
- `list`  
  A list of characters from the input string.

---

## `load_ms(msFile, ancFile=None, n=None, flip_alleles=False)`

Loads a gzipped or uncompressed `ms` file (and optionally an `anc` file) and returns genotype matrices, introgressed allele matrices (if applicable), position vectors, and the parameters.

### Parameters
- `msFile`: `str`  
  Path to the `ms` file (gzipped or plain text).
- `ancFile`: `str`, optional  
  Path to the `anc` file containing introgressed allele information.
- `n`: `int` or `None`, optional  
  The number of individuals to be loaded (if specified).
- `flip_alleles`: `bool`, default=`False`  
  If `True`, flips the alleles in the genotype matrix based on the majority allele.

### Returns
- `X`: `list` of `np.ndarray`  
  List of genotype matrices (one per chunk in the `ms` file).
- `Y`: `list` of `np.ndarray` or `None`  
  List of introgressed allele matrices (if an `anc` file is provided).
- `P`: `list` of `np.ndarray`  
  List of position vectors.
- `params`: `list` of `list`  
  List of parameter lists, one for each chunk.

---