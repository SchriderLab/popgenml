# -*- coding: utf-8 -*-
import sys

def main():
    args = sys.argv[1:]
    
    # Parse command line arguments
    if len(args) == 3:
        infile = args[0]
        outfile = args[1]
        nsites = float(args[2])
    elif len(args) == 2:
        infile = args[0]
        outfile = args[1]
        nsites = 1.0
    else:
        print("###################################################")
        print("Usage: python ms2haps.py infile.ms outfile nsite\n")
        print("infile.ms: Input filename with file extension.")
        print("outfile: Output filename without file extension.")
        print("nsites: (Optional) Number of simulated sites. Default value is 1. This is multiplied to the positions.")
        print("###################################################")
        sys.exit(1)

    # Read file as a list of strings
    try:
        with open(infile, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
    except IOError:
        print(f"Error reading file: {infile}")
        sys.exit(1)

    # Find lines starting with "//"
    startlines = [i for i, line in enumerate(lines) if line.startswith("//")]

    if not startlines:
        print("No input in ms format.")
        sys.exit(1)

    num_blocks = len(startlines)
    
    # Append the length of lines to act as the end boundary for the last block
    startlines.append(len(lines))

    # Loop through each block (works for both single and multiple realisations)
    for i in range(num_blocks):
        start = startlines[i]
        end = startlines[i+1]

        # Extract positions (lines[start+2])
        pos_line = lines[start+2]
        pos_parts = pos_line.split()
        
        # skip "positions:" string, multiply by nsites, and round to integer
        pos = [round(float(x) * nsites) for x in pos_parts[1:]]

        # Extract sequences
        seq_lines = [line for line in lines[start+3:end] if line != ""]

        # Convert sequences to a matrix and transpose
        # Using zip(*...) transposes the lists: columns (individuals) become rows (sites)
        seq_matrix = [list(site_chars) for site_chars in zip(*seq_lines)]

        # Find sites with multiple mutations (duplicates)
        # We count occurrences to identify duplicates and remove ALL instances of them
        pos_counts = {}
        for p in pos:
            pos_counts[p] = pos_counts.get(p, 0) + 1

        dup_pos = {p for p, count in pos_counts.items() if count > 1}

        # Filter out duplicated positions
        if dup_pos:
            filtered_pos = []
            filtered_seq = []
            for p, seq_row in zip(pos, seq_matrix):
                if p not in dup_pos:
                    filtered_pos.append(p)
                    filtered_seq.append(seq_row)
            pos = filtered_pos
            seq_matrix = filtered_seq

        if len(pos) == 0:
            print("No segsites\nBP have to be integers! (Use third argument)")
            if num_blocks == 1:
                sys.exit(1)
            else:
                continue

        N = len(seq_matrix[0])  # Number of individuals (haplotypes)
        L = len(seq_matrix)     # Number of sites (SNPs)

        ##### Prepare output filenames #####
        if num_blocks == 1:
            sample_file = f"{outfile}.sample"
            haps_file = f"{outfile}.haps"
            chr_id = "1"
        else:
            sample_file = f"{outfile}_chr{i+1}.sample"
            haps_file = f"{outfile}_chr{i+1}.haps"
            chr_id = str(i+1)

        ##### Write outfile.sample #####
        with open(sample_file, 'w') as f_samp:
            f_samp.write("ID_1 ID_2 missing\n")
            f_samp.write("0 0 0\n")
            # Diploid individuals (N/2)
            for j in range(1, (N // 2) + 1):
                f_samp.write(f"UNR{j} UNR{j} 0\n")

        ##### Write outfile.haps #####
        with open(haps_file, 'w') as f_haps:
            for j in range(L):
                # Format: chr_id, SNP_id, position, ref_allele, alt_allele, ...sequence
                row = [chr_id, f"SNP{j+1}", str(pos[j]), "A", "T"]
                row.extend(seq_matrix[j])
                f_haps.write(" ".join(row) + "\n")

if __name__ == "__main__":
    main()
