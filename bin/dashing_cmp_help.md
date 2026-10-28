#Calling Dashing2 version v2.1.20 with command '/scratch4/blangme2/sjhwang/ok_k_mer/bin/dashing2 cmp --help'
dashing2 cmp <opts> [fastas... (optional)]
We use only m-mers; if w <= k, however, this reduces to k-mers if the -w/--window-size is unspecified.
--presketched	 To compute distances using a pre-sketched method (e.g., dashing2 sketch -o path), use this flag and pass in a single positional argument.
General usage:
dashing2 <subcommand> <flags> [optional: positional arguments]

dashing2 takes all positional arguments as sketch objects.
Alternatively, -F/--ffile takes inputs from a file in addition to positional arguments.Threading parallelism is set with -p/--threads.

There are two main subcommands: sketch and cmp

Sketch only generates sketches, while cmp performs comparisons across inputs, but sketches if necessary.


File Format Options --
Dashing2 can sketch 4 kinds of files:Fastq/Fasta, which has specific encoding options (default)
--bed to sketch BED files for interval sets
--bigwig to sketch BigWig files for coverage vectors
and --leafcutter to sketch LeafCutter splicing output


Sequence Parsing Options --
If parsing fasta or fastq data, you can set several options:
  1. Alphabet [Default: DNA]. See 'Sequence Alphabet Options' below for more details.
  2. K-mer length (-k/--kmer-length) [Default: Maximum expressible in uint64_t (32 for DNA)]
  3. Window size (-w/--window-size) [Default: k-mer length]. Selects minimum-hash item from window length. Larger windows yields fewer minimizers.
  4. K-mer spacing. (--spacing) [Default: unspaced]. Allows for some positions to be ignored and others used. See below for more detail.

  5. K-mer encoding size. Defaults to uint64_t. 128-bit integers (up to 64bp DNA) can be enabled with --long-kmers.
     Dashing2 supports unbounded k-mer length by switching to rolling hashing if k is greater than the maximum possible length for the alphabet chosen.
  6. Disabling Canonicalization (--no-canon). By default, DNA alphabet k-mers are canonicalized to abstract strand. --no-canon causes Dashing2 to be strand-specific.
  7. Seed (--seed). To draw samples from a new hash function, you can provide a seed to the analysis.
  8. K-mer filtering, such as from a set of sequences, downsampling, or minimum count. See Detailed Filtering Options below for details.
Detailed Sequence Parsing Options
-k/--kmer-length: set k. Defaults to the largest k expressible directly in uint64_t.
If k is greater than this limit (31 for DNA, 14 for --protein, 22 for --protein8, 24 for --protein6), then rolling hashes will be generated instead of exact k-mer encodings.
-w/--window-size: set window size for winnowing; by default, all k-mers are used. If w > k, then only the minimum-hash k-mer in each window is processed
This can be useful to speed up sketching or to reduce the number of items sketched in exact mode.
--entmin: If -w/--window-size is enabled, this option weights the hash value by the entropy of the k-mer itself.
This is only valid for k-mers short enough to be encoded exactly in 64-bit or 128-bit integers, depending on if --long-kmers is enabled.
--spacing: Set a spacing scheme for spaced minimizers
This must have 1 less integer than the k-mer length.
e.g., -k5 --spacing 0,1,1,0 specifies a match length of 5 with a match pattern of `KK$K$KK`, where $ positions are ignored and `K` positions are kept.
This can also be run-length compressed in <space, num> format.
For example, --spacing 0,1x2,0 is equivalent to 0,1,1,0.
-2/--128bit/long-kmers: Use 128-bit k-mer hashes instead of 64-bit
Detailed Sequence Alphabet Options

Dashing2 sketches DNA by default. This can be changed with the following flags; this will disable canonicalization.
--enable-protein: Use 20 character amino acid alphabet.
--protein14: Use 14 character amino acid alphabet.
--protein6: Use 6 character amino acid alphabet.
--protein8: Use 8 character (3-bit) amino acid alphabet.
--no-canon: If DNA is being encoded, this disables canonicalization. By default, DNA sequence is canonicalized with its reverse-complement.
            Otherwise, this is ignored
--seed: Set a seed for k-mer hashing; If 0, this disables k-mer XORing and k-mers are encoded directly if a k-mer type can represent it.
        Otherwise, this changes the hash function applied to k-mers when generated sorted hash sets. This makes it easy to decode quickly, but we can still get good bottom-k estimates using these hashes
        the xor value for u64 kmers (unless --long-kmers is enabled) is the Wang 64-bit hash of the seed.
        u128 kmers (--long-kmers) have the same lower 64 bits, but the upper 64 bits are the Wang 64-bit hash of the u64 xor value.
Detailed Filtering Options

--downsample	 Downsample minimizers at fraction <arg> . Default is 1: IE, all minimizers pass.
-m/--threshold/--count-threshold <arg>: Set a count threshold for inclusion. If set to > 1, this will only sketch k-mers with count >= <arg>
If there are a set of common k-mers or artefactual sequence, you can specify --filterset to skip k-mers in this file when sketching other files.
By default, this converts it into a sorted hash set and skips k-mers which are found in the set.
`--filterset [path]` yields this.


Input File Options --
By default, dashing2 reads positional arguments and sketches them. You may want to use flags instructing it
to read from paths in <file>. Additionally, you can put multiple files separated by spaces into a single line to place them all into a single sketch instead.
-F/--ffile: read paths from file in addition to positional arguments.
-Q/--qfile: read query paths from file; this is used for asymmetric queries (e.g., containment).
If multiple files are space-delimited in a single line, they will be sketched jointly.


Sketch options
-S/--sketchsize: Set sketchsize (1024)
Warning: this is a significant change from Dashing1, where sketch size was required to be a power of 2, and so the sketch size would be 2**<arg>.
This restriction is no longer in place, and so any positive integer can be selected.
e.g. - instead of -S12 in Dashing, you would specify -S 4096
For convenience, we offer another argument to specify the size in log2.
-L/--sketch-size-l2: Set sketchsize to 2^<arg>. Must be > 0 and < 64
--cache/--cache-sketches: Save sketches to disk instead of re-computing each time.
	Dependent option:
	                 --outprefix: specifies directory in which to save sketches instead of adjacent to the input files.
	                 aliases: --prefix.
	                 Note: You must have permission to write in the specified folder.

Sketching Mode Options -- 

Inputs can be summarized into several structures, and flags determine which is chosen.
1. SetSketch (one-permutation). Treats inputs as sets, ignoring multiplicities. The fastest option.
   This is faster at sketching, but has a small probability of failure which grows with sketch size. Big one-permutation sketches may perform poorly.
   (Default setting.)
2. FullSetSketch. Also treats inputs as sets but has better behavior for larger sketches and small sets.
   FullSetSketch is slower than one-permutation at sketching, and equally fast at comparisons than One-Permutation.
   --full/--full-setsketch to enable.

 We provide to weighted sketching algorithms -- WeightedSetSketch and DiscreteProbabilitySetSketch
 Both require counting for sequence data, but do not for input methods with counting already performed, e.g. BigWig.
 Full k-mer counting is enabled by default, but memory requirements can be fixed by using a count-min sketch during sketching.
 Enabled by --countmin-size [number-registers], this allows for weighted sketching with fixed memory usage at the expense of some approximation.
 This is only relevant to WeightedSetSketch and DiscreteProbabilitySetSketch.
3. WeightedSetSketch: Weighted Sets sketched by BagMinHash
   Multiset sketching via BagMinHash is an LSH for the weighted Jaccard similarity, which treats k-mer counts as weighted sets.
   -B/--multiset/--bagminhash to enable.
4. DiscreteProbabilitySetSketch: Discrete Probability Distributions using ProbMinHash
   ProbMinHash is an LSH for the probability Jaccard index, which normalizes observations by total counts, yielding a discrete probability distribution for each collection.
   ProbMinHash is most applicable for datasets where sampling fractions are important, such as expression or splicing counts.
   --prob/--pminhash
   ProbMinHash is 2-10x as fast as BagMinHash at sketching.
 We also provide some exact comparison and sketching modes.
 These include full k-mer sets (--set), a k-mer count dictionary (--countdict), or a sequence of minimizers (--seq)
 5. K-mer Sets.
   This generates a sorted hash set for k-mers in the data. If the parser is windowed (-w is fairly large), this could even be rather small.
   -H/--set to enable
   If an LSH table is generated, then weighted bottom-k hashes are used to build an LSH table
 6. Full k-mer countdict. 
    This generates a sorted hash set for k-mers in the data, and additionally saves the associated counts for these k-mers.
    If an LSH table is generated, then weighted bottom-k hashes as in Cohen, E. "Summarizing Data using Bottom-K Sketches"
   -J/--countdict to enable 
 7. Full k-mer (or minimizer) sequence. This faster than building the hash set, and can be used to build a minimizer index afterwards
          If you use --parse-by-seq with this and an output path is provided, then the stacked minimizer sequences will be written to it.
          The format is the similar to the standard stacked sketches, except that the cardinality fields instead represent minimizer sequence lengths (in 64-bit registers).
          Specifically, it consists of a header: [uint64_t nitems, uint32_t k, uint32_t w], followed by `nitems` [double], specifying sequence lengths of 64-bit registers
   -G/--seq to enable.
    Dependent option:
          --hp-compress:
              Minimizer sequence will be homopolymer-compressed before emission. 
              This makes the sequences ignore the lengths of minimizer stretches.


Other Sketching Options -- 
--parse-by-seq: Parse each sequence in each file as a separate entity. For workloads using edit distance, or for the --greedy mode, this will store all sequences in a temporary file in $TMPDIR.
                Previous versions of Dashing2 stored all sequences in memory with high memory usage for --parse-by-seq. This is reduced in v2.1.18.
                For faster use but more memory (restoring previous behavior), add --seqs-in-ram to avoid spilling to disk.
-s/--save-kmers: Save k-mers. This puts the k-mers saved into .kmer files to correspond with the minhash samples. 
  If an output path is specified for dashing2 and --save-kmers is enabled, stacked k-mers will be written to <arg>.kmer64, and names will be written to <arg>.kmer.names.txt
  This has a 16-byte header containing a 32-bit integer describing the alphabet used, 32 bits describing sketch size, one 32-bit integer for k, and one 32-bit integer for window-length.
  This database can be used for dashing2 contain.
-N/--save-kmercounts: Save k-mer counts for sketches. This puts the k-mer counts saved into .kmercounts.f64 files to correspond with the k-mers.
-o/--outfile: sketches are stacked into a single file and written to <arg>
  This is the path for the stacked sketches; to set output location, use --cmpout instead. (This is the distance matrix betweek sketches).
  This can also reduce memory requirements, as the destination file is memory-mapped instead of held in memory.


Comparison Options -- 
We provide exhaustive (all-vs-all) comparisons, top-k nearest neighbor (KNN) graph generation and clustering. 
  Within Exhaustive Comparisons, we have dense and sparse. The first 3 are dense:
    1. Upper Triangular PHYLIP (default)
    2. Square distance matrix (--asymmetric-all-pairs/--square)
    3. Rectangular distance matrix (--qfile/-Q)
  Instead of performing pairwise comparisons across one set, you can instead compare all in set X against all in set Y.
  We call this rectangular. When enabled with a path, entries on each line of the file at that path are treated as entities.
  positional arguments and -F paths are treated as a reference set;
  Paths provided in -Q/--qfile are are treated as a query set.
  Performing --asymmetric-all-pairs with the same input for -F and -Q should yield equivalent results.
  The output shape then has |F| rows and |Q| columns, (F, Q) in row-major format

 We also support Sparse Exhaustive Comparisons, where the results are limited to more important entries.
  These include: 
  1. Top-K, (--topk) only the top-k nearest neighbors are emitted for each item
  2. Thresholded, (--similarity-threshold), which emits entries only if the similarity exceeds the given threshold.
Both change the output format from a full matrix into compressed-sparse row (CSR) format listing only the filtered entries;
All of these are powered by the use of an LSH table built over the sketches, with the exception of exact mode (--countdict or --set), which use an LSH index built over their bottom-k hashes.
For details on LSH table parameters, see `LSH Options` below.
Top-K (K-Nearest-Neighbor) mode -- 
--topk/--top-k <arg>	Maximum number of nearest neighbors to list. If <arg> is greater than N - 1, pairwise distances are instead emitted.

Thresholded Mode -- 
--similarity-threshold <arg>	Minimum fraction similarity for inclusion.
	If this is enabled, only pairwise similarities over <arg> will be emitted.


Greedy HIT Clustering Options --
In addition to exhaustive comparisons, we also perform greedy clustering using the CD High-Identity with Tolerance (CD-HIT) algorithm.
--greedy <float (0-1]> For greedy clustering by a given similarity threshold.
This uses an LSH index by default. 
    To compare all points to all clusters, add E to the end of the flag. (e.g., '--greedy 0.8E')    By default, this emits the names of the entities (sequence names, if --parse-by-seq, and filenames otherwise).
    You may want to emit fasta-formatted output. You can do this by adding F to the end of the --greedy argument.
    Example: '--greedy 0.8F' or '--greedy 0.8FE'.
    This is only allowed for --parse-by-seq.
  As this number approaches 1, the number and uniformity of clusters grows.
  For human-readable output, this emits one line per cluster listing its constituents, ordered by similarity
  For machine-readable output, this file consists of 2 64-bit integers (nclusters, nsets), followed by (nclusters + 1) 64-bit integers, followed by nsets 64-bit integers, identifying which sets belonged to which clusters.
  This is a vector in Compressed-Sparse notation.
  Python code for parsing the binary representation is available at https://github.com/dnbaker/dashing2/blob/main/python/parse.py.

Distance Register Size Options --
By default, we compare items with full hash function registers; to trade accuracy for speed, these sketches can be compressed before comparisons.
To truncate for faster comparisons, you can either select a register size to which to truncate to after generating full floating-point values, which will allow Dashing2 to use as much resolution as possible in a fixed number of bits.
On the other hand, this means that initial sketching needs more memory.
If you provide register values ahead of time, you can accumulate in smaller registers directly to reduce memory requirements.
--fastcmp/--regsize <arg>	Enable faster comparisons using n-byte signatures rather than full registers. By default, these are logarithmically-compressed
  You can use this first approach with --fastcmp/--regsize. These are logarithmically compressed.
  For example, --fastcmp 1 uses byte-sized sketches, with a and b parameters inferred by the data to minimize information loss
  <arg> may be 8 (64 bits), 4 (32 bits), 2 (16 bits), or 1 (8 bits)
  Results may even be somewhat stabilized by the use of smaller registers.
	Dependent Options:
	          --setsketch-ab <float1>,<float2>
	            Example: --setsketch-ab 0.4,1.005
	            If you specify a, b before using this method, then SetSketches of --fastcmp size bytes will be generated directly, rather than truncating after sketching all items.
	            However, this is only supported for the SetSketch. (e.g., not for --bagminhash or --probminhash).
	            This can allow you to scale to larger collections.
	            We also provide several pre-set parameter values.
	          --fastcmp-bytes sets a and b to 20 and 1.2, and sets --fastcmp to 1
	          --fastcmp-shorts sets a and b to .06 and 1.0005, and sets --fastcmp to 2.
	          --fastcmp-words sets a and b to 19.77 and 1.0000000109723500835 and sets --fastcmp to 4.

If you instead want to truncate to the bottom-b bits of the signature --
	          --bbit-sigs: truncate to bottom-<arg> bytes of signatures instead of logarithmically-compressed.
The runtime is effectively equivalent to the setsketch.


Comparison Function Options --
The default comparison emitted is similarity. For MinHash/HLL/SetSketch sketches, this is the fraction of shared registers.
This can be changed to a distance (--mash-distance) for k-mer similarity, where it can be used for hierarchical clustering.
--mash-distance/--poisson-distance/--distance	 Emit distances, as estimated by the Poisson model for k-mer distances.
--symmetric-containment	 Use symmetric containment as the distance. e.g., (|A & B| / min(|A|, |B|))
--containment	 Use containment as the distance. e.g., (|A & B| / |A|). This is asymmetric, so you must consider that when deciding the output shape.
--intersection-size/--intersection	 Emit the cardinality of the intersection between entities. IE, the number of k-mers shared between the two.
--union-size	 Emit the cardinality of the union between entities. IE, the number of k-mers in the union of the two.
--compute-edit-distance	 For edit distance, perform actual edit distance calculations rather than returning the distance in LSH space.
                       	 This means that the LSH index eliminates the quadratic barrier in candidate generation, but they are refined using actual edit distance.

Distance Output Options --
--binary-output	Emit binary output rather than human-readable.

	 For symmetric pairwise, this emits condensed distance matrix in f32
	 For -Q/--qfile usage, this emits a full rectangular distance matrix in of shape (|F|, |Q|)
	 For asymmetric pairwise, this emits a full distance matrix in f32 in row-major storage.
	 For asymmetric pairwise, this emits a flat distance matrix in f32
	 For top-k filtered, this emits a matrix of min(k, |N|) x |N| of IDs and distances
In `dashing2 sketch`, distances are not automatically computed; If set, they are written to <arg>
In `dashing2 cmp`, this defaults to stdout.
--cmpout/--distout/--cmp-outfile	Compute distances and emit them to <arg>.
	 For similarity-thresholded distances, this emits a compressed-sparse row (CSR) formatted matrix with 64-bit indptr, 32-bit indices, and 32-bit floats for distances


LSH Options --
There are a variety of heuristics in the LSH tables; however, the most important besides sketch size is the number of hash tables used.
--nLSH <int=2>	
   Aliases: --nlsh
This sets the number of LSH tables. The first 3 tables use register grouping sizes of powers of 2, and subsequent tables use 2 times the index.
If 2 is used (default), these will be of sizes (1, 2), but 4 yields (1, 2, 4, 6) and 5 yields (1, 2, 4, 6, 8) registers per composite key.
Increase this number to pay more memory/time for higher accuracy.
Decrease this number for higher speed and lower accuracy.
This is ignored for exact sketching (--countdict or --set), where a single permutation is generated and a single hash table is used.
--maxcand <int>	 Set the maximum number of candidates to fetch from the LSH index before evaluating distances against them.
                  This is always used in --greedy mode.
                  By default, this number is heuristically selected by the number of items in the index.
                  If N <= 100000, uses max(N / 50, max(cbrt(N), 3)). Otherwise, uses (log(N)^3).
                  Note: this is a maximum; if fewer items have matches, fewer will be reported.
                  This option is ignored in --topk mode, as the number of samples is ceil(3.5 * <topk>).
                  If set in --similarity-threshold mode, the number of items compared will be truncated to <maxcand> even if further samples are above the similarity threshold.
                  This can prevent quadratic complexity for the (rare) case that all items are within threshold distaance of each other.
