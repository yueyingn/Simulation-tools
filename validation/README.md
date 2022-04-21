
## Validation pipeline for the production run


## Process the BH detail file

* `bh-detail-reduce.py` : write the BlackholeDetails-R to seperate physical columns in bigfile format

```bash
mpirun -n 60 python bh-detail-reduce.py --bhdetail-dir "$bhdetail_dir" --output-dir "$output_dir" --snap $i
```

## Summary Plots

* `plot-selected-bh-detail.py` : plot BH details for some random selected BHs
* `plot-stats.py` : plot statistics of BH mass function, luminosity function and galaxy mass function
* `plot-stellar-bhs-traj.py` : plot host galaxy on 10ckpc scale of some selected BHs, and stellar field + BH trajectories in 100ckpc scale

```bash
python plot-selected-bh-detail.py --bhdetail-file "$bhdetail_file" --pig-file "$pig_file" --output-dir "$output_dir"
python plot-stats.py --pig-file "$pig_file" --output-dir "$output_dir"
python plot-stellar-bhs-traj.py --bhdetail-file "$bhdetail_file" --pig-file "$pig_file" --output-dir "$output_dir"
```
