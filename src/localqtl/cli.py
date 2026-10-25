"""
``python -m localqtl`` / ``localqtl``: command-line interface with tensorQTL's
modes, arguments and output file names, plus local-ancestry options.

Output files (``<output_dir>/<prefix>...``), as in tensorQTL::

    cis               <prefix>.cis_qtl.txt.gz
    cis_nominal       <prefix>.cis_qtl_pairs.<chr>.parquet
                      <prefix>.cis_qtl_top_assoc.txt.gz        (interaction scans)
                      <prefix>.cis_qtl.signif_pairs.parquet    (with --cis_output)
    cis_independent   <prefix>.cis_independent_qtl.txt.gz
    cis_conditional   <prefix>.cis_conditional_qtl.txt.gz
    cis_susie         <prefix>.SuSiE_summary.parquet, <prefix>.SuSiE.pickle
    trans             <prefix>.trans_qtl_pairs.parquet | .txt.gz   (sparse)
                      <prefix>.trans_qtl_{pval,beta,beta_se,af}.parquet (dense)
    trans_susie       <prefix>.SuSiE_summary.parquet, <prefix>.SuSiE.pickle
    log               <prefix>.localQTL.<mode>.log

Result columns use tensorQTL's spelling (``true_df``, ``pval_true_df``,
``b_g_se`` ...) unless ``--localqtl_names`` is given.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import os
import pickle
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from . import genotypeio, pgen, susie, trans
from .cis import map_conditional_dosage, map_independent, map_nominal, map_permutations
from .cis.postproc import get_significant_pairs
from .compat import from_tensorqtl_columns, to_tensorqtl_columns
from .phenotypeio import read_phenotype_bed
from .stats import calculate_qvalues
from .utils import SimpleLogger

__all__ = ["build_parser", "main"]

MODES = ['cis', 'cis_nominal', 'cis_independent', 'cis_conditional', 'cis_susie',
         'trans', 'trans_susie']


def _version() -> str:
    try:
        return importlib.metadata.version("localqtl")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localqtl",
                                     description='localQTL: GPU-based, ancestry-aware QTL mapper (tensorQTL-compatible CLI)')
    parser.add_argument('genotype_path', help='Genotypes in PLINK 2 (pgen/psam/pvar) or PLINK 1 (bed/bim/fam) format, '
                                              'or a parquet / tab-delimited dosage matrix (trans mode)')
    parser.add_argument('phenotypes', help="Phenotypes in BED format (.bed, .bed.gz, .bed.parquet), or optionally for "
                                           "'trans' mode, parquet or tab-delimited.")
    parser.add_argument('prefix', help='Prefix for output file names')
    parser.add_argument('--mode', type=str, default='cis', choices=MODES, help='Mapping mode. Default: cis')
    parser.add_argument('--covariates', default=None, help='Covariates file, tab-delimited (covariates x samples)')
    parser.add_argument('--paired_covariate', default=None,
                        help='Single phenotype-specific covariate, tab-delimited (phenotypes x samples)')
    parser.add_argument('--permutations', type=int, default=10000, help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str,
                        help='Tab-delimited file mapping sample ID to interaction value(s) (if multiple interaction '
                             'terms are used, the file must include a header with variable names)')
    parser.add_argument('--cis_output', default=None, type=str,
                        help="Output from 'cis' mode with q-values. Required for independent cis-QTL "
                             "mapping; in 'cis_conditional' mode give the signals to re-fit (the "
                             "'cis_independent' or 'cis' output).")
    parser.add_argument('--phenotype_groups', default=None, type=str,
                        help='Phenotype groups. Header-less TSV with two columns: phenotype_id, group_id')
    parser.add_argument('--window', default=1000000, type=int, help='Cis-window size, in bases. Default: 1000000.')
    parser.add_argument('--pval_threshold', default=1e-5, type=float,
                        help='Output only significant phenotype-variant pairs with a p-value below threshold. '
                             'Default: 1e-5 for trans-QTL')
    parser.add_argument('--logp', action='store_true', help='Report nominal p-values as -log10(P) (trans mode)')
    parser.add_argument('--maf_threshold', default=None, type=float,
                        help='Include only genotypes with minor allele frequency >= maf_threshold. '
                             'Default: 0 (cis), 0.05 (trans)')
    parser.add_argument('--maf_threshold_interaction', default=0.05, type=float,
                        help='MAF threshold for interactions, applied to lower and upper half of samples')
    parser.add_argument('--dosages', action='store_true', help='Load dosages instead of genotypes (PLINK 2 pgen input).')
    parser.add_argument('--return_dense', action='store_true', help='Return dense output for trans-QTL.')
    parser.add_argument('--return_r2', action='store_true', help='Return r2 (only for sparse trans-QTL output)')
    parser.add_argument('--best_only', action='store_true',
                        help='Only write lead association for each phenotype (interaction mode only)')
    parser.add_argument('--output_text', action='store_true',
                        help='Write output in txt.gz format instead of parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=20000,
                        help='GPU batch size (trans-QTLs only). Reduce this if encountering OOM errors.')
    parser.add_argument('--chunk_size', default=None,
                        help="For cis-QTL mapping, load genotypes into CPU memory in chunks of chunk_size variants, "
                             "or by chromosome if chunk_size is 'chr' (requires PLINK 2 input).")
    parser.add_argument('--susie_loci', default=None,
                        help="Table (parquet or tsv) with loci to fine-map (phenotype_id, chr, pos) with mode 'trans_susie'.")
    parser.add_argument('--disable_beta_approx', action='store_true',
                        help='Disable Beta-distribution approximation of empirical p-values (not recommended).')
    parser.add_argument('--warn_monomorphic', action='store_true',
                        help='Accepted for tensorQTL compatibility (monomorphic variants are always dropped).')
    parser.add_argument('--max_effects', type=int, default=10,
                        help='Maximum number of non-zero effects in the SuSiE regression model.')
    parser.add_argument('--fdr', default=0.05, type=float, help='FDR for cis-QTLs')
    parser.add_argument('--qvalue_lambda', default=None, type=float, help='lambda parameter for pi0est in qvalue.')
    parser.add_argument('--seed', default=None, type=int, help='Seed for permutations.')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory')
    # localQTL additions
    la = parser.add_argument_group('local ancestry (localQTL)')
    la.add_argument('--haplotypes', default=None,
                    help='Local ancestry: RFMix output prefix/directory, FLARE output prefix, or an rfmix_reader Zarr cache')
    la.add_argument('--la_format', default='auto', choices=['auto', 'rfmix', 'flare', 'cache'],
                    help="How to open --haplotypes (default: auto-detect; a directory containing a Zarr store is a cache)")
    la.add_argument('--la_source', default='msp', choices=['msp', 'fb'], help='RFMix source files (default: msp)')
    la.add_argument('--la_cache_dir', default=None, help='rfmix_reader cache directory for RFMix/FLARE input')
    la.add_argument('--use_posterior', action='store_true',
                    help='Use posterior-expected ancestry counts (RFMix .fb input with a cache)')
    la.add_argument('--ancestry_model', default='main', choices=['main', 'interaction', 'dosage'],
                    help="Local-ancestry model for cis modes: 'main' (covariates; the only option for trans), "
                         "'interaction' (GxH), 'dosage' (per-ancestry effects; 'cis' and 'cis_nominal')")
    la.add_argument('--condition_on', default='genotype', choices=['genotype', 'dosage', 'none'],
                    help="'cis_conditional' mode: project out the other signals of a phenotype as "
                         "their pooled dosage (genotype), their per-ancestry dosages (dosage), or "
                         "not at all (none)")
    la.add_argument('--min_ancestry_ac', default=0.0, type=float,
                    help='Dosage model: mask ancestries with fewer in-sample minor alleles than this')
    misc = parser.add_argument_group('localQTL options')
    misc.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Compute device')
    misc.add_argument('--perm_chunk', default=4096, type=int, help='Permutations per GPU chunk (cis modes)')
    misc.add_argument('--localqtl_names', action='store_true',
                      help="Keep localQTL column names (true_dof, se_g, ...) instead of tensorQTL's")
    misc.add_argument('--version', action='version', version=f'localqtl {_version()}')
    return parser


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------
def _read_table(path: str, **kw) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep='\t', **kw)


def _read_interaction(path: str, sample_ids) -> pd.DataFrame:
    with open(path) as f:
        f.readline()
        second = f.readline().strip()
    if len(second.split('\t')) == 2:  # index + value, header optional
        df = pd.read_csv(path, sep='\t', index_col=0, header=None)
        try:
            df = df.astype(np.float32)
        except ValueError:  # first row was a header
            df = pd.read_csv(path, sep='\t', index_col=0)
    else:
        df = pd.read_csv(path, sep='\t', index_col=0)
    df.index = df.index.astype(str)
    missing = [s for s in sample_ids if s not in df.index]
    if missing:
        raise ValueError(f"{len(missing)} samples missing from the interaction file (first: {missing[:5]})")
    df = df.loc[list(sample_ids)].astype(np.float32)
    if df.columns.dtype != object:
        df.columns = ['i'] if df.shape[1] == 1 else [f'i{k + 1}' for k in range(df.shape[1])]
    return df


def _read_groups(path: str, phenotype_df: pd.DataFrame) -> pd.Series:
    group_s = pd.read_csv(path, sep='\t', index_col=0, header=None).squeeze('columns').rename(None)
    group_s.index = group_s.index.astype(str)
    group_dict = group_s.to_dict()
    previous_group, parsed_groups = '', 0
    for i in phenotype_df.index:
        if group_dict[i] != previous_group:
            parsed_groups += 1
            previous_group = group_dict[i]
    if parsed_groups != len(group_s.unique()):
        raise ValueError('Groups defined in input do not match phenotype file (check sort order).')
    return group_s


def _is_zarr_store(path: str) -> bool:
    return os.path.isdir(path) and (path.endswith('.zarr') or any(
        os.path.exists(os.path.join(path, f)) for f in ('zarr.json', '.zgroup', '.zattrs')))


def load_haplotypes(args, sample_ids, logger: SimpleLogger):
    """Open the local-ancestry panel named by ``--haplotypes`` restricted to ``sample_ids``."""
    from .haplotypeio import FlareReader, RFMixReader, open_panel
    path = args.haplotypes
    fmt = args.la_format
    if fmt == 'auto':
        if _is_zarr_store(path):
            fmt = 'cache'
        elif os.path.exists(f"{path}.anc.vcf.gz") or glob.glob(f"{path}*.anc.vcf.gz"):
            fmt = 'flare'
        else:
            fmt = 'rfmix'
    logger.write(f'  * reading local ancestry ({path}; format={fmt})')
    samples = [str(s) for s in sample_ids]
    if fmt == 'cache':
        return open_panel(path, select_samples=samples, use_posterior=args.use_posterior)
    if fmt == 'flare':
        return FlareReader(path, select_samples=samples, cache_dir=args.la_cache_dir,
                           use_posterior=args.use_posterior, verbose=False)
    return RFMixReader(path, select_samples=samples, source=args.la_source, cache_dir=args.la_cache_dir,
                       keep_posteriors=bool(args.use_posterior and args.la_source == 'fb'),
                       use_posterior=args.use_posterior, verbose=False)


def _names(df: pd.DataFrame, args) -> pd.DataFrame:
    return df if args.localqtl_names else to_tensorqtl_columns(df)


def _write_tsv(df: pd.DataFrame, path: str, index: bool) -> None:
    df.to_csv(path, sep='\t', index=index, float_format='%.6g')


def _trans_chunk_bounds(num_variants: int, chunk_size: int):
    n, rem = np.divmod(num_variants, int(chunk_size))
    bounds = [0] + n * [int(chunk_size)]
    if rem != 0:
        bounds.append(int(rem))
    return np.cumsum(bounds)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == 'cis_independent' and (args.cis_output is None or not os.path.exists(args.cis_output)):
        raise ValueError("Output from 'cis' mode must be provided (--cis_output).")
    if args.interaction is not None and args.mode not in ['cis_nominal', 'trans']:
        raise ValueError("Interactions are only supported in 'cis_nominal' or 'trans' mode.")
    if args.ancestry_model == 'dosage' and args.mode not in ('cis', 'cis_nominal'):
        raise ValueError("--ancestry_model dosage is only available in 'cis' and 'cis_nominal' modes.")
    if args.mode == 'cis_conditional':
        if args.cis_output is None or not os.path.exists(args.cis_output):
            raise ValueError("Signals to re-fit must be provided for 'cis_conditional' (--cis_output).")
        if args.haplotypes is None:
            raise ValueError("'cis_conditional' fits the ancestry-specific dosage model and requires --haplotypes.")
        if args.chunk_size is not None:
            raise ValueError("'cis_conditional' reads only the listed signals and does not support --chunk_size.")
        if args.phenotype_groups is not None:
            raise ValueError("'cis_conditional' fits one signal at a time and does not support --phenotype_groups.")
    if args.ancestry_model == 'interaction' and args.mode not in ('cis', 'cis_nominal', 'cis_independent'):
        raise ValueError("--ancestry_model interaction is only available in cis modes.")
    if args.haplotypes is not None and args.mode in ('cis_susie', 'trans_susie'):
        raise ValueError("Local ancestry is not supported in the SuSiE modes.")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(args.output_dir, f'{args.prefix}.localQTL.{args.mode}.log'))
    logger.write(f'[{datetime.now().strftime("%b %d %H:%M:%S")}] Running localQTL v{_version()}: '
                 f'{args.mode.split("_")[0]}-QTL mapping')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        logger.write('  * WARNING: CUDA requested but unavailable; using CPU')
        device = 'cpu'
    if device == 'cuda':
        logger.write(f'  * using GPU ({torch.cuda.get_device_name(torch.cuda.current_device())})')
    else:
        logger.write('  * WARNING: using CPU!')
    if args.seed is not None:
        logger.write(f'  * using seed {args.seed}')

    # phenotypes
    logger.write(f'  * reading phenotypes ({args.phenotypes})')
    phenotype_pos_df = None
    if args.mode.startswith('cis') or args.phenotypes.lower().endswith(('.bed', '.bed.gz', '.bed.parquet')):
        if not args.phenotypes.lower().endswith(('.bed', '.bed.gz', '.bed.parquet')):
            raise ValueError("For cis modes, phenotypes must be in BED format.")
        phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotypes)
        if 'pos' in phenotype_pos_df.columns:
            logger.write(f"  * cis-window detected as position ± {args.window:,}")
        else:
            logger.write(f"  * cis-window detected as [start - {args.window:,}, end + {args.window:,}]")
    else:
        phenotype_df = _read_table(args.phenotypes, index_col=0)
    phenotype_df.columns = phenotype_df.columns.astype(str)
    sample_ids = list(phenotype_df.columns)

    covariates_df = None
    if args.covariates is not None:
        logger.write(f'  * reading covariates ({args.covariates})')
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        covariates_df.index = covariates_df.index.astype(str)
        if not covariates_df.index.equals(phenotype_df.columns):
            missing = [s for s in sample_ids if s not in covariates_df.index]
            if missing:
                raise ValueError(f"{len(missing)} phenotype samples missing from covariates (first: {missing[:5]})")
            covariates_df = covariates_df.loc[sample_ids]

    paired_covariate_df = None
    if args.paired_covariate is not None:
        paired_covariate_df = pd.read_csv(args.paired_covariate, sep='\t', index_col=0)
        paired_covariate_df.columns = paired_covariate_df.columns.astype(str)
        if not paired_covariate_df.index.isin(phenotype_df.index).all():
            raise ValueError("Paired covariate phenotypes must be present in phenotype matrix.")
        if not paired_covariate_df.columns.equals(phenotype_df.columns):
            paired_covariate_df = paired_covariate_df[sample_ids]

    interaction_df = None
    if args.interaction is not None:
        logger.write(f'  * reading interaction term(s) ({args.interaction})')
        interaction_df = _read_interaction(args.interaction, sample_ids)

    maf_threshold = args.maf_threshold
    if maf_threshold is None:
        maf_threshold = 0.05 if args.mode == 'trans' else 0.0

    group_s = _read_groups(args.phenotype_groups, phenotype_df) if args.phenotype_groups is not None else None

    # genotypes
    pgr = None
    genotype_df = variant_df = None
    if args.chunk_size is None:
        logger.write('  * loading genotype dosages' if args.dosages else '  * loading genotypes')
        genotype_df, variant_df = genotypeio.load_genotypes(args.genotype_path, select_samples=sample_ids,
                                                            dosages=args.dosages)
        genotype_df.columns = genotype_df.columns.astype(str)
        if list(genotype_df.columns) != sample_ids:
            genotype_df = genotype_df[sample_ids]
        if variant_df is None and args.mode.startswith('cis'):
            raise ValueError("Genotype data without variant positions is only supported for mode='trans'.")
    else:
        if not all(os.path.exists(f"{args.genotype_path}.{ext}") for ext in ['pgen', 'psam', 'pvar']):
            raise ValueError("Processing in chunks requires PLINK 2 pgen/psam/pvar files.")
        pgr = pgen.PgenReader(args.genotype_path, select_samples=sample_ids)

    haplotypes = load_haplotypes(args, sample_ids, logger) if args.haplotypes is not None else None
    la_kwargs = dict(haplotypes=haplotypes, ancestry_model=args.ancestry_model) if haplotypes is not None else {}

    def _chunks():
        return pgen.generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, args.chunk_size,
                                           window=args.window, dosages=args.dosages, verbose=True)

    if args.mode == 'cis':
        kw = dict(covariates_df=covariates_df, group_s=group_s, paired_covariate_df=paired_covariate_df,
                  nperm=args.permutations, window=args.window, beta_approx=not args.disable_beta_approx,
                  maf_threshold=maf_threshold, logger=logger, seed=args.seed, device=device,
                  perm_chunk=args.perm_chunk, min_ancestry_ac=args.min_ancestry_ac, **la_kwargs)
        if pgr is None:
            res_df = map_permutations(genotype_df, variant_df, phenotype_df, phenotype_pos_df, **kw)
        else:
            res_df = pd.concat([map_permutations(gt, var, p, ppos, **kw) for gt, var, p, ppos, _ in _chunks()],
                               ignore_index=True)
        logger.write('  * writing output')
        res_df = calculate_qvalues(res_df, fdr=args.fdr, qvalue_lambda=args.qvalue_lambda, logger=logger)
        _write_tsv(_names(res_df, args), os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.txt.gz'), index=False)

    elif args.mode == 'cis_nominal':
        has_int = interaction_df is not None
        kw = dict(covariates_df=covariates_df, paired_covariate_df=paired_covariate_df,
                  interaction_df=interaction_df,
                  maf_threshold_interaction=args.maf_threshold_interaction if has_int else 0.0,
                  group_s=group_s, window=args.window, maf_threshold=maf_threshold,
                  run_eigenmt=has_int, write_top=True, write_stats=not args.best_only,
                  out_dir=args.output_dir, out_prefix=f'{args.prefix}.cis_qtl_pairs',
                  top_assoc_path=os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_top_assoc.txt.gz'),
                  logger=logger, device=device, min_ancestry_ac=args.min_ancestry_ac, **la_kwargs)
        if pgr is None:
            map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, **kw)
            if args.cis_output is not None:
                cis_df = from_tensorqtl_columns(pd.read_csv(args.cis_output, sep='\t'))
                signif_df = get_significant_pairs(
                    cis_df, os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_pairs.*.parquet'),
                    group_s=group_s, fdr=args.fdr, logger=logger)
                signif_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.cis_qtl.signif_pairs.parquet'))
        else:
            chunks = []
            for gt, var, p, ppos, ci in _chunks():
                prefix = f"{args.prefix}.chunk{ci + 1}"
                chunks.append(prefix)
                ckw = dict(kw, out_prefix=f'{prefix}.cis_qtl_pairs',
                           top_assoc_path=os.path.join(args.output_dir, f'{prefix}.cis_qtl_top_assoc.txt.gz'))
                map_nominal(gt, var, p, ppos, **ckw)
            chunk_files = glob.glob(os.path.join(args.output_dir, f"{args.prefix}.chunk*.cis_qtl_pairs.*.parquet"))
            if args.chunk_size == 'chr':
                for f in chunk_files:
                    x = re.findall(rf"{re.escape(args.prefix)}\.(chunk\d+)", os.path.basename(f))
                    os.rename(f, f.replace(f"{x[0]}.", ""))
            elif chunk_files:
                chunk_df = pd.DataFrame({
                    'file': chunk_files,
                    'chunk': [int(re.findall(rf"{re.escape(args.prefix)}\.chunk(\d+)", os.path.basename(i))[0]) for i in chunk_files],
                    'chr': [re.findall(r"\.cis_qtl_pairs\.(.*)\.parquet", os.path.basename(i))[0] for i in chunk_files],
                }).sort_values('chunk')
                for chrom, chr_df in chunk_df.groupby('chr', sort=False):
                    pd.concat([pd.read_parquet(f) for f in chr_df['file']]).reset_index(drop=True).to_parquet(
                        os.path.join(args.output_dir, f"{args.prefix}.cis_qtl_pairs.{chrom}.parquet"))
                    for f in chr_df['file']:
                        os.remove(f)
            if has_int:
                tops = [os.path.join(args.output_dir, f"{c}.cis_qtl_top_assoc.txt.gz") for c in chunks]
                tops = [t for t in tops if os.path.exists(t)]
                if tops:
                    pd.concat([pd.read_csv(f, sep='\t', index_col=0, dtype=str) for f in tops]).to_csv(
                        os.path.join(args.output_dir, f"{args.prefix}.cis_qtl_top_assoc.txt.gz"), sep='\t')
                    for f in tops:
                        os.remove(f)
        if has_int and not args.localqtl_names:
            path = os.path.join(args.output_dir, f'{args.prefix}.cis_qtl_top_assoc.txt.gz')
            if os.path.exists(path):
                top = pd.read_csv(path, sep='\t', index_col=0)
                to_tensorqtl_columns(top).to_csv(path, sep='\t', float_format='%.6g')

    elif args.mode == 'cis_independent':
        summary_df = from_tensorqtl_columns(pd.read_csv(args.cis_output, sep='\t'))
        summary_df.rename(columns={'minor_allele_samples': 'ma_samples', 'minor_allele_count': 'ma_count'}, inplace=True)
        if 'phenotype_id' not in summary_df.columns:  # tensorQTL writes phenotype_id as the index
            summary_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
            summary_df.index.name = 'phenotype_id'
            summary_df = from_tensorqtl_columns(summary_df.reset_index())
        kw = dict(covariates_df=covariates_df, group_s=group_s, fdr=args.fdr, nperm=args.permutations,
                  window=args.window, maf_threshold=maf_threshold, logger=logger, seed=args.seed,
                  device=device, beta_approx=not args.disable_beta_approx, perm_chunk=args.perm_chunk,
                  paired_covariate_df=paired_covariate_df, **la_kwargs)
        if pgr is None:
            res_df = map_independent(genotype_df, variant_df, summary_df, phenotype_df, phenotype_pos_df, **kw)
        else:
            res_df = pd.concat([map_independent(gt, var, summary_df, p, ppos, **kw)
                                for gt, var, p, ppos, _ in _chunks()], ignore_index=True)
        logger.write('  * writing output')
        _write_tsv(_names(res_df, args), os.path.join(args.output_dir, f'{args.prefix}.cis_independent_qtl.txt.gz'),
                   index=False)

    elif args.mode == 'cis_conditional':
        signals_df = _read_table(args.cis_output)
        if 'phenotype_id' not in signals_df.columns:
            signals_df = _read_table(args.cis_output, index_col=0)
            signals_df.index.name = 'phenotype_id'
            signals_df = signals_df.reset_index()
        signals_df = signals_df[[c for c in ('phenotype_id', 'variant_id', 'rank')
                                 if c in signals_df.columns]]
        res_df = map_conditional_dosage(
            genotype_df, variant_df, phenotype_df, phenotype_pos_df, signals_df, haplotypes,
            covariates_df=covariates_df, condition_on=args.condition_on,
            min_ancestry_ac=args.min_ancestry_ac, device=device, logger=logger,
        )
        logger.write('  * writing output')
        _write_tsv(_names(res_df, args),
                   os.path.join(args.output_dir, f'{args.prefix}.cis_conditional_qtl.txt.gz'),
                   index=False)

    elif args.mode == 'cis_susie':
        if args.cis_output is None:
            raise ValueError("--cis_output (cis results with q-values, or significant pairs) is required for cis_susie.")
        signif_df = _read_table(args.cis_output)
        if 'qval' in signif_df:
            signif_df = signif_df[signif_df['qval'] <= args.fdr]
        if 'phenotype_id' not in signif_df.columns:
            signif_df = pd.read_csv(args.cis_output, sep='\t', index_col=0).reset_index(names='phenotype_id')
            if 'qval' in signif_df:
                signif_df = signif_df[signif_df['qval'] <= args.fdr]
        phenotype_ids = phenotype_df.index[phenotype_df.index.isin(signif_df['phenotype_id'].unique())]
        phenotype_df = phenotype_df.loc[phenotype_ids]
        phenotype_pos_df = phenotype_pos_df.loc[phenotype_ids]
        kw = dict(covariates_df=covariates_df, paired_covariate_df=paired_covariate_df,
                  L=args.max_effects, maf_threshold=maf_threshold, max_iter=500,
                  window=args.window, summary_only=False, logger=logger, device=device)
        if pgr is None:
            summary_df, res = susie.map(genotype_df, variant_df, phenotype_df, phenotype_pos_df, **kw)
        else:
            summary_df, res = [], {}
            for gt, var, p, ppos, _ in _chunks():
                s, r = susie.map(gt, var, p, ppos, **kw)
                summary_df.append(s); res |= r
            summary_df = pd.concat(summary_df).reset_index(drop=True)
        summary_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.SuSiE_summary.parquet'))
        with open(os.path.join(args.output_dir, f'{args.prefix}.SuSiE.pickle'), 'wb') as f:
            pickle.dump(res, f)

    elif args.mode == 'trans_susie':
        if args.susie_loci is None:
            raise ValueError("--susie_loci is required for trans_susie.")
        locus_df = _read_table(args.susie_loci)
        locus_df.rename(columns={'position': 'pos'}, inplace=True)
        if pgr is not None:
            raise NotImplementedError("trans_susie does not support --chunk_size.")
        if variant_df is None:
            raise ValueError("trans_susie requires genotypes with variant positions.")
        summary_df, res = susie.map_loci(locus_df, genotype_df, variant_df, phenotype_df, covariates_df,
                                         maf_threshold=maf_threshold, max_iter=500, window=args.window,
                                         logger=logger, device=device)
        summary_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.SuSiE_summary.parquet'))
        with open(os.path.join(args.output_dir, f'{args.prefix}.SuSiE.pickle'), 'wb') as f:
            pickle.dump(res, f)

    elif args.mode == 'trans':
        return_sparse = not args.return_dense
        if return_sparse:
            logger.write(f'  * p-value threshold: {args.pval_threshold:.2g}')
        interaction_s = None
        if interaction_df is not None:
            if interaction_df.shape[1] > 1:
                raise NotImplementedError('trans-QTL mapping currently only supports a single interaction.')
            interaction_s = interaction_df.iloc[:, 0]
        kw = dict(covariates_df=covariates_df, interaction_s=interaction_s, return_sparse=return_sparse,
                  pval_threshold=args.pval_threshold, maf_threshold=maf_threshold, batch_size=args.batch_size,
                  return_r2=args.return_r2, logp=args.logp, logger=logger, device=device)
        if haplotypes is not None:
            kw.update(haplotypes=haplotypes, variant_df=variant_df if pgr is None else None)
        if pgr is None:
            pairs_df = trans.map_trans(genotype_df, phenotype_df, **kw)
        else:
            if haplotypes is not None:
                raise NotImplementedError("Local ancestry in trans mode requires in-memory genotypes (no --chunk_size).")
            parts = []
            bounds = _trans_chunk_bounds(pgr.num_variants, int(args.chunk_size))
            for i in range(len(bounds) - 1):
                print(f"Processing genotype chunk {i + 1}/{len(bounds) - 1}")
                arr = pgr.read_range(int(bounds[i]), int(bounds[i + 1]) - 1, impute_mean=False, dosages=args.dosages,
                                     dtype=np.float32 if args.dosages else np.int8)
                if torch.is_tensor(arr):
                    arr = arr.detach().cpu().numpy()
                gt_df = pd.DataFrame(arr, index=pgr.variant_ids[int(bounds[i]):int(bounds[i + 1])],
                                     columns=pgr.sample_ids)
                parts.append(trans.map_trans(gt_df, phenotype_df, **kw))
            if return_sparse:
                pairs_df = pd.concat(parts).reset_index(drop=True)
            else:
                pairs_df = tuple(pd.concat([p[k] for p in parts]) for k in range(len(parts[0])))
            variant_df = pgr.variant_df
        if return_sparse:
            if variant_df is not None and phenotype_pos_df is not None:
                logger.write('  * filtering out cis-QTLs (within +/-5Mb)')
                pairs_df = trans.filter_cis(pairs_df, phenotype_pos_df, variant_df, window=5_000_000)
            logger.write('  * writing sparse output')
            pairs_df = _names(pairs_df, args)
            if not args.output_text:
                pairs_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.parquet'))
            else:
                _write_tsv(pairs_df, os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pairs.txt.gz'), index=False)
        else:
            logger.write('  * writing dense output')
            if interaction_s is None:
                pval_df, b_df, b_se_df, af_s = pairs_df
                pval_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_pval.parquet'))
                b_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_beta.parquet'))
                b_se_df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_beta_se.parquet'))
                af_s.to_frame().to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_af.parquet'))
            else:
                pval_g_df, pval_i_df, pval_gi_df, af_s, ma_samples_s, ma_count_s = pairs_df
                for name, df in (('pval_g', pval_g_df), ('pval_i', pval_i_df), ('pval_gi', pval_gi_df)):
                    df.to_parquet(os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_{name}.parquet'))
                pd.concat([af_s, ma_samples_s, ma_count_s], axis=1).to_parquet(
                    os.path.join(args.output_dir, f'{args.prefix}.trans_qtl_af.parquet'))

    logger.write(f'[{datetime.now().strftime("%b %d %H:%M:%S")}] Finished mapping')
    logger.close()
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
