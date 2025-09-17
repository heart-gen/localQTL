import torch
import pandas as pd

from .haplotypeio import InputGeneratorCis

def map_cis_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                    covariates_df=None, haplotype_reader=None, window=1_000_000,
                    nperm=None, device="cuda"):
    """
    Run nominal (and optionally permutation) cis-QTL mapping.
    """
    # Residualize phenotypes once
    if covariates_df is not None:
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        Y_resid = residualize_phenotypes(phenotype_df.values, covariates_df.values, device=device)
        phenotype_df_resid = pd.DataFrame(Y_resid, index=phenotype_df.index, columns=phenotype_df.columns)
    else:
        phenotype_df_resid = phenotype_df

    # Create cis-window generator
    igc = InputGeneratorCis(genotype_df, variant_df,
                            phenotype_df_resid, phenotype_pos_df,
                            window=window, haplotype_reader=haplotype_reader)

    results = []
    for pheno_id, y, G_block, var_df, H_block in igc.generate_data():
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Permuted phenotypes (if requested)
        if nperm is not None:
            y_perm = torch.stack([y_t[torch.randperm(len(y_t))] for _ in range(nperm)], dim=1)
        else:
            y_perm = None

        # Run batched regression
        betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
            y_t, G_t, H=H_t, y_perm=y_perm, device=device
        )

        df = pd.DataFrame({
            "phenotype_id": pheno_id,
            "variant_id": var_df.index.values,
            "tstat": tstats[:,0].cpu().numpy(),
            "beta": betas[:,0].cpu().numpy(),
            "se": ses[:,0].cpu().numpy(),
            "pos": var_df["pos"].values,
        })

        if nperm is not None:
            df["max_r2_perm"] = r2_perm.cpu().numpy().max()  # store empirical null

        results.append(df)

    return pd.concat(results, axis=0).reset_index(drop=True)


