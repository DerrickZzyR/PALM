import argparse
import random
from pathlib import Path

import numpy as np
import torch
from config import CRPModelConfig,CRPTrainConfig,TCCMModelConfig,TCCMTrainConfig,DataConfig,SSEModelConfig,SSETrainConfig
from src.engine import test_crp, test_sse, train_crp, train_tccm, train_sse


def setup_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    """Build the single entry-point parser, grouped by model and function."""
    parser = argparse.ArgumentParser(
        description="PALM training, evaluation, and evidence-chain entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False,
    )

    # ==========================
    # Core runtime and actions
    # ==========================
    core = parser.add_argument_group("Core runtime and actions")
    core.add_argument("--device",default="cuda")
    core.add_argument("--seed",type=int,default=42)
    core.add_argument("--batch_size",type=int,default=32)
    core.add_argument("--train_tccm",type=int,choices=[0, 1],default=0)
    core.add_argument("--train_sse",type=int,choices=[0, 1],default=1)
    core.add_argument("--train_crp",type=int,choices=[0, 1],default=0)
    core.add_argument("--test_sse",type=int,choices=[0, 1],default=0)
    core.add_argument("--test_crp",type=int,choices=[0, 1],default=0)

    # ==========================
    # Data and evaluation
    # ==========================
    data = parser.add_argument_group("Data and evaluation")
    data.add_argument("--data_path",type=Path,default="")
    data.add_argument("--text_path",type=Path,default="")
    data.add_argument("--multi_port_path",type=Path,default="")
    data.add_argument("--splits",nargs="+",choices=["train", "test"],default=["train", "test"])
    data.add_argument("--threshold", type=float, default=0.5)
    data.add_argument("--time_steps", type=int, default=200)
    data.add_argument("--num_channels", type=int, default=15)
    data.add_argument("--num_classes", type=int, default=2)

    # ==========================
    # TCCM architecture and optimization
    # ==========================
    tccm = parser.add_argument_group("TCCM: Topology-Constrained Causal Model")
    tccm.add_argument("--win_size", type=int, default=6)
    tccm.add_argument("--tccm_epochs", type=int, default=500)
    tccm.add_argument("--tccm_learning_rate",type=float,default=5e-2,)
    tccm.add_argument("--group_penalty", type=float, default=0.05)
    tccm.add_argument("--ridge_penalty", type=float, default=1e-2)
    tccm.add_argument("--penalty", choices=["H", "GL", "GSGL"], default="GSGL")
    tccm.add_argument("--use_topology_mask",type=int,choices=[0, 1],default=1)
    tccm.add_argument("--topology_mask_path",type=Path,default=Path("topology_mask_ali.json"))
    tccm.add_argument("--topology_lam", type=float, default=1e-4)

    # ==========================
    # SSE
    # ==========================
    sse = parser.add_argument_group("SSE: StateSignatureExtractor")
    sse.add_argument("--embedding_dim", type=int, default=128)
    sse.add_argument("--shape_size", type=int, default=4)
    sse.add_argument("--shape_stride", type=int, default=2)
    sse.add_argument("--sparse_rate", type=float, default=0.5)
    sse.add_argument("--num_experts", type=int, default=4)
    sse.add_argument("--alpha", type=float, default=0.7)
    sse.add_argument("--attention_hidden_dim",type=int,default=8)
    sse.add_argument("--sse_dropout", type=float, default=0.15)
    sse.add_argument("--selector_gate_strength",type=float,default=0.0,)
    sse.add_argument("--sse_revin", type=int, choices=[0, 1], default=1)
    sse.add_argument("--sse_affine", type=int, choices=[0, 1], default=0)
    sse.add_argument("--sse_subtract_last", type=int, choices=[0, 1], default=0)
    sse.add_argument("--sse_epochs", type=int, default=100)
    sse.add_argument("--warmup_epochs", type=int, default=40)
    sse.add_argument("--sse_learning_rate",type=float,default=1e-4)
    sse.add_argument("--sse_weight_decay", type=float, default=0.0)
    sse.add_argument("--moe_loss_weight", type=float, default=1e-3)
    sse.add_argument("--selector_loss_weight",type=float,default=0.1)
    sse.add_argument("--warmup_start_ratio",type=float,default=0.1)
    sse.add_argument("--plateau_factor", type=float, default=0.5)
    sse.add_argument("--plateau_patience",type=int,default=10)
    sse.add_argument("--sse_minimum_learning_rate",type=float,default=1e-6)
    sse.add_argument("--sse_gradient_clip_norm",type=float,default=1.0)

    # ==========================
    # Co-Refinement Predictor
    # ==========================
    crp = parser.add_argument_group("CRP: Co-Refinement Predictor")
    crp.add_argument("--text_feature_tag", default="longclip_b_ctx248_trunc_model_gpt_5")
    crp.add_argument("--crp_hidden_dim", type=int, default=512)
    crp.add_argument("--crp_num_heads", type=int, default=4)
    crp.add_argument("--crp_dropout", type=float, default=0.3)
    crp.add_argument("--crp_multi_ports", type=int, default=8)
    crp.add_argument("--crp_multi_input_channels",type=int,default=15)
    crp.add_argument("--crp_multi_hidden_dim",type=int,default=64)
    crp.add_argument("--crp_multi_context_dim",type=int,default=128)
    crp.add_argument("--crp_multi_dilations",type=int,nargs="+",default=[1, 2, 4, 8, 16, 32])
    crp.add_argument("--crp_multi_dropout", type=float, default=0.2)
    crp.add_argument("--crp_multi_revin_affine",type=int,choices=[0, 1],default=1)
    crp.add_argument("--crp_multi_revin_eps",type=float,default=1e-5)
    crp.add_argument("--crp_epochs", type=int, default=50)
    crp.add_argument("--crp_learning_rate",type=float,default=1e-4)
    crp.add_argument("--crp_weight_decay", type=float, default=0.0)
    crp.add_argument("--crp_restart_period", type=int, default=10)
    crp.add_argument("--crp_restart_multiplier",type=int,default=2)
    crp.add_argument("--crp_minimum_learning_rate",type=float,default=1e-6)
    crp.add_argument("--crp_gradient_clip_norm",type=float,default=1.0)

    # ==========================
    # Evidence Chain Generation
    # ==========================
    evidence = parser.add_argument_group("Evidence Chain Generation")
    evidence.add_argument("--gene_des",type=int,choices=[0, 1],default=0)
    evidence.add_argument("--llm_model", default="gpt-5")
    evidence.add_argument("--llm_base_url", default=None)
    evidence.add_argument("--llm_api_key",default=None)
    evidence.add_argument("--llm_temperature", type=float, default=0.1)
    evidence.add_argument("--llm_max_tokens", type=int, default=400)
    evidence.add_argument("--llm_retries", type=int, default=3)
    evidence.add_argument("--evidence_context_points",type=int,default=8,)
    evidence.add_argument("--evidence_max_tiles", type=int, default=4)
    evidence.add_argument("--evidence_merge_gap", type=int, default=2)
    evidence.add_argument("--evidence_min_interval_points",type=int,default=3,)
    evidence.add_argument("--evidence_y_floor", type=float, default=5.0)
    evidence.add_argument("--feature_names_json",type=Path,default=None,)
    evidence.add_argument("--health_prior_path",type=Path,default=Path("healthy_causal_relations.json"))
    evidence.add_argument("--evidence_output",type=Path,default=Path("dataset/evidence_chain.json"))
    evidence.add_argument("--description_model_tag",default="")
    evidence.add_argument("--evidence_save_every", type=int, default=50)
    evidence.add_argument("--gene_splits",nargs="+",choices=["train", "test"],default=["train", "test"],)

    # ==========================
    # Checkpoints
    # ==========================
    checkpoints = parser.add_argument_group("Checkpoints")
    checkpoints.add_argument("--tccm_checkpoint", type=Path, default="")
    checkpoints.add_argument("--sse_checkpoint", type=Path, default="")
    checkpoints.add_argument("--crp_checkpoint",type=Path,default="")
    return parser


def data_config_from_args(args: argparse.Namespace) -> DataConfig:
    return DataConfig(time_steps=args.time_steps,num_channels=args.num_channels,num_classes=args.num_classes)


def tccm_model_config_from_args(args: argparse.Namespace,) -> TCCMModelConfig:
    return TCCMModelConfig(lag=args.win_size)


def sse_model_config_from_args(args: argparse.Namespace) -> SSEModelConfig:
    return SSEModelConfig(
        embedding_dim=args.embedding_dim,
        shape_size=args.shape_size,
        shape_stride=args.shape_stride,
        sparse_rate=args.sparse_rate,
        num_experts=args.num_experts,
        alpha=args.alpha,
        attention_hidden_dim=args.attention_hidden_dim,
        dropout=args.sse_dropout,
        selector_gate_strength=args.selector_gate_strength,
        use_revin=bool(args.sse_revin),
        affine=bool(args.sse_affine),
        subtract_last=bool(args.sse_subtract_last),
    )


def crp_model_config_from_args(args: argparse.Namespace) -> CRPModelConfig:
    return CRPModelConfig(
        hidden_dim=args.crp_hidden_dim,
        num_heads=args.crp_num_heads,
        dropout=args.crp_dropout,
        multi_ports=args.crp_multi_ports,
        multi_input_channels=args.crp_multi_input_channels,
        multi_hidden_dim=args.crp_multi_hidden_dim,
        multi_context_dim=args.crp_multi_context_dim,
        multi_dilations=tuple(args.crp_multi_dilations),
        multi_dropout=args.crp_multi_dropout,
        multi_revin_affine=bool(args.crp_multi_revin_affine),
        multi_revin_eps=args.crp_multi_revin_eps,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_reproducibility(args.seed)
    data_config = data_config_from_args(args)
    tccm_model_config = tccm_model_config_from_args(args)

    if args.train_tccm == 1:
        tccm_train_config = TCCMTrainConfig(
            epochs=args.tccm_epochs,
            learning_rate=args.tccm_learning_rate,
            group_penalty=args.group_penalty,
            ridge_penalty=args.ridge_penalty,
            penalty=args.penalty,
            use_topology_mask=bool(args.use_topology_mask),
            topology_mask_path=args.topology_mask_path,
            topology_lam=args.topology_lam,
        )
        train_tccm(
            data_config=data_config,
            tccm_model_config=tccm_model_config,
            tccm_train_config=tccm_train_config,
            data_root=args.data_path,
            output_checkpoint=args.tccm_checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        return

    if args.train_sse == 1:
        sse_model_config = sse_model_config_from_args(args)
        sse_train_config = SSETrainConfig(
            epochs=args.sse_epochs,
            warmup_epochs=args.warmup_epochs,
            learning_rate=args.sse_learning_rate,
            weight_decay=args.sse_weight_decay,
            moe_loss_weight=args.moe_loss_weight,
            selector_loss_weight=args.selector_loss_weight,
            warmup_start_ratio=args.warmup_start_ratio,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            minimum_learning_rate=args.sse_minimum_learning_rate,
            gradient_clip_norm=args.sse_gradient_clip_norm,
        )
        train_sse(
            data_config=data_config,
            tccm_model_config=tccm_model_config,
            sse_model_config=sse_model_config,
            sse_train_config=sse_train_config,
            data_root=args.data_path,
            tccm_checkpoint=args.tccm_checkpoint,
            output_checkpoint=args.sse_checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        return

    if args.test_sse == 1:
        sse_model_config = sse_model_config_from_args(args)
        test_sse(
            data_config=data_config,
            tccm_model_config=tccm_model_config,
            sse_model_config=sse_model_config,
            data_root=args.data_path,
            tccm_checkpoint=args.tccm_checkpoint,
            sse_checkpoint=args.sse_checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            threshold=args.threshold,
        )
        return

    if args.train_crp == 1:
        sse_model_config = sse_model_config_from_args(args)
        crp_model_config = crp_model_config_from_args(args)
        crp_train_config = CRPTrainConfig(
            epochs=args.crp_epochs,
            learning_rate=args.crp_learning_rate,
            weight_decay=args.crp_weight_decay,
            restart_period=args.crp_restart_period,
            restart_multiplier=args.crp_restart_multiplier,
            minimum_learning_rate=args.crp_minimum_learning_rate,
            gradient_clip_norm=args.crp_gradient_clip_norm,
        )
        train_crp(
            data_config=data_config,
            tccm_model_config=tccm_model_config,
            sse_model_config=sse_model_config,
            crp_model_config=crp_model_config,
            crp_train_config=crp_train_config,
            data_root=args.data_path,
            tccm_checkpoint=args.tccm_checkpoint,
            sse_checkpoint=args.sse_checkpoint,
            crp_checkpoint=args.crp_checkpoint,
            text_root=args.text_path,
            text_feature_tag=args.text_feature_tag,
            batch_size=args.batch_size,
            device=args.device,
            multi_port_root=args.multi_port_path,
        )
        return

    if args.test_crp == 1:
        sse_model_config = sse_model_config_from_args(args)
        crp_model_config = crp_model_config_from_args(args)
        test_crp(
            data_config=data_config,
            tccm_model_config=tccm_model_config,
            sse_model_config=sse_model_config,
            crp_model_config=crp_model_config,
            data_root=args.data_path,
            tccm_checkpoint=args.tccm_checkpoint,
            sse_checkpoint=args.sse_checkpoint,
            crp_checkpoint=args.crp_checkpoint,
            text_root=args.text_path,
            text_feature_tag=args.text_feature_tag,
            batch_size=args.batch_size,
            device=args.device,
            threshold=args.threshold,
            multi_port_root=args.multi_port_path,
        )
        return


if __name__ == "__main__":
    main()
