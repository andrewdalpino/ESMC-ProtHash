import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, LinearLR
from torch.utils.data import DataLoader, random_split
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from torchdata.stateful_dataloader import StatefulDataLoader

from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC

from src.prothash.model import ESMCProtHash

from data import UniRef50, LengthBucketBatchSampler, SortedLengthBatchSampler
from loss import DecomposedTokenRepresentationLoss, WeightedCombinedLoss
from metrics import CosineSimilarity, LinearCKA, Top1MacroF1

from tqdm import tqdm

AVAILABLE_TEACHERS = {"esmc_300m", "esmc_600m"}

TEACHER_LAYER_ANCHOR_POINTS = {
    "esmc_300m": (7, 15, 22, 29),
    "esmc_600m": (8, 18, 27, 35),
}


def main():
    parser = ArgumentParser(
        description="Distill a larger ESMC model into a smaller one."
    )

    parser.add_argument(
        "--teacher_name", choices=AVAILABLE_TEACHERS, default="esmc_300m"
    )

    parser.add_argument("--dataset_path", default="dataset/uniref50.fasta", type=str)
    parser.add_argument("--num_length_buckets", default=100, type=int)
    parser.add_argument("--num_dataset_processes", default=2, type=int)
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--quantization_aware_training", action="store_true")
    parser.add_argument("--quant_group_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--anneal_learning_rate", action="store_true")
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--max_steps", default=150000, type=int)
    parser.add_argument("--stage1_direction_weight", default=0.25, type=float)
    parser.add_argument("--stage1_magnitude_weight", default=0.0025, type=float)
    parser.add_argument("--stage2_direction_weight", default=0.5, type=float)
    parser.add_argument("--stage2_magnitude_weight", default=0.005, type=float)
    parser.add_argument("--stage3_direction_weight", default=0.75, type=float)
    parser.add_argument("--stage3_magnitude_weight", default=0.0075, type=float)
    parser.add_argument("--stage4_direction_weight", default=1.0, type=float)
    parser.add_argument("--stage4_magnitude_weight", default=0.01, type=float)
    parser.add_argument("--stage5_direction_weight", default=0.1, type=float)
    parser.add_argument("--stage5_magnitude_weight", default=0.001, type=float)
    parser.add_argument("--loss_norm_epsilon", default=1e-8, type=float)
    parser.add_argument("--embedding_dimensions", default=512, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--hidden_ratio", default=4, type=int)
    parser.add_argument("--num_stage1_layers", default=3, type=int)
    parser.add_argument("--num_stage2_layers", default=3, type=int)
    parser.add_argument("--num_stage3_layers", default=3, type=int)
    parser.add_argument("--num_stage4_layers", default=3, type=int)
    parser.add_argument("--eval_interval", default=200, type=int)
    parser.add_argument("--num_eval_samples", default=10000, type=int)
    parser.add_argument("--checkpoint_interval", default=200, type=int)

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.max_sequence_length > 2048:
        raise ValueError(
            f"Maximum sequence length cannot exceed 2048, {args.max_sequence_length} given."
        )

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.max_steps < 1:
        raise ValueError(f"Must train for at least 1 step, {args.max_steps} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.num_eval_samples < 1:
        raise ValueError(
            f"Number of evaluation samples must be greater than 0, {args.num_eval_samples} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = EsmSequenceTokenizer()

    dataset = UniRef50(
        path=args.dataset_path,
        tokenizer=tokenizer,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )

    num_training_samples = len(dataset) - args.num_eval_samples

    training, testing = random_split(
        dataset, (num_training_samples, args.num_eval_samples)
    )

    bucket_sampler = LengthBucketBatchSampler(
        training, args.batch_size, args.num_length_buckets
    )

    train_loader = StatefulDataLoader(
        training,
        batch_sampler=bucket_sampler,
        collate_fn=dataset.collate_pad_right,
        pin_memory="cuda" in args.device,
        snapshot_every_n_steps=args.gradient_accumulation_steps,
        num_workers=args.num_dataset_processes,
    )

    sorted_length_sampler = SortedLengthBatchSampler(testing, args.batch_size)

    test_loader = DataLoader(
        testing,
        batch_sampler=sorted_length_sampler,
        collate_fn=dataset.collate_pad_right,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    teacher = ESMC.from_pretrained(args.teacher_name)

    # Freeze teacher model parameters.
    teacher.requires_grad_(False)

    teacher = teacher.to(args.device)

    teacher.eval()

    anchor_points = TEACHER_LAYER_ANCHOR_POINTS[args.teacher_name]

    print("Teacher model loaded successfully")

    model_args = {
        "vocabulary_size": tokenizer.vocab_size,
        "padding_index": tokenizer.pad_token_id,
        "context_length": args.max_sequence_length,
        "teacher_dimensions": teacher.embed.embedding_dim,
        "embedding_dimensions": args.embedding_dimensions,
        "num_attention_heads": args.num_attention_heads,
        "hidden_ratio": args.hidden_ratio,
        "num_stage1_layers": args.num_stage1_layers,
        "num_stage2_layers": args.num_stage2_layers,
        "num_stage3_layers": args.num_stage3_layers,
        "num_stage4_layers": args.num_stage4_layers,
    }

    student = ESMCProtHash(**model_args)

    student.add_sequence_head()

    if args.quantization_aware_training:
        student.add_fake_quantized_tensors(args.quant_group_size)

    student = student.to(args.device)

    print(f"Number of parameters: {student.num_params:,}")

    loss_function = DecomposedTokenRepresentationLoss(args.loss_norm_epsilon)

    combined_loss_function = WeightedCombinedLoss(
        [
            args.stage1_direction_weight,
            args.stage1_magnitude_weight,
            args.stage2_direction_weight,
            args.stage2_magnitude_weight,
            args.stage3_direction_weight,
            args.stage3_magnitude_weight,
            args.stage4_direction_weight,
            args.stage4_magnitude_weight,
            args.stage5_direction_weight,
            args.stage5_magnitude_weight,
        ]
    )

    combined_loss_function = combined_loss_function.to(args.device)

    optimizer = AdamW(student.parameters(), lr=args.learning_rate)

    step = 1

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

        train_loader.load_state_dict(checkpoint["train_loader"])

        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        step += checkpoint["step"]

        print("Previous checkpoint resumed successfully")

    if args.anneal_learning_rate:
        total_iters = args.max_steps - step

        scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
        )
    else:
        scheduler = ConstantLR(optimizer, factor=1.0)

    student.train()

    stage1_cosine_similarity_metric = CosineSimilarity()
    stage2_cosine_similarity_metric = CosineSimilarity()
    stage3_cosine_similarity_metric = CosineSimilarity()
    stage4_cosine_similarity_metric = CosineSimilarity()

    stage1_linear_cka_metric = LinearCKA()
    stage2_linear_cka_metric = LinearCKA()
    stage3_linear_cka_metric = LinearCKA()
    stage4_linear_cka_metric = LinearCKA()

    f1_metric = Top1MacroF1()

    new_progress_bar = partial(
        tqdm,
        total=args.gradient_accumulation_steps,
        leave=False,
    )

    total_stage1_direction_loss, total_stage1_magnitude_loss = 0.0, 0.0
    total_stage2_direction_loss, total_stage2_magnitude_loss = 0.0, 0.0
    total_stage3_direction_loss, total_stage3_magnitude_loss = 0.0, 0.0
    total_stage4_direction_loss, total_stage4_magnitude_loss = 0.0, 0.0
    total_stage5_direction_loss, total_stage5_magnitude_loss = 0.0, 0.0

    num_batches = 0

    print("Distilling ...")

    progress_bar = new_progress_bar(desc=f"Step {step:,}")

    for index, x in enumerate(train_loader, start=1):
        x = x.to(args.device, non_blocking=True)

        mask = x != tokenizer.pad_token_id

        with amp_context:
            with torch.no_grad():
                out_teacher = teacher.forward(x)

            y1_student, y2_student, y3_student, y4_student, y5_student = (
                student.forward_with_adapters(x)
            )

            assert (
                out_teacher.hidden_states is not None
            ), "Teacher model must return hidden states."

            y1_teacher = out_teacher.hidden_states[anchor_points[0]]
            y2_teacher = out_teacher.hidden_states[anchor_points[1]]
            y3_teacher = out_teacher.hidden_states[anchor_points[2]]
            y4_teacher = out_teacher.hidden_states[anchor_points[3]]

            # Trim off unused vocabulary tokens.
            y5_teacher = out_teacher.sequence_logits[..., : tokenizer.vocab_size]

            stage1_direction_loss, stage1_magnitude_loss = loss_function.forward(
                y1_student, y1_teacher, mask
            )

            stage2_direction_loss, stage2_magnitude_loss = loss_function.forward(
                y2_student, y2_teacher, mask
            )

            stage3_direction_loss, stage3_magnitude_loss = loss_function.forward(
                y3_student, y3_teacher, mask
            )

            stage4_direction_loss, stage4_magnitude_loss = loss_function.forward(
                y4_student, y4_teacher, mask
            )

            stage5_direction_loss, stage5_magnitude_loss = loss_function.forward(
                y5_student, y5_teacher, mask
            )

            combined_loss = combined_loss_function.forward(
                torch.stack(
                    [
                        stage1_direction_loss,
                        stage1_magnitude_loss,
                        stage2_direction_loss,
                        stage2_magnitude_loss,
                        stage3_direction_loss,
                        stage3_magnitude_loss,
                        stage4_direction_loss,
                        stage4_magnitude_loss,
                        stage5_direction_loss,
                        stage5_magnitude_loss,
                    ]
                )
            )

            scaled_loss = combined_loss / args.gradient_accumulation_steps

        scaled_loss.backward()

        total_stage1_direction_loss += stage1_direction_loss.item()
        total_stage1_magnitude_loss += stage1_magnitude_loss.item()
        total_stage2_direction_loss += stage2_direction_loss.item()
        total_stage2_magnitude_loss += stage2_magnitude_loss.item()
        total_stage3_direction_loss += stage3_direction_loss.item()
        total_stage3_magnitude_loss += stage3_magnitude_loss.item()
        total_stage4_direction_loss += stage4_direction_loss.item()
        total_stage4_magnitude_loss += stage4_magnitude_loss.item()
        total_stage5_direction_loss += stage5_direction_loss.item()
        total_stage5_magnitude_loss += stage5_magnitude_loss.item()

        num_batches += 1

        progress_bar.update(1)

        if index % args.gradient_accumulation_steps == 0:
            norm = clip_grad_norm_(student.parameters(), args.max_gradient_norm)

            optimizer.step()

            optimizer.zero_grad()

            scheduler.step()

            progress_bar.close()

            average_stage1_direction_loss = total_stage1_direction_loss / num_batches
            average_stage1_magnitude_loss = total_stage1_magnitude_loss / num_batches
            average_stage2_direction_loss = total_stage2_direction_loss / num_batches
            average_stage2_magnitude_loss = total_stage2_magnitude_loss / num_batches
            average_stage3_direction_loss = total_stage3_direction_loss / num_batches
            average_stage3_magnitude_loss = total_stage3_magnitude_loss / num_batches
            average_stage4_direction_loss = total_stage4_direction_loss / num_batches
            average_stage4_magnitude_loss = total_stage4_magnitude_loss / num_batches
            average_stage5_direction_loss = total_stage5_direction_loss / num_batches
            average_stage5_magnitude_loss = total_stage5_magnitude_loss / num_batches

            gradient_norm = norm.item()

            logger.add_scalar(
                "Stage 1 Direction L2", average_stage1_direction_loss, step
            )

            logger.add_scalar(
                "Stage 1 Magnitude L2", average_stage1_magnitude_loss, step
            )

            logger.add_scalar(
                "Stage 2 Direction L2", average_stage2_direction_loss, step
            )

            logger.add_scalar(
                "Stage 2 Magnitude L2", average_stage2_magnitude_loss, step
            )

            logger.add_scalar(
                "Stage 3 Direction L2", average_stage3_direction_loss, step
            )

            logger.add_scalar(
                "Stage 3 Magnitude L2", average_stage3_magnitude_loss, step
            )

            logger.add_scalar(
                "Stage 4 Direction L2", average_stage4_direction_loss, step
            )

            logger.add_scalar(
                "Stage 4 Magnitude L2", average_stage4_magnitude_loss, step
            )

            logger.add_scalar(
                "Stage 5 Direction L2", average_stage5_direction_loss, step
            )

            logger.add_scalar(
                "Stage 5 Magnitude L2", average_stage5_magnitude_loss, step
            )

            logger.add_scalar("Gradient Norm", gradient_norm, step)

            print(
                f"Step {step:,}:",
                f"Stage 1 Direction L2: {average_stage1_direction_loss:.5f},",
                f"Stage 1 Magnitude L2: {average_stage1_magnitude_loss:.5f},",
                f"Stage 2 Direction L2: {average_stage2_direction_loss:.5f},",
                f"Stage 2 Magnitude L2: {average_stage2_magnitude_loss:.5f},",
                f"Stage 3 Direction L2: {average_stage3_direction_loss:.5f},",
                f"Stage 3 Magnitude L2: {average_stage3_magnitude_loss:.5f},",
                f"Stage 4 Direction L2: {average_stage4_direction_loss:.5f},",
                f"Stage 4 Magnitude L2: {average_stage4_magnitude_loss:.5f},",
                f"Stage 5 Direction L2: {average_stage5_direction_loss:.5f},",
                f"Stage 5 Magnitude L2: {average_stage5_magnitude_loss:.5f},",
                f"Gradient Norm: {gradient_norm:.5f}",
            )

            total_stage1_direction_loss, total_stage1_magnitude_loss = 0.0, 0.0
            total_stage2_direction_loss, total_stage2_magnitude_loss = 0.0, 0.0
            total_stage3_direction_loss, total_stage3_magnitude_loss = 0.0, 0.0
            total_stage4_direction_loss, total_stage4_magnitude_loss = 0.0, 0.0
            total_stage5_direction_loss, total_stage5_magnitude_loss = 0.0, 0.0

            num_batches = 0

            if step % args.eval_interval == 0:
                student.eval()

                for x in tqdm(test_loader, desc="Testing", leave=False):
                    x = x.to(args.device, non_blocking=True)

                    mask = x != tokenizer.pad_token_id

                    with torch.inference_mode():
                        out_teacher = teacher.forward(x)

                    assert (
                        out_teacher.hidden_states is not None
                    ), "Teacher model must return hidden states."

                    y1_teacher = out_teacher.hidden_states[anchor_points[0]]
                    y2_teacher = out_teacher.hidden_states[anchor_points[1]]
                    y3_teacher = out_teacher.hidden_states[anchor_points[2]]
                    y4_teacher = out_teacher.hidden_states[anchor_points[3]]

                    # Trim off unused vocabulary tokens.
                    y5_teacher = out_teacher.sequence_logits[
                        ..., : tokenizer.vocab_size
                    ]

                    with torch.no_grad():
                        z1, z2, z3, z4, logits = student.forward_with_adapters(x)

                    stage1_cosine_similarity_metric.update(z1, y1_teacher, mask)

                    stage2_cosine_similarity_metric.update(z2, y2_teacher, mask)

                    stage3_cosine_similarity_metric.update(z3, y3_teacher, mask)

                    stage4_cosine_similarity_metric.update(z4, y4_teacher, mask)

                    stage1_linear_cka_metric.update(z1, y1_teacher, mask)
                    stage2_linear_cka_metric.update(z2, y2_teacher, mask)
                    stage3_linear_cka_metric.update(z3, y3_teacher, mask)
                    stage4_linear_cka_metric.update(z4, y4_teacher, mask)

                    f1_metric.update(logits, y5_teacher, mask)

                average_stage1_cosine_similarity = (
                    stage1_cosine_similarity_metric.compute()
                )

                average_stage2_cosine_similarity = (
                    stage2_cosine_similarity_metric.compute()
                )

                average_stage3_cosine_similarity = (
                    stage3_cosine_similarity_metric.compute()
                )

                average_stage4_cosine_similarity = (
                    stage4_cosine_similarity_metric.compute()
                )

                average_stage1_linear_cka = stage1_linear_cka_metric.compute()
                average_stage2_linear_cka = stage2_linear_cka_metric.compute()
                average_stage3_linear_cka = stage3_linear_cka_metric.compute()
                average_stage4_linear_cka = stage4_linear_cka_metric.compute()

                f1, precision, recall = f1_metric.compute()

                logger.add_scalar(
                    "Stage 1 Cosine Similarity", average_stage1_cosine_similarity, step
                )

                logger.add_scalar(
                    "Stage 2 Cosine Similarity", average_stage2_cosine_similarity, step
                )

                logger.add_scalar(
                    "Stage 3 Cosine Similarity", average_stage3_cosine_similarity, step
                )

                logger.add_scalar(
                    "Stage 4 Cosine Similarity", average_stage4_cosine_similarity, step
                )

                logger.add_scalar("Stage 1 CKA", average_stage1_linear_cka, step)
                logger.add_scalar("Stage 2 CKA", average_stage2_linear_cka, step)
                logger.add_scalar("Stage 3 CKA", average_stage3_linear_cka, step)
                logger.add_scalar("Stage 4 CKA", average_stage4_linear_cka, step)
                logger.add_scalar("F1 Score", f1, step)
                logger.add_scalar("Precision", precision, step)
                logger.add_scalar("Recall", recall, step)

                print(
                    f"Stage 1 Cosine Similarity: {average_stage1_cosine_similarity:.4f},",
                    f"Stage 2 Cosine Similarity: {average_stage2_cosine_similarity:.4f},",
                    f"Stage 3 Cosine Similarity: {average_stage3_cosine_similarity:.4f},",
                    f"Stage 4 Cosine Similarity: {average_stage4_cosine_similarity:.4f}",
                )

                print(
                    f"Stage 1 CKA: {average_stage1_linear_cka:.4f},",
                    f"Stage 2 CKA: {average_stage2_linear_cka:.4f},",
                    f"Stage 3 CKA: {average_stage3_linear_cka:.4f},",
                    f"Stage 4 CKA: {average_stage4_linear_cka:.4f}",
                )

                print(
                    f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
                )

                stage1_cosine_similarity_metric.reset()
                stage2_cosine_similarity_metric.reset()
                stage3_cosine_similarity_metric.reset()
                stage4_cosine_similarity_metric.reset()

                stage1_linear_cka_metric.reset()
                stage2_linear_cka_metric.reset()
                stage3_linear_cka_metric.reset()
                stage4_linear_cka_metric.reset()

                f1_metric.reset()

                student.train()

            if step % args.checkpoint_interval == 0:
                checkpoint = {
                    "step": step,
                    "train_loader": train_loader.state_dict(),
                    "model_args": model_args,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                torch.save(checkpoint, args.checkpoint_path)

                print("Checkpoint saved")

            if step >= args.max_steps:
                break

            step += 1

            progress_bar = new_progress_bar(desc=f"Step {step:,}")

    logger.close()

    print("Done!")


if __name__ == "__main__":
    main()
