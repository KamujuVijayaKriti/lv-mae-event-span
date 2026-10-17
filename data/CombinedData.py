import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
# 1280
class CombinedData(Dataset):
    def __init__(self, csv_file, duration_min=10, duration_max=1280, embedding_dim=1024, num_patches=120, mask_ratio=.4, importance_masking_strategy=False):
        """
        Args:
            csv_file (string): Path to the csv file.
            duration (int): Duration in seconds to determine the subset of paths.
        """
        self.df = pd.read_csv(csv_file)

        self.duration_min = duration_min
        self.duration_max = duration_max

        self.SEGMENT_LENGTH = 5
        # self.SEGMENT_LENGTH = 10
        self.max_len = num_patches
        self.pad_token = torch.nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)

        self.mask_ratio = mask_ratio

        self.importance_masking_strategy = importance_masking_strategy


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        max_attempts = 10  # Prevent infinite loop

        for _ in range(max_attempts):
            try:
                # Get the list of paths and the corresponding length
                row = self.df.iloc[idx]
                path_list = eval(row['segment_paths'])
                number_of_segments = row['number_of_segments']

                while number_of_segments * (1 - self.mask_ratio) < 1:
                    idx = random.randint(0, len(self.df) - 1)
                    row = self.df.iloc[idx]
                    path_list = eval(row['segment_paths'])
                    number_of_segments = row['number_of_segments']

                # Sample video duration between duration_min and duration_max
                self.duration = torch.randint(
                    self.duration_min,
                    self.duration_max + 1,
                    (1,)
                ).item()

                # Calculate number of segments
                subset_length = self.duration // self.SEGMENT_LENGTH

                # Do not exceed available segments
                if number_of_segments < subset_length:
                    subset_length = number_of_segments

                # Sample starting index
                max_j = number_of_segments - subset_length
                j = torch.randint(0, max_j + 1, (1,)).item()

                # Extract subset of paths
                subset_paths = path_list[j:j + subset_length]

                # Load and stack embeddings
                embeddings = []

                for path in subset_paths:
                    data = torch.load(path)
                    embeddings.append(data['embedding'])

                # Handle very short videos
                if number_of_segments < 2:
                    embeddings = embeddings * 8
                    subset_length = subset_length * 8

                elif number_of_segments < 5:
                    embeddings = embeddings * 3
                    subset_length = subset_length * 3

                stacked_embeddings = torch.stack(
                    embeddings,
                    dim=0
                )

                normalized_stacked_embeddings = torch.nn.functional.normalize(
                    stacked_embeddings,
                    p=2,
                    dim=-1
                )

                # Apply masking strategy.
                # For downstream attentive probing with mask_ratio=0,
                # preserve the original temporal order and do not
                # randomly permute the embeddings.
                if self.mask_ratio == 0.0:

                    masked_embeddings = normalized_stacked_embeddings

                    masked_length = normalized_stacked_embeddings.shape[0]

                    mask = torch.zeros(
                        masked_length,
                        dtype=normalized_stacked_embeddings.dtype
                    )

                    ids_shuffle = torch.arange(
                        masked_length,
                        dtype=torch.long
                    )

                    ids_restore = torch.arange(
                        masked_length,
                        dtype=torch.long
                    )

                elif self.importance_masking_strategy:

                    masked_embeddings, mask, ids_restore, ids_shuffle = (
                        self.event_span_masking(
                            normalized_stacked_embeddings
                        )
                    )

                    masked_length = masked_embeddings.shape[0]

                else:
                    raise ValueError("This repository implements event_span masking only: "
                                     "pretrain with --masking_strategy, or probe with mask_ratio=0.")

                n_pad_masked = self.max_len - masked_length
                n_pad_original = self.max_len - subset_length

                # Pad masked embeddings
                if n_pad_masked > 0:

                    masked_padded_embeddings = torch.cat(
                        (
                            masked_embeddings,
                            torch.stack(
                                [self.pad_token] * n_pad_masked,
                                dim=0
                            )
                        ),
                        dim=0
                    )

                    attn_mask = torch.cat(
                        (
                            torch.zeros(
                                masked_length + 1,
                                dtype=torch.bool
                            ),
                            torch.ones(
                                n_pad_masked,
                                dtype=torch.bool
                            )
                        ),
                        dim=0
                    )

                else:

                    masked_padded_embeddings = masked_embeddings

                    attn_mask = torch.zeros(
                        masked_length + 1,
                        dtype=torch.bool
                    )

                # Pad original embeddings and masking information
                if n_pad_original > 0:

                    normalized_stacked_embeddings = torch.cat(
                        (
                            normalized_stacked_embeddings,
                            torch.stack(
                                [self.pad_token] * n_pad_original,
                                dim=0
                            )
                        ),
                        dim=0
                    )

                    mask = torch.cat(
                        (
                            mask,
                            torch.zeros(
                                n_pad_original,
                                dtype=mask.dtype
                            )
                        ),
                        dim=0
                    )

                    ids_shuffle = torch.cat(
                        (
                            ids_shuffle,
                            torch.zeros(
                                n_pad_original,
                                dtype=ids_shuffle.dtype
                            )
                        ),
                        dim=0
                    )

                    ids_restore = torch.cat(
                        (
                            ids_restore,
                            torch.zeros(
                                n_pad_original,
                                dtype=ids_restore.dtype
                            )
                        ),
                        dim=0
                    )

                # Return embeddings + metadata for downstream evaluation
                return {
                    'masked_padded_embeddings': masked_padded_embeddings,
                    'attn_mask': attn_mask,
                    'normalized_stacked_embeddings': normalized_stacked_embeddings,
                    'mask': mask,
                    'ids_restore': ids_restore,
                    'ids_shuffle': ids_shuffle,
                    'masked_length': masked_length,
                    'length': subset_length,

                    # Metadata for downstream evaluation
		    'label': int(row["label"]),
		    'label_name': str(row['label_name']),
		    'video_filename': str(row['video_filename'])
		}
            except Exception as e:

                print(
                    f"Error loading file {path if 'path' in locals() else 'unknown'}: {str(e)}"
                )

                # Sample a new index
                idx = random.randint(
                    0,
                    len(self.df) - 1
                )

        # If all attempts fail
        raise RuntimeError(
            f"Failed to load data after {max_attempts} attempts"
        )


# ===================== event-span masking (novelty) =====================
import os as _os
from data.event_span_masking import event_span_masking as _event_span_masking


def _apply_event_span(self, x, *args, **kwargs):
    tau = float(_os.environ.get('LVMAE_TAU', '0.5'))   # set every epoch by main_pretrain.py (tau_schedule)
    return _event_span_masking(x, self.mask_ratio, tau=tau)


CombinedData.event_span_masking = _apply_event_span
