import torch
from torch import nn

class Knee3DCNN(nn.Module):

    """
    Three-branch 3D CNN for predicting 12 knee abnormalities

    Inputs:
        sagittal: (batch, depth, height, width)
        coronal:  (batch, depth, height, width)
        axial:    (batch, depth, height, width)

    Output:
        logits: (batch, 12)
    """

    NUM_TARGETS = 12

    def __init__(self) -> None:
        super().__init__()

        # -----------------------------------------------------------------
        # Independent feature extractor for each MRI plane.
        # -----------------------------------------------------------------

        self.sagittal_branch = self._create_branch()
        self.coronal_branch = self._create_branch()
        self.axial_branch = self._create_branch()

        # -----------------------------------------------------------------
        # Each branch produces 8 features.
        # Three branches therefore produce 24 combined features.
        # -----------------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.linear(24,32),
            nn.ReLU(),
            nn.Linear(32, self.NUM_TARGETS)
        )

        @staticmethod
        def _create_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(
                    in_channels=1,
                    out_channels=4,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2),
                nn.Conv3d(
                    in_channels=4,
                    out_channels=8,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2),

                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten(start_dim=1)
            )

        def forward(
                self,
                sagittal: torch.Tensor,
                coronal: torch.Tensor,
                axial: torch.Tensor            
        ) -> torch.Tensor:

            volumes = (
                ("sagittal", sagittal),
                ("coronal", coronal),
                ("axial", axial)
            )

            batch_size = sagittal.shape[0]

            for name, volume in volumes:
                if volume.ndim != 4:
                    raise ValueError(f"{name} must have shape (batch, depth, height, width)")

                if volume.shape[0] != batch_size:
                    raise ValueError("All three MRI planes must have the same batch size")

            # ----------------------------------------------------------------------------
            # Conv3d expects (batch, channels, depth, height, width)
            # ----------------------------------------------------------------------------

            sagittal = sagittal.unsqueeze(1)
            coronal = coronal.unsqueeze(1) 
            axial = axial.unsqueeze(1)                  

            sagittal_features = self.sagittal_branch(sagittal)
            coronal_features = self.coronal_branch(coronal)
            axial_features = self.axial_branch(axial)

            combined_features = torch.cat(
                (
                    sagittal_features,
                    combined_features,
                    axial_features
                ),
                dim=1
            )

            # ----------------------------------------------------------------------------
            # Return raw logits, not probabilities
            # ----------------------------------------------------------------------------

            return self.classifier(combined_features)