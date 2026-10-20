
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeDiagnostics:
    original_shape: tuple[int, int, int]
    resampled_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]

    @property
    def crop_before(self) -> tuple[int, int, int]:
        return tuple(
            max(0, current - target) // 2
            for current, target in zip(
                self.resampled_shape, self.target_shape
            )
        )

    @property
    def crop_after(self) -> tuple[int, int, int]:
        return tuple(
            max(0, current - target)
            - before
            for current, target, before in zip(
                self.resampled_shape,
                self.target_shape,
                self.crop_before,
            )
        )

    @property
    def padding_before(self) -> tuple[int, int, int]:
        return tuple(
            max(0, target - current) // 2
            for current, target in zip(
                self.resampled_shape, self.target_shape
            )
        )

    @property
    def padding_after(self) -> tuple[int, int, int]:
        return tuple(
            max(0, target - current) - before
            for current, target, before in zip(
                self.resampled_shape,
                self.target_shape,
                self.padding_before,
            )
        )

    def print_report(self) -> None:
        print(f"Original shape:    {self.original_shape}")
        print(f"Resampled shape:   {self.resampled_shape}")
        print(f"Target shape:      {self.target_shape}")
        print(f"Crop before:       {self.crop_before}")
        print(f"Crop after:        {self.crop_after}")
        print(f"Padding before:    {self.padding_before}")
        print(f"Padding after:     {self.padding_after}")
