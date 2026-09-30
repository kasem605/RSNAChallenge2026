from dataclasses import dataclass

@dataclass(frozen=True)
class KneeLabels:
    """
    Abnormality labels for one MRI study

    Each value represents whether the corresponding
    abnormality is present

    Label convention:
        0 - negative
        1 = positive
    """

    acl: int    # anterior cruciate ligament
    mcl: int    # medial collateral ligament
    medial_meniscus: int    
    lateral_meniscus: int
    medial_oa: int   # medial compartmental osteoarthritis
    lateral_oa: int  # lateral compartmental osteoarthritis
    pf_oa: int       # patellofemoral osteoarthritis
    effusion: int
    synovitis: int
    bakers: int
    contusion: int
    fracture: int

    def __post_init__(self)->None:
        labels = (
            self.acl,
            self.mcl,
            self.medial_meniscus,
            self.lateral_meniscus,
            self.medial_oa,
            self.lateral_oa,
            self.pf_oa,
            self.effusion,
            self.synovitis,
            self.bakers,
            self.contusion,
            self.fracture
        )

        if any(label not in (0, 1) for label in labels):
            raise ValueError("Knee abnormality lables must be either 0 or 1.")

    @property
    def as_tuple(self) -> tuple[int, ...]:
        return(
            self.acl,
            self.mcl,
            self.medial_meniscus,
            self.lateral_meniscus,
            self.medial_oa,
            self.lateral_oa,
            self.pf_oa,
            self.effusion,
            self.synovitis,
            self.bakers,
            self.contusion,
            self.fracture
        )

    @property
    def count(self) -> int:
        return len(self.as_tuple)