"""adversarial case generator: produces CitizenProfiles the current advisor
struggles with. trained against -advisor_reward + diversity penalty + validity gate."""

from nyaya_mitra.case_gen.diversity import DiversityTracker
from nyaya_mitra.case_gen.generator import (
    GeneratedCase,
    build_generator_advisor,
    score_generation,
)
from nyaya_mitra.case_gen.validator import ProfileValidator, ValidationResult

__all__ = [
    "DiversityTracker",
    "GeneratedCase",
    "ProfileValidator",
    "ValidationResult",
    "build_generator_advisor",
    "score_generation",
]
