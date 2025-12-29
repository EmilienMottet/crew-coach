"""Centralized validation thresholds for macro compliance checks.

Single source of truth for all tolerance values used across:
- NutritionalValidationAgent (backstory)
- MealRecipeReviewerTask (task description)
- crew_mealy.py _validate_daily_totals (Python validation)
- macro_calculator.py check_macro_compliance (pre-calc validation)

Thresholds are intentionally lenient to avoid false rejections.
Real-world meal planning has natural variance from portion estimates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MacroThresholds:
    """Tolerance thresholds for macro validation.

    Uses BOTH percentage AND absolute thresholds (whichever is more lenient).
    This prevents rejecting plans that are close enough in absolute terms
    even if the percentage is slightly over.

    The values are intentionally generous because:
    - Portion sizes are estimates
    - Nutritional data has variance
    - Athletes need flexibility, not robotic precision
    """

    # Percentage tolerances (expressed as decimals)
    # Using < comparison (not <=) so 18% passes an 18% threshold
    calories_pct: float = 0.12  # +/- 12% for calories
    protein_pct: float = 0.18  # +/- 18% for protein
    carbs_pct: float = 0.20  # +/- 20% for carbs
    fat_pct: float = 0.20  # +/- 20% for fat

    # Absolute tolerances (in grams/kcal)
    # Used as fallback when percentage would be too strict
    calories_abs: float = 300  # +/- 300 kcal absolute
    protein_abs: float = 15  # +/- 15g absolute
    carbs_abs: float = 25  # +/- 25g absolute
    fat_abs: float = 12  # +/- 12g absolute


# Default thresholds instance - import this in other modules
DEFAULT_THRESHOLDS = MacroThresholds()

# Relaxed thresholds (+50%) for 4th attempt when strict validation fails
# This provides a fallback to avoid complete plan failures
RELAXED_THRESHOLDS = MacroThresholds(
    calories_pct=0.18,  # ±18% (vs ±12%)
    protein_pct=0.27,  # ±27% (vs ±18%)
    carbs_pct=0.30,  # ±30% (vs ±20%)
    fat_pct=0.30,  # ±30% (vs ±20%)
    calories_abs=450,  # ±450 (vs ±300)
    protein_abs=22,  # ±22g (vs ±15g)
    carbs_abs=38,  # ±38g (vs ±25g)
    fat_abs=18,  # ±18g (vs ±12g)
)

# Variety thresholds
MIN_UNIQUE_MEALS_RATIO = 0.85  # 85% unique meals acceptable (e.g., 24/28 is OK)


def get_adaptive_thresholds(
    target_calories: int, use_relaxed: bool = False
) -> MacroThresholds:
    """Get thresholds adapted to the target calorie level.

    High-energy days (>3000 kcal) get looser thresholds because:
    - Larger absolute values make percentage harder to hit exactly
    - Athletes training heavily need flexibility
    - Small percentage miss at high calories = large absolute miss

    Args:
        target_calories: Daily calorie target
        use_relaxed: If True, use RELAXED_THRESHOLDS (4th attempt fallback)

    Returns:
        MacroThresholds instance appropriate for the calorie level
    """
    if use_relaxed:
        return RELAXED_THRESHOLDS

    if target_calories > 3000:
        # High-energy day: use intermediate thresholds
        return MacroThresholds(
            calories_pct=0.15,  # ±15% (vs ±12%)
            protein_pct=0.20,  # ±20% (vs ±18%)
            carbs_pct=0.25,  # ±25% (vs ±20%)
            fat_pct=0.25,  # ±25% (vs ±20%)
            calories_abs=400,  # ±400 (vs ±300)
            protein_abs=20,  # ±20g (vs ±15g)
            carbs_abs=35,  # ±35g (vs ±25g)
            fat_abs=15,  # ±15g (vs ±12g)
        )

    return DEFAULT_THRESHOLDS


def is_within_tolerance(
    actual: float,
    target: float,
    pct_tolerance: float,
    abs_tolerance: float,
) -> bool:
    """Check if actual is within tolerance of target.

    Uses the MORE LENIENT of percentage or absolute tolerance.
    Uses strict less-than (<) to handle boundary cases properly.
    Example: 18.0% variance passes +/-18% threshold.

    Args:
        actual: Actual value
        target: Target value
        pct_tolerance: Percentage tolerance (as decimal, e.g., 0.18 for 18%)
        abs_tolerance: Absolute tolerance

    Returns:
        True if within tolerance (either percentage OR absolute)
    """
    if target == 0:
        return True  # Can't calculate percentage, assume OK

    diff = abs(actual - target)
    pct_diff = diff / target

    # Pass if EITHER tolerance is met (more lenient)
    # Use < (not <=) for proper boundary handling
    return pct_diff < pct_tolerance or diff < abs_tolerance


def check_compliance(
    actual: Dict[str, float],
    target: Dict[str, float],
    thresholds: MacroThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Check if actual macros comply with targets.

    Args:
        actual: Dict with calories, protein_g, carbs_g, fat_g
        target: Dict with target values (same keys)
        thresholds: MacroThresholds instance

    Returns:
        Dict with:
        - compliant: bool - True if all macros within tolerance
        - issues: List[str] - Human-readable list of failed macros
        - details: Dict[str, Dict] - Per-macro analysis with actual, target, diff, etc.
    """
    issues: List[str] = []
    details: Dict[str, Dict[str, Any]] = {}

    # Define checks: (actual_key, target_key, pct_tolerance, abs_tolerance, unit)
    checks = [
        (
            "calories",
            "calories",
            thresholds.calories_pct,
            thresholds.calories_abs,
            "kcal",
        ),
        ("protein_g", "protein_g", thresholds.protein_pct, thresholds.protein_abs, "g"),
        ("carbs_g", "carbs_g", thresholds.carbs_pct, thresholds.carbs_abs, "g"),
        ("fat_g", "fat_g", thresholds.fat_pct, thresholds.fat_abs, "g"),
    ]

    for actual_key, target_key, pct_tol, abs_tol, unit in checks:
        actual_val = actual.get(actual_key, 0) or 0
        target_val = target.get(target_key, 0) or 0

        if target_val == 0:
            details[actual_key] = {
                "actual": actual_val,
                "target": target_val,
                "diff": 0,
                "pct_diff": 0,
                "within_tolerance": True,
            }
            continue

        diff = actual_val - target_val
        pct_diff = (diff / target_val) * 100 if target_val else 0

        within = is_within_tolerance(actual_val, target_val, pct_tol, abs_tol)

        details[actual_key] = {
            "actual": round(actual_val, 1),
            "target": round(target_val, 1),
            "diff": round(diff, 1),
            "pct_diff": round(pct_diff, 1),
            "within_tolerance": within,
        }

        if not within:
            direction = "over" if diff > 0 else "under"
            macro_name = actual_key.replace("_g", "").replace("_", " ").capitalize()
            issues.append(
                f"{macro_name}: {actual_val:.0f}{unit} is {abs(diff):.0f}{unit} "
                f"({abs(pct_diff):.1f}%) {direction} target {target_val:.0f}{unit}"
            )

    return {
        "compliant": len(issues) == 0,
        "issues": issues,
        "details": details,
    }


def check_variety(
    unique_meals: int,
    expected_meals: int,
    min_ratio: float = MIN_UNIQUE_MEALS_RATIO,
) -> Dict[str, Any]:
    """Check if meal variety is sufficient.

    Args:
        unique_meals: Number of unique meal names
        expected_meals: Expected number of meals (days * 4)
        min_ratio: Minimum acceptable ratio (default 0.95 = 95%)

    Returns:
        Dict with:
        - acceptable: bool
        - ratio: float
        - message: str
    """
    if expected_meals == 0:
        return {"acceptable": True, "ratio": 1.0, "message": "No meals expected"}

    ratio = unique_meals / expected_meals
    acceptable = ratio >= min_ratio

    if acceptable:
        message = (
            f"Good variety: {unique_meals}/{expected_meals} unique meals ({ratio:.0%})"
        )
    else:
        message = f"Low variety: {unique_meals}/{expected_meals} unique meals ({ratio:.0%}), need {min_ratio:.0%}"

    return {
        "acceptable": acceptable,
        "ratio": round(ratio, 3),
        "message": message,
    }


def format_thresholds_for_prompt() -> str:
    """Format thresholds as a string for LLM prompts.

    Returns a human-readable description of the thresholds.
    """
    t = DEFAULT_THRESHOLDS
    return f"""UNIFIED MACRO THRESHOLDS (pass if EITHER percentage OR absolute is met):
- Calories: +/-{t.calories_pct:.0%} OR +/-{t.calories_abs:.0f} kcal
- Protein: +/-{t.protein_pct:.0%} OR +/-{t.protein_abs:.0f}g
- Carbs: +/-{t.carbs_pct:.0%} OR +/-{t.carbs_abs:.0f}g
- Fat: +/-{t.fat_pct:.0%} OR +/-{t.fat_abs:.0f}g
- Variety: {MIN_UNIQUE_MEALS_RATIO:.0%} unique meals acceptable (e.g., 27/28 is OK)

NOTE: Boundary values PASS (e.g., 18% variance passes +/-18% threshold)."""


# =============================================================================
# Dietary Restrictions Configuration
# =============================================================================


@dataclass(frozen=True)
class DietaryRestriction:
    """A single dietary restriction rule.

    Attributes:
        name: Identifier for the restriction (e.g., "cheese")
        forbidden_keywords: Keywords that trigger rejection (case-insensitive)
        exceptions: Allowed exceptions (e.g., mascarpone for cheese)
        severity: "critical" = block validation, "warning" = log only
    """

    name: str
    forbidden_keywords: tuple  # Use tuple for frozen dataclass
    exceptions: tuple = ()
    severity: str = "critical"


# Default dietary restrictions for the user
DEFAULT_DIETARY_RESTRICTIONS: List[DietaryRestriction] = [
    DietaryRestriction(
        name="cheese",
        forbidden_keywords=(
            "cheese",
            "fromage",
            "parmesan",
            "cheddar",
            "mozzarella",
            "gruyere",
            "gruyère",
            "brie",
            "camembert",
            "feta",
            "gouda",
            "emmental",
            "comté",
            "comte",
            "roquefort",
            "gorgonzola",
            "pecorino",
            "ricotta",
            "halloumi",
            "paneer",
        ),
        exceptions=("mascarpone",),  # Mascarpone is allowed
        severity="critical",
    ),
    DietaryRestriction(
        name="smoked_salmon",
        forbidden_keywords=(
            "smoked salmon",
            "saumon fume",
            "saumon fumé",
            "lox",
            "gravlax",
            "nova",
        ),
        exceptions=(),
        severity="critical",
    ),
    DietaryRestriction(
        name="carbonated_drinks",
        forbidden_keywords=(
            "soda",
            "cola",
            "sprite",
            "fanta",
            "pepsi",
            "carbonated",
            "sparkling soda",
            "fizzy drink",
        ),
        exceptions=("sparkling water", "eau gazeuse", "perrier", "san pellegrino"),
        severity="warning",  # Non-blocking
    ),
]


def check_dietary_restrictions(
    ingredients: List[str],
    restrictions: Optional[List[DietaryRestriction]] = None,
) -> Dict[str, Any]:
    """Check ingredients against dietary restrictions.

    Args:
        ingredients: List of ingredient strings to check
        restrictions: List of DietaryRestriction rules (default: DEFAULT_DIETARY_RESTRICTIONS)

    Returns:
        Dict with:
        - passed: bool - True if no critical violations
        - violations: List[Dict] - List of critical violations found
        - warnings: List[Dict] - List of non-critical warnings
    """
    if restrictions is None:
        restrictions = DEFAULT_DIETARY_RESTRICTIONS

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for ingredient in ingredients:
        if not ingredient:
            continue
        ingredient_lower = ingredient.lower()

        for restriction in restrictions:
            # Check if any exception applies first
            is_exception = any(
                exc.lower() in ingredient_lower for exc in restriction.exceptions
            )
            if is_exception:
                continue

            # Check if any forbidden keyword matches
            for keyword in restriction.forbidden_keywords:
                if keyword.lower() in ingredient_lower:
                    violation = {
                        "ingredient": ingredient,
                        "restriction": restriction.name,
                        "matched_keyword": keyword,
                        "severity": restriction.severity,
                    }
                    if restriction.severity == "critical":
                        violations.append(violation)
                    else:
                        warnings.append(violation)
                    break  # Only report first match per restriction

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }
