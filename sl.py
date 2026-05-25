"""
Subjective Logic opinion types: BinomialOpinion and MultinomialOpinion.

Implements the opinion algebra of Jøsang (2016), including the conjunction
operator (∧, Definition 14.4) and mappings from Beta/Dirichlet distributions.
"""

from typing import Dict

_EPS = 1e-12


class BinomialOpinion:
    """
    A Subjective Logic binomial opinion ω = (b, d, u, a) where
    b + d + u = 1 and the projected probability is E[ω] = b + a·u.
    """

    def __init__(self, belief: float, disbelief: float, uncertainty: float,
                 base_rate: float = 0.5):
        if round(belief + disbelief + uncertainty, 6) != 1:
            raise ValueError("Belief, disbelief, and uncertainty must sum to 1.")
        self.b = belief
        self.d = disbelief
        self.u = uncertainty
        self.a = base_rate

    def expectation(self) -> float:
        return self.b + self.a * self.u

    def __repr__(self):
        return (f"BinomialOpinion(b={self.b:.3f}, d={self.d:.3f}, "
                f"u={self.u:.3f}, a={self.a}, E={self.expectation():.3f})")

    @staticmethod
    def from_beta(alpha: float, beta: float, base_rate: float = 0.5) -> "BinomialOpinion":
        """Map a Beta(α, β) distribution to a BinomialOpinion via r = α−1, s = β−1."""
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive.")
        r, s = alpha - 1, beta - 1
        K = r + s
        return BinomialOpinion(r / (K + 2), s / (K + 2), 2 / (K + 2), base_rate)

    def conjunction(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """
        Subjective Logic conjunction ω_{x∧y} (Jøsang 2016, Definition 14.4):

            b_{x∧y} = bx·by + [(1−ax)·ay·bx·uy + ax·(1−ay)·ux·by] / (1 − ax·ay)
            d_{x∧y} = dx + dy − dx·dy
            u_{x∧y} = ux·uy + [(1−ay)·bx·uy + (1−ax)·ux·by] / (1 − ax·ay)
            a_{x∧y} = ax·ay
        """
        bx, dx, ux, ax = self.b, self.d, self.u, self.a
        by, dy, uy, ay = other.b, other.d, other.u, other.a

        denom = 1.0 - ax * ay
        if abs(denom) < _EPS:
            raise ValueError("Conjunction undefined: 1 − ax·ay ≈ 0.")

        b = bx * by + ((1 - ax) * ay * bx * uy + ax * (1 - ay) * ux * by) / denom
        d = dx + dy - dx * dy
        u = ux * uy + ((1 - ay) * bx * uy + (1 - ax) * ux * by) / denom
        a = ax * ay

        total = b + d + u
        return BinomialOpinion(round(b / total, 10), round(d / total, 10),
                               round(u / total, 10), a)

    def __and__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        return self.conjunction(other)


class MultinomialOpinion:
    """
    A Subjective Logic multinomial opinion ω = (b, u, a) over K outcomes,
    where Σb_k + u = 1 and Σa_k = 1.
    """

    def __init__(self, belief: Dict[str, float], uncertainty: float,
                 base_rate: Dict[str, float]):
        if abs(sum(belief.values()) + uncertainty - 1) > 1e-6:
            raise ValueError("Sum of beliefs + uncertainty must equal 1.")
        if abs(sum(base_rate.values()) - 1) > 1e-6:
            raise ValueError("Base rates must sum to 1.")
        if set(belief.keys()) != set(base_rate.keys()):
            raise ValueError("Belief and base-rate keys must match.")
        self.belief = belief
        self.u = uncertainty
        self.base_rate = base_rate

    def expectation(self) -> Dict[str, float]:
        return {x: self.belief[x] + self.base_rate[x] * self.u for x in self.belief}

    def __repr__(self):
        exp = ", ".join(f"{k}: {v:.3f}" for k, v in self.expectation().items())
        return (f"MultinomialOpinion(belief={self.belief}, u={self.u:.3f}, "
                f"base_rate={self.base_rate}, E={{ {exp} }})")

    @staticmethod
    def from_dirichlet(alpha: Dict[str, float],
                       base_rate: Dict[str, float] = None) -> "MultinomialOpinion":
        """Map a Dirichlet(α) distribution to a MultinomialOpinion via r_k = α_k."""
        alpha = {k: max(v, _EPS) for k, v in alpha.items()}
        outcomes = list(alpha.keys())
        K = sum(alpha.values())
        W = len(outcomes)
        belief = {x: alpha[x] / (K + W) for x in outcomes}
        u = W / (K + W)
        if base_rate is None:
            base_rate = {x: 1.0 / W for x in outcomes}
        return MultinomialOpinion(belief, u, base_rate)


if __name__ == "__main__":
    print(BinomialOpinion.from_beta(alpha=3, beta=5))
    print(MultinomialOpinion.from_dirichlet({"R": 2, "G": 3, "B": 4}))
