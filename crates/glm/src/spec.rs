//! Adaptive draft-block length for speculative decoding.
//!
//! The original engine captures a CUDA graph with one verify branch per draft length
//! and lets a trained confidence head pick the branch on the GPU. GLM checkpoints
//! ship no such head, so the same idea is driven by measured acceptance instead: the
//! policy tracks the per-token acceptance rate and picks the block that maximises
//! expected emitted tokens per unit of step time.

/// Relative cost constants, normalised so that one verify row costs 1.0.
/// A drafted token is one extra single-row pass through the MTP layer, hence cheap
/// relative to the full-model verify pass.
pub const DRAFT_COST: f32 = 0.12;
pub const OVERHEAD: f32 = 0.35;

#[derive(Clone, Debug)]
pub struct SpecPolicy {
    pub min_block: usize,
    pub max_block: usize,
    /// EWMA of the per-draft-token acceptance probability.
    acc: f32,
    pub warmup: u32,
}

impl SpecPolicy {
    pub fn new(min_block: usize, max_block: usize) -> Self {
        assert!(min_block >= 1 && min_block <= max_block);
        SpecPolicy { min_block, max_block, acc: 0.5, warmup: 4 }
    }

    pub fn acceptance(&self) -> f32 {
        self.acc
    }

    /// Fold one step's outcome (`accepted` drafts out of `drafted`) into the estimate.
    pub fn observe(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let sample = accepted as f32 / drafted as f32;
        let alpha = if self.warmup > 0 {
            self.warmup -= 1;
            0.5
        } else {
            0.15
        };
        self.acc = (1.0 - alpha) * self.acc + alpha * sample;
        self.acc = self.acc.clamp(0.001, 0.999);
    }

    /// Expected number of committed tokens for a block of `b` drafts, assuming the
    /// acceptance probability is independent per token: 1 + a + a^2 + ... + a^b.
    pub fn expected_tokens(&self, b: usize) -> f32 {
        let mut t = 1.0;
        let mut p = 1.0;
        for _ in 0..b {
            p *= self.acc;
            t += p;
        }
        t
    }

    /// Relative step cost: one extra verify row per draft, plus the draft passes.
    pub fn step_cost(&self, b: usize) -> f32 {
        OVERHEAD + (b + 1) as f32 + DRAFT_COST * b as f32
    }

    pub fn score(&self, b: usize) -> f32 {
        self.expected_tokens(b) / self.step_cost(b)
    }

    /// Best draft block for the current acceptance estimate.
    pub fn block(&self) -> usize {
        let mut best = self.min_block;
        let mut best_score = f32::NEG_INFINITY;
        for b in self.min_block..=self.max_block {
            let s = self.score(b);
            if s > best_score {
                best_score = s;
                best = b;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_grows_with_acceptance() {
        let mut p = SpecPolicy::new(1, 7);
        for _ in 0..16 {
            p.observe(0, 7);
        }
        assert_eq!(p.block(), 1, "acceptance {} -> block {}", p.acceptance(), p.block());
        for _ in 0..64 {
            p.observe(7, 7);
        }
        assert_eq!(p.block(), 7, "acceptance {}", p.acceptance());
    }

    #[test]
    fn expected_tokens_bounds_hold() {
        let mut p = SpecPolicy::new(1, 7);
        p.observe(0, 1);
        let low = p.expected_tokens(4);
        for _ in 0..40 {
            p.observe(1, 1);
        }
        let high = p.expected_tokens(4);
        assert!(low >= 1.0 && low < 1.5, "{low}");
        assert!(high > 3.5 && high <= 5.0, "{high}");
        assert!(p.expected_tokens(0) == 1.0);
    }

    #[test]
    fn zero_drafts_does_not_poison_the_estimate() {
        let mut p = SpecPolicy::new(1, 3);
        let before = p.acceptance();
        p.observe(0, 0);
        assert_eq!(before, p.acceptance());
    }

    #[test]
    fn clamp_keeps_the_block_in_range() {
        let p = SpecPolicy::new(2, 5);
        assert!((2..=5).contains(&p.block()));
    }
}

/// Host mirror of `glm_spec_pick`.
///
/// The device version decides how wide the next verification pass is: it smooths the
/// observed acceptance rate and maps it onto `1..=blk` rows, then hands the branch index
/// to the graph's conditional node. Keeping the arithmetic here lets the policy be tested
/// without a device, and lets a future host fallback reach the same decision.
///
/// `spec` is laid out like the device buffer: `[0]` emitted, `[1]` committed, `[4]` the
/// smoothed rate in 1/256 units, `[5]` the width chosen last round.
pub fn pick(spec: &mut [i32], blk: usize) -> usize {
    let blk = blk.max(1) as i64;
    let drafted = if spec[5] > 0 { spec[5] as i64 } else { blk };
    let mut accepted = spec[1] as i64 - 1;
    accepted = accepted.clamp(0, drafted);
    let inst = 256 * accepted / drafted.max(1);
    let ema = if spec[4] <= 0 { inst } else { (spec[4] as i64 * 3 + inst) / 4 };
    spec[4] = ema as i32;
    let l = 1 + (ema * (blk - 1) + 128) / 256;
    let l = l.clamp(1, blk) as i32;
    spec[5] = l;
    l as usize
}

#[cfg(test)]
mod pick_tests {
    use super::pick;

    #[test]
    fn full_acceptance_widens_the_pass() {
        let mut spec = [0i32; 8];
        spec[1] = 8; // committed = 7 drafts + the anchor row
        assert_eq!(pick(&mut spec, 7), 7);
    }

    #[test]
    fn repeated_rejection_narrows_to_one_row() {
        let mut spec = [0i32; 8];
        spec[4] = 256; // start optimistic
        spec[1] = 1; // committed the anchor only: nothing accepted
        let mut widths = Vec::new();
        for _ in 0..14 {
            widths.push(pick(&mut spec, 7));
        }
        // monotone down to a single draft, never to zero
        assert!(widths.windows(2).all(|w| w[1] <= w[0]), "{widths:?}");
        assert_eq!(*widths.last().unwrap(), 1, "{widths:?}");
    }

    #[test]
    fn the_choice_never_leaves_the_block() {
        let mut spec = [0i32; 8];
        for committed in 0..=9 {
            spec[1] = committed;
            for blk in 1..8 {
                let l = pick(&mut spec, blk);
                assert!((1..=blk).contains(&l), "committed {committed} blk {blk} -> {l}");
            }
        }
    }
}
