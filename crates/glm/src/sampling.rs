//! Host-side mirror of the device samplers.
//!
//! `glm_sample` and `glm_spec_accept` in `cu/sample.cu` split the vocabulary into
//! `blockDim.x` contiguous slices and walk the CDF in index order with a splitmix64
//! stream. The functions here reproduce that exactly, so the sampling contract can be
//! tested (and replayed in logs) without a GPU.


pub const SLICES: usize = 1024;

/// splitmix64, matching `rng_next` in `common.cuh`.
pub fn rng_next(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 32) as u32
}

/// `rng_unit`: 24-bit uniform in [0, 1).
pub fn rng_unit(state: &mut u64) -> f32 {
    (rng_next(state) >> 8) as f32 / 16_777_216.0
}

pub fn seed_for(seed: u64, step: u32) -> u64 {
    seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn slice_bounds(i: usize, n: usize, slices: usize) -> (usize, usize) {
    let lo = (i as u128 * n as u128 / slices as u128) as usize;
    let hi = ((i + 1) as u128 * n as u128 / slices as u128) as usize;
    (lo, hi)
}

/// Deterministic temperature sample, identical to `glm_sample`.
pub fn sample(logits: &[f32], temp: f32, seed: u64, step: u32) -> usize {
    let n = logits.len();
    assert!(n > 0, "empty logits");
    let inv = if temp > 0.0 { 1.0 / temp } else { 1.0 };
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut slice_sum = [0f32; SLICES];
    for (i, s) in slice_sum.iter_mut().enumerate() {
        let (lo, hi) = slice_bounds(i, n, SLICES);
        *s = logits[lo..hi].iter().map(|v| ((v - mx) * inv).exp()).sum();
    }
    let mut prefix = [0f32; SLICES];
    let mut acc = 0f32;
    for i in 0..SLICES {
        prefix[i] = acc;
        acc += slice_sum[i];
    }
    let total = acc;
    let mut st = seed_for(seed, step);
    let u = rng_unit(&mut st) * if total > 0.0 { total } else { 1.0 };
    let mut pick = SLICES - 1;
    for i in 0..SLICES {
        if u >= prefix[i] && u < prefix[i] + slice_sum[i] {
            pick = i;
            break;
        }
    }
    let (lo, hi) = slice_bounds(pick, n, SLICES);
    let mut c = 0f32;
    let mut chosen = hi.saturating_sub(1);
    for (off, v) in logits[lo..hi].iter().enumerate() {
        c += ((v - mx) * inv).exp();
        if u < prefix[pick] + c {
            chosen = lo + off;
            break;
        }
    }
    chosen
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] || (*v == logits[best] && i < best) {
            best = i;
        }
    }
    best
}

/// One acceptance decision, mirroring the `greedy == 0` branch of `glm_spec_accept`.
pub struct AcceptStep {
    pub accepted: bool,
    pub replacement: Option<usize>,
}

pub fn acceptance_greedy(target_row: &[f32], draft: usize) -> AcceptStep {
    let am = argmax(target_row);
    AcceptStep { accepted: am == draft, replacement: (!(am == draft)).then_some(am) }
}

pub fn acceptance_sampling(
    target_row: &[f32],
    draft_row: &[f32],
    draft: usize,
    temp: f32,
    seed: u64,
    step: u32,
) -> AcceptStep {
    assert!(!target_row.is_empty() && target_row.len() == draft_row.len(), "logit row width mismatch");
    let p = softmax_at(target_row, temp);
    let q = softmax_at(draft_row, temp);
    let mut st = seed_for(seed, step);
    let u = rng_unit(&mut st);
    let p_d = p[draft];
    let q_d = q[draft];
    let acc = if q_d > 0.0 { (p_d / q_d).min(1.0) } else { 1.0 };
    if u < acc {
        return AcceptStep { accepted: true, replacement: None };
    }
    // residual draw from max(0, p - q)
    let mut resid: Vec<f32> = p.iter().zip(q.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
    let total: f32 = resid.iter().sum();
    if total <= 0.0 {
        return AcceptStep { accepted: false, replacement: Some(argmax(target_row)) };
    }
    for v in resid.iter_mut() {
        *v /= total;
    }
    let mut st = seed_for(seed ^ 0xD1B5_4A32_D192_ED03, step);
    let u = rng_unit(&mut st) * total;
    let mut c = 0.0;
    for (i, v) in resid.iter().enumerate() {
        c += *v * total;
        if u < c {
            return AcceptStep { accepted: false, replacement: Some(i) };
        }
    }
    AcceptStep { accepted: false, replacement: Some(resid.len() - 1) }
}

fn softmax_at(row: &[f32], temp: f32) -> Vec<f32> {
    let inv = if temp > 0.0 { 1.0 / temp } else { 1.0 };
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = row.iter().map(|v| ((v - mx) * inv).exp()).collect();
    let s: f32 = out.iter().sum();
    for v in out.iter_mut() {
        *v /= s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_the_cuda_splitmix() {
        // reference values produced by the same splitmix64 constants
        let mut st = 0u64;
        assert_eq!(rng_next(&mut st), 0xE220_A839u32);
        let mut st = 0u64;
        let u = rng_unit(&mut st);
        assert!((0.0..1.0).contains(&u), "{u}");
    }

    #[test]
    fn sampling_is_deterministic_and_prefers_the_mode() {
        let logits = [0.1f32, 5.0, 0.2, -3.0];
        let a = sample(&logits, 1.0, 7, 0);
        let b = sample(&logits, 1.0, 7, 0);
        assert_eq!(a, b);
        let mut hits = [0usize; 4];
        for step in 0..2000 {
            hits[sample(&logits, 1.0, 42, step)] += 1;
        }
        assert!(hits[1] > 1900, "{hits:?}");
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn low_temperature_matches_argmax() {
        let logits = [1.0f32, 1.5, 0.2];
        for step in 0..64 {
            assert_eq!(sample(&logits, 0.01, 3, step), 1);
        }
    }

    #[test]
    fn greedy_acceptance_follows_the_target_argmax() {
        let row = [0.0f32, 9.0, 1.0];
        assert!(acceptance_greedy(&row, 1).accepted);
        let r = acceptance_greedy(&row, 2);
        assert!(!r.accepted);
        assert_eq!(r.replacement, Some(1));
    }

    #[test]
    fn stochastic_acceptance_accepts_when_drafter_matches_target() {
        let t = [0.0f32, 4.0, 0.0];
        let q = [0.0f32, 4.0, 0.0];
        for step in 0..32 {
            assert!(acceptance_sampling(&t, &q, 1, 1.0, 5, step).accepted);
        }
        // a drafter that prefers a token the target almost never picks is rejected
        let q_bad = [4.0f32, 0.0, 0.0];
        let mut rejected = 0;
        for step in 0..64 {
            if !acceptance_sampling(&t, &q_bad, 0, 1.0, 5, step).accepted {
                rejected += 1;
            }
        }
        assert!(rejected > 40, "{rejected}");
    }
}
