//! Kernel registry and device facts used by the GLM engine.

pub use glm_kernels::*;

/// SM count of the active device, with the B300 default when no device is open
/// (unit tests and `--check` runs load this crate without a GPU).
pub fn sm_count() -> usize {
    static SMS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SMS.get_or_init(|| {
        // cudarc panics when libcuda is absent (CPU-only test runs); fall back to the
        // B300 SM count so planning code stays testable off-device.
        std::panic::catch_unwind(|| deepseek_ultrafast_cuda::current_sm_count().unwrap_or(148))
            .unwrap_or(148)
            .max(1) as usize
    })
}

/// Decode-step GEMV occupancy target: keep at least one CTA per SM per head group.
pub fn decode_cta_target(n: usize) -> usize {
    (n / 64).min(sm_count() * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_count_is_positive_without_a_device() {
        assert!(sm_count() > 0);
        assert!(decode_cta_target(4096) <= sm_count() * 2);
    }
}
