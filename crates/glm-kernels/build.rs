//! Compiles `cu/*.cu` into cubins for the target Blackwell arch and embeds them.
//!
//! Unlike `crates/kernels` this crate holds only the kernels a single-GPU GLM
//! deployment needs, so a build failure here is specific to the GLM path.
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let arch = match env::var("GLM_SM") {
        Ok(v) if v.starts_with("-arch=") => v,
        Ok(v) => format!("-arch={v}"),
        Err(_) => "-arch=sm_103a".into(),
    };
    if !arch.starts_with("-arch=sm_1") {
        panic!("GLM_SM must be an sm_100a/sm_103a-class arch, got {arch}");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=GLM_SM");
    println!("cargo:rerun-if-changed=cu");

    let mut srcs: Vec<PathBuf> = fs::read_dir("cu").unwrap().map(|e| e.unwrap().path()).collect();
    srcs.retain(|p| p.extension().is_some_and(|e| e == "cu"));
    srcs.sort();

    // Tensor-core kernels are reused from the upstream corpus instead of being rewritten:
    // they already carry the tcgen05/TMA plumbing (TMEM allocation, mbarrier, cluster
    // splits) that a hand port would only re-derive. They compile from their own directory
    // so their `#include "common.cuh"` resolves to the upstream header, not ours.
    let upstream = PathBuf::from("../kernels/cu");
    for name in ["tcgemv.cu", "tcgemm.cu", "tcmoe.cu", "variant_zzzzzzzzzzmoe.cu", "gemv.cu", "tcgemv2.cu"] {
        let p = upstream.join(name);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
            srcs.push(p);
        }
    }

    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
    let mut running: std::collections::VecDeque<(PathBuf, std::process::Child)> = Default::default();
    let mut entries = Vec::new();
    for src in srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        // Upstream sources get a distinct stem so they can never collide with ours (both
        // corpora have a gemv.cu, and a silent overwrite would swap the kernel set).
        let mut stem = src.file_stem().unwrap().to_str().unwrap().to_string();
        if !src.starts_with("cu") {
            stem = format!("u_{stem}");
        }
        let cubin = out.join(format!("{stem}.cubin"));
        if running.len() >= jobs {
            let (s, mut c) = running.pop_front().unwrap();
            assert!(c.wait().expect("nvcc").success(), "nvcc failed on {}", s.display());
        }
        let child = Command::new(&nvcc)
            .args([&arch, "-cubin", "-O3", "-std=c++17", "-lineinfo", "-Icu", "-I../kernels/cu"])
            .args(["-diag-suppress", "177", "-diag-suppress", "221", "-diag-suppress", "550"])
            .arg(&src)
            .arg("-o")
            .arg(&cubin)
            .spawn()
            .expect("nvcc (set NVCC=/path/to/nvcc)");
        running.push_back((src, child));
        entries.push(format!("({stem:?}, include_bytes!({:?}) as &[u8])", cubin.to_str().unwrap()));
    }
    for (s, mut c) in running {
        assert!(c.wait().expect("nvcc").success(), "nvcc failed on {}", s.display());
    }
    fs::write(
        out.join("cubins.rs"),
        format!("pub const CUBINS: &[(&str, &[u8])] = &[{}];\n", entries.join(",\n")),
    )
    .unwrap();
}
