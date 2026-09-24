//! Feature availability only; no arithmetic benchmark or production dispatch.
fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        let raw = unsafe { std::arch::x86_64::__cpuid_count(7, 0) };
        println!(
            "{{\"os\":\"{}\",\"arch\":\"x86_64\",\"avx2\":{},\"fma\":{},\"avx512f\":{},\"avx512bw\":{},\"avx512vl\":{},\"avx512vnni\":{},\"avx512bf16\":{},\"avxvnni\":{},\"avx512fp16\":{},\"raw_cpuid_amx_int8\":{},\"raw_cpuid_amx_tile\":{}}}",
            std::env::consts::OS,
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx512f"),
            std::is_x86_feature_detected!("avx512bw"),
            std::is_x86_feature_detected!("avx512vl"),
            std::is_x86_feature_detected!("avx512vnni"),
            std::is_x86_feature_detected!("avx512bf16"),
            std::is_x86_feature_detected!("avxvnni"),
            std::is_x86_feature_detected!("avx512fp16"),
            raw.edx & (1 << 25) != 0,
            raw.edx & (1 << 24) != 0,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    println!(
        "{{\"arch\":\"{}\",\"x86_features\":false}}",
        std::env::consts::ARCH
    );
}
