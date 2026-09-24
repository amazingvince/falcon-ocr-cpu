//! Host CPU topology for default thread counts.
//!
//! Prefill is compute-bound and gains from every logical CPU (SMT siblings
//! included). Decode streams weights and caches from memory, where two SMT
//! siblings of one core only contend, so its default is one thread per
//! physical core. On a Ryzen 9 7950X (16 cores, 32 threads) 32 prefill threads
//! are 22% faster than 16, while 32 decode threads are 60% slower than 16.

/// Logical CPUs available to this process (at least 1).
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Physical cores, or the logical CPU count when the topology is unknown.
pub fn physical_cores() -> usize {
    let logical = logical_cpus();
    detect().filter(|&n| n > 0).map_or(logical, |n| n.min(logical))
}

#[cfg(windows)]
fn detect() -> Option<usize> {
    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION is 32 bytes on 64-bit Windows:
    // ULONG_PTR mask, u32 relationship (+4 padding), 16-byte union.
    const ENTRY: usize = 32;
    const RELATION_PROCESSOR_CORE: u32 = 0;
    unsafe extern "system" {
        fn GetLogicalProcessorInformation(buffer: *mut u8, length: *mut u32) -> i32;
    }
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let mut length = 0u32;
    // SAFETY: a null buffer only queries the required length.
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut length) };
    if length == 0 {
        return None;
    }
    let mut buffer = vec![0u8; length as usize];
    // SAFETY: the buffer holds `length` bytes, as the call requires.
    if unsafe { GetLogicalProcessorInformation(buffer.as_mut_ptr(), &mut length) } == 0 {
        return None;
    }
    let cores = buffer[..length as usize]
        .chunks_exact(ENTRY)
        .filter(|entry| u32::from_le_bytes(entry[8..12].try_into().unwrap()) == RELATION_PROCESSOR_CORE)
        .count();
    Some(cores)
}

#[cfg(target_os = "linux")]
fn detect() -> Option<usize> {
    let mut cores = std::collections::HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) || name.len() == 3 {
            continue;
        }
        let topology = entry.path().join("topology");
        let read = |file: &str| std::fs::read_to_string(topology.join(file)).ok();
        if let (Some(package), Some(core)) = (read("physical_package_id"), read("core_id")) {
            cores.insert((package.trim().to_owned(), core.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(target_os = "macos")]
fn detect() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> i32;
    }
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    // SAFETY: the name is NUL-terminated and `value` has `size` bytes.
    let status = unsafe {
        sysctlbyname(
            c"hw.physicalcpu".as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn detect() -> Option<usize> {
    None
}

/// Physical performance cores on a hybrid CPU (Intel P-cores, Arm big
/// cores, Apple performance cores): `None` when every core is alike, when
/// the OS does not say, or when the answer is not below the core count.
pub fn performance_cores() -> Option<usize> {
    detect_performance().filter(|&p| p > 0 && p < physical_cores())
}

#[cfg(windows)]
fn detect_performance() -> Option<usize> {
    const RELATION_PROCESSOR_CORE: u32 = 0;
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: u32, buffer: *mut u8, length: *mut u32) -> i32;
    }
    let mut length = 0u32;
    // SAFETY: a null buffer only queries the required length.
    unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut length) };
    if length == 0 {
        return None;
    }
    let mut buffer = vec![0u8; length as usize];
    // SAFETY: the buffer holds `length` bytes, as the call requires.
    if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer.as_mut_ptr(), &mut length) } == 0 {
        return None;
    }
    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: u32 Relationship, u32 Size,
    // then PROCESSOR_RELATIONSHIP { u8 Flags, u8 EfficiencyClass, .. }; a
    // higher class is a faster core.
    let bytes = &buffer[..length as usize];
    let mut classes = Vec::new();
    let mut offset = 0;
    while offset + 10 <= bytes.len() {
        let relationship = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if size == 0 {
            return None;
        }
        if relationship == RELATION_PROCESSOR_CORE {
            classes.push(bytes[offset + 9]);
        }
        offset += size;
    }
    let top = *classes.iter().max()?;
    if classes.iter().all(|&c| c == top) {
        return None;
    }
    Some(classes.iter().filter(|&&c| c == top).count())
}

#[cfg(target_os = "linux")]
fn detect_performance() -> Option<usize> {
    // Intel hybrid CPUs list the P-cores under the core PMU; Arm DynamIQ
    // systems expose a capacity per CPU. Either way, count physical cores
    // among the fastest CPUs.
    let cpus: Vec<usize> = match std::fs::read_to_string("/sys/devices/cpu_core/cpus") {
        Ok(list) => parse_cpu_list(&list),
        Err(_) => {
            let mut capacities = Vec::new();
            for cpu in 0..logical_cpus() * 2 {
                let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
                match std::fs::read_to_string(path) {
                    Ok(text) => capacities.push((cpu, text.trim().parse::<u64>().ok()?)),
                    Err(_) if cpu >= logical_cpus() => break,
                    Err(_) => continue,
                }
            }
            let top = capacities.iter().map(|&(_, c)| c).max()?;
            if capacities.iter().all(|&(_, c)| c == top) {
                return None;
            }
            capacities
                .iter()
                .filter(|&&(_, c)| c == top)
                .map(|&(cpu, _)| cpu)
                .collect()
        }
    };
    let mut cores = std::collections::HashSet::new();
    for cpu in cpus {
        let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let read = |file: &str| std::fs::read_to_string(format!("{topology}/{file}")).ok();
        if let (Some(package), Some(core)) = (read("physical_package_id"), read("core_id")) {
            cores.insert((package.trim().to_owned(), core.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// CPUs in a sysfs list such as `0-7,16-23`.
#[cfg(target_os = "linux")]
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    cpus.extend(a..=b);
                }
            }
            None => cpus.extend(part.parse::<usize>().ok()),
        }
    }
    cpus
}

#[cfg(target_os = "macos")]
fn detect_performance() -> Option<usize> {
    // Apple silicon reports its performance level 0 (P-cores) separately.
    (sysctl_i32(c"hw.nperflevels")? > 1)
        .then(|| sysctl_i32(c"hw.perflevel0.physicalcpu"))
        .flatten()
}

#[cfg(target_os = "macos")]
fn sysctl_i32(name: &std::ffi::CStr) -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            old_len: *mut usize,
            new: *mut std::ffi::c_void,
            new_len: usize,
        ) -> i32;
    }
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    // SAFETY: the name is NUL-terminated and `value` has `size` bytes.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn detect_performance() -> Option<usize> {
    None
}

/// Memory read bandwidth, GB/s per pass: `bytes` of memory (touched first)
/// summed by `threads` threads, `passes` times.
pub fn read_bandwidth_gb_s(bytes: usize, threads: usize, passes: usize) -> Vec<f64> {
    use rayon::prelude::*;
    let words = (bytes / 8).max(1);
    let data: Vec<u64> = (0..words as u64).collect();
    let threads = threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("bandwidth probe pool");
    let chunk = words.div_ceil(threads * 4).max(1024);
    (0..passes)
        .map(|_| {
            let started = std::time::Instant::now();
            let total = pool.install(|| {
                data.par_chunks(chunk)
                    .map(|part| {
                        let mut acc = [0u64; 4];
                        for lanes in part.chunks_exact(4) {
                            for (a, v) in acc.iter_mut().zip(lanes) {
                                *a = a.wrapping_add(*v);
                            }
                        }
                        acc.iter().fold(0u64, |a, &b| a.wrapping_add(b))
                    })
                    .reduce(|| 0u64, u64::wrapping_add)
            });
            std::hint::black_box(total);
            (words * 8) as f64 / started.elapsed().as_secs_f64() / 1e9
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn bandwidth_probe_returns_one_positive_figure_per_pass() {
        let passes = super::read_bandwidth_gb_s(8 << 20, 2, 3);
        assert_eq!(passes.len(), 3);
        assert!(passes.iter().all(|&g| g.is_finite() && g > 0.0), "{passes:?}");
    }

    #[test]
    fn physical_cores_are_positive_and_at_most_logical() {
        let (physical, logical) = (super::physical_cores(), super::logical_cpus());
        assert!(physical >= 1 && physical <= logical, "{physical} of {logical}");
        if let Some(performance) = super::performance_cores() {
            assert!(
                performance >= 1 && performance < physical,
                "{performance} of {physical}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_lists_parse() {
        assert_eq!(super::parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert!(super::parse_cpu_list("").is_empty());
    }
}
