//! Host CPU topology for default thread counts.
//!
//! Prefill is compute-bound and gains from every logical CPU (SMT siblings
//! included). Decode streams weights and caches from memory, where two SMT
//! siblings of one core only contend, so its default is one thread per
//! physical core. On a Ryzen 9 7950X (16 cores, 32 threads) 32 prefill threads
//! are 22% faster than 16, while 32 decode threads are 60% slower than 16.

/// Logical CPUs available to this process (at least 1).
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Physical cores, or the logical CPU count when the topology is unknown.
pub fn physical_cores() -> usize {
    let logical = logical_cpus();
    detect()
        .filter(|&n| n > 0)
        .map_or(logical, |n| n.min(logical))
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
        .filter(|entry| {
            u32::from_le_bytes(entry[8..12].try_into().unwrap()) == RELATION_PROCESSOR_CORE
        })
        .count();
    Some(cores)
}

#[cfg(target_os = "linux")]
fn detect() -> Option<usize> {
    let mut cores = std::collections::HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu")
            || !name[3..].chars().all(|c| c.is_ascii_digit())
            || name.len() == 3
        {
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

#[cfg(test)]
mod tests {
    #[test]
    fn physical_cores_are_positive_and_at_most_logical() {
        let (physical, logical) = (super::physical_cores(), super::logical_cpus());
        assert!(
            physical >= 1 && physical <= logical,
            "{physical} of {logical}"
        );
    }
}
