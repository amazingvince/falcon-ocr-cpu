//! Minimal dynamic binding to the pinned AOCL-DLP classic FP32 GEMM ABI.
//! This module is diagnostic-only and does not add a production dependency.

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// include/classic/aocl_gemm_interface_apis.h at the pinned revision declares
// this exact C signature; include/classic/dlp_base_types.h defines md_t=int64_t.
type Gemm = unsafe extern "C" fn(
    c_char,
    c_char,
    c_char,
    i64,
    i64,
    i64,
    f32,
    *const f32,
    i64,
    c_char,
    *const f32,
    i64,
    c_char,
    f32,
    *mut f32,
    i64,
    *mut c_void,
);

pub struct Aocl {
    // The library remains loaded for the lifetime of the function pointer.
    _library: platform::Library,
    gemm: Gemm,
    digest: String,
}

impl Aocl {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self> {
        ensure!(
            expected_sha256.len() == 64 && expected_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "expected library SHA-256 must contain 64 hexadecimal digits"
        );
        let path = path.canonicalize().context("resolve AOCL library path")?;
        let mut file = platform::open_library_file(&path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let digest = format!("{:x}", hash.finalize());
        ensure!(
            digest.eq_ignore_ascii_case(expected_sha256),
            "AOCL library SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        );
        file.seek(SeekFrom::Start(0))?;
        // Hash validation completes before any native library initializer runs.
        let library = platform::Library::load(&path, file)?;
        let symbol = library.symbol(c"aocl_gemm_f32f32f32of32")?;
        // SAFETY: The non-null symbol comes from the explicitly hash-approved
        // pinned library, whose C prototype is recorded above. Both supported
        // platforms represent C function pointers as pointer-sized addresses.
        // `library` is retained in Self and cannot unload before this pointer.
        let gemm: Gemm = unsafe { std::mem::transmute(symbol) };
        Ok(Self {
            _library: library,
            gemm,
            digest,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.digest
    }

    /// C[m,n] = alpha * A[m,k] * W[n,k]^T + beta * C[m,n]. All matrices are
    /// contiguous row-major. Zero-size calls are rejected rather than exposing
    /// native pointer/leading-dimension edge cases this probe does not qualify.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        weight: &[f32],
        channels: usize,
        output: &mut [f32],
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        ensure!(
            rows > 0 && width > 0 && channels > 0,
            "AOCL dimensions must be positive"
        );
        ensure!(
            alpha.is_finite() && beta.is_finite(),
            "AOCL alpha and beta must be finite"
        );
        let a = rows.checked_mul(width).context("AOCL A shape overflow")?;
        let b = channels
            .checked_mul(width)
            .context("AOCL W shape overflow")?;
        let c = rows
            .checked_mul(channels)
            .context("AOCL C shape overflow")?;
        ensure!(input.len() == a, "AOCL A shape mismatch");
        ensure!(weight.len() == b, "AOCL W shape mismatch");
        ensure!(output.len() == c, "AOCL C shape mismatch");
        let (m, n, k) = (
            i64::try_from(rows)?,
            i64::try_from(channels)?,
            i64::try_from(width)?,
        );
        // SAFETY: Exact checked lengths cover every row-major strided access.
        // Borrowing excludes input/output aliasing. `md_t` conversions cannot
        // truncate. R/N/T interprets W[n,k] as the transposed right operand;
        // normal-memory flags N disable packed-memory assumptions. Null metadata
        // requests no post-operations. Native execution is synchronous and the
        // loaded library/function pointer and all three slices remain alive.
        unsafe {
            (self.gemm)(
                b'R' as c_char,
                b'N' as c_char,
                b'T' as c_char,
                m,
                n,
                k,
                alpha,
                input.as_ptr(),
                k,
                b'N' as c_char,
                weight.as_ptr(),
                k,
                b'N' as c_char,
                beta,
                output.as_mut_ptr(),
                n,
                std::ptr::null_mut(),
            );
        }
        Ok(())
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::CStr;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::OpenOptionsExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(path: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub fn open_library_file(path: &Path) -> Result<File> {
        // FILE_SHARE_READ denies concurrent replacement/modification while the
        // handle is retained, closing the hash-to-load path race on Windows.
        Ok(std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(path)?)
    }

    pub struct Library {
        handle: *mut c_void,
        _verified_file: File,
    }

    impl Library {
        pub fn load(path: &Path, verified_file: File) -> Result<Self> {
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            ensure!(!wide.contains(&0), "library path contains NUL");
            wide.push(0);
            // SAFETY: `wide` is a NUL-terminated absolute path and stays alive
            // for this synchronous call. Search only its directory and System32
            // for dependencies; no process-global search path is changed.
            let handle =
                unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0x100 | 0x800) };
            if handle.is_null() {
                return Err(std::io::Error::last_os_error()).context("LoadLibraryExW AOCL");
            }
            Ok(Self {
                handle,
                _verified_file: verified_file,
            })
        }

        pub fn symbol(&self, name: &CStr) -> Result<*mut c_void> {
            // SAFETY: This owns a live LoadLibrary handle and name is terminated.
            let symbol = unsafe { GetProcAddress(self.handle, name.as_ptr()) };
            if symbol.is_null() {
                return Err(std::io::Error::last_os_error())
                    .context("GetProcAddress AOCL FP32 GEMM");
            }
            Ok(symbol)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: This unique owner releases its successful load exactly once.
            unsafe {
                FreeLibrary(self.handle);
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
        fn dlerror() -> *const c_char;
    }

    pub fn open_library_file(path: &Path) -> Result<File> {
        Ok(File::open(path)?)
    }
    pub struct Library {
        handle: *mut c_void,
        _verified_file: File,
    }

    fn last_error() -> String {
        // SAFETY: dlerror returns either null or a thread-local terminated string,
        // which is copied immediately before another loader call can replace it.
        unsafe {
            let error = dlerror();
            if error.is_null() {
                "no dynamic loader detail".to_owned()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        }
    }

    impl Library {
        pub fn load(path: &Path, verified_file: File) -> Result<Self> {
            let name = CString::new(path.as_os_str().as_bytes())?;
            // SAFETY: name is a stable terminated absolute path. RTLD_NOW resolves
            // symbols synchronously and default RTLD_LOCAL avoids global exports.
            // Linux callers must not modify/replace this trusted library during
            // the probe; the retained read handle is not a mandatory write lock.
            let handle = unsafe { dlopen(name.as_ptr(), 2) };
            ensure!(!handle.is_null(), "dlopen AOCL: {}", last_error());
            Ok(Self {
                handle,
                _verified_file: verified_file,
            })
        }
        pub fn symbol(&self, name: &CStr) -> Result<*mut c_void> {
            // SAFETY: handle is live and name is terminated for the synchronous call.
            let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
            ensure!(!symbol.is_null(), "dlsym AOCL: {}", last_error());
            Ok(symbol)
        }
    }
    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: This unique owner releases its successful load exactly once.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
compile_error!("The isolated AOCL probe supports Windows and Linux dynamic loaders only");
